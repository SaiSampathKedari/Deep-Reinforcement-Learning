"""DQN and Double DQN, for discrete actions numbered 0 to num_actions - 1.

The idea
--------
A Q-network takes a batch of B observations and returns one number per
action, shaped ``(B, num_actions)``. Each number, Q(s, a), estimates the
discounted return of taking action a in state s and acting well afterwards.
Choosing an action is then easy: take the one with the largest Q-value.

* While collecting data, the behaviour policy is epsilon-greedy: with
  probability epsilon it takes a uniformly random action (to explore),
  otherwise the greedy one.
* For evaluation and deployment, the inference policy takes the greedy
  action, without epsilon.

Learning
--------
For each replay minibatch, push Q(s, a) of the stored action toward a
Bellman target::

    target = reward + gamma * (1 - terminated) * next_q

* DQN: next_q is the largest next-state Q-value, according to the target
  network.
* Double DQN: the online network picks the next action and the target
  network scores it. Taking the max over noisy estimates favours the ones
  that happen to be too high, so plain DQN tends to overestimate values.
  Choosing with one network and scoring with another reduces that bias.

Only a true termination (the episode really ended, for example the pole fell)
sets next_q to 0. A truncation (for example a time limit) still
bootstraps: the episode was cut off, not finished, and the collector stored
the true next observation.

Target network
--------------
The target network is a lagging copy of the online network, so that an
update does not also move the target it is chasing. ``update()`` trains only
the online network; ``update_targets()`` moves the target toward it once;
the trainer decides when to call it. The defaults (``tau=0.005``,
``target_update_interval=1``) take a small soft step after every completed
minibatch update. The 2015 DQN paper instead copied the whole network every
10,000 updates; set ``tau=1`` and ``target_update_interval=10_000`` for that.

References: Mnih et al., 2015 (DQN); van Hasselt et al., 2016 (Double DQN).
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from math import isfinite
from numbers import Real
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from deeprl import learner_state
from deeprl.algorithms.off_policy import OffPolicyConfig, OffPolicyLearner
from deeprl.buffers import ReplayBatch
from deeprl.data_spec import TensorData
from deeprl.normalization import RunningNormalizer
from deeprl.processors import ObservationProcessor
from deeprl.stats import TrainingProgress, TrainingStats


@dataclass(frozen=True)
class DQNConfig(OffPolicyConfig):
    """DQN settings: optimization, Double DQN and epsilon-greedy exploration.

    Run length, replay, the update schedule and the target-network settings
    (``tau``, ``target_update_interval``) come from ``OffPolicyConfig``. Their
    defaults take a small soft target step after every completed minibatch
    update (not every environment step).

    Attributes:
        lr_q: Learning rate for the online Q-network's optimizer. 0 is
            allowed: gradients are still computed and the optimizer still
            runs, but standard optimizers such as SGD and Adam then leave
            the weights unchanged.
        max_grad_norm: Largest allowed global L2 norm of the online
            network's gradients. Why: a few transitions with large TD
            errors can produce one huge gradient. Clipping scales the whole
            gradient down uniformly until its global L2 norm is at most
            this value. The optimizer then turns that gradient into the
            actual parameter update (with Adam, the update size is not
            simply proportional to the gradient).
        double_q: True for Double DQN (the online network picks the next
            action, the target network scores it). False for DQN (the
            target network does both).
        exploration_initial_eps: Epsilon at the start of training, in [0, 1].
            1 means every action is random.
        exploration_final_eps: Epsilon after ``exploration_steps``, in
            [0, 1]. Usually a little above 0 (default 0.05), so the agent
            never stops exploring completely.
        exploration_steps: Number of collected transitions (across all
            environments, counted from the start of the run, warm-up
            included) over which epsilon falls in a straight line from the
            initial to the final value. 0 uses the final value right away.
            Separately, ``behaviour_policy`` takes only random actions until
            ``learning_starts`` transitions have been collected.
        q_optimizer_class: PyTorch optimizer class for the online network.
        q_optimizer_kwargs: Extra optimizer arguments, for example Adam's
            ``eps``. Not ``params`` or ``lr``: the learner passes the online
            network's parameters itself, and the learning rate comes only
            from ``lr_q``.
    """

    lr_q            : float = 1e-4
    max_grad_norm   : float = 10.0
    double_q        : bool  = False   # False: DQN. True: Double DQN.

    # Linear epsilon schedule over a fixed number of collected transitions.
    exploration_initial_eps : float = 1.0
    exploration_final_eps   : float = 0.05
    exploration_steps       : int   = 100_000  # 0 skips the decay; warm-up still applies.

    q_optimizer_class  : type[torch.optim.Optimizer] = torch.optim.Adam
    q_optimizer_kwargs : dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Check the shared settings, then the DQN settings.

        Raises:
            ValueError: If a numeric setting is out of range, or
                ``q_optimizer_kwargs`` contains ``params`` or ``lr``.
            TypeError: If a device, ``double_q`` or an optimizer setting
                has the wrong type.
        """
        # 0. The shared off-policy settings (counts, gamma, tau, devices).
        super().__post_init__()

        # 1. Learning settings and epsilons must be finite numbers. The
        #    learning rate may be 0; the gradient-norm limit must be
        #    positive (0 would erase every gradient).
        for name in (
            "lr_q", "max_grad_norm",
            "exploration_initial_eps", "exploration_final_eps",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number")

        if self.lr_q < 0 or self.max_grad_norm <= 0:
            raise ValueError(
                "lr_q must be nonnegative and max_grad_norm positive"
            )

        # 2. Epsilon is a probability, so it must be in [0, 1]. The schedule
        #    length counts transitions, so it is a whole number >= 0.
        for name in ("exploration_initial_eps", "exploration_final_eps"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")

        if (
            isinstance(self.exploration_steps, bool)
            or not isinstance(self.exploration_steps, int)
            or self.exploration_steps < 0
        ):
            raise ValueError("exploration_steps must be an integer >= 0")

        # 3. The Double DQN flag and the optimizer must have the declared
        #    types. params and lr are passed by the learner, so they must
        #    not also appear in the extra arguments.
        if not isinstance(self.double_q, bool):
            raise TypeError("double_q must be boolean")

        if not isinstance(self.q_optimizer_class, type) or not issubclass(
            self.q_optimizer_class, torch.optim.Optimizer
        ):
            raise TypeError(
                "q_optimizer_class must be a PyTorch optimizer class"
            )

        if not isinstance(self.q_optimizer_kwargs, dict):
            raise TypeError("q_optimizer_kwargs must be a dictionary")

        if {"params", "lr"} & self.q_optimizer_kwargs.keys():
            raise ValueError(
                "q_optimizer_kwargs must not contain params or lr"
            )


class DQNLearner(OffPolicyLearner):
    """Learn action values from replay, with an online and a target Q-network.

    There is no separate actor: the online Q-network supplies both the
    epsilon-greedy actions used for collection and the greedy actions used
    for evaluation and deployment.

    Observation processing (selection, normalization) lives outside both
    networks, in one shared ``obs_processor``. The trainer updates its
    statistics during collection and saves them separately from this
    learner's state. Replay keeps raw observations; this learner processes
    each sampled minibatch when it uses it.

    Wiring it up. The trainer needs two policies around the same network and
    processor: ``learner.behaviour_policy`` (explores) and an
    ``InferencePolicy`` (greedy)::

        learner = DQNLearner(q_network, cfg, obs_processor=processor)
        inference_policy = InferencePolicy(
            learner.q_network,
            observation_processor=learner.obs_processor,
            action_selector=greedy_action,
        )
        train_off_policy(
            envs=envs,
            learner=learner,
            behaviour_policy=learner.behaviour_policy,
            inference_policy=inference_policy,
        )

    Attributes:
        cfg: DQN settings, including the shared off-policy schedule.
        q_network: The online network, trained by the optimizer. Maps a
            batch of observations to ``(B, num_actions)`` Q-values.
        target_q_network: The lagging copy used for Bellman targets. It has
            its own storage, no gradients, and stays in evaluation mode.
            Only ``update_targets`` (and loading a checkpoint) changes it;
            optimizer steps never do.
        obs_processor: The shared observation processor, or None for
            unprocessed input.
        optim_q: Optimizer for the online network only.
    """

    algorithm_id = "dqn"
    mutable_on_resume = OffPolicyLearner.mutable_on_resume | {"lr_q"}

    # ------------------------------------------------------------------
    # 0. Networks and optimizer
    # ------------------------------------------------------------------
    def __init__(
        self,
        q_network: nn.Module,
        cfg: DQNConfig,
        *,
        obs_processor: ObservationProcessor | None = None,
    ) -> None:
        """Build the learner: online network, target copy, optimizer.

        Args:
            q_network: Online network returning one Q-value per discrete
                action: raw values, not probabilities or a distribution.
            cfg: DQN optimization, exploration and training settings.
            obs_processor: Optional processor, shared with the inference
                policy. Pass normalization here, not inside q_network.

        Raises:
            ValueError: If q_network contains a ``RunningNormalizer`` or an
                ``ObservationProcessor``.
        """
        super().__init__(cfg)
        self.cfg: DQNConfig = cfg

        # 1. Observation normalization must live in one place: the shared
        #    obs_processor. Its statistics change only when the trainer
        #    explicitly calls its update() during collection, and the
        #    trainer saves them separately. A normalizer embedded in
        #    q_network would not receive the trainer's statistics updates,
        #    and copying the target network would duplicate its statistics.
        if any(
            isinstance(module, (RunningNormalizer, ObservationProcessor))
            for module in q_network.modules()
        ):
            raise ValueError(
                "use obs_processor instead of embedding observation normalization"
            )

        # 2. The target starts as an exact copy of the online network.
        #    deepcopy gives it separate storage, so optimizer steps on the
        #    online network do not change it. It is never trained, so its
        #    gradients are off; evaluation mode turns off training-only
        #    behaviour (such as dropout) so its targets are not noisy.
        self.q_network = q_network.to(cfg.device)
        self.target_q_network = (
            deepcopy(self.q_network).requires_grad_(False).eval()
        )

        # 3. Move the processor to the same device. ``.to`` moves it in
        #    place and returns the same object, so an InferencePolicy built
        #    with ``learner.obs_processor`` shares these statistics.
        self.obs_processor = (
            obs_processor.to(cfg.device) if obs_processor is not None else None
        )

        # 4. The optimizer sees only the online network. The target changes
        #    only through update_targets.
        self.optim_q = cfg.q_optimizer_class(
            self.q_network.parameters(),
            lr=cfg.lr_q,
            **cfg.q_optimizer_kwargs,
        )

    # ------------------------------------------------------------------
    # 1. Learner checkpoint state
    # ------------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        """Return the online network, target network and optimizer state.

        Network states include persistent buffers as well as parameters.
        The target is saved separately, not rebuilt from the online network
        on resume. Why: the target deliberately lags behind; resetting it to
        the online weights would make the next Bellman targets jump.

        This is only the learner's part of a checkpoint. The trainer adds
        progress, configs, observation statistics and random-number state.
        """
        return {
            "q_network": self.q_network.state_dict(),
            "target_q_network": self.target_q_network.state_dict(),
            "optim_q": self.optim_q.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a state made by ``state_dict`` into the existing components.

        The target gets its own saved weights back, not a copy of the
        online network. Observation statistics and progress are restored
        by the trainer.

        Loading is not all-or-nothing: if loading one component's tensors
        fails after earlier components were restored, build a new learner
        before trying again.

        Args:
            state: The saved learner state, not the whole checkpoint.

        Raises:
            ValueError: If a component key is missing or unexpected.
        """
        # 1. Reject missing or unexpected component keys before loading any
        #    tensors, so a malformed state fails without touching the
        #    networks. This does not prove the state came from DQN: the
        #    trainer separately checks the algorithm and the config.
        learner_state.validate_state_keys(
            state,
            {
                "q_network",
                "target_q_network",
                "optim_q",
            },
        )

        # 2. Copy the saved values into the existing networks and optimizer.
        self.q_network.load_state_dict(state["q_network"])
        self.target_q_network.load_state_dict(state["target_q_network"])
        self.optim_q.load_state_dict(state["optim_q"])

        # 3. Loading copies values only; gradient flags and train/eval mode
        #    are not part of a state dict. Set them again anyway, so the
        #    target is guaranteed to stay untrainable and in evaluation mode.
        self.target_q_network.requires_grad_(False).eval()

    # ------------------------------------------------------------------
    # 2. Resume configuration
    # ------------------------------------------------------------------
    def apply_resume_overrides(self, overrides: dict[str, Any]) -> None:
        """Apply a changed ``lr_q`` after the optimizer state was restored.

        The optimizer's saved state includes the old learning rate, so the
        new one is written into every parameter group afterwards. Other
        allowed changes, such as ``total_timesteps``, are read from the
        current config and need nothing here.

        Args:
            overrides: Changed fields and their new values, from
                ``validate_resume_config``.
        """
        if "lr_q" in overrides:
            for group in self.optim_q.param_groups:
                group["lr"] = overrides["lr_q"]

    # ------------------------------------------------------------------
    # 3. Collection: epsilon-greedy action selection
    # ------------------------------------------------------------------
    def exploration_rate(self, global_step: int) -> float:
        """Return epsilon after ``global_step`` collected transitions.

        Epsilon falls in a straight line, then stays at the final value::

            epsilon
              initial  *
                         *
                           *
              final          * * * * * * * *
                       0   exploration_steps   ->  global_step

        Example with the defaults (1.0 to 0.05 over 100,000 transitions):
        epsilon is 0.525 at 50,000 and 0.05 from 100,000 on.

        Why a fixed number of transitions, not a fraction of
        ``total_timesteps``? Raising ``total_timesteps`` to train longer on
        resume would otherwise change epsilon in the middle of a run. On
        resume the saved ``global_step`` simply continues the schedule.

        The initial random-only warm-up is applied by ``behaviour_policy``,
        not here.

        Args:
            global_step: Transitions collected so far, across all
                environments.

        Raises:
            ValueError: If global_step is not a nonnegative integer.
        """
        # 1. The schedule needs a real transition count.
        if (
            isinstance(global_step, bool)
            or not isinstance(global_step, int)
            or global_step < 0
        ):
            raise ValueError("global_step must be a nonnegative integer")

        # 2. How far along the schedule we are: 0 at the start, 1 at the end
        #    and afterwards. exploration_steps=0 means "already at the end".
        fraction = (
            min(global_step / self.cfg.exploration_steps, 1.0)
            if self.cfg.exploration_steps else 1.0
        )

        # 3. Move that fraction of the way from the initial to the final value.
        return self.cfg.exploration_initial_eps + fraction * (
            self.cfg.exploration_final_eps - self.cfg.exploration_initial_eps
        )

    @torch.no_grad()
    def behaviour_policy(
        self,
        observations: TensorData,
        global_step: int,
    ) -> torch.Tensor:
        """Choose epsilon-greedy actions for N environments.

        Pass this method to the trainer as ``behaviour_policy``. It takes
        raw observations (tensor leaves shaped ``(N, *feature_shape)``) and
        returns one action index per environment, shaped ``(N,)``, dtype
        torch.int64.

        Each environment decides on its own: with probability epsilon it
        takes a uniformly random action, otherwise the greedy one (argmax
        Q). A random action can also happen to be the greedy one.

        Warm-up: while ``global_step`` (transitions collected before this
        vector step) is below ``learning_starts``, epsilon is forced to 1.
        Why: the network is still untrained, and its greedy choices are
        arbitrary but systematic (often the same action again and again).
        Uniform random actions give the replay buffer a broader first
        dataset. After a resume, ``global_step`` is already past
        ``learning_starts``, so the replay refill does not restart this
        warm-up or the epsilon schedule.

        No gradients are recorded (acting needs none), and the processor's
        statistics are not changed: the trainer updates them after the step.
        This leaves the online network in evaluation mode; ``update`` sets
        the modes it needs.

        Args:
            observations: Raw observations for the N environments.
            global_step: Transitions collected so far, across all
                environments.

        Returns:
            Action indices shaped ``(N,)``.

        Raises:
            ValueError: If the network output is not ``(N, num_actions)``
                with at least one action.
        """
        # 1. Choose epsilon from the cumulative transition count, and force
        #    fully random actions during the initial warm-up.
        epsilon = self.exploration_rate(global_step)
        if global_step < self.cfg.learning_starts:
            epsilon = 1.0

        # 2. Evaluation mode (no dropout while acting). Process the raw
        #    observations the same way the learner and inference policy do,
        #    without updating the normalization statistics.
        self.q_network.eval()

        if self.obs_processor is not None:
            observations = self.obs_processor(observations, group="actor")

        # 3. Compute Q(s, a) for every action and take the greedy actions.
        q_values = self.q_network(observations)

        if q_values.ndim != 2 or q_values.shape[1] == 0:
            raise ValueError("q_network must return (N, num_actions)")

        actions = q_values.argmax(dim=1)

        # 4. Explore: each environment independently replaces its greedy
        #    action with a random one, with probability epsilon.
        if epsilon > 0:
            random_actions = torch.randint(
                q_values.shape[1],
                actions.shape,
                device=actions.device,
            )
            explore = torch.rand(
                actions.shape,
                device=actions.device,
            ) < epsilon
            actions = torch.where(explore, random_actions, actions)

        return actions

    # ------------------------------------------------------------------
    # 4. Target-network updates
    # ------------------------------------------------------------------
    @torch.no_grad()
    def update_targets(self) -> None:
        """Move the target network one step toward the online network.

        The trainer calls this after every ``target_update_interval``
        completed minibatch updates; this method keeps no schedule of its
        own and takes no optimizer step.

        * Parameters (weights, biases) are blended:
          ``target = (1 - tau) * target + tau * online``. ``tau=1`` gives
          an exact copy; a small tau moves the target a little each time.
        * Buffers (state that is not trained by gradients, such as BatchNorm
          running statistics and its integer batch counter) are copied
          directly, even when tau < 1. Integers cannot be blended, and
          copying keeps the target's normalization the same as the online
          network's. Stable-Baselines3's DQN copies BatchNorm statistics the
          same way.
        """
        # 1. Blend the parameters. lerp_ computes
        #    target + tau * (online - target), in place.
        for target, online in zip(
            self.target_q_network.parameters(),
            self.q_network.parameters(),
            strict=True,
        ):
            target.lerp_(online, self.cfg.tau)

        # 2. Copy the buffers.
        for target, online in zip(
            self.target_q_network.buffers(),
            self.q_network.buffers(),
            strict=True,
        ):
            target.copy_(online)

        # 3. Keep the target in evaluation mode.
        self.target_q_network.eval()

    # ------------------------------------------------------------------
    # 5. One replay-minibatch update
    # ------------------------------------------------------------------
    def update(
        self,
        batch: ReplayBatch,
        *,
        progress: TrainingProgress,
    ) -> TrainingStats:
        """Fit the online Q-values to Bellman targets on one replay minibatch.

        Steps:

        0. Check the batch fields.
        1. Build the targets (DQN or Double DQN) without gradients. Only a
           true termination removes the next-state value; truncations
           bootstrap.
        2. Compute Q(s, a) for each stored action and the Huber loss
           against the targets.
        3. Clip the online network's gradients and take one optimizer step.

        The targets are constants for the loss (no gradient flows into
        them). Why: the goal is to move Q(s, a) toward the target, not the
        target toward Q(s, a). Current and next observations are processed
        with the same statistics, which do not change during the update.
        Target refreshes happen in the trainer, after this method returns.

        Args:
            batch: B replay transitions on ``cfg.device``. Observations are
                raw; actions are integer indices shaped ``(B,)``. Rewards
                and the boolean termination/truncation flags are ``(B,)``.
            progress: Trainer counters before this update. Part of the
                shared learner interface; DQN does not use them.

        Returns:
            Metrics, with ``gradient_steps=1``:

            * ``losses/q_loss``: the Huber loss.
            * ``values/q_mean``: mean Q(s, a) of the stored actions.
            * ``values/target_mean``: mean Bellman target.
            * ``diagnostics/td_error_abs``: mean ``|target - Q(s, a)|``.
            * ``grads/q_norm``: gradient norm, measured before clipping.

            Afterwards the online network is in training mode and the target
            network in evaluation mode.

        Raises:
            ValueError: If a field has the wrong shape, the batch is empty,
                the network output has the wrong shape, or an action index
                is out of range.
            TypeError: If actions are not int32/int64 or the flags are not
                boolean.
        """
        # 0. Check the replay fields. Shape mistakes are dangerous here
        #    because PyTorch broadcasting does not always raise: a (B,)
        #    tensor combined with a (B, 1) tensor silently becomes (B, B).
        #    Actions must be flat integer indices, not (B, 1).
        if not isinstance(batch.actions, torch.Tensor) or batch.actions.ndim != 1:
            raise ValueError("actions must be an integer tensor shaped (B,)")

        if batch.actions.dtype not in (torch.int32, torch.int64):
            raise TypeError("actions must have int32 or int64 dtype")

        batch_size = batch.actions.shape[0]
        if batch_size == 0:
            raise ValueError("cannot update from an empty batch")

        for name in ("rewards", "terminations", "truncations"):
            if getattr(batch, name).shape != (batch_size,):
                raise ValueError(f"{name} must have shape (B,)")

        if (
            batch.terminations.dtype != torch.bool
            or batch.truncations.dtype != torch.bool
        ):
            raise TypeError("terminations and truncations must be boolean")

        # 1. Build the targets in evaluation mode (no dropout noise) and
        #    without gradients (they are constants for the loss).
        self.q_network.eval()
        self.target_q_network.eval()

        with torch.no_grad():
            # Terminated transitions keep next_value = 0: there is no next
            # state to bootstrap from. Truncated ones are nonterminal.
            next_values = torch.zeros_like(batch.rewards)
            nonterminal = ~batch.terminations

            # Skip the network calls when their result would be multiplied
            # by zero anyway (gamma = 0, or every transition terminated).
            if self.cfg.gamma > 0 and nonterminal.any().item():
                next_observations = batch.next_observations[nonterminal]

                if self.obs_processor is not None:
                    next_observations = self.obs_processor(
                        next_observations,
                        group="actor",
                    )

                # K is the number of nonterminal transitions in this batch.
                target_q_values = self.target_q_network(next_observations)  # (K, num_actions)

                if self.cfg.double_q:
                    # Double DQN: the online network picks the next action,
                    # the target network scores it.
                    next_actions = self.q_network(
                        next_observations
                    ).argmax(dim=1)

                    next_values[nonterminal] = target_q_values.gather(
                        1,
                        next_actions[:, None],
                    ).squeeze(1)
                else:
                    # DQN: the target network both picks and scores.
                    next_values[nonterminal] = target_q_values.max(
                        dim=1
                    ).values

            targets = batch.rewards + self.cfg.gamma * next_values  # (B,)

        # 2. Back in training mode, compute Q(s, a) for the stored actions,
        #    with gradients this time.
        self.q_network.train()
        observations = batch.observations

        if self.obs_processor is not None:
            observations = self.obs_processor(
                observations,
                group="actor",
            )

        q_values = self.q_network(observations)  # (B, num_actions)

        if (
            q_values.ndim != 2
            or q_values.shape[0] != batch_size
            or q_values.shape[1] == 0
        ):
            raise ValueError("q_network must return (B, num_actions)")

        # Each stored action must name one of the network's output columns.
        actions = batch.actions.long()
        if ((actions < 0) | (actions >= q_values.shape[1])).any().item():
            raise ValueError("actions must index the Q-network's action columns")

        # Pick, in each row, the Q-value of the action that was taken.
        selected_q = q_values.gather(
            1,
            actions[:, None],
        ).squeeze(1)  # (B,)

        # Huber loss (smooth L1, threshold 1): squared error for small TD
        # errors, absolute error for large ones. Its derivative with respect
        # to each predicted Q-value is at most 1 in size, so large TD errors
        # pull less hard than with a squared loss. (The gradients of the
        # weights also depend on the network itself, so this is not a hard
        # limit on them.)
        q_loss = F.smooth_l1_loss(selected_q, targets)

        # 3. One optimizer step on the online network only. The norm is
        #    measured before clipping, so a logged value above
        #    max_grad_norm means the clip was active.
        self.optim_q.zero_grad(set_to_none=True)
        q_loss.backward()

        grad_norm = nn.utils.clip_grad_norm_(
            self.q_network.parameters(),
            self.cfg.max_grad_norm,
        )
        self.optim_q.step()

        return TrainingStats(
            metrics={
                "losses/q_loss": q_loss.item(),
                "values/q_mean": selected_q.detach().mean().item(),
                "values/target_mean": targets.mean().item(),
                "diagnostics/td_error_abs": (
                    targets - selected_q.detach()
                ).abs().mean().item(),
                "grads/q_norm": grad_norm.item(),
            },
            gradient_steps=1,
        )
