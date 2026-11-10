"""PPO-Clip: Proximal Policy Optimization with a clipped objective.

The idea
--------
A policy-gradient step is only trustworthy while the policy being improved
stays close to the policy that collected the data. PPO wants to reuse one
rollout for several passes of minibatch gradient steps, so it needs a way to
stop the policy from drifting too far from the data. It does this by clipping
the probability ratio ``r = pi_theta(a|s) / pi_old(a|s)`` inside the
objective::

    L(theta) = mean( min( r * A,  clip(r, 1 - eps, 1 + eps) * A ) )

``A`` is the advantage of the stored action. Once an action's ratio leaves the
band ``[1 - eps, 1 + eps]`` in the direction its advantage favours, the
clipped term takes over and that action's gradient becomes zero. So one
rollout cannot push the policy arbitrarily far, however many epochs run over
it (Schulman et al., 2017).

Who does what
-------------
* ``train_on_policy`` (``training/``) collects each rollout and calls the
  learner.
* ``OnPolicyLearner`` (``on_policy.py``) owns the networks, the observation
  processor, the inference policy, and ``prepare_targets``: the GAE
  advantages and value targets, computed once per rollout with the networks
  that collected it, and then held fixed.
* This module owns PPO's own part, ``update``: ``update_epochs`` passes over
  the rollout in shuffled minibatches, each with one critic step and one
  clipped policy step.

Options, and where each comes from
----------------------------------
Each option matches a known reference implementation, so recipes from those
libraries can be reproduced:

* ``value_clip_range``: a clipped value loss (OpenAI Baselines, CleanRL,
  RSL-RL).
* ``normalize_advantage``: standardize advantages, over the whole rollout
  (RSL-RL) or per minibatch (SB3, CleanRL).
* ``target_kl``: stop an update early once the policy has moved too far (SB3,
  CleanRL).
* ``lr_schedule="adaptive"``: raise or lower the learning rate so each step
  moves the policy by about ``desired_kl`` (RSL-RL; the default in Isaac Lab
  locomotion recipes).
* ``log_ratio_clip``: a numerical guard against overflow in ``exp``.

The actor and critic have separate optimizers, learning rates and gradient
clips, because they are separate networks. RSL-RL instead uses one optimizer
for both; that is a known difference when comparing against it.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from math import isfinite
from numbers import Real
from typing import Any, Literal

import torch
import torch.nn as nn
from torch.distributions import Distribution, Independent, Normal, kl_divergence

from deeprl import learner_state
from deeprl.algorithms.on_policy import (
    OnPolicyConfig,
    OnPolicyLearner,
)
from deeprl.buffers import RolloutBuffer
from deeprl.inference import InferencePolicy
from deeprl.normalization import RewardScaler
from deeprl.processors import ActionProcessor, ObservationProcessor
from deeprl.stats import TrainingStats
from deeprl.utils import explained_variance


def _normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
    """Standardize advantages to mean 0 and standard deviation 1.

    Why: the raw advantages carry the scale of the rewards, so without this
    the size of a policy step would depend on how the task's rewards happen
    to be scaled. After standardizing, about half the actions are pushed up
    and half down, by comparable amounts.

    The population standard deviation (dividing by n, not n - 1) is used
    everywhere: for the whole rollout and for single minibatches.

    Args:
        advantages: A whole rollout's or one minibatch's advantages.

    Returns:
        The standardized advantages, same shape. A single value is returned
        unchanged, because its standard deviation is undefined.
    """
    if advantages.numel() <= 1:
        return advantages
    return (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + 1e-8
    )


def _gaussian_action_stats(
    distribution: Distribution,
) -> tuple[torch.Tensor, int] | None:
    """Summarize a batch of Gaussian action distributions, for logging.

    Why: the standard deviation of a Gaussian policy is its exploration
    noise. Watching it shows whether exploration is collapsing (std towards
    0) or exploding, and the size of the means shows whether actions drift
    towards the edges of their range.

    Args:
        distribution: The policy's action distribution for one minibatch.

    Returns:
        ``None`` if the distribution is not a diagonal Gaussian
        (``Independent`` over ``Normal``). Otherwise ``(stats, count)``:
        ``stats`` holds ``(std sum, std min, std max, |mean| sum, |mean|
        max)`` over every action component of the batch, and ``count`` is
        the number of those components, so sums can later become means.
    """
    if not (
        isinstance(distribution, Independent)
        and isinstance(distribution.base_dist, Normal)
    ):
        return None

    std = distribution.stddev
    mean_abs = distribution.mean.abs()
    return torch.stack((
        std.sum(),
        std.min(),
        std.max(),
        mean_abs.sum(),
        mean_abs.max(),
    )), std.numel()


@dataclass(frozen=True)
class PPOConfig(OnPolicyConfig):
    """Settings for PPO-Clip.

    The run, rollout and GAE settings (``total_timesteps``, ``rollout_steps``,
    ``gamma``, ``gae_lambda``, ...) come from ``OnPolicyConfig``.

    Attributes:
        lr_critic: Learning rate of the critic's optimizer.
        lr_actor: Learning rate of the actor's optimizer.
        batch_size: M, the transitions per minibatch. Every epoch reshuffles
            the whole rollout of T * N transitions and splits it into
            minibatches of M; the last one may be smaller. A recipe written
            as "k minibatches per epoch" (RSL-RL's ``num_mini_batches``)
            corresponds to ``batch_size = T * N / k``.
        update_epochs: Passes over the rollout per update.
        max_grad_norm: Largest allowed global L2 norm of the gradients,
            applied separately to the actor and to the critic. Clipping
            scales the whole gradient down and keeps its direction.
        clip_range: epsilon, the half-width of the ratio band
            ``[1 - eps, 1 + eps]`` in the clipped policy objective.
        value_clip_range: ``None`` for the ordinary squared-error critic
            loss. Otherwise the clipped value loss: a prediction may move at
            most this far from its collection-time value before the loss
            stops rewarding the move.
        ent_coef: Weight of the entropy bonus. A positive value rewards a
            broader action distribution, which keeps the policy exploring.
        normalize_advantage: Standardize advantages to mean 0 and std 1
            before they weight the policy objective.
        normalize_advantage_per_minibatch: Standardize each minibatch on its
            own (SB3, CleanRL) instead of the whole rollout once (RSL-RL).
            Only used when ``normalize_advantage`` is on.
        log_ratio_clip: ``None``, or a bound on ``|log r|`` applied before
            ``exp`` in the policy objective. It is an overflow guard: for
            extreme ratios it prevents ``inf``, at the cost of changing the
            loss there. Diagnostics still use the original log-ratios.
        target_kl: ``None``, or a KL size at which to stop the update early.
            Before each optimizer step, if the approximate KL between the
            collection policy and the current policy exceeds
            ``1.5 * target_kl`` (SB3's rule), the remaining minibatches and
            epochs are skipped.
        lr_schedule: ``"fixed"`` keeps the learning rates as configured.
            ``"adaptive"`` changes both before each minibatch step, from the
            analytic KL between the collection policy and the current one
            (RSL-RL's rule, see ``PPOLearner._adapt_learning_rates``). The
            collection policy is a frozen copy of the actor, taken at the
            start of each update. That copy reproduces collection exactly for
            networks whose output is the same in eval and train mode (for
            example plain MLPs without dropout).
        desired_kl: The KL per minibatch step that the adaptive schedule
            aims for. Required when ``lr_schedule="adaptive"``.
        lr_factor: How much one adjustment multiplies or divides the
            learning rates; greater than 1.
        min_learning_rate: The adaptive schedule never goes below this.
        max_learning_rate: The adaptive schedule never goes above this.
        actor_optimizer_class: PyTorch optimizer class for the actor.
        actor_optimizer_kwargs: Extra optimizer arguments for the actor,
            excluding ``params`` and ``lr`` (the learner passes those).
        critic_optimizer_class: PyTorch optimizer class for the critic.
        critic_optimizer_kwargs: Extra optimizer arguments for the critic,
            excluding ``params`` and ``lr``.
    """

    # Optimization
    lr_critic           :   float        = 3e-4
    lr_actor            :   float        = 1e-3
    batch_size          :   int          = 64
    update_epochs       :   int          = 10
    max_grad_norm       :   float        = 0.5

    # PPO clipped objective
    clip_range          :   float        = 0.2
    value_clip_range    :   float | None = None
    ent_coef            :   float        = 0.0
    normalize_advantage :   bool         = True
    normalize_advantage_per_minibatch: bool = False
    log_ratio_clip      :   float | None = None

    # Optional policy-KL early stopping
    target_kl           :   float | None = None

    # Optional RSL-RL-style learning-rate adaptation, evaluated per minibatch.
    lr_schedule         :   Literal["fixed", "adaptive"] = "fixed"
    desired_kl          :   float | None = None
    lr_factor           :   float = 1.5
    min_learning_rate   :   float = 1e-5
    max_learning_rate   :   float = 1e-2

    # PyTorch optimizer classes and options, excluding params and lr.
    actor_optimizer_class   :   type[torch.optim.Optimizer] = torch.optim.Adam
    actor_optimizer_kwargs  :   dict[str, Any] = field(default_factory=dict)
    critic_optimizer_class  :   type[torch.optim.Optimizer] = torch.optim.Adam
    critic_optimizer_kwargs :   dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Check the settings when the config is created.

        A wrong setting caught here fails immediately, instead of hours into
        a run.

        Raises:
            ValueError: If a count is not a positive integer, a coefficient
                is not a finite number >= 0, the learning-rate schedule
                settings are inconsistent, or the adaptive schedule is
                combined with ``target_kl``.
        """
        # 0. The shared on-policy settings first.
        super().__post_init__()

        # 1. Counts must be whole numbers of at least 1 (booleans are
        #    rejected, although Python treats them as integers).
        for name, value in (
            ("batch_size", self.batch_size),
            ("update_epochs", self.update_epochs),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise ValueError(
                    f"{name} must be a positive integer, got {value!r}"
                )

        # 2. Coefficients must be finite numbers >= 0. The optional ones may
        #    also be None, which switches their feature off.
        for name, value in (
            ("lr_actor", self.lr_actor),
            ("lr_critic", self.lr_critic),
            ("ent_coef", self.ent_coef),
            ("max_grad_norm", self.max_grad_norm),
            ("clip_range", self.clip_range),
            ("value_clip_range", self.value_clip_range),
            ("log_ratio_clip", self.log_ratio_clip),
            ("target_kl", self.target_kl),
            ("lr_factor", self.lr_factor),
            ("min_learning_rate", self.min_learning_rate),
            ("max_learning_rate", self.max_learning_rate),
        ):
            if value is None and name in {
                "value_clip_range", "log_ratio_clip", "target_kl",
            }:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not isfinite(value)
                or value < 0.0
            ):
                raise ValueError(
                    f"{name} must be a finite number >= 0, got {value!r}"
                )

        # 3. The learning-rate schedule: a known name, a factor that actually
        #    changes the rate, and positive bounds in the right order.
        if self.lr_schedule not in ("fixed", "adaptive"):
            raise ValueError("lr_schedule must be 'fixed' or 'adaptive'")
        if self.lr_factor <= 1.0:
            raise ValueError("lr_factor must be greater than 1")
        if self.min_learning_rate <= 0 or self.max_learning_rate <= 0:
            raise ValueError("learning-rate bounds must be positive")
        if self.min_learning_rate > self.max_learning_rate:
            raise ValueError("min_learning_rate must not exceed max_learning_rate")

        # 4. desired_kl, when given, is a finite positive number.
        if self.desired_kl is not None and (
            isinstance(self.desired_kl, bool)
            or not isinstance(self.desired_kl, Real)
            or not isfinite(self.desired_kl)
            or self.desired_kl <= 0
        ):
            raise ValueError("desired_kl must be a finite positive number")

        # 5. The adaptive schedule needs its target, and it cannot be combined
        #    with target_kl: two KL controllers on one update (one changing
        #    the step size, one stopping the update) would work against each
        #    other.
        if self.lr_schedule == "adaptive":
            if self.desired_kl is None:
                raise ValueError("adaptive lr_schedule requires desired_kl")
            if self.target_kl is not None:
                raise ValueError("target_kl and adaptive lr_schedule cannot be combined")


class PPOLearner(OnPolicyLearner):
    """The PPO-Clip learner: optimizers and the multi-epoch minibatch update.

    The base class (``OnPolicyLearner``) provides the networks, the shared
    observation processor, the inference policy used for collection, and
    ``prepare_targets`` (advantages and value targets for each rollout). This
    class adds PPO's optimizers and its ``update``. Train it with
    ``train_on_policy``.

    Attributes:
        cfg: The PPO settings.
        optim_actor: Optimizer of the actor (``policy``).
        optim_critic: Optimizer of the critic (``value_fn``).
        algorithm_id: ``"ppo"``, recorded in checkpoints.
        mutable_on_resume: Settings that may change when resuming from a
            checkpoint: the base class's, plus the learning rates, the
            entropy coefficient, ``target_kl`` and ``batch_size``.
    """

    algorithm_id = "ppo"

    mutable_on_resume = OnPolicyLearner.mutable_on_resume | {
        "lr_actor", "lr_critic", "ent_coef", "target_kl", "batch_size",
    }

    def __init__(
        self,
        policy      :   nn.Module,
        value_fn    :   nn.Module,
        cfg         :   PPOConfig,
        *,
        obs_processor: ObservationProcessor | None = None,
        action_processor: ActionProcessor | None = None,
        reward_scaler: RewardScaler | None = None,
    ) -> None:
        """Set up the networks (in the base class), then the optimizers.

        Args:
            policy: The actor: maps observations to action distributions.
            value_fn: The critic: maps observations to value estimates.
            cfg: The PPO settings.
            obs_processor: Optional shared observation processor, used by
                both networks and by the inference policy.
            action_processor: Optional conversion from sampled actions to
                environment commands.
            reward_scaler: Optional training-only reward scaling, applied when
                the targets are prepared.
        """
        super().__init__(
            policy,
            value_fn,
            cfg,
            obs_processor=obs_processor,
            action_processor=action_processor,
            reward_scaler=reward_scaler,
        )
        self.cfg: PPOConfig = cfg
        self.configure_optimizers()

    def configure_optimizers(self) -> None:
        """Create one optimizer for the actor and one for the critic.

        Called once, at the end of construction, after the base class has
        moved the networks to ``cfg.device``. An optimizer holds references
        to the parameters it updates, so it must be built after the move.

        Raises:
            RuntimeError: If the optimizers already exist.
        """
        if hasattr(self, "optim_actor") or hasattr(self, "optim_critic"):
            raise RuntimeError("optimizers are already configured")
        self.optim_actor = self.cfg.actor_optimizer_class(
            self.policy.parameters(),
            lr=self.cfg.lr_actor,
            **self.cfg.actor_optimizer_kwargs,
        )
        self.optim_critic = self.cfg.critic_optimizer_class(
            self.value_fn.parameters(),
            lr=self.cfg.lr_critic,
            **self.cfg.critic_optimizer_kwargs,
        )

    def state_dict(self) -> dict[str, Any]:
        """Return the actor, critic and optimizer states, and the reward scaler's.

        The optimizer states include the current learning rates, so a rate
        changed by the adaptive schedule is restored on resume. This is only
        the learner's part of a checkpoint; the trainer adds progress,
        configs, observation statistics and random-number state.
        """
        state = {
            "policy": self.policy.state_dict(),
            "value_fn": self.value_fn.state_dict(),
            "optim_actor": self.optim_actor.state_dict(),
            "optim_critic": self.optim_critic.state_dict(),
        }
        if self.reward_scaler is not None:
            state["reward_scaler"] = {
                "config": self.reward_scaler.config,
                "state_dict": self.reward_scaler.state_dict(),
            }
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a state made by ``state_dict`` into the existing components.

        The keys are checked first, so a state from another learner or setup
        fails before anything is changed. Loading is not all-or-nothing: if
        loading one component's tensors fails after earlier ones were
        restored, build a new learner before trying again.

        Args:
            state: The saved learner state, not the whole checkpoint.

        Raises:
            ValueError: If a key is missing or unexpected, or the saved reward
                scaler does not match this one.
        """
        self.validate_reward_scaler_state(state)
        expected_keys = {"policy", "value_fn", "optim_actor", "optim_critic"}
        if self.reward_scaler is not None:
            expected_keys.add("reward_scaler")
        learner_state.validate_state_keys(state, expected_keys)
        self.policy.load_state_dict(state["policy"])
        self.value_fn.load_state_dict(state["value_fn"])
        self.optim_actor.load_state_dict(state["optim_actor"])
        self.optim_critic.load_state_dict(state["optim_critic"])
        if self.reward_scaler is not None:
            self.reward_scaler.load_state_dict(state["reward_scaler"]["state_dict"])

    def apply_resume_overrides(self, overrides: dict[str, Any]) -> None:
        """Apply learning rates changed for a resumed run.

        Called after ``load_state_dict``: the restored optimizer states carry
        the old rates, so a new ``lr_actor`` or ``lr_critic`` is written into
        every parameter group afterwards. With the adaptive schedule, the
        rate then keeps adapting from this new value. Other allowed changes
        (for example ``ent_coef`` or ``batch_size``) are read from the
        current config and need nothing here.

        Args:
            overrides: Changed settings and their new values, from
                ``validate_resume_config``.
        """
        if "lr_actor" in overrides:
            for group in self.optim_actor.param_groups:
                group["lr"] = overrides["lr_actor"]
        if "lr_critic" in overrides:
            for group in self.optim_critic.param_groups:
                group["lr"] = overrides["lr_critic"]

    def _adapt_learning_rates(self, kl: torch.Tensor) -> None:
        """Raise or lower both learning rates from one minibatch's KL.

        RSL-RL's rule, applied before each optimizer step of the adaptive
        schedule:

        * KL above ``2 * desired_kl``: the policy is moving too fast, so
          divide the rates by ``lr_factor`` (not below ``min_learning_rate``);
        * KL between 0 and ``desired_kl / 2``: it is moving too slowly, so
          multiply them by ``lr_factor`` (not above ``max_learning_rate``);
        * otherwise: keep the rates.

        Both optimizers change by the same factor, so the actor-to-critic
        rate ratio stays as configured (RSL-RL has a single rate for both).

        Args:
            kl: The analytic KL from the collection policy to the current
                policy, averaged over the minibatch (a scalar tensor).

        Raises:
            ValueError: If the KL is not finite.
        """
        desired_kl = self.cfg.desired_kl
        assert desired_kl is not None

        # 1. A non-finite KL means the update has already gone wrong; changing
        #    the rates from it would only hide that.
        kl_value = kl.item()
        if not isfinite(kl_value):
            raise ValueError("analytic policy KL must be finite")

        # 2. Decide the direction (or no change) from the KL.
        if kl_value > 2.0 * desired_kl:
            direction = "decrease"
        elif 0.0 < kl_value < desired_kl / 2.0:
            direction = "increase"
        else:
            return

        # 3. Apply it to every parameter group of both optimizers, within the
        #    bounds.
        for optimizer in (self.optim_actor, self.optim_critic):
            for group in optimizer.param_groups:
                # A zero rate deliberately freezes that parameter group.
                rate = group["lr"]
                if rate == 0.0:
                    continue
                if direction == "decrease":
                    group["lr"] = max(
                        self.cfg.min_learning_rate, rate / self.cfg.lr_factor
                    )
                else:
                    group["lr"] = min(
                        self.cfg.max_learning_rate, rate * self.cfg.lr_factor
                    )

    def update(
        self,
        buffer          :   RolloutBuffer,
        advantages      :   torch.Tensor,
        value_targets   :   torch.Tensor,
    ) -> TrainingStats:
        """Improve the actor and critic on one complete rollout.

        The advantages and value targets were computed once, before this
        call, and stay fixed throughout: the update trains towards them
        instead of chasing numbers that would change with every step.

        Steps:

        0. Setup, once: take a frozen copy of the actor (adaptive schedule
           only), switch the networks to training mode, and standardize the
           advantages over the whole rollout (unless per-minibatch).
        1. For each of ``update_epochs`` epochs, the rollout's B = T * N
           transitions are reshuffled into minibatches of M. Per minibatch:
           re-evaluate the action distribution, log-probabilities and values
           with the current networks;
        2. compute the value loss (plain or clipped);
        3. compute the clipped policy objective, with the entropy bonus;
        4. measure how far the policy has moved (KL), then stop early
           (``target_kl``) or adapt the learning rates (adaptive schedule);
        5. take one critic step and one actor step, with gradient clipping;
        6. add the minibatch's metrics, weighted by its size.
        7. Reduce the metrics once, with a single transfer to the CPU.

        Shapes: within a minibatch, log-probabilities, ratios, advantages,
        values and targets are (M,).

        Args:
            buffer: The full rollout buffer, with collection-time values and
                log-probabilities.
            advantages: Fixed advantages, shaped (T, N).
            value_targets: Fixed critic targets, shaped (T, N).

        Returns:
            The update's metrics (sample-weighted means over accepted
            minibatches; minimum and maximum for the action-std extremes) and
            the number of completed optimizer steps.
        """

        # ------------------------------------------------------------------
        # 0. Setup
        # ------------------------------------------------------------------

        # Adaptive schedule only: a frozen copy of the actor, taken before any
        # step, is the policy that collected this rollout (the actor has not
        # changed since, and the observation statistics stay frozen until
        # after the update). It goes through the same inference path as the
        # current actor, in evaluation mode like during collection, so its
        # distributions match collection.
        old_inference_policy = None
        if self.cfg.lr_schedule == "adaptive":
            old_policy = deepcopy(self.policy).eval()
            old_policy.requires_grad_(False)
            old_inference_policy = InferencePolicy(
                old_policy,
                observation_processor=self.obs_processor,
            )

        # Collection ran the networks in evaluation mode; training uses
        # training mode. The forward passes below build a fresh autograd graph
        # for every minibatch.
        self.policy.train()
        self.value_fn.train()

        # Metrics are accumulated on the device as sums weighted by minibatch
        # size, and transferred once at the end (step 7). Only minibatches
        # that led to an optimizer step count.
        metric_names = (
            "losses/policy_loss",
            "losses/value_loss",
            "diagnostics/entropy",
            "diagnostics/approximate_kl",
            "diagnostics/clip_fraction",
            "grads/policy_norm",
            "grads/value_norm",
        )
        if old_inference_policy is not None:
            metric_names += ("diagnostics/analytic_kl",)
        metric_sums = torch.zeros(
            len(metric_names),
            dtype=advantages.dtype,
            device=self.cfg.device,
        )
        num_samples = 0
        gradient_steps = 0
        gaussian_stats = None
        num_action_components = 0

        # Standardize once over the whole rollout, so every epoch uses the
        # same advantage weights. (Per-minibatch mode does it in step 1.)
        if (
            self.cfg.normalize_advantage
            and not self.cfg.normalize_advantage_per_minibatch
        ):
            advantages = _normalize_advantages(advantages)

        # Set when target_kl stops the update, to leave the epoch loop too.
        stop_early = False

        for _ in range(self.cfg.update_epochs):

            # Each call reshuffles the whole rollout and yields flat
            # minibatches of M transitions.
            for batch in buffer.get(
                advantages=advantages,
                value_targets=value_targets,
                batch_size=self.cfg.batch_size,
            ):

                # --------------------------------------------------------------
                # 1. Minibatch re-evaluation
                # --------------------------------------------------------------

                # pi_theta(.|S), log pi_theta(A|S) and V_phi(S) with the
                # current parameters, for the stored observations and actions.
                # Each scalar output has shape (M,).
                action_distributions: Distribution = self.inference_policy(batch.observations)
                log_probs: torch.Tensor = action_distributions.log_prob(batch.actions)
                values: torch.Tensor = self.evaluate_values(batch.observations)
                batch_advantages = batch.advantages
                if (
                    self.cfg.normalize_advantage
                    and self.cfg.normalize_advantage_per_minibatch
                ):
                    batch_advantages = _normalize_advantages(batch_advantages)

                # --------------------------------------------------------------
                # 2. Value loss
                # --------------------------------------------------------------

                # The value targets are fixed TD(lambda) regression labels, so
                # the critic step is a semi-gradient step: no gradient flows
                # into the targets.
                if self.cfg.value_clip_range is None:
                    value_loss = 0.5 * (
                        batch.value_targets - values
                    ).square().mean()
                else:
                    # The clipped version keeps the new prediction near the
                    # collection-time value V_old(S):
                    #   clipped = V_old + clip(V - V_old, -eps_v, +eps_v)
                    clipped_values = batch.old_values + (
                        values - batch.old_values
                    ).clamp(
                        min=-self.cfg.value_clip_range,
                        max=self.cfg.value_clip_range,
                    )
                    unclipped_value_loss = (batch.value_targets - values).square()
                    clipped_value_loss = (
                        batch.value_targets - clipped_values
                    ).square()

                    # Take the larger (pessimistic) loss, so the critic gains
                    # nothing from moving farther than eps_v in one update.
                    value_loss = 0.5 * torch.maximum(
                        unclipped_value_loss,
                        clipped_value_loss,
                    ).mean()

                # --------------------------------------------------------------
                # 3. Clipped policy objective
                # --------------------------------------------------------------

                # The entropy keeps its gradient: with ent_coef > 0 it is part
                # of the objective and rewards a broader action distribution.
                entropy = action_distributions.entropy().mean()

                # ratio = pi_theta(A|S) / pi_old(A|S), computed in log space.
                # The old log-probabilities were stored during collection, so
                # the denominator is a fixed number. The optional log_ratio_clip
                # is an overflow guard for exp, not part of PPO's objective.
                log_ratios = log_probs - batch.old_log_probs
                objective_log_ratios = log_ratios
                if self.cfg.log_ratio_clip is not None:
                    objective_log_ratios = log_ratios.clamp(
                        -self.cfg.log_ratio_clip,
                        self.cfg.log_ratio_clip,
                    )
                ratios = torch.exp(objective_log_ratios)

                # For each transition take the smaller of the plain and the
                # clipped term, so moving the ratio outside the band can only
                # lower the objective, never raise it. The loss is the
                # negative because the optimizers minimize.
                unclipped_objective = ratios * batch_advantages
                clipped_objective = ratios.clip(
                    min=1.0 - self.cfg.clip_range,
                    max=1.0 + self.cfg.clip_range,
                ) * batch_advantages
                policy_loss = -(
                    torch.min(unclipped_objective, clipped_objective).mean()
                    + self.cfg.ent_coef * entropy
                )

                # --------------------------------------------------------------
                # 4. How far the policy has moved: KL, early stop, schedule
                # --------------------------------------------------------------

                # Diagnostics only: no gradients. They use the unclipped
                # log-ratios, so they report the policy's true movement.
                with torch.no_grad():
                    diagnostic_ratios = (
                        log_ratios.exp() if self.cfg.log_ratio_clip is not None else ratios
                    )

                    # Sample-based KL estimate, mean of (r - 1) - log r (always
                    # >= 0): what target_kl compares against. clip_fraction is
                    # the share of transitions whose ratio left the band.
                    approximate_kl = ((diagnostic_ratios - 1.0) - log_ratios).mean()
                    clip_fraction = (
                        (diagnostic_ratios - 1.0).abs() > self.cfg.clip_range
                    ).float().mean()

                    # Adaptive schedule only: the exact KL from the collection
                    # policy (the frozen copy) to the current policy, averaged
                    # over the minibatch. It needs a distribution with a known
                    # closed-form KL, such as a Gaussian.
                    analytic_kl = None
                    if old_inference_policy is not None:
                        old_distribution = old_inference_policy(batch.observations)
                        try:
                            per_sample_kl = kl_divergence(
                                old_distribution, action_distributions
                            )
                        except (NotImplementedError, TypeError) as exc:
                            raise TypeError(
                                "adaptive PPO requires a registered analytic KL "
                                "for the policy distribution"
                            ) from exc
                        if (
                            not isinstance(per_sample_kl, torch.Tensor)
                            or per_sample_kl.shape != log_probs.shape
                        ):
                            raise ValueError(
                                "analytic policy KL must have one value per transition"
                            )
                        analytic_kl = per_sample_kl.mean()

                    # Gaussian policies: exploration-noise statistics from this
                    # minibatch's distributions (no extra pass over the rollout).
                    gaussian_batch_stats = _gaussian_action_stats(action_distributions)

                # target_kl: stop before another optimizer step once the
                # policy has moved well beyond the target. This minibatch is
                # not used, and the remaining minibatches and epochs are
                # skipped.
                if (
                    self.cfg.target_kl is not None
                    and approximate_kl > 1.5 * self.cfg.target_kl
                ):
                    stop_early = True
                    break

                # Adaptive schedule: set the learning rates for this step from
                # the KL measured just before it.
                if analytic_kl is not None:
                    self._adapt_learning_rates(analytic_kl)

                # --------------------------------------------------------------
                # 5. Optimizer steps
                # --------------------------------------------------------------

                # Critic: one step on the value loss, with its gradient
                # clipped to max_grad_norm.
                self.optim_critic.zero_grad(set_to_none=True)
                value_loss.backward()
                value_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.value_fn.parameters(),
                    max_norm=self.cfg.max_grad_norm,
                )
                self.optim_critic.step()

                # Actor: one step on the clipped policy loss, with its gradient
                # clipped to max_grad_norm.
                self.optim_actor.zero_grad(set_to_none=True)
                policy_loss.backward()
                policy_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    max_norm=self.cfg.max_grad_norm,
                )
                self.optim_actor.step()

                # --------------------------------------------------------------
                # 6. Minibatch metrics
                # --------------------------------------------------------------

                # Weight by the number of transitions, so a smaller last
                # minibatch counts in proportion. Only completed steps count.
                num_batch_samples = batch.observations.shape[0]
                num_samples += num_batch_samples
                gradient_steps += 1

                minibatch_metrics = (
                    policy_loss.detach(),
                    value_loss.detach(),
                    entropy.detach(),
                    approximate_kl.detach(),
                    clip_fraction.detach(),
                    policy_grad_norm.detach(),
                    value_grad_norm.detach(),
                )
                if analytic_kl is not None:
                    minibatch_metrics += (analytic_kl.detach(),)
                metric_sums.add_(torch.stack(minibatch_metrics), alpha=num_batch_samples)

                # Gaussian statistics: sums add up, extremes keep the running
                # minimum or maximum.
                if gaussian_batch_stats is not None:
                    batch_stats, batch_components = gaussian_batch_stats
                    if gaussian_stats is None:
                        gaussian_stats = batch_stats
                    else:
                        gaussian_stats[0] += batch_stats[0]
                        gaussian_stats[1] = torch.minimum(
                            gaussian_stats[1], batch_stats[1]
                        )
                        gaussian_stats[2] = torch.maximum(
                            gaussian_stats[2], batch_stats[2]
                        )
                        gaussian_stats[3] += batch_stats[3]
                        gaussian_stats[4] = torch.maximum(
                            gaussian_stats[4], batch_stats[4]
                        )
                    num_action_components += batch_components

            if stop_early:
                break

        # ------------------------------------------------------------------
        # 7. Metric reduction
        # ------------------------------------------------------------------

        # One transfer to the CPU for everything accumulated on the device.
        # The weighted sums become means; the Gaussian sums become means per
        # action component, and the extremes are reported as they are.
        metrics = {}
        if num_samples > 0:
            totals = (
                torch.cat((metric_sums, gaussian_stats))
                if gaussian_stats is not None else metric_sums
            ).tolist()
            metrics = {
                name: total / num_samples
                for name, total in zip(metric_names, totals)
            }
            if gaussian_stats is not None:
                std_sum, std_min, std_max, mean_abs_sum, mean_abs_max = totals[
                    len(metric_names):
                ]
                metrics.update({
                    "diagnostics/action_std_mean": std_sum / num_action_components,
                    "diagnostics/action_std_min": std_min,
                    "diagnostics/action_std_max": std_max,
                    "diagnostics/action_mean_abs_mean": mean_abs_sum / num_action_components,
                    "diagnostics/action_mean_abs_max": mean_abs_max,
                })

        # How much work the update actually did: whether target_kl stopped it,
        # how many steps it took, and how many epochs' worth of transitions.
        rollout_size = buffer.rollout_steps * buffer.num_envs
        metrics.update({
            "diagnostics/early_stop": float(stop_early),
            "diagnostics/optimizer_steps": float(gradient_steps),
            "diagnostics/effective_epochs": num_samples / rollout_size,
        })

        # Explained variance compares the collection-time values with the
        # fixed TD(lambda) targets over the whole rollout (1 = the critic
        # explains the targets fully), as the other on-policy learners do.
        # The learning rates show the adaptive schedule's current values.
        metrics.update(
            {
                "diagnostics/explained_variance": explained_variance(
                    buffer.values.flatten(),
                    value_targets.flatten(),
                ),
                "charts/policy_learning_rate": self.optim_actor.param_groups[0]["lr"],
                "charts/value_learning_rate": self.optim_critic.param_groups[0]["lr"],
            }
        )

        return TrainingStats(
            metrics=metrics,
            gradient_steps=gradient_steps,
        )
