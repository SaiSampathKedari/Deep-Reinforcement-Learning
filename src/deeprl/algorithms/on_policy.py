"""Shared settings and the learner interface for on-policy algorithms.

On-policy algorithms (PPO, A2C, NPG, TRPO) learn from data collected by the
current policy, then throw that data away. This module holds what they share:

* ``evaluate_next_values``: value estimates of next observations, used to
  "bootstrap" returns.
* ``OnPolicyConfig``: settings every on-policy algorithm needs (run length,
  rollout length, discount factor, GAE lambda).
* ``OnPolicyLearner``: the base class each algorithm extends. It owns the
  networks, the processors and the reward scaler, prepares training targets,
  and defines the methods the training loop calls.

One training iteration, as run by ``training/train_on_policy.py``::

    collect a rollout     (learner.inference_policy, learner.evaluate_values)
    prepare targets       (learner.prepare_targets)
    update the networks   (learner.update)
    update statistics     (learner.obs_processor.update, called by the trainer)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Any

import torch
import torch.nn as nn

from deeprl import learner_state
from deeprl.advantages import generalized_advantage_estimate
from deeprl.buffers import RolloutBuffer
from deeprl.data_spec import TensorData
from deeprl.inference import InferencePolicy
from deeprl.normalization import RewardScaler, RunningNormalizer
from deeprl.processors import ActionProcessor, ObservationProcessor
from deeprl.stats import TrainingStats


@torch.no_grad()
def evaluate_next_values(
    value_fn: Callable[[TensorData], torch.Tensor],
    next_observations: TensorData,
    terminations: torch.Tensor,
) -> torch.Tensor:
    """Return the value of each next observation, for bootstrapping.

    Training targets need the value of the state after each step:

    * If the episode **terminated** (a true end, such as a fall), nothing
      follows, so the next value is 0.
    * If the episode was **truncated** (a time limit), the task would have
      continued, so the value of the true final observation is used. It is
      never the value of the next episode's first observation.

    Args:
        value_fn: Maps raw observations to values shaped ``(K,)``.
        next_observations: Next observations, each feature shaped
            ``(T, N, *feature_shape)``.
        terminations: True where an episode terminated, shaped ``(T, N)``.

    Returns:
        Next values shaped like ``terminations``, usually ``(T, N)``, with 0
        wherever the episode terminated.
    """
    # 1. Find the steps that need a next value: everything except terminations.
    nonterminal_mask = ~terminations

    # 2. Start with zeros, which is the correct value for terminated steps.
    next_values = torch.zeros(
        terminations.shape,
        dtype=torch.float32,
        device=terminations.device,
    )

    # 3. Evaluate only the non-terminal next observations, in one batch, and
    #    fill them in. Terminated steps keep their zero.
    next_values[nonterminal_mask] = value_fn(next_observations[nonterminal_mask])
    return next_values


@dataclass(frozen=True)
class OnPolicyConfig:
    """Settings shared by all on-policy algorithms.

    Each algorithm's own config (for example ``PPOConfig``) extends this class
    with its optimization settings.

    Attributes:
        total_timesteps: Total environment steps to train for, counted across
            all environments.
        seed: Random seed. The trainer uses it for action sampling,
            minibatch shuffling and the environments' first reset. The
            networks are built before the trainer runs, so their initial
            weights follow this seed only if the caller seeds before building
            them (the examples call ``seed_torch(cfg.seed)`` first).
        device: Device for the networks and training, for example ``cpu`` or
            ``cuda``.
        log_every: Write a log entry every this many environment steps.
            0 turns periodic logging off.
        rollout_steps: T, the number of steps collected from each environment
            before every update. One rollout holds ``T * N`` transitions.
        gamma: Discount factor: how much future rewards count, between 0 and 1.
        gae_lambda: Lambda of generalized advantage estimation, between 0
            (one-step estimates: low variance, more bias) and 1 (sum the
            rewards up to the end of the episode or the rollout: high
            variance, less bias). Even with 1, the critic's value is still
            used at the end of a rollout and after a truncation, so it is not
            always a complete Monte Carlo return.
    """

    # Run control
    total_timesteps     :   int          = 500_000  # Total transitions across all environments.
    seed                :   int          = 0
    device              :   torch.device = torch.device("cpu")
    log_every           :   int          = 1000

    # Rollout and advantage estimation
    rollout_steps       :   int          = 128      # T: transitions collected per environment.
    gamma               :   float        = 0.99     # Reward discount factor.
    gae_lambda          :   float        = 0.95     # Lambda in generalized advantage estimation.

    def __post_init__(self) -> None:
        """Check the settings when the config is created.

        Raises:
            ValueError: If a count is not a large enough integer, or ``gamma``
                or ``gae_lambda`` is not a finite number between 0 and 1.
        """
        # 1. Counts must be whole numbers (booleans are rejected) and at
        #    least the given minimum.
        for name, value, minimum in (
            ("total_timesteps", self.total_timesteps, 1),
            ("rollout_steps", self.rollout_steps, 1),
            ("log_every", self.log_every, 0),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < minimum
            ):
                raise ValueError(
                    f"{name} must be an integer >= {minimum}, got {value!r}"
                )

        # 2. gamma and gae_lambda must be finite numbers between 0 and 1.
        for name, value in (
            ("gamma", self.gamma),
            ("gae_lambda", self.gae_lambda),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(
                    f"{name} must be a finite number between 0 and 1, got {value!r}"
                )


class OnPolicyLearner(ABC):
    """Base class for on-policy algorithms.

    It owns the actor (``policy``) and the critic (``value_fn``), the
    observation and action processors, and the optional reward scaler.

    What this base class provides:

    * ``inference_policy``: the complete policy (processors + actor), used for
      data collection, evaluation and export.
    * ``evaluate_values``: critic values for raw observations.
    * ``prepare_targets``: advantages and value targets for one rollout.
    * Checks for the reward scaler and for config changes when resuming.

    What each algorithm must implement:

    * ``update``: its own training step on one rollout.
    * ``state_dict`` / ``load_state_dict``: save and restore its networks,
      optimizers and other learning state, explicitly. The trainer saves the
      rest of a checkpoint (progress, random-number state, observation
      statistics, configs).
    * ``apply_resume_overrides``: apply allowed config changes when resuming.

    Args:
        policy: The actor network, returning an action distribution.
        value_fn: The critic network, returning one value per observation.
        cfg: The algorithm's config.
        obs_processor: Turns raw observations into network inputs. Shared by
            actor and critic.
        action_processor: Turns sampled actions into environment commands.
        reward_scaler: Training-only reward scaling.

    Attributes:
        mutable_on_resume: Config fields allowed to change when training
            resumes from a checkpoint. Algorithms extend this set, for example
            with learning rates.
    """

    mutable_on_resume = frozenset({"total_timesteps", "device", "log_every"})

    def __init__(
        self,
        policy      :   nn.Module,
        value_fn    :   nn.Module,
        cfg         :   OnPolicyConfig,
        *,
        obs_processor: ObservationProcessor | None = None,
        action_processor: ActionProcessor | None = None,
        reward_scaler: RewardScaler | None = None,
    ) -> None:
        """Check the parts, move them to the training device and build the policy.

        Raises:
            ValueError: If the reward scaler's gamma differs from the config,
                the actor and critic share trainable parameters, or a network
                contains its own observation normalization.
        """
        # 1. The reward scaler discounts returns with its own gamma; it must be
        #    the same discount the learner uses.
        if reward_scaler is not None and reward_scaler.gamma != cfg.gamma:
            raise ValueError("reward_scaler gamma does not match the learner")

        # 2. Independent actor/critic updates require disjoint trainable
        #    parameters. The actor and the critic are updated separately (PPO
        #    and A2C use two optimizers; NPG and TRPO update the actor with a
        #    natural-gradient step and only the critic with an optimizer), so
        #    a shared trainable parameter would be changed by both updates.
        policy_ids = {id(p) for p in policy.parameters() if p.requires_grad}
        value_ids = {id(p) for p in value_fn.parameters() if p.requires_grad}
        if policy_ids & value_ids:
            raise ValueError("policy and value_fn must not share trainable parameters")

        # 3. Observation normalization lives in one place, the obs_processor.
        #    A normalizer hidden inside a network would be a second, separate
        #    copy of the statistics.
        if any(
            isinstance(module, (RunningNormalizer, ObservationProcessor))
            for network in (policy, value_fn)
            for module in network.modules()
        ):
            raise ValueError(
                "use obs_processor rather than embedding observation normalization in networks"
            )

        # 4. Move the networks and the reward scaler to the training device.
        self.cfg = cfg
        self.policy = policy.to(self.cfg.device)
        self.value_fn = value_fn.to(self.cfg.device)
        self.reward_scaler = (
            reward_scaler.to(cfg.device) if reward_scaler is not None else None
        )

        # 5. Move the processors to the training device too.
        if obs_processor is not None:
            obs_processor.to(cfg.device)
        self.obs_processor = obs_processor
        if action_processor is not None:
            action_processor.to(cfg.device)

        # 6. Build the complete policy once. It wraps the same actor network,
        #    so every training update is immediately visible to it.
        self.inference_policy = InferencePolicy(
            self.policy,
            observation_processor=obs_processor,
            action_processor=action_processor,
        )

    def evaluate_values(self, observations: TensorData) -> torch.Tensor:
        """Return the critic's value estimates for raw observations.

        The observation processor gives the critic its features, including
        privileged ones. Gradients flow, so ``update`` can train the critic
        with this method too.

        Args:
            observations: Raw observations, batched.

        Returns:
            One value per observation, shaped ``(B,)``.
        """
        if self.obs_processor is not None:
            observations = self.obs_processor(observations, group="critic")
        return self.value_fn(observations)

    @torch.no_grad()
    def prepare_targets(
        self,
        buffer: RolloutBuffer,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute advantages and value targets for one complete rollout.

        Runs without gradients: the results are fixed numbers that ``update``
        trains towards, not things to differentiate through. The observation
        statistics are the same ones used during collection. Rewards are
        scaled in a copy, so the buffer's rewards stay raw for logging.

        Args:
            buffer: A full rollout buffer.

        Returns:
            ``(advantages, value_targets)``, both shaped ``(T, N)``.

        Raises:
            RuntimeError: If the rollout is not complete.
        """
        # 0. Targets need every step of the rollout.
        if not buffer.full or buffer.next_observations is None:
            raise RuntimeError("cannot prepare targets from an incomplete rollout")

        # 1. Value of each next state; terminated steps get 0 (see
        #    evaluate_next_values).
        next_values = evaluate_next_values(
            self.evaluate_values,
            buffer.next_observations,
            buffer.terminations,
        )

        # 2. Scale the rewards for training, if enabled. This returns a copy;
        #    the buffer keeps the raw rewards.
        rewards = buffer.rewards
        if self.reward_scaler is not None:
            rewards = self.reward_scaler(rewards, buffer.dones)

        # 3. Generalized advantage estimation (advantages.py): how much better
        #    each action was than expected (advantages), and what the critic
        #    should predict (value targets).
        return generalized_advantage_estimate(
            rewards,
            buffer.values,
            next_values=next_values,
            terminations=buffer.terminations,
            truncations=buffer.truncations,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
        )

    def validate_reward_scaler_state(self, state: Mapping[str, Any]) -> None:
        """Check that a saved learner state matches this learner's reward scaling.

        Called before anything is restored, so a mismatch never leaves the
        learner half-loaded.

        Args:
            state: A learner state from a checkpoint.

        Raises:
            ValueError: If reward scaling is on in one and off in the other,
                or its settings differ.
        """
        # 1. The state must be a dictionary.
        if not isinstance(state, Mapping):
            raise ValueError("checkpoint learner state must be a mapping")

        # 2. Reward scaling must be on in both or off in both.
        if (self.reward_scaler is not None) != ("reward_scaler" in state):
            raise ValueError("checkpoint reward scaling does not match this learner")
        if self.reward_scaler is None:
            return

        # 3. The saved entry must hold the settings and the statistics.
        saved = state["reward_scaler"]
        if not isinstance(saved, Mapping) or set(saved) != {"config", "state_dict"}:
            raise ValueError("checkpoint reward_scaler must contain config and state_dict")

        # 4. The settings must match; statistics from a different setup would
        #    mean something else.
        if saved["config"] != self.reward_scaler.config:
            raise ValueError("checkpoint reward scaling configuration does not match this learner")
        if not isinstance(saved["state_dict"], Mapping):
            raise ValueError("checkpoint reward_scaler state_dict must be a mapping")

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Return the learner's own state.

        That includes the networks, optimizers and auxiliary state such as the
        reward scaler, each under its own key. It is not everything needed to
        resume: the trainer saves the rest of the checkpoint (progress,
        random-number state, observation statistics, configs).
        """
        raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a state made by ``state_dict``.

        Implementations check the keys first, then restore into the existing
        networks and optimizers.
        """
        raise NotImplementedError

    def validate_resume_config(
        self,
        saved_config: dict[str, Any],
        current_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Check that a resumed run only changes settings it is allowed to change.

        Args:
            saved_config: The config stored in the checkpoint.
            current_config: The config of the run that is resuming.

        Returns:
            The allowed fields that changed, with their new values, for
            ``apply_resume_overrides``.

        Raises:
            ValueError: If a field outside ``mutable_on_resume`` changed.
        """
        return learner_state.validate_resume_config(
            saved_config,
            current_config,
            mutable_on_resume=self.mutable_on_resume,
        )

    @abstractmethod
    def apply_resume_overrides(self, overrides: dict[str, Any]) -> None:
        """Apply allowed config changes after the saved state was restored.

        For example, set a new learning rate in the restored optimizer.

        Args:
            overrides: Changed fields and their new values, from
                ``validate_resume_config``.
        """
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        buffer          :   RolloutBuffer,
        advantages      :   torch.Tensor,
        value_targets   :   torch.Tensor,
    ) -> TrainingStats:
        """Train the actor and critic on one complete rollout.

        Args:
            buffer: The full rollout buffer.
            advantages: From ``prepare_targets``, shaped ``(T, N)``.
            value_targets: From ``prepare_targets``, shaped ``(T, N)``.

        Returns:
            Training statistics, such as losses and gradient steps, for
            logging.
        """
        raise NotImplementedError
