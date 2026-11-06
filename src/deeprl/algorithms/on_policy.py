"""Configuration and learner contract for on-policy algorithms."""

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
    """Return bootstrap values matching terminations, normally shaped (T, N).

    Each next-observation leaf has shape (T, N, *feature_shape). Truncations
    still bootstrap from their true final observation, not the reset state.
    True terminations receive zero bootstrap value.
    """
    nonterminal_mask = ~terminations
    next_values = torch.zeros(
        terminations.shape,
        dtype=torch.float32,
        device=terminations.device,
    )
    next_values[nonterminal_mask] = value_fn(next_observations[nonterminal_mask])
    return next_values


@dataclass(frozen=True)
class OnPolicyConfig:
    """Run, rollout, and GAE settings shared by on-policy algorithms."""

    # Run control
    total_timesteps     :   int          = 500_000 # Total transitions across all environments.
    seed                :   int          = 0
    device              :   torch.device = torch.device("cpu")
    log_every           :   int          = 1000

    # Rollout and advantage estimation
    rollout_steps       :   int          = 128      # T: transitions collected per environment.
    gamma               :   float        = 0.99     # Reward discount factor.
    gae_lambda          :   float        = 0.95     # Lambda in generalized advantage estimation.

    def __post_init__(self) -> None:
        """Validate shared run and advantage-estimation settings."""
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
    """On-policy update, explicit learner state, and resume contract."""

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
        """Place separate networks and their shared processing on the run device."""
        # Independent actor/critic updates require disjoint trainable parameters.
        if reward_scaler is not None and reward_scaler.gamma != cfg.gamma:
            raise ValueError("reward_scaler gamma does not match the learner")
        policy_ids = {id(p) for p in policy.parameters() if p.requires_grad}
        value_ids = {id(p) for p in value_fn.parameters() if p.requires_grad}
        if policy_ids & value_ids:
            raise ValueError("policy and value_fn must not share trainable parameters")
        if any(
            isinstance(module, (RunningNormalizer, ObservationProcessor))
            for network in (policy, value_fn)
            for module in network.modules()
        ):
            raise ValueError(
                "use obs_processor rather than embedding observation normalization in networks"
            )

        self.cfg = cfg
        self.policy = policy.to(self.cfg.device)
        self.value_fn = value_fn.to(self.cfg.device)
        self.reward_scaler = (
            reward_scaler.to(cfg.device) if reward_scaler is not None else None
        )
        if obs_processor is not None:
            obs_processor.to(cfg.device)
        self.obs_processor = obs_processor
        if action_processor is not None:
            action_processor.to(cfg.device)
        self.inference_policy = InferencePolicy(
            self.policy,
            observation_processor=obs_processor,
            action_processor=action_processor,
        )

    def evaluate_values(self, observations: TensorData) -> torch.Tensor:
        """Map raw batched observations to scalar values (B,), preserving gradients."""
        if self.obs_processor is not None:
            observations = self.obs_processor(observations, group="critic")
        return self.value_fn(observations)

    @torch.no_grad()
    def prepare_targets(
        self,
        buffer: RolloutBuffer,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare fixed GAE advantages and value targets from a complete rollout.

        Use the same observation statistics and critic mode as collection.
        Reward scaling produces a copy; stored rewards remain raw for reporting.
        Both returned tensors have shape (T, N), matching rollout rewards/values.
        """
        if not buffer.full or buffer.next_observations is None:
            raise RuntimeError("cannot prepare targets from an incomplete rollout")

        # 1. Bootstrap successors; only true terminations have zero next value.
        next_values = evaluate_next_values(
            self.evaluate_values,
            buffer.next_observations,
            buffer.terminations,
        )

        # 2. Scale training rewards without changing the raw buffer.
        rewards = buffer.rewards
        if self.reward_scaler is not None:
            rewards = self.reward_scaler(rewards, buffer.dones)

        # 3. Compute fixed advantage weights and critic regression targets.
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
        """Check training-only scaling settings before restoring any components."""
        if not isinstance(state, Mapping):
            raise ValueError("checkpoint learner state must be a mapping")
        if (self.reward_scaler is not None) != ("reward_scaler" in state):
            raise ValueError("checkpoint reward scaling does not match this learner")
        if self.reward_scaler is None:
            return
        saved = state["reward_scaler"]
        if not isinstance(saved, Mapping) or set(saved) != {"config", "state_dict"}:
            raise ValueError("checkpoint reward_scaler must contain config and state_dict")
        if saved["config"] != self.reward_scaler.config:
            raise ValueError("checkpoint reward scaling configuration does not match this learner")
        if not isinstance(saved["state_dict"], Mapping):
            raise ValueError("checkpoint reward_scaler state_dict must be a mapping")

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Return all network, optimizer, and auxiliary learning state."""
        raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Validate component names and restore existing learning objects."""
        raise NotImplementedError

    def validate_resume_config(
        self,
        saved_config: dict[str, Any],
        current_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate configuration and return changed allowed fields by name."""
        return learner_state.validate_resume_config(
            saved_config,
            current_config,
            mutable_on_resume=self.mutable_on_resume,
        )

    @abstractmethod
    def apply_resume_overrides(self, overrides: dict[str, Any]) -> None:
        """Apply explicit config changes after restoring learning state."""
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        buffer          :   RolloutBuffer,
        advantages      :   torch.Tensor,
        value_targets   :   torch.Tensor,
    ) -> TrainingStats:
        """Update the policy and value function from one completed rollout."""
        raise NotImplementedError
