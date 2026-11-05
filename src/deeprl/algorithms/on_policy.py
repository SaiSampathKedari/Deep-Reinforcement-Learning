"""Configuration and learner contract for on-policy algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Any

import torch
import torch.nn as nn

from deeprl import learner_state
from deeprl.buffers import RolloutBuffer
from deeprl.stats import TrainingStats


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
    normalize_observations_during_collection: bool = False

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
    ) -> None:
        policy_ids = {id(p) for p in policy.parameters() if p.requires_grad}
        value_ids = {id(p) for p in value_fn.parameters() if p.requires_grad}
        if policy_ids & value_ids:
            raise ValueError("policy and value_fn must not share trainable parameters")

        self.cfg = cfg
        self.policy = policy.to(self.cfg.device)
        self.value_fn = value_fn.to(self.cfg.device)

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
