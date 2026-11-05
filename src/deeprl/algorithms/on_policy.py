"""Configuration and learner contract for on-policy algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Any, ClassVar

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
    """On-policy update, named state, and resume-validation contract."""

    mutable_on_resume = frozenset({"total_timesteps", "device", "log_every"})
    learning_rate_fields: ClassVar[dict[str, str]] = {}

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

    @property
    def modules_by_role(self) -> dict[str, nn.Module]:
        """Return checkpointed modules under their stable state keys."""
        return {
            "policy": self.policy,
            "value_fn": self.value_fn,
        }

    @property
    def optimizers_by_role(self) -> dict[str, torch.optim.Optimizer]:
        """Return optimizer state; manual-update learners may return none."""
        return {}

    def state_dict(self) -> dict[str, Any]:
        """Return named module and optimizer states using the existing flat layout."""
        return learner_state.learner_state_dict(
            self.modules_by_role, self.optimizers_by_role
        )

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Check component names before restoring modules and optimizers."""
        learner_state.load_learner_state_dict(
            state, self.modules_by_role, self.optimizers_by_role
        )

    def validate_resume_config(
        self,
        saved_config: dict[str, Any],
        current_config: dict[str, Any],
    ) -> dict[str, float]:
        """Validate configuration and return changed learning rates by optimizer role."""
        return learner_state.validate_resume_config(
            saved_config,
            current_config,
            mutable_on_resume=self.mutable_on_resume,
            learning_rate_fields=self.learning_rate_fields,
        )

    def apply_resume_overrides(self, learning_rates: dict[str, float]) -> None:
        """Apply configured learning-rate changes after restoring optimizer state."""
        learner_state.apply_resume_overrides(learning_rates, self.optimizers_by_role)

    @abstractmethod
    def update(
        self,
        buffer          :   RolloutBuffer,
        advantages      :   torch.Tensor,
        value_targets   :   torch.Tensor,
    ) -> TrainingStats:
        """Update the policy and value function from one completed rollout."""
        raise NotImplementedError
