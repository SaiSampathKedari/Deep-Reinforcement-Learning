"""Configuration and learner contract for off-policy training."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Any

import torch

from deeprl import learner_state
from deeprl.buffers import ReplayBatch
from deeprl.stats import TrainingProgress, TrainingStats


@dataclass(frozen=True)
class OffPolicyConfig:
    """Run, replay, and update settings shared by off-policy algorithms."""

    # Run control.
    total_timesteps : int = 500_000     # Total transitions across all environments.
    seed            : int = 0           # Random seed.
    device          : torch.device = torch.device("cpu")  # Networks and sampled batches.
    log_every       : int = 1_000       # Transition logging interval; 0 disables.

    # Discount and replay storage.
    gamma           : float = 0.99      # Discount factor for Bellman targets.
    buffer_size     : int = 1_000_000   # Maximum number of stored transitions.
    replay_device   : torch.device = torch.device("cpu")  # Complete replay storage.
    batch_size      : int = 256         # Transitions sampled per learner update.

    # Collection and learning schedule.
    learning_starts : int = 1_000       # Minimum collected transitions before learning.
    train_freq      : int = 1           # Vector steps collected per learning phase.
    gradient_steps  : int = 1           # Replay-minibatch updates per learning phase.

    def __post_init__(self) -> None:
        """Validate counts, discount factor, and device types."""
        for name, minimum in (
            ("total_timesteps", 1),
            ("buffer_size", 1),
            ("batch_size", 1),
            ("train_freq", 1),
            ("gradient_steps", 1),
            ("seed", 0),
            ("learning_starts", 0),
            ("log_every", 0),
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < minimum
            ):
                raise ValueError(
                    f"{name} must be an integer >= {minimum}, got {value!r}"
                )

        if (
            isinstance(self.gamma, bool)
            or not isinstance(self.gamma, Real)
            or not isfinite(self.gamma)
            or not 0.0 <= self.gamma <= 1.0
        ):
            raise ValueError(
                f"gamma must be finite and between 0 and 1, got {self.gamma!r}"
            )

        for name in ("device", "replay_device"):
            if not isinstance(getattr(self, name), torch.device):
                raise TypeError(f"{name} must be a torch.device")


class OffPolicyLearner(ABC):
    """Shared configuration and resume settings for off-policy learners."""

    mutable_on_resume = frozenset({
        "total_timesteps",
        "device",
        "replay_device",
        "log_every",
    })

    def __init__(
        self,
        cfg: OffPolicyConfig,
    ) -> None:
        self.cfg = cfg

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Return learning state, including target networks and optimizers."""
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
        batch: ReplayBatch,
        *,
        progress: TrainingProgress,
    ) -> TrainingStats:
        """Process one replay minibatch using pre-update training progress."""
        raise NotImplementedError
