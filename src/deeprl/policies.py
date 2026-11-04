from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Categorical, Independent, Normal

from deeprl.buffers import TensorData


class CategoricalPolicy(nn.Module):
    """Categorical policy backed by a caller-supplied logits network."""

    def __init__(self, logits_network: nn.Module) -> None:
        super().__init__()
        self.logits_network = logits_network

    def forward(self, observations: TensorData) -> Categorical:
        logits = self.logits_network(observations)  # (B, num_actions)
        
        batch_size = observations.shape[0]
        if (
            logits.ndim != 2
            or logits.shape[0] != batch_size
            or logits.shape[1] == 0
        ):
            raise ValueError(
                f"logits_network must return shape "
                f"({batch_size}, num_actions) with num_actions > 0, "
                f"got {tuple(logits.shape)}"
            )
        
        return Categorical(logits=logits)


class DiagonalGaussianPolicy(nn.Module):
    """Gaussian policy with learned, state-independent standard deviation."""

    def __init__(
        self,
        mean_network: nn.Module,
        action_dim: int,
        log_std_init: float = 0.0,
        log_std_bounds: tuple[float, float] | None = None,
    ) -> None:
        super().__init__()
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if log_std_bounds is not None:
            low, high = log_std_bounds
            if not (math.isfinite(low) and math.isfinite(high) and low < high):
                raise ValueError("log_std_bounds must be finite and strictly increasing")

        self.mean_network = mean_network
        self.log_std_bounds = log_std_bounds
        self.log_std = nn.Parameter(
            torch.full((action_dim,), log_std_init, dtype=torch.float32)
        )

    def forward(self, observations: TensorData) -> Independent:
        mean = self.mean_network(observations)  # (B, action_dim)
        expected_shape = (observations.shape[0], self.log_std.numel())

        if mean.shape != expected_shape:
            raise ValueError(
                f"mean_network must return shape {expected_shape}, "
                f"got {tuple(mean.shape)}"
            )

        # Optional numerical bounds constrain the distribution, not stored actions.
        log_std = self.log_std
        if self.log_std_bounds is not None:
            log_std = log_std.clamp(*self.log_std_bounds)
        std = log_std.exp().expand_as(mean)  # (B, action_dim)

        # Sum log-probabilities and entropy over the action components.
        return Independent(Normal(mean, std), 1)
