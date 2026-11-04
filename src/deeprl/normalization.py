"""Running statistics for observation normalization and reward scaling."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class RunningNormalizer(nn.Module):
    """Standardize features using explicitly updated running statistics."""
    
    def __init__(
        self,
        shape   :   tuple[int, ...] | list[int],
        epsilon :   float = 1e-8,
        clip    :   float | None = 10.0,
    ) -> None:
        super().__init__()
        if epsilon <= 0 or (clip is not None and clip <= 0):
            raise ValueError("epsilon and clip must be positive")
        
        self.shape = tuple(shape)
        self.epsilon = epsilon
        self.clip = clip
        
        # Buffers move with the module and are included in its state_dict.
        self.register_buffer("mean", torch.zeros(self.shape, dtype=torch.float64))
        self.register_buffer("var", torch.ones(self.shape, dtype=torch.float64))
        self.register_buffer("count", torch.zeros((), dtype=torch.float64))
    
    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """Update feature statistics, pooling all leading batch dimensions."""
        
        if not isinstance(x, torch.Tensor):
            raise TypeError("expected a Tensor; normalize structured inputs per key")
        
        if self.shape and tuple(x.shape[-len(self.shape):]) != self.shape:
            raise ValueError(
                f"expected trailing shape {self.shape}, got {tuple(x.shape)}"
            )
        
        x = x.reshape(-1, *self.shape).to(
            device=self.mean.device, dtype=torch.float64
        )
        batch_count = x.shape[0]
        if batch_count == 0:
            return

        batch_var, batch_mean = torch.var_mean(x, dim=0, unbiased=False)
        total = self.count + batch_count
        delta = batch_mean - self.mean

        # Merge existing and incoming moments, including their mean difference.
        sum_squares = (
            self.var * self.count
            + batch_var * batch_count
            + delta.square() * (self.count * batch_count / total)
        )
        self.mean.add_(delta * (batch_count / total))
        self.var.copy_(sum_squares / total)
        self.count.copy_(total)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply current statistics without updating them; return float32."""
        normalized = (x.to(torch.float64) - self.mean) / torch.sqrt(
            self.var + self.epsilon
        )
        if self.clip is not None:
            normalized = normalized.clamp(-self.clip, self.clip)
        return normalized.to(torch.float32)


class RewardScaler(nn.Module):
    """Scale rewards by return std, optionally updating per step and clipping."""

    def __init__(
        self,
        num_envs: int,
        gamma: float,
        epsilon: float = 1e-8,
        update_per_step: bool = False,
        clip: float | None = None,
    ) -> None:
        super().__init__()
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if not isinstance(update_per_step, bool):
            raise TypeError("update_per_step must be boolean")
        if clip is not None and (not math.isfinite(clip) or clip <= 0):
            raise ValueError("clip must be finite and positive")

        self.gamma = gamma
        self.epsilon = epsilon
        self.update_per_step = update_per_step
        self.clip = clip
        self.return_stats = RunningNormalizer(shape=(), clip=None)

        # Episode accumulators are not restored: resume starts fresh episodes.
        self.register_buffer(
            "returns",
            torch.zeros(num_envs, dtype=torch.float64),
            persistent=False,
        )

    @torch.no_grad()
    def forward(
        self,
        rewards: torch.Tensor,  # (T, N), raw rewards
        dones: torch.Tensor,    # (T, N), terminated or truncated
    ) -> torch.Tensor:
        """Update return statistics and produce a scaled copy of the rewards."""
        if rewards.ndim != 2 or rewards.shape[1] != self.returns.numel():
            raise ValueError("rewards must have shape (T, num_envs)")
        if not rewards.is_floating_point():
            raise TypeError("rewards must be floating-point tensors")
        if dones.shape != rewards.shape or dones.dtype != torch.bool:
            raise ValueError("dones must be boolean with the same shape as rewards")

        if self.update_per_step:
            scaled = torch.empty_like(rewards)
        else:
            discounted_returns = torch.empty_like(rewards, dtype=torch.float64)

        for t in range(rewards.shape[0]):
            self.returns.mul_(self.gamma).add_(rewards[t])
            if self.update_per_step:
                self.return_stats.update(self.returns)
                std = torch.sqrt(self.return_stats.var + self.epsilon)
                scaled[t] = rewards[t] / std
            else:
                discounted_returns[t] = self.returns

            # Include the final reward before clearing completed episodes.
            self.returns[dones[t]] = 0.0

        if not self.update_per_step:
            # By default, pool statistics across time and environments.
            self.return_stats.update(discounted_returns)
            std = torch.sqrt(self.return_stats.var + self.epsilon)
            scaled = (rewards / std).to(dtype=rewards.dtype)

        # Preserve raw rewards; only the returned copy feeds training targets.
        if self.clip is not None:
            scaled = scaled.clamp(-self.clip, self.clip)
        return scaled
