"""Running statistics for observation normalization and reward scaling.

This module has two classes:

* ``RunningNormalizer`` keeps a running mean and variance of some data and
  standardizes new data with them: ``(x - mean) / sqrt(var + epsilon)``.
  ``ObservationProcessor`` (``processors.py``) uses one per normalized
  observation feature.
* ``RewardScaler`` divides rewards by the running standard deviation of the
  discounted return. It is a training-only tool: it changes the rewards used
  for training targets, never the rewards that are logged or stored.

Both keep their statistics in float64 buffers, so the values are saved by
``state_dict()`` and stay accurate after millions of samples.
"""

from __future__ import annotations

import math
from numbers import Real

import torch
import torch.nn as nn


class RunningNormalizer(nn.Module):
    """Running mean and variance of a feature, used to standardize it.

    The statistics cover the trailing ``shape`` axes; all leading axes are
    treated as samples. For example, with ``shape=(11,)``, an input shaped
    ``(T, N, 11)`` adds ``T * N`` samples of an 11-number feature, and the
    mean and variance each have shape ``(11,)``.

    Statistics change only when ``update`` is called. Applying the normalizer
    (``forward``) never changes them. Before the first update, the normalizer
    is identity: values pass through unchanged (only converted to float32),
    without clipping.

    Args:
        shape: Shape of one sample, without leading axes, for example ``(11,)``.
        epsilon: Small number added to the variance before the square root, so
            the normalizer never divides by zero.
        clip: Normalized values are limited to ``[-clip, clip]``. ``None`` turns
            clipping off.

    Attributes:
        mean: Running mean, float64, shaped ``shape``.
        var: Running variance, float64, shaped ``shape``. Starts at 1.
        count: Number of samples seen so far, float64 scalar.
    """

    def __init__(
        self,
        shape   :   tuple[int, ...] | list[int],
        epsilon :   float = 1e-8,
        clip    :   float | None = 10.0,
    ) -> None:
        """Check the settings and create empty statistics.

        Raises:
            ValueError: If ``epsilon`` or ``clip`` is not a finite positive
                number.
        """
        super().__init__()

        # 1. epsilon and clip must be finite positive numbers (clip may be None).
        for name, value in (("epsilon", epsilon), ("clip", clip)):
            if name == "clip" and value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")

        # 2. Store the settings.
        self.shape = tuple(shape)
        self.epsilon = epsilon
        self.clip = clip

        # 3. Create the statistics as buffers: they move with ``.to(device)``
        #    and are saved by ``state_dict()``, but are not trained by an
        #    optimizer. float64 keeps them accurate over millions of samples.
        self.register_buffer("mean", torch.zeros(self.shape, dtype=torch.float64))
        self.register_buffer("var", torch.ones(self.shape, dtype=torch.float64))
        self.register_buffer("count", torch.zeros((), dtype=torch.float64))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """Add new samples to the running mean and variance.

        All leading axes are pooled into one set of samples. The new samples are
        merged with everything seen before, so updating in several pieces gives
        the same result as one update with all the data, up to floating-point
        rounding.

        Args:
            x: Samples shaped ``(..., *shape)``.

        Raises:
            TypeError: If ``x`` is not a real-valued tensor.
            ValueError: If the trailing shape is wrong, a sample is NaN or
                infinite, or the merged statistics would overflow. The
                statistics are left unchanged in every error case.
        """
        # 1. Only real-valued tensors can be normalized. Structured data
        #    (TensorDicts) must be normalized feature by feature.
        if not isinstance(x, torch.Tensor):
            raise TypeError("expected a Tensor; normalize structured inputs per key")
        if x.is_complex():
            raise TypeError("normalization samples must be real-valued")

        # 2. The trailing axes must match the feature shape.
        if self.shape and tuple(x.shape[-len(self.shape):]) != self.shape:
            raise ValueError(
                f"expected trailing shape {self.shape}, got {tuple(x.shape)}"
            )

        # 3. Flatten all leading axes into one sample axis: (T, N, 11) becomes
        #    (T * N, 11). Compute in float64 on the statistics' device.
        x = x.reshape(-1, *self.shape).to(
            device=self.mean.device,
            dtype=torch.float64,
        )

        # 4. Nothing to add for an empty batch.
        batch_count = x.shape[0]
        if batch_count == 0:
            return

        # 5. One NaN or infinity would corrupt the statistics for good.
        if not torch.isfinite(x).all().item():
            raise ValueError("normalization samples must be finite")

        # 6. Mean and variance of the new samples alone.
        batch_var, batch_mean = torch.var_mean(x, dim=0, unbiased=False)
        total = self.count + batch_count
        delta = batch_mean - self.mean

        # 7. Merge old and new statistics (Chan's parallel algorithm):
        #    - the mean moves toward the new mean by the new data's share of
        #      all samples;
        #    - the total spread is the old spread plus the new spread plus a
        #      term for how far apart the two means were.
        sum_squares = (
            self.var * self.count
            + batch_var * batch_count
            + delta.square() * (self.count * batch_count / total)
        )
        mean = (self.mean + delta * (batch_count / total)).to(self.mean)
        var = (sum_squares / total).to(self.var)
        total = total.to(self.count)

        # 8. Check the new values before saving them, so an overflow never
        #    leaves the statistics half-updated.
        valid_moments = (
            torch.isfinite(mean).all()
            & torch.isfinite(var).all()
            & torch.isfinite(total)
        )
        if not valid_moments.item():
            raise ValueError("normalization statistics overflowed; state was not updated")

        # 9. Save the merged statistics in place.
        self.mean.copy_(mean)
        self.var.copy_(var)
        self.count.copy_(total)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standardize ``x`` with the current statistics.

        Computes ``(x - mean) / sqrt(var + epsilon)`` and clips the result to
        ``[-clip, clip]``. Before the first update (``count == 0``) it returns
        ``x`` unchanged instead. The statistics themselves never change here.

        Args:
            x: Values shaped ``(..., *shape)``.

        Returns:
            A new float32 tensor with the same shape as ``x``.
        """
        # 1. Standardize in float64 for accuracy.
        normalized = (x.to(torch.float64) - self.mean) / torch.sqrt(
            self.var + self.epsilon
        )

        # 2. Optionally limit extreme values.
        if self.clip is not None:
            normalized = normalized.clamp(-self.clip, self.clip)

        # 3. With no statistics yet, return the input unchanged (identity),
        #    including values beyond clip. The result is always float32.
        return torch.where(
            self.count > 0,
            normalized,
            x.to(torch.float64),
        ).to(torch.float32)


class RewardScaler(nn.Module):
    """Scale rewards by the running standard deviation of the discounted return.

    Raw rewards can have very different sizes from task to task. Dividing them
    by a running estimate of the return's spread keeps value targets in a
    similar range, which helps PPO-style training. Rewards are only divided,
    not shifted: the mean is not subtracted, so the sign of each reward stays
    the same.

    How it works: each environment keeps a running discounted return,
    ``R = gamma * R + reward``, which restarts at the end of every episode.
    The running variance of these returns, shared by all environments, gives
    the scale: ``scaled_reward = reward / sqrt(var + epsilon)``.

    Two modes decide when the statistics are updated:

    * ``update_per_step=False`` (default): process the whole rollout first,
      update the statistics once, then scale every reward with the same value.
    * ``update_per_step=True``: update after every step and scale each step's
      rewards with the statistics at that moment. This matches the order used
      by Stable-Baselines3's ``VecNormalize``.

    The input rewards are never changed; a scaled copy is returned, so logged
    returns stay raw. This is a training-only tool and is not part of the
    saved policy.

    Args:
        num_envs: Number of parallel environments, N.
        gamma: Discount factor used for the running return, between 0 and 1.
        epsilon: Small number added to the variance before the square root.
        update_per_step: Which update mode to use (see above).
        clip: Scaled rewards are limited to ``[-clip, clip]``. ``None`` turns
            clipping off.
    """

    def __init__(
        self,
        num_envs: int,
        gamma: float,
        epsilon: float = 1e-8,
        update_per_step: bool = False,
        clip: float | None = None,
    ) -> None:
        """Check the settings and create the return statistics and accumulators.

        Raises:
            ValueError: If ``num_envs``, ``gamma``, ``epsilon`` or ``clip`` is
                invalid.
            TypeError: If ``update_per_step`` is not a boolean.
        """
        super().__init__()

        # 1. Check every setting.
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")
        if (
            isinstance(gamma, bool)
            or not isinstance(gamma, Real)
            or not math.isfinite(gamma)
            or not 0.0 <= gamma <= 1.0
        ):
            raise ValueError("gamma must be finite and between 0 and 1")
        for name, value in (("epsilon", epsilon), ("clip", clip)):
            if name == "clip" and value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(update_per_step, bool):
            raise TypeError("update_per_step must be boolean")

        # 2. Store the settings.
        self.gamma = gamma
        self.epsilon = epsilon
        self.update_per_step = update_per_step
        self.clip = clip

        # 3. Running statistics of the discounted returns: one scalar mean and
        #    variance shared by all environments. Saved with the checkpoint.
        self.return_stats = RunningNormalizer(shape=(), clip=None)

        # 4. One running discounted return per environment. Not saved
        #    (persistent=False): a resumed run starts fresh episodes, so these
        #    restart at zero.
        self.register_buffer(
            "returns",
            torch.zeros(num_envs, dtype=torch.float64),
            persistent=False,
        )

    @property
    def config(self) -> dict[str, int | float | bool | None]:
        """Settings needed to interpret the saved return statistics, as plain data."""
        return {
            "num_envs": self.returns.numel(),
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "update_per_step": self.update_per_step,
            "clip": self.clip,
        }

    @torch.no_grad()
    def forward(
        self,
        rewards: torch.Tensor,  # (T, N), raw rewards
        dones: torch.Tensor,    # (T, N), terminated or truncated
    ) -> torch.Tensor:
        """Update the return statistics and return a scaled copy of the rewards.

        Args:
            rewards: Raw rewards in time order, shaped ``(T, N)``: T steps from
                N environments.
            dones: True where an episode ended (terminated or truncated),
                shaped ``(T, N)``.

        Returns:
            Scaled rewards shaped ``(T, N)``. The input ``rewards`` is not
            modified.

        Raises:
            ValueError: If the shapes are wrong or a reward is NaN or infinite.
                Nothing is updated in that case.
            TypeError: If ``rewards`` is not a floating-point tensor.
        """
        # 1. Check shapes and dtypes.
        if rewards.ndim != 2 or rewards.shape[1] != self.returns.numel():
            raise ValueError("rewards must have shape (T, num_envs)")
        if not rewards.is_floating_point():
            raise TypeError("rewards must be floating-point tensors")
        if dones.shape != rewards.shape or dones.dtype != torch.bool:
            raise ValueError("dones must be boolean with the same shape as rewards")

        # 2. Check the whole rollout before changing any running return, so a
        #    bad reward never leaves the accumulators half-updated.
        if not torch.isfinite(rewards).all().item():
            raise ValueError("rewards must be finite")

        # 3. Prepare the output for the chosen mode: scaled rewards directly
        #    (per-step mode), or the returns at every step (default mode).
        if self.update_per_step:
            scaled = torch.empty_like(rewards)
        else:
            discounted_returns = torch.empty_like(rewards, dtype=torch.float64)

        # 4. Walk through the rollout in time order.
        for t in range(rewards.shape[0]):
            # Running discounted return of each environment: R = gamma * R + r.
            self.returns.mul_(self.gamma).add_(rewards[t])
            if self.update_per_step:
                # Per-step mode: update the statistics now, then scale this step.
                self.return_stats.update(self.returns)
                std = torch.sqrt(self.return_stats.var + self.epsilon)
                scaled[t] = rewards[t] / std
            else:
                # Default mode: remember the returns; scale after the loop.
                discounted_returns[t] = self.returns

            # Restart the return of environments whose episode just ended. The
            # final reward was already included above.
            self.returns[dones[t]] = 0.0

        # 5. Default mode: update the statistics once with all T * N returns,
        #    then scale every reward with the same standard deviation.
        if not self.update_per_step:
            self.return_stats.update(discounted_returns)
            std = torch.sqrt(self.return_stats.var + self.epsilon)
            scaled = (rewards / std).to(dtype=rewards.dtype)

        # 6. Optionally limit extreme values. Raw rewards stay unchanged; only
        #    this copy feeds the training targets.
        if self.clip is not None:
            scaled = scaled.clamp(-self.clip, self.clip)
        return scaled
