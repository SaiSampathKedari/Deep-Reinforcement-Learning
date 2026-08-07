"""Metric recording for training loops.

One object owns everything an algorithm would otherwise repeat: the global step
counter, per-environment episode accumulators, interval averaging of training
diagnostics, and the optional TensorBoard mirror. An algorithm makes exactly one
call per vector step and returns `logger.finish()` at the end.

Metrics land in two places at once:

    history   dict[str, list[(global_step, value)]] -- returned to the caller,
              used by tests, notebooks and multi-seed aggregation.
    writer    an optional caller-owned SummaryWriter, for live curves.

The writer is imported only under TYPE_CHECKING, so `tensorboard` stays an
optional extra (`uv sync --extra tb`) and importing this module never requires it.

Episode statistics are tracked here rather than with
`gymnasium.wrappers.vector.RecordEpisodeStatistics`, which is unusable with
`AutoresetMode.DISABLED`: its `reset()` inspects `options["reset_mask"]` *after*
`SyncVectorEnv.reset()` has already popped that key, so a partial reset silently
clears the counters of every environment, not just the finished ones.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

import torch

from deeprl.utils import explained_variance

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

MetricPoint: TypeAlias = tuple[int, float]
MetricHistory: TypeAlias = dict[str, list[MetricPoint]]
Scalar: TypeAlias = int | float | torch.Tensor


def _scalar(value: Scalar) -> float:
    """Coerce a python number or single-element tensor to a float."""
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(
                f"only scalar metrics can be recorded; got shape {tuple(value.shape)}"
            )
        return float(value.detach().item())
    return float(value)


class Logger:
    """Records episode statistics and training diagnostics for one run.

    Args:
        num_envs: number of vectorized environments; one `log_step` call
            advances the global step by this much.
        log_every: emit averaged diagnostics every this many environment steps.
            0 disables them (episodes are still recorded).
        writer: optional SummaryWriter. Constructed and closed by the caller.
        device: device for the episode accumulators; match the env's.
        ev_window: number of (value, target) pairs explained variance is
            measured over. Too small and the denominator Var[target] collapses,
            which makes EV swing wildly on a perfectly healthy critic.
        ev_name: metric name for explained variance. Name it for the target the
            algorithm actually supplies -- a one-step TD target and a GAE return
            are different quantities and should not share a key.

    Typical use::

        logger = Logger(envs.num_envs, cfg.log_every, writer, cfg.device)
        ...
        logger.log_step(rewards, dones, metrics={...}, values=v, td_targets=t)
        ...
        return logger.finish()
    """

    def __init__(
        self,
        num_envs: int,
        log_every: int = 1000,
        writer: SummaryWriter | None = None,
        device: torch.device = torch.device("cpu"),
        ev_window: int = 2000,
        ev_name: str = "diagnostics/explained_variance",
    ) -> None:
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        if log_every < 0:
            raise ValueError(f"log_every must be non-negative, got {log_every}")

        self.num_envs = num_envs
        self.log_every = log_every
        self.writer = writer
        self.device = torch.device(device)
        self.global_step = 0

        self._history: defaultdict[str, list[MetricPoint]] = defaultdict(list)

        # Per-environment episode accumulators.
        self._ep_returns = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self._ep_lengths = torch.zeros(num_envs, dtype=torch.int64, device=self.device)

        # Interval accumulators. Counts are per metric, not shared: an algorithm
        # may record a metric on only some steps (PPO's clipfrac, SAC's alpha).
        self._sums: defaultdict[str, float] = defaultdict(float)
        self._counts: defaultdict[str, int] = defaultdict(int)

        # Explained variance needs paired arrays over a window, not a scalar.
        self._ev_name = ev_name
        self._ev_values: deque[float] = deque(maxlen=ev_window)
        self._ev_targets: deque[float] = deque(maxlen=ev_window)

        self._start_time = time.perf_counter()
        self._next_log_step = log_every

    # ----------------------------------------------------------------- record

    def record(self, name: str, value: Scalar, step: int | None = None) -> None:
        """Record one scalar under its full name, immediately.

        Use for values that should not be interval-averaged, e.g.
        `logger.record("eval/mean_return", r)`.
        """
        scalar = _scalar(value)
        at = self.global_step if step is None else int(step)
        self._history[name].append((at, scalar))
        if self.writer is not None:
            self.writer.add_scalar(name, scalar, global_step=at)

    def log_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        metrics: Mapping[str, Scalar] | None = None,
        values: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
    ) -> None:
        """Call once per vector step, after the update.

        Advances the global step, accumulates episode return and length, emits
        `charts/episodic_return` and `charts/episodic_length` for every episode
        that just finished, and averages `metrics` over the logging interval.

        Args:
            rewards: (num_envs,) rewards from this step.
            dones: (num_envs,) bool, `terminated | truncated`.
            metrics: scalar diagnostics under their full names, e.g.
                `{"losses/actor_loss": ..., "grads/actor_norm": ...}`. Each is
                averaged over the interval independently.
            values: optional (num_envs,) critic predictions.
            targets: optional (num_envs,) critic regression targets. Supplied
                together with `values`, these feed a rolling window from which
                `diagnostics/explained_variance` is computed at each flush.

                The number's meaning depends on what the caller passes: a
                one-step TD target here, empirical rollout returns in PPO. It
                sits under `diagnostics/` rather than `losses/` so the two are
                not silently compared.
        """
        if rewards.shape != (self.num_envs,):
            raise ValueError(
                f"rewards must have shape ({self.num_envs},), got {tuple(rewards.shape)}"
            )
        if dones.shape != (self.num_envs,):
            raise ValueError(
                f"dones must have shape ({self.num_envs},), got {tuple(dones.shape)}"
            )

        # Normalize onto the logger's device so the boolean indexing below
        # cannot fail on an env that returns tensors from somewhere else.
        rewards = rewards.detach().to(device=self.device, dtype=self._ep_returns.dtype)
        dones = dones.detach().to(device=self.device, dtype=torch.bool)

        self.global_step += self.num_envs

        self._ep_returns += rewards
        self._ep_lengths += 1

        if metrics:
            for name, value in metrics.items():
                self._sums[name] += _scalar(value)
                self._counts[name] += 1

        if values is not None and targets is not None:
            self._ev_values.extend(values.detach().flatten().tolist())
            self._ev_targets.extend(targets.detach().flatten().tolist())

        if dones.any().item():
            for ep_return, ep_length in zip(
                self._ep_returns[dones].tolist(),
                self._ep_lengths[dones].tolist(),
                strict=True,
            ):
                self.record("charts/episodic_return", ep_return)
                self.record("charts/episodic_length", ep_length)
            self._ep_returns[dones] = 0.0
            self._ep_lengths[dones] = 0

        if self.log_every > 0 and self.global_step >= self._next_log_step:
            self._flush()
            while self._next_log_step <= self.global_step:
                self._next_log_step += self.log_every

    def _flush(self) -> None:
        """Emit interval-averaged diagnostics and clear the accumulators."""
        for name, total in self._sums.items():
            count = self._counts[name]
            if count > 0:
                self.record(name, total / count)
        self._sums.clear()
        self._counts.clear()

        if self._ev_values:
            self.record(
                self._ev_name,
                explained_variance(
                    torch.tensor(list(self._ev_values)),
                    torch.tensor(list(self._ev_targets)),
                ),
            )

        self.record("charts/SPS", self.global_step / max(self._elapsed, 1e-8))

    # ------------------------------------------------------------------- read

    @property
    def history(self) -> MetricHistory:
        """All recorded metrics as `{name: [(global_step, value), ...]}`."""
        return {name: list(points) for name, points in self._history.items()}

    @property
    def _elapsed(self) -> float:
        return time.perf_counter() - self._start_time

    def recent(self, name: str, n: int = 10) -> float:
        """Mean of the last `n` recorded values of `name`; nan if none."""
        points = self._history.get(name, [])[-n:]
        if not points:
            return float("nan")
        return sum(v for _, v in points) / len(points)

    def finish(self) -> MetricHistory:
        """Flush the final partial interval and return the history.

        Does not close the writer -- the caller owns it.
        """
        if self.log_every > 0 and (self._sums or self._ev_values):
            self._flush()
        return self.history

    # ------------------------------------------------------------------ write

    def log_hyperparameters(self, cfg: Any) -> None:
        """Write the config to TensorBoard as a markdown table (CleanRL's trick)."""
        if self.writer is None:
            return
        from dataclasses import asdict, is_dataclass

        fields = asdict(cfg) if is_dataclass(cfg) else dict(vars(cfg))
        rows = "\n".join(f"|{k}|{v}|" for k, v in fields.items())
        self.writer.add_text("hyperparameters", "|param|value|\n|-|-|\n" + rows)

    def save(self, path: str | Path) -> None:
        """Write the history to JSON. Called by train.py, never by an algorithm."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.history, f)
