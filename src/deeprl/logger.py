"""Run-level metric collection, reduction, and persistence."""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import torch

from deeprl.stats import TrainingStats
from deeprl.utils import explained_variance

if TYPE_CHECKING:
    from _io import TextIOWrapper
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm

MetricPoint: TypeAlias = tuple[int, float]
MetricHistory: TypeAlias = dict[str, list[MetricPoint]]
Scalar: TypeAlias = int | float | torch.Tensor


def _scalar(value: Scalar) -> float:
    """Convert a Python number or scalar tensor to a detached float."""
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(
                f"only scalar metrics can be recorded; got shape {tuple(value.shape)}"
            )
        return float(value.detach().item())
    return float(value)


class Logger:
    """Track and write metrics for one training run and one random seed.

    Collection calls update persistent episode state. Update calls submit
    diagnostics that have already been reduced within an algorithm update.
    Periodic output happens only through :meth:`maybe_dump`, so a report always
    describes complete collection/update cycles.
    """

    def __init__(
        self,
        num_envs: int,
        log_every: int = 1000,
        writer: SummaryWriter | None = None,
        device: torch.device = torch.device("cpu"),
        *,
        metrics_path: str | Path | None = None,
        episode_window: int = 100,
        ev_window: int = 2000,
        ev_name: str = "diagnostics/td_target_explained_variance",
        history_size: int | None = 10_000,
        total_timesteps: int | None = None,
        show_progress: bool = False,
    ) -> None:
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        if log_every < 0:
            raise ValueError(f"log_every must be non-negative, got {log_every}")
        if history_size is not None and history_size < 0:
            raise ValueError(
                f"history_size must be non-negative or None, got {history_size}"
            )

        self.num_envs = num_envs
        self.log_every = log_every
        self.writer = writer
        self.device = torch.device(device)
        self.global_step = 0
        self.gradient_step = 0

        self._history: defaultdict[str, deque[MetricPoint]] = defaultdict(
            lambda: deque(maxlen=history_size)
        )

        self._episode_returns = torch.zeros(
            num_envs, dtype=torch.float32, device=self.device
        )
        self._episode_lengths = torch.zeros(
            num_envs, dtype=torch.int64, device=self.device
        )
        self._recent_returns: deque[float] = deque(maxlen=episode_window)
        self._recent_lengths: deque[float] = deque(maxlen=episode_window)

        # Update-level metrics are combined using the number of learner
        # minibatches represented by each TrainingStats submission.
        self._metric_sums: defaultdict[str, float] = defaultdict(float)
        self._metric_weights: defaultdict[str, int] = defaultdict(int)

        # One-step actor-critic has too few samples per update for a meaningful
        # explained variance, so it supplies paired samples for this window.
        self._ev_name = ev_name
        self._ev_values: deque[float] = deque(maxlen=ev_window)
        self._ev_targets: deque[float] = deque(maxlen=ev_window)

        self._start_time = time.perf_counter()
        self._last_dump_time = self._start_time
        self._last_dump_step = 0
        self._next_log_step = log_every if log_every > 0 else None

        self._metrics_file: TextIOWrapper | None = None
        if metrics_path is not None:
            path = Path(metrics_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._metrics_file = path.open("a", encoding="utf-8")

        self._progress: tqdm | None = None
        if show_progress and total_timesteps is not None:
            from tqdm import tqdm

            self._progress = tqdm(total=total_timesteps, unit="step", dynamic_ncols=True)

    # -------------------------------------------------------------- collection

    def log_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        *,
        values: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
    ) -> None:
        """Accumulate one vector-environment step without dumping metrics."""
        if rewards.shape != (self.num_envs,):
            raise ValueError(
                f"rewards must have shape ({self.num_envs},), got {tuple(rewards.shape)}"
            )
        if dones.shape != (self.num_envs,):
            raise ValueError(
                f"dones must have shape ({self.num_envs},), got {tuple(dones.shape)}"
            )
        if (values is None) != (targets is None):
            raise ValueError("values and targets must be provided together")
        if values is not None and values.shape != targets.shape:
            raise ValueError("values and targets must have the same shape")

        rewards = rewards.detach().to(
            device=self.device, dtype=self._episode_returns.dtype
        )
        dones = dones.detach().to(device=self.device, dtype=torch.bool)

        self.global_step += self.num_envs
        self._episode_returns += rewards
        self._episode_lengths += 1

        if self._progress is not None:
            remaining = max(self._progress.total - self._progress.n, 0)
            self._progress.update(min(self.num_envs, remaining))

        if values is not None and targets is not None:
            self._ev_values.extend(values.detach().flatten().tolist())
            self._ev_targets.extend(targets.detach().flatten().tolist())

        if dones.any().item():
            for episode_return, episode_length in zip(
                self._episode_returns[dones].tolist(),
                self._episode_lengths[dones].tolist(),
                strict=True,
            ):
                self._recent_returns.append(episode_return)
                self._recent_lengths.append(float(episode_length))
                self._emit(
                    "episode",
                    {
                        "charts/episodic_return": episode_return,
                        "charts/episodic_length": episode_length,
                    },
                )
            self._episode_returns[dones] = 0.0
            self._episode_lengths[dones] = 0

    def log_rollout(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        """Accumulate a time-major ``(T, N)`` rollout without dumping."""
        if rewards.ndim != 2 or rewards.shape[1] != self.num_envs:
            raise ValueError(
                f"rewards must have shape (T, {self.num_envs}), got {tuple(rewards.shape)}"
            )
        if dones.shape != rewards.shape:
            raise ValueError(
                f"dones must have shape {tuple(rewards.shape)}, got {tuple(dones.shape)}"
            )

        for step_rewards, step_dones in zip(rewards, dones, strict=True):
            self.log_step(step_rewards, step_dones)

    # ------------------------------------------------------------------ update

    def log_update(self, stats: TrainingStats) -> None:
        """Accumulate one complete algorithm-update summary."""
        weight = stats.gradient_steps
        if weight <= 0:
            raise ValueError("gradient_steps must be positive")

        self.gradient_step += weight
        for name, value in stats.metrics.items():
            scalar = _scalar(value)
            self._metric_sums[name] += scalar * weight
            self._metric_weights[name] += weight

    def maybe_dump(self) -> None:
        """Write one report when the environment-step threshold is crossed."""
        if self._next_log_step is None or self.global_step < self._next_log_step:
            return

        self._dump()
        while self._next_log_step <= self.global_step:
            self._next_log_step += self.log_every

    def _dump(self) -> None:
        now = time.perf_counter()
        metrics = {
            name: total / self._metric_weights[name]
            for name, total in self._metric_sums.items()
            if self._metric_weights[name] > 0
        }

        if self._recent_returns:
            metrics["charts/episodic_return_mean"] = sum(self._recent_returns) / len(
                self._recent_returns
            )
            metrics["charts/episodic_length_mean"] = sum(self._recent_lengths) / len(
                self._recent_lengths
            )

        if self._ev_values:
            metrics[self._ev_name] = explained_variance(
                torch.tensor(list(self._ev_values)),
                torch.tensor(list(self._ev_targets)),
            )

        elapsed = now - self._start_time
        interval_elapsed = now - self._last_dump_time
        interval_steps = self.global_step - self._last_dump_step
        metrics["charts/SPS"] = self.global_step / max(elapsed, 1e-8)
        metrics["charts/interval_SPS"] = interval_steps / max(interval_elapsed, 1e-8)

        self._emit("train", metrics, elapsed_seconds=elapsed)
        self._metric_sums.clear()
        self._metric_weights.clear()
        self._last_dump_step = self.global_step
        self._last_dump_time = now

        if self._progress is not None:
            postfix: dict[str, str] = {}
            if self._recent_returns:
                postfix["return"] = f"{metrics['charts/episodic_return_mean']:.2f}"
            if "losses/policy_loss" in metrics:
                postfix["policy_loss"] = f"{metrics['losses/policy_loss']:.3g}"
            self._progress.set_postfix(postfix)

    # --------------------------------------------------------------- evaluation

    def log_evaluation(
        self,
        episode_returns: torch.Tensor,
        episode_lengths: torch.Tensor,
        *,
        deterministic: bool,
    ) -> None:
        """Record raw episodes and their within-run evaluation summary."""
        returns = episode_returns.detach().flatten().to(dtype=torch.float32, device="cpu")
        lengths = episode_lengths.detach().flatten().to(dtype=torch.float32, device="cpu")
        if returns.shape != lengths.shape:
            raise ValueError("episode_returns and episode_lengths must have the same shape")
        if returns.numel() == 0:
            raise ValueError("evaluation must contain at least one episode")

        mode = "deterministic" if deterministic else "stochastic"
        for episode_return, episode_length in zip(
            returns.tolist(), lengths.tolist(), strict=True
        ):
            self._emit(
                "eval_episode",
                {
                    f"eval/{mode}_episodic_return": episode_return,
                    f"eval/{mode}_episodic_length": episode_length,
                },
            )

        self._emit(
            "eval",
            {
                f"eval/{mode}_return_mean": returns.mean().item(),
                f"eval/{mode}_return_std": returns.std(unbiased=False).item(),
                f"eval/{mode}_length_mean": lengths.mean().item(),
            },
        )

    # ------------------------------------------------------------------- output

    def record(self, name: str, value: Scalar, step: int | None = None) -> None:
        """Immediately emit one custom scalar metric."""
        self._emit("metric", {name: _scalar(value)}, step=step)

    def _emit(
        self,
        kind: Literal["episode", "train", "eval_episode", "eval", "metric"],
        metrics: Mapping[str, Scalar],
        *,
        step: int | None = None,
        elapsed_seconds: float | None = None,
    ) -> None:
        at = self.global_step if step is None else int(step)
        scalars = {name: _scalar(value) for name, value in metrics.items()}

        for name, value in scalars.items():
            self._history[name].append((at, value))
            if self.writer is not None:
                self.writer.add_scalar(name, value, global_step=at)

        if self._metrics_file is not None:
            row: dict[str, Any] = {
                "kind": kind,
                "global_step": at,
                "gradient_step": self.gradient_step,
            }
            if elapsed_seconds is not None:
                row["elapsed_seconds"] = elapsed_seconds
            row.update(
                {
                    name: value if math.isfinite(value) else None
                    for name, value in scalars.items()
                }
            )
            self._metrics_file.write(json.dumps(row, allow_nan=False) + "\n")
            self._metrics_file.flush()

    def log_hyperparameters(self, cfg: Any) -> None:
        """Mirror a resolved configuration into TensorBoard."""
        if self.writer is None:
            return
        fields = asdict(cfg) if is_dataclass(cfg) else dict(vars(cfg))
        rows = "\n".join(f"|{key}|{value}|" for key, value in fields.items())
        self.writer.add_text("hyperparameters", "|param|value|\n|-|-|\n" + rows)

    @property
    def history(self) -> MetricHistory:
        """Recent in-memory metrics as ``{name: [(step, value), ...]}``."""
        return {name: list(points) for name, points in self._history.items()}

    def recent(self, name: str, n: int = 10) -> float:
        """Return the mean of the last ``n`` in-memory values, or NaN."""
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        points = list(self._history.get(name, ()))[-n:]
        if not points:
            return float("nan")
        return sum(value for _, value in points) / len(points)

    def finish(self) -> MetricHistory:
        """Force the final report, flush outputs, and return recent history."""
        if self.global_step != self._last_dump_step or self._metric_sums:
            self._dump()
        if self.writer is not None:
            self.writer.flush()
        if self._metrics_file is not None:
            self._metrics_file.close()
            self._metrics_file = None
        if self._progress is not None:
            self._progress.close()
            self._progress = None
        return self.history
