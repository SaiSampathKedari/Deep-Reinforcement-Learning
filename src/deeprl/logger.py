"""Record what happens during one training run: episodes, training, evaluation, speed.

Where the metrics go:

* ``metrics.jsonl`` (optional, ``metrics_path``): the permanent record, one
  JSON object per line. Every row has a ``kind``:

  - ``"episode"``: one finished training episode (return and length);
  - ``"train"``: one periodic report (averaged losses and diagnostics,
    recent episode means, steps per second);
  - ``"eval_episode"``: one evaluation episode;
  - ``"eval"``: the summary of one evaluation (mean and spread);
  - ``"metric"``: a custom value from ``record``.

  An example row::

      {"kind": "train", "global_step": 16384, "gradient_step": 320,
       "elapsed_seconds": 12.3, "losses/policy_loss": -0.012, "charts/SPS": 1331.0}

* TensorBoard (optional, ``writer``): the same values as curves.
* Memory: recent values per metric (``history``, ``recent``), for tests and
  quick checks.
* A progress bar (optional, ``show_progress``).

How the trainer uses it (see ``training/train_on_policy.py``)::

    for each update:
        collect a rollout and update the learner
        logger.log_rollout(rewards, dones, progress=progress)   # training episodes
        logger.log_update(stats, progress=progress)              # losses and diagnostics
        logger.maybe_dump()                                      # report if log_every steps passed
    logger.finish()                                              # final report, close the metrics file

The trainer owns the step counters (``TrainingProgress``). The logger only
copies the counters it is given; it never advances them itself.

Logged training rewards are the rewards the training environment gives,
before the learner's reward scaling. Reward wrappers on the environment
itself are included, since the logger only sees what ``env.step`` returns.
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import torch

from deeprl.stats import TrainingProgress, TrainingStats
from deeprl.utils import explained_variance

if TYPE_CHECKING:
    from _io import TextIOWrapper

    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm

# One recorded value: (environment step, value).
MetricPoint: TypeAlias = tuple[int, float]

# All recorded values: {metric name: [(step, value), ...]}.
MetricHistory: TypeAlias = dict[str, list[MetricPoint]]

# A value that can be logged: a Python number or a one-element tensor.
Scalar: TypeAlias = int | float | torch.Tensor


def _scalar(value: Scalar) -> float:
    """Convert a Python number or a one-element tensor to a plain float.

    Raises:
        ValueError: If a tensor holds more than one number.
    """
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(
                f"only scalar metrics can be recorded; got shape {tuple(value.shape)}"
            )
        return float(value.detach().item())
    return float(value)


class Logger:
    """Collect and write the metrics of one training run (one random seed).

    The logger receives three kinds of input:

    * **Collection** (``log_step``, ``log_rollout``): rewards and episode ends,
      step by step. The logger keeps a running return and length for each
      environment and records each episode when it ends.
    * **Updates** (``log_update``): one ``TrainingStats`` per learner update,
      already summarized by the algorithm (for example, the mean policy loss
      over its minibatches). These are averaged until the next report.
    * **Evaluation** (``log_evaluation``): written immediately.

    Periodic ``"train"`` reports are written only by ``maybe_dump``. The
    trainer calls it after a complete collect-and-update cycle, so a report
    never describes half of an update.

    Args:
        num_envs: N, the number of training environments.
        log_every: A report becomes due every this many environment steps. 0 turns
            periodic reports off; ``finish`` still writes a final one.
        writer: A TensorBoard ``SummaryWriter``, or ``None``.
        device: Where the running episode sums are kept. Use the device the
            rewards are on, to avoid copies.
        metrics_path: The ``metrics.jsonl`` file, or ``None``. It is opened
            for appending, so a resumed run continues the same file.
        episode_window: The episode means in a report cover the last this
            many finished episodes.
        ev_window: How many recent (value, target) pairs are kept for
            explained variance (see ``log_step``).
        ev_name: The metric name for that explained variance.
        history_size: How many recent points per metric are kept in memory.
            ``None`` keeps all of them.
        total_timesteps: Length of the progress bar.
        show_progress: Show a progress bar (needs ``total_timesteps``).

    Example:
        Illustrative; ``stats`` and ``progress`` come from the training loop.

        >>> logger = Logger(8, log_every=10_000, metrics_path="runs/hopper/metrics.jsonl")
        >>> logger.log_update(stats, progress=progress)
        >>> logger.maybe_dump()
        >>> history = logger.finish()
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
        """Check the settings, prepare the running sums and open the outputs.

        Raises:
            ValueError: If ``num_envs`` is not positive, or ``log_every`` or
                ``history_size`` is negative.
        """
        # 1. Check the settings.
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        if log_every < 0:
            raise ValueError(f"log_every must be non-negative, got {log_every}")
        if history_size is not None and history_size < 0:
            raise ValueError(
                f"history_size must be non-negative or None, got {history_size}"
            )

        # 2. Store the settings. The copy of the trainer's counters starts at
        #    zero.
        self.num_envs = num_envs
        self.log_every = log_every
        self.writer = writer
        self.device = torch.device(device)
        self._training_progress = TrainingProgress()

        # 3. In-memory history: one queue of (step, value) per metric name.
        #    A full queue drops its oldest point.
        self._history: defaultdict[str, deque[MetricPoint]] = defaultdict(
            lambda: deque(maxlen=history_size)
        )

        # 4. The running return and length of each environment's current
        #    episode, shaped (N,), and the last finished episodes.
        self._episode_returns = torch.zeros(
            num_envs,
            dtype=torch.float32,
            device=self.device,
        )
        self._episode_lengths = torch.zeros(
            num_envs,
            dtype=torch.int64,
            device=self.device,
        )
        self._recent_returns: deque[float] = deque(maxlen=episode_window)
        self._recent_lengths: deque[float] = deque(maxlen=episode_window)

        # 5. Update metrics waiting for the next report: a weighted sum and the
        #    total weight per metric. The weight is the number of gradient
        #    steps (learner minibatches) in each TrainingStats.
        self._metric_sums: defaultdict[str, float] = defaultdict(float)
        self._metric_weights: defaultdict[str, int] = defaultdict(int)

        # 6. Explained variance from individual (value, target) pairs. One-step
        #    actor-critic has too few samples per update for a meaningful
        #    explained variance, so it supplies pairs for this window instead.
        self._ev_name = ev_name
        self._ev_values: deque[float] = deque(maxlen=ev_window)
        self._ev_targets: deque[float] = deque(maxlen=ev_window)

        # 7. Timing for steps per second, and the step of the next report
        #    (None when periodic reports are off).
        self._start_time = time.perf_counter()
        self._start_step = 0
        self._last_dump_time = self._start_time
        self._last_dump_step = 0
        self._next_log_step = log_every if log_every > 0 else None

        # 8. Open metrics.jsonl for appending, creating its folder if needed.
        self._metrics_file: TextIOWrapper | None = None
        if metrics_path is not None:
            path = Path(metrics_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._metrics_file = path.open("a", encoding="utf-8")

        # 9. Optional progress bar. tqdm is imported only when it is used.
        self._progress: tqdm | None = None
        if show_progress and total_timesteps is not None:
            from tqdm import tqdm

            self._progress = tqdm(
                total=total_timesteps,
                unit="step",
                dynamic_ncols=True,
            )

    @property
    def global_step(self) -> int:
        """Environment steps, from the last counters the trainer passed in."""
        return self._training_progress.global_step

    @property
    def gradient_step(self) -> int:
        """Gradient steps, from the last counters the trainer passed in."""
        return self._training_progress.gradient_step

    def _set_training_progress(self, progress: TrainingProgress) -> None:
        """Copy the trainer's counters and move the progress bar forward.

        This only copies the counters; it does not change when the next
        report is due.
        """
        self._training_progress = progress

        # Move the bar to the new step: never backwards, never past the end.
        if self._progress is not None:
            position = min(progress.global_step, self._progress.total)
            self._progress.update(max(position - self._progress.n, 0))

    def resume(self, *, progress: TrainingProgress) -> None:
        """Prepare the logger for a run resumed from a checkpoint.

        The counters continue from the checkpoint. Everything else starts
        over: the environments begin fresh episodes on resume, so unfinished
        episode sums, recent episodes, pending update metrics and timing are
        cleared. The next report is due at the first multiple of ``log_every``
        after the restored step; for example, with ``log_every=10_000`` and a
        resume at step 25,000, the next report is at 30,000.

        Args:
            progress: The restored counters.
        """
        # 1. Continue from the restored counters.
        self._training_progress = progress

        # 2. Clear everything collected so far.
        self._history.clear()
        self._episode_returns.zero_()
        self._episode_lengths.zero_()
        self._recent_returns.clear()
        self._recent_lengths.clear()
        self._metric_sums.clear()
        self._metric_weights.clear()
        self._ev_values.clear()
        self._ev_targets.clear()

        # 3. Restart timing from now, and schedule the next report.
        self._start_time = time.perf_counter()
        self._start_step = progress.global_step
        self._last_dump_time = self._start_time
        self._last_dump_step = progress.global_step
        self._next_log_step = (
            (progress.global_step // self.log_every + 1) * self.log_every
            if self.log_every > 0
            else None
        )

        # 4. Start the progress bar at the restored step.
        if self._progress is not None:
            self._progress.reset()
            position = min(progress.global_step, self._progress.total)
            self._progress.n = position
            self._progress.initial = position
            self._progress.last_print_n = position
            self._progress.refresh()

    # -------------------------------------------------------------- collection

    def log_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        *,
        progress: TrainingProgress,
        values: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
    ) -> None:
        """Record one step of all N environments.

        Adds each reward to its environment's running episode return and
        counts one step. For every environment whose episode ended, records
        the episode (an ``"episode"`` row, and the recent-episode window) and
        starts that environment's sums again from zero. No report is written
        here; that is ``maybe_dump``'s job.

        Args:
            rewards: Raw rewards, shaped ``(N,)``.
            dones: True where an episode ended (terminated or truncated),
                shaped ``(N,)``.
            progress: The counters after this step.
            values: Optional value predictions for explained variance. Give
                ``values`` and ``targets`` together, with the same shape.
                One-step actor-critic uses this.
            targets: The matching value targets.

        Raises:
            ValueError: If a shape is wrong, or only one of ``values`` and
                ``targets`` is given.
        """
        # 1. Check the shapes.
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

        # 2. Bring the inputs to the logger's device: rewards as float32,
        #    episode ends as booleans.
        rewards = rewards.detach().to(
            device=self.device,
            dtype=self._episode_returns.dtype,
        )
        dones = dones.detach().to(device=self.device, dtype=torch.bool)

        # 3. Copy the counters, then add this step to every running episode.
        self._set_training_progress(progress)
        self._episode_returns += rewards
        self._episode_lengths += 1

        # 4. Keep the (value, target) pairs for explained variance.
        if values is not None and targets is not None:
            self._ev_values.extend(values.detach().flatten().tolist())
            self._ev_targets.extend(targets.detach().flatten().tolist())

        # 5. Record every episode that just ended, then reset its sums.
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
        *,
        progress: TrainingProgress,
    ) -> None:
        """Record a whole rollout of T steps from N environments, step by step.

        ``progress`` holds the counters at the end of collection. The step
        count of each step inside the rollout is worked out backwards, so each
        episode is recorded at the step where it really ended. For example,
        with N=8, T=128 and ``progress.global_step=2048``, the rollout started
        at step 1024; its first row ends at 1032 and its last at 2048.

        Args:
            rewards: Raw rewards, shaped ``(T, N)``.
            dones: Episode ends (terminated or truncated), shaped ``(T, N)``.
            progress: The counters at the end of collection.

        Raises:
            ValueError: If a shape is wrong, or the rollout would start before
                step 0.
        """
        # 1. Check the shapes.
        if rewards.ndim != 2 or rewards.shape[1] != self.num_envs:
            raise ValueError(
                f"rewards must have shape (T, {self.num_envs}), got {tuple(rewards.shape)}"
            )
        if dones.shape != rewards.shape:
            raise ValueError(
                f"dones must have shape {tuple(rewards.shape)}, got {tuple(dones.shape)}"
            )

        # 2. The step count where this rollout started.
        start_step = progress.global_step - rewards.shape[0] * self.num_envs
        if start_step < 0:
            raise ValueError("rollout cannot start before global_step zero")

        # 3. Record each row with its own step count: row t ends at
        #    start_step + (t + 1) * N.
        for index, (step_rewards, step_dones) in enumerate(
            zip(rewards, dones, strict=True)
        ):
            step_progress = replace(
                progress,
                global_step=start_step + (index + 1) * self.num_envs,
            )
            self.log_step(step_rewards, step_dones, progress=step_progress)

        # 4. End with the trainer's counters for the whole rollout.
        self._set_training_progress(progress)

    # ------------------------------------------------------------------ update

    def log_update(
        self,
        stats: TrainingStats,
        *,
        progress: TrainingProgress,
    ) -> None:
        """Add one learner update's summary to the next report.

        Each metric is added with a weight, ``max(gradient_steps, 1)``: the
        number of gradient steps in this update, or 1 for an update with no
        gradient step (so its diagnostics still count). The report divides by
        the total weight, so it shows a weighted average of the submitted
        update summaries since the last report.

        For example, an update with 10 gradient steps and a loss of 0.5, then
        one with 30 steps and a loss of 0.1, are reported as
        ``(10 * 0.5 + 30 * 0.1) / 40 = 0.2``. When every update has at least
        one gradient step, this equals the average per gradient step. A
        zero-step update counts as weight 1, so then it is not exactly that.

        Args:
            stats: The update's summary, from ``learner.update``.
            progress: The counters after this update.

        Raises:
            ValueError: If ``stats.gradient_steps`` is negative.
        """
        # 1. Check the input.
        if stats.gradient_steps < 0:
            raise ValueError("gradient_steps must be non-negative")

        # 2. Copy the counters.
        self._set_training_progress(progress)

        # 3. Add each metric with its weight, max(gradient_steps, 1). An update
        #    without any gradient step still counts once, so its diagnostics
        #    are not lost.
        weight = max(stats.gradient_steps, 1)
        for name, value in stats.metrics.items():
            scalar = _scalar(value)
            self._metric_sums[name] += scalar * weight
            self._metric_weights[name] += weight

    def maybe_dump(self) -> None:
        """Write a ``"train"`` report if one is due.

        A report becomes due when the step count reaches the next multiple of
        ``log_every``. It is written at the first call to this method after
        that, which the trainer makes after each complete collect-and-update
        cycle. So the report's step is usually somewhat past the multiple.

        Training advances in chunks (one rollout is T * N steps), so several
        report steps may pass between two calls. Only one report is written,
        then the next report step moves past the current step. Nothing is
        written when ``log_every`` is 0.
        """
        # 1. Not due yet (or reports off): nothing to do.
        if self._next_log_step is None or self.global_step < self._next_log_step:
            return

        # 2. Write one report.
        self._dump()

        # 3. Move the next report step past the current step, skipping any
        #    report steps that were passed together.
        while self._next_log_step <= self.global_step:
            self._next_log_step += self.log_every

    def _dump(self) -> None:
        """Write one ``"train"`` report and start a new reporting interval.

        The report contains:

        * each pending update metric, averaged (see ``log_update``);
        * ``charts/episodic_return_mean`` and ``charts/episodic_length_mean``
          over the last ``episode_window`` finished episodes, once at least
          one episode has finished;
        * explained variance over the (value, target) window, if pairs were
          given;
        * ``charts/SPS``: environment steps per second since the start (or
          the resume), and ``charts/interval_SPS``: since the last report.

        Afterwards the pending update metrics are cleared. The episode window
        is kept: it always covers the most recent episodes.
        """
        # 1. Average the pending update metrics.
        now = time.perf_counter()
        metrics = {
            name: total / self._metric_weights[name]
            for name, total in self._metric_sums.items()
            if self._metric_weights[name] > 0
        }

        # 2. Means of the recent finished episodes.
        if self._recent_returns:
            metrics["charts/episodic_return_mean"] = sum(self._recent_returns) / len(
                self._recent_returns
            )
            metrics["charts/episodic_length_mean"] = sum(self._recent_lengths) / len(
                self._recent_lengths
            )

        # 3. Explained variance of the stored (value, target) pairs.
        if self._ev_values:
            metrics[self._ev_name] = explained_variance(
                torch.tensor(list(self._ev_values)),
                torch.tensor(list(self._ev_targets)),
            )

        # 4. Speed: since the start, and since the last report.
        elapsed = now - self._start_time
        interval_elapsed = now - self._last_dump_time
        interval_steps = self.global_step - self._last_dump_step
        metrics["charts/SPS"] = (self.global_step - self._start_step) / max(elapsed, 1e-8)
        metrics["charts/interval_SPS"] = interval_steps / max(interval_elapsed, 1e-8)

        # 5. Write the report and start the next interval.
        self._emit("train", metrics, elapsed_seconds=elapsed)
        self._metric_sums.clear()
        self._metric_weights.clear()
        self._last_dump_step = self.global_step
        self._last_dump_time = now

        # 6. Show the latest return and policy loss next to the progress bar.
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
        progress: TrainingProgress,
        deterministic: bool,
        episode_seeds: Sequence[int] | None = None,
    ) -> None:
        """Record the results of one evaluation in one action mode.

        Writes one ``"eval_episode"`` row per episode and one ``"eval"``
        summary row with the mean return, the spread of the returns (standard
        deviation across episodes) and the mean length. Names start with
        ``eval/deterministic_`` or ``eval/stochastic_``. Everything is written
        immediately, not held for the next report.

        Args:
            episode_returns: One return per episode.
            episode_lengths: One length per episode.
            progress: The counters at the time of evaluation.
            deterministic: Which action mode was evaluated.
            episode_seeds: Optional environment seed of each episode, saved in
                the rows. A seed identifies how the environment was
                initialized for that episode (its starting state), not the
                whole trajectory: with sampled actions, later episodes also
                depend on the random draws of earlier ones.

        Raises:
            ValueError: If the inputs are empty, differ in size, or the seeds
                are not one integer per episode.
        """
        # 1. Flatten to float32 on the CPU and check the sizes.
        returns = episode_returns.detach().flatten().to(
            dtype=torch.float32,
            device="cpu",
        )
        lengths = episode_lengths.detach().flatten().to(
            dtype=torch.float32,
            device="cpu",
        )
        if returns.shape != lengths.shape:
            raise ValueError("episode_returns and episode_lengths must have the same shape")
        if returns.numel() == 0:
            raise ValueError("evaluation must contain at least one episode")

        # 2. Seeds, if given: exactly one integer per episode.
        if episode_seeds is not None:
            episode_seeds = tuple(episode_seeds)
            if len(episode_seeds) != returns.numel() or any(
                type(seed) is not int for seed in episode_seeds
            ):
                raise ValueError("episode_seeds must contain one integer per evaluation episode")

        # 3. Copy the counters and name the mode.
        self._set_training_progress(progress)
        mode = "deterministic" if deterministic else "stochastic"

        # 4. One row per episode, with its seed.
        for index, (episode_return, episode_length) in enumerate(zip(
            returns.tolist(),
            lengths.tolist(),
            strict=True,
        )):
            self._emit(
                "eval_episode",
                {
                    f"eval/{mode}_episodic_return": episode_return,
                    f"eval/{mode}_episodic_length": episode_length,
                },
                metadata={
                    "evaluation": {
                        "mode": mode,
                        "deterministic": deterministic,
                        "seed": None if episode_seeds is None else episode_seeds[index],
                    },
                },
            )

        # 5. One summary row for the whole evaluation.
        self._emit(
            "eval",
            {
                f"eval/{mode}_return_mean": returns.mean().item(),
                f"eval/{mode}_return_std": returns.std(unbiased=False).item(),
                f"eval/{mode}_length_mean": lengths.mean().item(),
            },
            metadata={
                "evaluation": {
                    "mode": mode,
                    "deterministic": deterministic,
                    "episode_seeds": None if episode_seeds is None else list(episode_seeds),
                },
            },
        )

    # ------------------------------------------------------------------- output

    def record(self, name: str, value: Scalar, step: int | None = None) -> None:
        """Write one custom value right away, as a ``"metric"`` row.

        Args:
            name: The metric name, for example ``"debug/learning_rate"``.
            value: A number or a one-element tensor.
            step: The environment step to record it at. ``None`` uses the
                current step.
        """
        self._emit("metric", {name: _scalar(value)}, step=step)

    def _emit(
        self,
        kind: Literal["episode", "train", "eval_episode", "eval", "metric"],
        metrics: Mapping[str, Scalar],
        *,
        step: int | None = None,
        elapsed_seconds: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Send a group of values to every output.

        * Memory: one ``(step, value)`` point per metric.
        * TensorBoard: one curve point per metric.
        * ``metrics.jsonl``: one row with ``kind``, ``global_step``,
          ``gradient_step``, the optional ``elapsed_seconds`` and
          ``metadata``, and the values. NaN and infinity are written as
          ``null``, because JSON has no NaN. Metadata (such as evaluation
          seeds) goes only to this file; TensorBoard takes numbers only.

        Python's file buffer is flushed after every row, so each row is handed
        to the operating system right away. This does not guarantee the row
        survives a system crash, and metrics still waiting for the next report
        are only in memory.

        Args:
            kind: The row kind (see the module description).
            metrics: Metric names and values.
            step: The environment step. ``None`` uses the current step.
            elapsed_seconds: Time since the start, for ``"train"`` rows.
            metadata: Extra descriptive data for the JSONL row.
        """
        # 1. The step to record at, and the values as plain floats.
        at = self.global_step if step is None else int(step)
        scalars = {name: _scalar(value) for name, value in metrics.items()}

        # 2. Memory and TensorBoard.
        for name, value in scalars.items():
            self._history[name].append((at, value))
            if self.writer is not None:
                self.writer.add_scalar(name, value, global_step=at)

        # 3. One JSONL row, written and flushed at once.
        if self._metrics_file is not None:
            row: dict[str, Any] = {
                "kind": kind,
                "global_step": at,
                "gradient_step": self.gradient_step,
            }
            if elapsed_seconds is not None:
                row["elapsed_seconds"] = elapsed_seconds
            if metadata is not None:
                row.update(metadata)
            row.update(
                {
                    name: value if math.isfinite(value) else None
                    for name, value in scalars.items()
                }
            )
            self._metrics_file.write(json.dumps(row, allow_nan=False) + "\n")
            self._metrics_file.flush()

    def log_hyperparameters(self, cfg: Any) -> None:
        """Show a config as a table in TensorBoard's Text tab.

        Does nothing without a TensorBoard writer.

        Args:
            cfg: A dataclass config, or any object with attributes.
        """
        if self.writer is None:
            return
        fields = asdict(cfg) if is_dataclass(cfg) else dict(vars(cfg))
        rows = "\n".join(f"|{key}|{value}|" for key, value in fields.items())
        self.writer.add_text("hyperparameters", "|param|value|\n|-|-|\n" + rows)

    @property
    def history(self) -> MetricHistory:
        """Recent values in memory, as ``{name: [(step, value), ...]}``, oldest first.

        At most ``history_size`` points per metric are kept.
        """
        return {name: list(points) for name, points in self._history.items()}

    def recent(self, name: str, n: int = 10) -> float:
        """Return the mean of a metric's last ``n`` values in memory.

        Useful in tests and quick checks, for example
        ``logger.recent("charts/episodic_return", 10)``.

        Args:
            name: The metric name.
            n: How many of the latest values to average.

        Returns:
            The mean, or NaN if the metric has no values yet.

        Raises:
            ValueError: If ``n`` is not positive.
        """
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        points = list(self._history.get(name, ()))[-n:]
        if not points:
            return float("nan")
        return sum(value for _, value in points) / len(points)

    def finish(self) -> MetricHistory:
        """End the run: write a final report, finish the outputs, return the history.

        The final report is written only if something happened since the last
        one (new steps or pending update metrics), so it is never a
        duplicate. Call this once, at the end of training.

        The metrics file and the progress bar are closed. The TensorBoard
        writer is only flushed, not closed: the caller created it, so the
        caller closes it.

        Returns:
            The in-memory history (see ``history``).
        """
        # 1. A final report for the last, possibly partial, interval.
        if self.global_step != self._last_dump_step or self._metric_sums:
            self._dump()

        # 2. Flush TensorBoard (the caller closes it), and close the metrics
        #    file and the progress bar.
        if self.writer is not None:
            self.writer.flush()
        if self._metrics_file is not None:
            self._metrics_file.close()
            self._metrics_file = None
        if self._progress is not None:
            self._progress.close()
            self._progress = None
        return self.history
