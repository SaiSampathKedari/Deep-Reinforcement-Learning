"""Measure how well a policy performs, and compare results across training seeds.

Evaluating one policy:

* ``evaluate_policy``: run a number of episodes on a separate environment and
  return each episode's total reward (return) and length.
* ``Evaluator``: run ``evaluate_policy`` on a schedule during training, for
  example every 10,000 environment steps and once at the end, and log the
  results.

Evaluation does not disturb training. It restores each module's train/eval
mode, and it saves and restores PyTorch's random-number state, so training
continues with the same PyTorch random numbers it would have used without
evaluation (Python's ``random`` and NumPy are not covered).

Returns are sums of the rewards the evaluation environment gives. The
learner's reward scaling is never applied, but reward wrappers on the
environment itself cannot be undone. To measure raw task returns, evaluate
on an environment without reward scaling or clipping.

Comparing runs with different seeds:

* ``load_metric_curve``: read one metric over time from a run's
  ``metrics.jsonl``.
* ``bootstrap_ci``: a pointwise confidence interval for the mean curve
  across runs.
* ``aggregate_seeds``: mean, standard deviation and confidence interval of
  one metric across several runs.

Each run counts as one sample. Results from one seed say little, because RL
results vary a lot between seeds. Resampling whole runs is the standard way
to put a confidence interval on them (see Agarwal et al., 2021, "Deep
Reinforcement Learning at the Edge of the Statistical Precipice").
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import gymnasium as gym
import torch
import torch.nn as nn

from deeprl.env import from_tensor_data, to_tensor_data
from deeprl.logger import Logger
from deeprl.stats import TrainingProgress


@torch.no_grad()
def evaluate_policy(
    env: gym.Env,
    policy: nn.Module,
    *,
    num_episodes: int,
    device: torch.device,
    deterministic: bool,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run complete episodes with a policy and record their returns and lengths.

    Episode ``i`` starts from environment seed ``seed + i``, so the same seed
    gives the same starting states. That makes results comparable across
    action modes and across points in training.

    Args:
        env: One plain Gymnasium environment (not vectorized), separate from
            the training environments.
        policy: An ``nn.Module`` with an ``act(observations, deterministic=...)``
            method that returns batched environment commands. An
            ``InferencePolicy`` has it; a custom module only needs to provide
            the same method.
        num_episodes: Number of episodes to run.
        device: Device the policy runs on.
        deterministic: Passed on to ``policy.act``. For a policy that returns
            a distribution, ``True`` takes the most likely action (or the
            mean) and ``False`` samples. Other action selectors decide for
            themselves; the greedy and direct-action selectors ignore it.
        seed: Seed of the first episode, and of the random numbers used for
            sampling actions.

    Returns:
        ``(returns, lengths)``: two CPU tensors shaped ``(num_episodes,)``.
        Returns are sums of the rewards ``env.step`` gave (float32); lengths
        are step counts (int64).

    Raises:
        ValueError: If ``num_episodes`` is not positive, or the policy does
            not return a batch of one action.
        TypeError: If the policy has no ``act`` method or ``deterministic`` is
            not a boolean.

    Note:
        Only PyTorch's CPU generator and the selected CUDA device's generator
        are seeded and restored. Python's ``random``, NumPy and custom
        generators are not covered.
    """
    # 0. Check the inputs. Remember each module's train/eval mode, then switch
    #    the whole policy to evaluation mode (for example, dropout off).
    if num_episodes <= 0:
        raise ValueError(f"num_episodes must be positive, got {num_episodes}")
    if not isinstance(policy, nn.Module) or not callable(getattr(policy, "act", None)):
        raise TypeError("evaluation policy must be an nn.Module with an act method")
    if not isinstance(deterministic, bool):
        raise TypeError("deterministic must be boolean")

    returns = torch.empty(num_episodes, dtype=torch.float32)
    lengths = torch.empty(num_episodes, dtype=torch.int64)
    training_modes = [
        (module, module.training) for module in policy.modules()
    ]
    policy.eval()

    try:
        # 1. Keep evaluation's random numbers separate from training's.
        #    fork_rng saves the CPU generator (and the selected CUDA device's
        #    generator) and restores them when the block ends. Training then
        #    continues as if evaluation had never drawn any random numbers.
        fork_devices: list[int] = []
        if device.type == "cuda":
            fork_devices.append(
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            )

        with torch.random.fork_rng(devices=fork_devices):

            # Seed the saved generators, so stochastic evaluation gives the
            # same actions every time it runs with the same seed.
            torch.random.default_generator.manual_seed(seed)
            if fork_devices:
                with torch.cuda.device(fork_devices[0]):
                    torch.cuda.manual_seed(seed)

            # 2. Run each episode from a fresh, seeded reset until it
            #    terminates or is truncated.
            for episode in range(num_episodes):
                observation, _ = env.reset(seed=seed + episode)
                episode_return = 0.0
                episode_length = 0
                done = False

                while not done:
                    # 2a. Give the policy a batch of one observation.
                    observation_tensor = to_tensor_data(observation, device=device)

                    # 2b. Choose the action. It is shaped (1,) for Discrete
                    #     or (1, *action_shape) for Box actions.
                    action = policy.act(observation_tensor, deterministic=deterministic)
                    action = to_tensor_data(action, batch_size=1)
                    if len(action.shape) == 0 or action.shape[0] != 1:
                        raise ValueError("evaluation actions must have leading batch dimension 1")

                    # 2c. Remove the batch axis and convert the action to the
                    #     NumPy form the plain environment expects.
                    action = action[0]
                    env_action = from_tensor_data(action, env.action_space, numpy=True)

                    # 2d. Step and add up the reward the environment gives.
                    observation, reward, terminated, truncated, _ = env.step(env_action)
                    episode_return += float(reward)
                    episode_length += 1
                    done = terminated or truncated

                returns[episode] = episode_return
                lengths[episode] = episode_length
    finally:
        # 3. Put every module back in its original mode, even if evaluation
        #    failed. Modes are restored one by one because modules in the same
        #    policy may have been in different modes.
        for module, was_training in training_modes:
            module.training = was_training

    return returns, lengths


class Evaluator:
    """Evaluate a policy on a schedule of environment steps during training.

    Evaluation happens when training first reaches or passes each multiple
    of ``eval_every`` (for example 10,000, 20,000, ...), and once more at the
    end. Each evaluation runs every requested mode on the same seeded
    episodes and logs the results (for example
    ``eval/deterministic_return_mean``).

    Args:
        env: One plain Gymnasium environment, used only for evaluation.
        eval_every: Evaluate every this many environment steps.
        num_episodes: Episodes per mode per evaluation.
        device: Device the policy runs on.
        seed: Seed of the first evaluation episode (see ``evaluate_policy``).
        modes: Which action modes to evaluate: ``"deterministic"``,
            ``"stochastic"``, or both.

    Example:
        Illustrative; ``eval_env``, ``device``, ``policy``, ``logger`` and
        ``progress`` come from the training script.

        >>> evaluator = Evaluator(
        ...     eval_env, eval_every=10_000, num_episodes=10, device=device, seed=1000
        ... )
        >>> # The trainer then calls, after each update:
        >>> evaluator.maybe_evaluate(policy, logger, progress=progress)
        >>> # and once at the end:
        >>> evaluator.evaluate_now(policy, logger, progress=progress)
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        eval_every: int,
        num_episodes: int,
        device: torch.device,
        seed: int,
        modes: Sequence[Literal["deterministic", "stochastic"]] = (
            "deterministic", "stochastic",
        ),
    ) -> None:
        """Check the settings and start the schedule.

        Raises:
            ValueError: If a count is not positive, or ``modes`` is empty,
                unknown or repeated.
        """
        # 1. Both counts must be positive.
        if eval_every <= 0:
            raise ValueError(f"eval_every must be positive, got {eval_every}")
        if num_episodes <= 0:
            raise ValueError(f"num_episodes must be positive, got {num_episodes}")

        # 2. Modes must be a non-empty list of known, distinct names. A single
        #    string such as "deterministic" is rejected, because it would be
        #    read letter by letter.
        if isinstance(modes, str):
            raise ValueError("modes must be a sequence of evaluation mode names")
        modes = tuple(modes)
        if (
            not modes
            or any(mode not in ("deterministic", "stochastic") for mode in modes)
            or len(set(modes)) != len(modes)
        ):
            raise ValueError("modes must contain unique deterministic/stochastic names")

        # 3. Store the settings.
        self.env = env
        self.eval_every = eval_every
        self.num_episodes = num_episodes
        self.device = device
        self.seed = seed
        self.modes = modes

        # 4. The schedule: the first evaluation is due at eval_every. No step
        #    has been evaluated yet.
        self._next_eval_step = eval_every
        self._last_eval_step: int | None = None

    def resume(self, global_step: int) -> None:
        """Continue the schedule after resuming training at ``global_step``.

        The next evaluation is due at the first multiple of ``eval_every``
        after ``global_step``. For example, with ``eval_every=10_000`` and a
        resume at step 25,000, the next one is at 30,000.

        Args:
            global_step: The restored environment step count.
        """
        self._next_eval_step = (global_step // self.eval_every + 1) * self.eval_every
        self._last_eval_step = None

    def _evaluate(
        self,
        policy: nn.Module,
        logger: Logger,
        *,
        progress: TrainingProgress,
    ) -> None:
        """Evaluate every requested mode now and log the results.

        All modes use the same seed, so they are measured on the same
        starting states.
        """
        for mode in self.modes:
            # 1. Run the episodes for this mode.
            deterministic = mode == "deterministic"
            returns, lengths = evaluate_policy(
                self.env,
                policy,
                num_episodes=self.num_episodes,
                device=self.device,
                deterministic=deterministic,
                seed=self.seed,
            )

            # 2. Log each episode and the summary, with the episode seeds.
            logger.log_evaluation(
                returns,
                lengths,
                progress=progress,
                deterministic=deterministic,
                episode_seeds=tuple(
                    self.seed + episode for episode in range(self.num_episodes)
                ),
            )

        # 3. Remember this step, so the final evaluation is not repeated.
        self._last_eval_step = progress.global_step

    def maybe_evaluate(
        self,
        policy: nn.Module,
        logger: Logger,
        *,
        progress: TrainingProgress,
    ) -> None:
        """Evaluate if training has reached the next scheduled step.

        Training advances in chunks (one rollout is T * N steps), so one call
        may pass several scheduled steps at once. It still evaluates only
        once. For example, with ``eval_every=10_000`` and rollouts of 16,384
        steps: evaluate at 16,384 (passed 10,000), then at 32,768 (passed
        20,000 and 30,000), then at 49,152 (passed 40,000).

        Args:
            policy: The policy to evaluate.
            logger: Where to log the results.
            progress: The current step counters.
        """
        # 1. Not due yet: nothing to do.
        if progress.global_step < self._next_eval_step:
            return

        # 2. Evaluate once.
        self._evaluate(policy, logger, progress=progress)

        # 3. Move the next due step past the current step, skipping any
        #    scheduled steps that were passed together.
        while self._next_eval_step <= progress.global_step:
            self._next_eval_step += self.eval_every

    def evaluate_now(
        self,
        policy: nn.Module,
        logger: Logger,
        *,
        progress: TrainingProgress,
    ) -> None:
        """Evaluate the final policy at the end of training.

        Skipped if ``maybe_evaluate`` already evaluated at this exact step, so
        the final policy is not logged twice.

        Args:
            policy: The final policy.
            logger: Where to log the results.
            progress: The final step counters.
        """
        if self._last_eval_step != progress.global_step:
            self._evaluate(policy, logger, progress=progress)


def load_metric_curve(
    metrics_path: str | Path,
    metric: str,
    *,
    kind: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read one metric over time from a run's ``metrics.jsonl``.

    The file has one JSON object per line, for example::

        {"kind": "eval", "global_step": 100000, "eval/deterministic_return_mean": 812.4, ...}

    Args:
        metrics_path: The run's ``metrics.jsonl``.
        metric: The metric name, for example
            ``"eval/deterministic_return_mean"``.
        kind: Only read rows of this kind (``"episode"``, ``"train"``,
            ``"eval_episode"``, ``"eval"`` or ``"metric"``). ``None`` reads
            all rows.

    Returns:
        ``(steps, values)``: the environment step (int64) and the metric
        value of each row that has the metric, in file order. Values use
        PyTorch's default floating dtype, normally float32.
    """
    steps: list[int] = []
    values: list[float] = []

    with Path(metrics_path).open(encoding="utf-8") as file:
        for line in file:
            # 1. Read one row; skip rows of another kind.
            row = json.loads(line)
            if kind is not None and row.get("kind") != kind:
                continue

            # 2. Skip rows without the metric; keep the step and the value.
            value = row.get(metric)
            if value is None:
                continue
            steps.append(int(row["global_step"]))
            values.append(float(value))

    return torch.tensor(steps, dtype=torch.int64), torch.tensor(values)


def bootstrap_ci(
    curves: torch.Tensor,
    *,
    num_bootstrap: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a pointwise confidence interval for the mean of several runs' curves.

    How it works (a percentile bootstrap): from S runs, draw S runs at random
    with replacement (the same run can be drawn more than once) and average
    them. Repeat this ``num_bootstrap`` times. The spread of those averages
    shows how uncertain the mean is. For a 95% interval, the bounds at each
    point are the 2.5% and 97.5% quantiles of the averages.

    Whole runs are drawn, not single points or episodes, because points
    within one run depend on each other. The runs themselves are assumed to
    be independent (different seeds).

    The interval is pointwise: each point has its own interval. It is not a
    band that covers the whole curve at once with that confidence.

    With few runs (for example 3) the interval is rough. With one run it
    collapses to that run's curve.

    Args:
        curves: One curve per run, shaped ``(num_seeds, num_points)``.
        num_bootstrap: How many resampled averages to draw.
        confidence: The nominal confidence level, between 0 and 1. With few
            runs, the real coverage can be lower.
        seed: Seed of the random draws. A private generator is used, so the
            global random numbers are not touched.

    Returns:
        ``(low, high)``, each shaped ``(num_points,)``.

    Raises:
        ValueError: If ``curves`` is not 2D with at least one run, or a
            setting is out of range.
    """
    # 1. Check the inputs.
    if curves.ndim != 2 or curves.shape[0] == 0:
        raise ValueError("curves must have shape (num_seeds, num_points)")
    if num_bootstrap <= 0:
        raise ValueError(f"num_bootstrap must be positive, got {num_bootstrap}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be between 0 and 1, got {confidence}")

    # 2. Draw run indices with replacement: num_bootstrap resamples of
    #    num_seeds runs each. Shape: (num_bootstrap, num_seeds).
    generator = torch.Generator().manual_seed(seed)
    num_seeds = curves.shape[0]
    indices = torch.randint(
        num_seeds,
        (num_bootstrap, num_seeds),
        generator=generator,
    )

    # 3. Average each resample. Shape: (num_bootstrap, num_points).
    bootstrap_means = curves[indices].mean(dim=1)

    # 4. Take the lower and upper quantiles at each point.
    tail = (1.0 - confidence) / 2.0
    quantiles = torch.tensor([tail, 1.0 - tail], dtype=curves.dtype)
    bounds = torch.quantile(bootstrap_means, quantiles, dim=0)
    return bounds[0], bounds[1]


def aggregate_seeds(
    metrics_paths: Sequence[str | Path],
    metric: str,
    *,
    kind: str | None = None,
    num_bootstrap: int = 10_000,
) -> dict[str, torch.Tensor]:
    """Summarize one metric across several runs with different seeds.

    Each run counts as one sample. All runs must have logged the metric at
    the same steps (the same evaluation schedule).

    Args:
        metrics_paths: One ``metrics.jsonl`` per run.
        metric: The metric name, for example
            ``"eval/deterministic_return_mean"``.
        kind: Only read rows of this kind (see ``load_metric_curve``).
        num_bootstrap: Resamples for the confidence interval.

    Returns:
        A dictionary of tensors:

        * ``steps``: the shared environment steps, shaped ``(P,)``;
        * ``curves``: every run's values, shaped ``(S, P)``;
        * ``mean``, ``std``: across runs at each step, shaped ``(P,)``
          (``std`` is 0 for a single run);
        * ``ci_low``, ``ci_high``: the pointwise 95% bootstrap confidence
          interval of the mean (see ``bootstrap_ci``), shaped ``(P,)``.

    Raises:
        ValueError: If no runs are given, the metric is missing, or the runs
            logged at different steps.

    Example:
        Illustrative; the paths are runs you trained, and ``plt`` is
        ``matplotlib.pyplot``.

        >>> result = aggregate_seeds(
        ...     ["runs/seed0/metrics.jsonl", "runs/seed1/metrics.jsonl"],
        ...     "eval/deterministic_return_mean",
        ...     kind="eval",
        ... )
        >>> plt.plot(result["steps"], result["mean"])
        >>> plt.fill_between(result["steps"], result["ci_low"], result["ci_high"])
    """
    # 1. At least one run is needed.
    if not metrics_paths:
        raise ValueError("metrics_paths must contain at least one run")

    # 2. Read every run's curve. The first run must have the metric, and all
    #    runs must have it at the same steps.
    loaded = [load_metric_curve(path, metric, kind=kind) for path in metrics_paths]
    reference_steps = loaded[0][0]
    if reference_steps.numel() == 0:
        raise ValueError(f"metric {metric!r} was not found in any selected rows")
    for steps, _ in loaded[1:]:
        if not torch.equal(steps, reference_steps):
            raise ValueError("metric steps are not aligned across seed runs")

    # 3. Stack the curves, (S, P), and summarize them at each step. The
    #    sample standard deviation needs at least two runs.
    curves = torch.stack([values for _, values in loaded])
    ci_low, ci_high = bootstrap_ci(curves, num_bootstrap=num_bootstrap)
    return {
        "steps": reference_steps,
        "curves": curves,
        "mean": curves.mean(dim=0),
        "std": curves.std(dim=0, unbiased=curves.shape[0] > 1),
        "ci_low": ci_low,
        "ci_high": ci_high,
    }
