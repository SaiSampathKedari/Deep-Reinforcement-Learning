"""Policy evaluation and aggregation across independent training seeds."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Distribution

from deeprl.buffers import TensorData
from deeprl.env import from_tensor_data, to_tensor_data
from deeprl.logger import Logger
from deeprl.stats import TrainingProgress


def _evaluation_action(
    distribution: Distribution,
    *,
    deterministic: bool,
) -> TensorData:
    if torch.is_tensor(distribution):
        raise TypeError(
            "tensor-valued actors and Q-functions require an explicit action_fn"
        )
    if not deterministic:
        return distribution.sample()
    try:
        return distribution.mode
    except NotImplementedError:
        return distribution.mean


@torch.no_grad()
def evaluate_policy(
    env: gym.Env,
    policy: nn.Module,
    *,
    num_episodes: int,
    device: torch.device,
    deterministic: bool,
    seed: int,
    action_fn: Callable[[TensorData, bool], TensorData] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate episodes while preserving default CPU/selected-CUDA Torch RNG.

    By default, policy returns a distribution. An explicit action_fn instead
    receives (batched_observations, deterministic) and returns batched actions.
    It must use policy or its submodules so evaluation mode is managed here.
    Python, NumPy, and custom generator state are outside this RNG guarantee.
    """
    if num_episodes <= 0:
        raise ValueError(f"num_episodes must be positive, got {num_episodes}")

    returns = torch.empty(num_episodes, dtype=torch.float32)
    lengths = torch.empty(num_episodes, dtype=torch.int64)
    training_modes = [
        (module, module.training) for module in policy.modules()
    ]
    policy.eval()

    try:
        # Stochastic evaluation should be repeatable without consuming the RNG
        # stream used by the training policy after evaluation returns.
        fork_devices: list[int] = []
        if device.type == "cuda":
            fork_devices.append(
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            )

        with torch.random.fork_rng(devices=fork_devices):
            
            # Seed only the generators restored by this evaluation context.
            torch.random.default_generator.manual_seed(seed)
            if fork_devices:
                with torch.cuda.device(fork_devices[0]):
                    torch.cuda.manual_seed(seed)
                    
            for episode in range(num_episodes):
                observation, _ = env.reset(seed=seed + episode)
                episode_return = 0.0
                episode_length = 0
                done = False

                while not done:
                    observation_tensor = to_tensor_data(observation, device=device)
                    if action_fn is None:
                        action = _evaluation_action(
                            policy(observation_tensor),
                            deterministic=deterministic,
                        )
                    else:
                        action = action_fn(observation_tensor, deterministic)
                    action = to_tensor_data(action, batch_size=1)
                    if len(action.shape) == 0 or action.shape[0] != 1:
                        raise ValueError("evaluation actions must have leading batch dimension 1")
                    action = action[0]
                    env_action = from_tensor_data(action, env.action_space, numpy=True)

                    observation, reward, terminated, truncated, _ = env.step(env_action)
                    episode_return += float(reward)
                    episode_length += 1
                    done = terminated or truncated

                returns[episode] = episode_return
                lengths[episode] = episode_length
    finally:
        for module, was_training in training_modes:
            module.training = was_training

    return returns, lengths


class Evaluator:
    """Run selected action modes on an environment-step schedule."""

    def __init__(
        self,
        env: gym.Env,
        *,
        eval_every: int,
        num_episodes: int,
        device: torch.device,
        seed: int,
        action_fn: Callable[[TensorData, bool], TensorData] | None = None,
        modes: Sequence[Literal["deterministic", "stochastic"]] = (
            "deterministic", "stochastic",
        ),
    ) -> None:
        if eval_every <= 0:
            raise ValueError(f"eval_every must be positive, got {eval_every}")
        if num_episodes <= 0:
            raise ValueError(f"num_episodes must be positive, got {num_episodes}")
        if isinstance(modes, str):
            raise ValueError("modes must be a sequence of evaluation mode names")
        modes = tuple(modes)
        if (
            not modes
            or any(mode not in ("deterministic", "stochastic") for mode in modes)
            or len(set(modes)) != len(modes)
        ):
            raise ValueError("modes must contain unique deterministic/stochastic names")

        self.env = env
        self.eval_every = eval_every
        self.num_episodes = num_episodes
        self.device = device
        self.seed = seed
        self.action_fn = action_fn
        self.modes = modes
        self._next_eval_step = eval_every
        self._last_eval_step: int | None = None

    def resume(self, global_step: int) -> None:
        """Continue the evaluation schedule after a restored training step."""
        self._next_eval_step = (global_step // self.eval_every + 1) * self.eval_every
        self._last_eval_step = None

    def _evaluate(
        self, policy: nn.Module, logger: Logger, *, progress: TrainingProgress
    ) -> None:
        for mode in self.modes:
            deterministic = mode == "deterministic"
            returns, lengths = evaluate_policy(
                self.env,
                policy,
                num_episodes=self.num_episodes,
                device=self.device,
                deterministic=deterministic,
                seed=self.seed,
                action_fn=self.action_fn,
            )
            logger.log_evaluation(
                returns,
                lengths,
                progress=progress,
                deterministic=deterministic,
            )
        self._last_eval_step = progress.global_step

    def maybe_evaluate(
        self, policy: nn.Module, logger: Logger, *, progress: TrainingProgress
    ) -> None:
        """Evaluate once after crossing one or more scheduled checkpoints."""
        if progress.global_step < self._next_eval_step:
            return

        self._evaluate(policy, logger, progress=progress)

        while self._next_eval_step <= progress.global_step:
            self._next_eval_step += self.eval_every

    def evaluate_now(
        self, policy: nn.Module, logger: Logger, *, progress: TrainingProgress
    ) -> None:
        """Evaluate the final policy unless it was evaluated at this step."""
        if self._last_eval_step != progress.global_step:
            self._evaluate(policy, logger, progress=progress)


def load_metric_curve(
    metrics_path: str | Path,
    metric: str,
    *,
    kind: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load one finite metric curve from a run's JSONL file."""
    steps: list[int] = []
    values: list[float] = []

    with Path(metrics_path).open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            if kind is not None and row.get("kind") != kind:
                continue
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
    """Bootstrap seed-level curves and return pointwise confidence bounds."""
    if curves.ndim != 2 or curves.shape[0] == 0:
        raise ValueError("curves must have shape (num_seeds, num_points)")
    if num_bootstrap <= 0:
        raise ValueError(f"num_bootstrap must be positive, got {num_bootstrap}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be between 0 and 1, got {confidence}")

    generator = torch.Generator().manual_seed(seed)
    num_seeds = curves.shape[0]
    indices = torch.randint(
        num_seeds,
        (num_bootstrap, num_seeds),
        generator=generator,
    )
    bootstrap_means = curves[indices].mean(dim=1)
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
    """Aggregate aligned curves while treating each run as one sample."""
    if not metrics_paths:
        raise ValueError("metrics_paths must contain at least one run")

    loaded = [load_metric_curve(path, metric, kind=kind) for path in metrics_paths]
    reference_steps = loaded[0][0]
    if reference_steps.numel() == 0:
        raise ValueError(f"metric {metric!r} was not found in any selected rows")
    for steps, _ in loaded[1:]:
        if not torch.equal(steps, reference_steps):
            raise ValueError("metric steps are not aligned across seed runs")

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