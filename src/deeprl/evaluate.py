"""Policy evaluation and aggregation across independent training seeds."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Distribution

from deeprl.logger import Logger


def _evaluation_action(
    distribution: Distribution,
    *,
    deterministic: bool,
) -> torch.Tensor:
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run complete evaluation episodes without changing training RNG state."""
    if num_episodes <= 0:
        raise ValueError(f"num_episodes must be positive, got {num_episodes}")

    returns = torch.empty(num_episodes, dtype=torch.float32)
    lengths = torch.empty(num_episodes, dtype=torch.int64)
    was_training = policy.training
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
            torch.manual_seed(seed)
            for episode in range(num_episodes):
                observation, _ = env.reset(seed=seed + episode)
                episode_return = 0.0
                episode_length = 0
                done = False

                while not done:
                    observation_tensor = torch.as_tensor(
                        observation,
                        dtype=torch.float32,
                        device=device,
                    ).unsqueeze(0)
                    distribution: Distribution = policy(observation_tensor)
                    action = _evaluation_action(
                        distribution,
                        deterministic=deterministic,
                    ).squeeze(0)

                    if isinstance(env.action_space, gym.spaces.Discrete):
                        env_action = int(action.item())
                    else:
                        env_action = action.detach().cpu().tolist()

                    observation, reward, terminated, truncated, _ = env.step(env_action)
                    episode_return += float(reward)
                    episode_length += 1
                    done = terminated or truncated

                returns[episode] = episode_return
                lengths[episode] = episode_length
    finally:
        policy.train(was_training)

    return returns, lengths


class Evaluator:
    """Run deterministic and stochastic evaluations on an env-step schedule."""

    def __init__(
        self,
        env: gym.Env,
        *,
        eval_every: int,
        num_episodes: int,
        device: torch.device,
        seed: int,
    ) -> None:
        if eval_every <= 0:
            raise ValueError(f"eval_every must be positive, got {eval_every}")
        if num_episodes <= 0:
            raise ValueError(f"num_episodes must be positive, got {num_episodes}")

        self.env = env
        self.eval_every = eval_every
        self.num_episodes = num_episodes
        self.device = device
        self.seed = seed
        self._next_eval_step = eval_every
        self._last_eval_step: int | None = None

    def _evaluate(self, policy: nn.Module, logger: Logger) -> None:
        for deterministic in (True, False):
            returns, lengths = evaluate_policy(
                self.env,
                policy,
                num_episodes=self.num_episodes,
                device=self.device,
                deterministic=deterministic,
                seed=self.seed,
            )
            logger.log_evaluation(
                returns,
                lengths,
                deterministic=deterministic,
            )
        self._last_eval_step = logger.global_step

    def maybe_evaluate(self, policy: nn.Module, logger: Logger) -> None:
        """Evaluate once after crossing one or more scheduled checkpoints."""
        if logger.global_step < self._next_eval_step:
            return

        self._evaluate(policy, logger)

        while self._next_eval_step <= logger.global_step:
            self._next_eval_step += self.eval_every

    def evaluate_now(self, policy: nn.Module, logger: Logger) -> None:
        """Evaluate the final policy unless it was evaluated at this step."""
        if self._last_eval_step != logger.global_step:
            self._evaluate(policy, logger)


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
