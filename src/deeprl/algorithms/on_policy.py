"""Shared training lifecycle for synchronous on-policy algorithms."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from gymnasium.vector import AutoresetMode, VectorEnv

from deeprl.advantages import generalized_advantage_estimate
from deeprl.buffers import RolloutBuffer
from deeprl.checkpoint import load_checkpoint, save_checkpoint
from deeprl.logger import Logger, MetricHistory
from deeprl.rollouts import collect_rollout, evaluate_next_values
from deeprl.stats import TrainingStats

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

    from deeprl.evaluate import Evaluator


@dataclass(frozen=True)
class OnPolicyConfig:
    """Run, rollout, and GAE settings shared by on-policy algorithms."""

    # Run control
    total_timesteps     :   int          = 500_000 # Total transitions across all environments.
    seed                :   int          = 0
    device              :   torch.device = torch.device("cpu")
    log_every           :   int          = 1000

    # Rollout and advantage estimation
    rollout_steps       :   int          = 128      # T: transitions collected per environment.
    gamma               :   float        = 0.99     # Reward discount factor.
    gae_lambda          :   float        = 0.95     # Lambda in generalized advantage estimation.


class OnPolicyLearner(ABC):
    """Policy, value function, and update contract for an on-policy algorithm."""

    def __init__(
        self,
        policy      :   nn.Module,
        value_fn    :   nn.Module,
        cfg         :   OnPolicyConfig,
    ) -> None:
        self.cfg = cfg
        self.policy = policy.to(self.cfg.device)
        self.value_fn = value_fn.to(self.cfg.device)

    def state_dict(self) -> dict[str, Any]:
        """Return the policy and value-function states."""
        return {
            "policy": self.policy.state_dict(),
            "value_fn": self.value_fn.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore the policy and value-function states."""
        self.policy.load_state_dict(state["policy"])
        self.value_fn.load_state_dict(state["value_fn"])

    @abstractmethod
    def update(
        self,
        buffer          :   RolloutBuffer,
        advantages      :   torch.Tensor,
        value_targets   :   torch.Tensor,
    ) -> TrainingStats:
        """Update the policy and value function from one completed rollout."""
        raise NotImplementedError


def train_on_policy(
    *,
    envs            :   VectorEnv,
    learner         :   OnPolicyLearner,
    writer          :   SummaryWriter | None = None,
    logger          :   Logger | None = None,
    evaluator       :   Evaluator | None = None,
    checkpoint_dir  :   str | Path | None = None,
    checkpoint_interval:   int | None = None,
    run_config      :   dict[str, Any] | None = None,
    resume_from     :   str | Path | None = None,
) -> MetricHistory:
    """Run the shared synchronous on-policy training lifecycle.

    Collects T steps from N environments, computes GAE, and delegates the
    algorithm-specific optimization schedule to the learner.

    Checkpoint intervals count completed rollout updates. Resume restores the
    learner and counters, then starts fresh episodes toward total_timesteps.
    Use the same environment and networks; simulator state is not restored.
    run_config is optional JSON-compatible experiment metadata.
    """
    cfg = learner.cfg

    if checkpoint_interval is not None and checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive or None")

    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    # Collection stores true final observations and resets completed
    # environments explicitly after their transitions have been recorded.
    assert envs.metadata.get("autoreset_mode") is AutoresetMode.DISABLED, (
        "On-policy rollout collection requires AutoresetMode.DISABLED to store "
        "true final observations before resetting completed environments."
    )

    # T steps from each of N environments produce B = T * N transitions.
    N = envs.num_envs
    T = cfg.rollout_steps
    rollout_size = cfg.rollout_steps * envs.num_envs

    # Only complete rollouts are collected and used for updates.
    num_updates = cfg.total_timesteps // rollout_size

    # Store config values, not executable optimizer classes, so the checkpoint
    # remains readable with torch.load(weights_only=True).
    learner_config = asdict(cfg)
    for name, value in learner_config.items():
        if isinstance(value, type):
            learner_config[name] = f"{value.__module__}.{value.__qualname__}"
        elif isinstance(value, torch.device):
            learner_config[name] = str(value)

    algorithm = f"{type(learner).__module__}.{type(learner).__qualname__}"
    start_update = 0
    checkpoint = None
    if resume_from is not None:
        # RNG states stay on CPU; native state loaders move model and optimizer
        # tensors to the devices of the learner's parameters.
        checkpoint = load_checkpoint(resume_from, map_location="cpu")
        if checkpoint.get("algorithm", algorithm) != algorithm:
            raise ValueError("checkpoint algorithm does not match the learner")
        if checkpoint.get("num_envs", N) != N:
            raise ValueError("checkpoint num_envs does not match the environment")

        saved_config = checkpoint.get("learner_config")
        if saved_config is not None:
            changed = [
                name for name, value in learner_config.items()
                if name not in {"total_timesteps", "device", "log_every"}
                and saved_config.get(name) != value
            ]
            if changed:
                raise ValueError(f"checkpoint learner config differs: {', '.join(changed)}")

        progress = checkpoint["progress"]
        start_update = progress["update"]
        if (
            start_update < 0
            or progress["gradient_step"] < 0
            or progress["global_step"] != start_update * rollout_size
        ):
            raise ValueError("checkpoint progress is inconsistent with the rollout size")
        if num_updates <= start_update:
            raise ValueError("total_timesteps leaves no remaining complete rollout")
        if run_config is None:
            run_config = checkpoint.get("config")

    if num_updates == 0:
        raise ValueError(
            f"total_timesteps={cfg.total_timesteps} is smaller than one rollout "
            f"({T} steps x {N} envs = {rollout_size}); nothing would be collected"
        )

    # Keep metadata as plain values accepted by the restricted checkpoint loader.
    if run_config is not None:
        run_config = json.loads(json.dumps(run_config))

    # Seed a new run; a resumed run restores its sampling RNG after setup.
    torch.manual_seed(cfg.seed)
    if cfg.device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
    if checkpoint is not None:
        learner.load_state_dict(checkpoint["learner"])

    # Reuse a caller-owned logger, or create one finalized by this function.
    owns_logger = logger is None
    if logger is None:
        logger = Logger(envs.num_envs, cfg.log_every, writer, cfg.device)
    if checkpoint is not None:
        logger.resume(
            global_step=progress["global_step"],
            gradient_step=progress["gradient_step"],
        )
        if evaluator is not None:
            evaluator.resume(progress["global_step"])
    logger.log_hyperparameters(cfg)

    # Initial observation batch: (N, *obs_shape). Resume also starts new episodes.
    observations, _ = envs.reset(seed=cfg.seed)

    # Time-major storage reused for every rollout: (T, N, ...).
    rollout_buffer = RolloutBuffer(
        rollout_steps=T,
        num_envs=N,
        device=cfg.device,
    )

    # Older checkpoints may contain learner/progress only. Restore RNG when
    # available, after environment reset and storage initialization are done.
    if checkpoint is not None and checkpoint.get("rng_state") is not None:
        rng_state = checkpoint["rng_state"]
        torch.set_rng_state(rng_state["torch"])
        if cfg.device.type == "cuda" and rng_state.get("cuda") is not None:
            torch.cuda.set_rng_state(rng_state["cuda"], device=cfg.device)
    checkpoint = None

    for update in range(start_update + 1, num_updates + 1):

        # --------------------------------------------------------------
        # 1. Rollout collection
        # --------------------------------------------------------------

        # Collection runs without gradients using the current policy and critic.
        learner.policy.eval()
        learner.value_fn.eval()

        # Fills the buffer with T x N transitions. The returned observations
        # have shape (N, *obs_shape) and start the next rollout.
        observations = collect_rollout(
            envs=envs,
            policy=learner.policy,
            value_fn=learner.value_fn,
            buffer=rollout_buffer,
            observations=observations,
        )

        # --------------------------------------------------------------
        # 2. Next-state value evaluation
        # --------------------------------------------------------------

        # Evaluate V(S_{t+1}) for every nonterminated transition.
        # Shape: (T, N). Terminated entries remain zero; truncated entries use
        # the true final observations stored before environment reset.
        next_values = evaluate_next_values(
            learner.value_fn,
            rollout_buffer.next_observations,
            rollout_buffer.terminations
        )

        # --------------------------------------------------------------
        # 3. GAE advantages and value targets
        # --------------------------------------------------------------

        # done_t = termination_t or truncation_t
        #
        # td_error_t = R_{t+1} + gamma * (1 - termination_t) * V(S_{t+1}) - V(S_t)
        #
        # advantage_t = td_error_t + gamma * gae_lambda * (1 - done_t) * advantage_{t+1}
        #
        # value_target_t = V(S_t) + advantage_t
        #
        # Advantages are TD(lambda) errors and value targets are the
        # corresponding TD(lambda) targets. Both have shape (T, N) and are
        # computed without gradients. The advantage is therefore a fixed
        # policy-loss weight, while the fixed value target gives the critic
        # its semi-gradient update through V(S_t) only.
        advantages, value_targets = generalized_advantage_estimate(
            rollout_buffer.rewards,
            rollout_buffer.values,
            next_values=next_values,
            terminations=rollout_buffer.terminations,
            truncations=rollout_buffer.truncations,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda
        )

        # --------------------------------------------------------------
        # 4. Parameter updates
        # --------------------------------------------------------------

        # Delegate rollout re-evaluation, batching, and parameter updates to the
        # algorithm. Each algorithm owns its complete optimization schedule.
        training_stats = learner.update(
            buffer=rollout_buffer,
            advantages=advantages,
            value_targets=value_targets,
        )

        # --------------------------------------------------------------
        # 5. Metrics and evaluation
        # --------------------------------------------------------------

        # Record collection and learner statistics before checking whether
        # the completed update has crossed a reporting boundary.
        # Both rewards and episode-boundary masks have shape (T, N).
        logger.log_rollout(
            rewards=rollout_buffer.rewards,
            dones=rollout_buffer.terminations | rollout_buffer.truncations,
        )
        logger.log_update(training_stats)
        logger.maybe_dump()

        # Evaluation also runs only at completed update boundaries.
        if evaluator is not None:
            evaluator.maybe_evaluate(learner.policy, logger)

        # Checkpoint intervals count completed rollout updates.
        if checkpoint_dir is not None:
            save_due = (
                checkpoint_interval is not None
                and update % checkpoint_interval == 0
            )

            if save_due or update == num_updates:
                checkpoint = {
                    "algorithm"         : algorithm,
                    "learner_config"    : learner_config,
                    "num_envs"          : N,
                    "learner"           : learner.state_dict(),
                    "config"            : run_config,
                    "progress"          : {
                        "global_step"   : logger.global_step,
                        "gradient_step" : logger.gradient_step,
                        "update"        : update,
                    },
                    "rng_state"         : {
                        "torch"         : torch.get_rng_state(),
                        "cuda"          : (
                            torch.cuda.get_rng_state(cfg.device)
                            if cfg.device.type == "cuda" else None
                        ),
                    },
                }
                save_checkpoint(checkpoint_dir / "last.pt", checkpoint)

                if save_due:
                    save_checkpoint(
                        checkpoint_dir / f"step_{logger.global_step}.pt",
                        checkpoint
                    )

    # ------------------------------------------------------------------
    # Final evaluation and cleanup
    # ------------------------------------------------------------------

    # Evaluate the final policy even when training ends between checkpoints.
    if evaluator is not None:
        evaluator.evaluate_now(learner.policy, logger)

    # Only finalize resources created inside this function.
    return logger.finish() if owns_logger else logger.history
