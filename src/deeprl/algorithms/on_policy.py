"""Shared training lifecycle for synchronous on-policy algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from gymnasium.vector import AutoresetMode, VectorEnv

from deeprl.advantages import generalized_advantage_estimate
from deeprl.buffers import RolloutBuffer
from deeprl.logger import Logger, MetricHistory
from deeprl.rollouts import collect_rollout, evaluate_next_values
from deeprl.stats import TrainingStats


if TYPE_CHECKING:
    from deeprl.evaluate import Evaluator
    from torch.utils.tensorboard import SummaryWriter


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
) -> MetricHistory:
    """Run the shared synchronous on-policy training lifecycle.

    Collects T steps from N environments, computes GAE, and delegates the
    algorithm-specific optimization schedule to the learner.
    """
    cfg = learner.cfg

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    # Seed PyTorch action sampling and rollout shuffling.
    torch.manual_seed(cfg.seed)
    if cfg.device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)

    # Collection stores true final observations and resets completed
    # environments explicitly after their transitions have been recorded.
    assert envs.metadata.get("autoreset_mode") is AutoresetMode.DISABLED, (
        "On-policy rollout collection requires AutoresetMode.DISABLED to store "
        "true final observations before resetting completed environments."
    )

    # Reuse a caller-owned logger, or create one finalized by this function.
    owns_logger = logger is None
    if logger is None:
        logger = Logger(envs.num_envs, cfg.log_every, writer, cfg.device)

    logger.log_hyperparameters(cfg)

    # Initial observation batch: (N, *obs_shape).
    observations, _ = envs.reset(seed=cfg.seed)

    # T steps from each of N environments produce B = T * N transitions.
    N = envs.num_envs
    T = cfg.rollout_steps
    rollout_size = cfg.rollout_steps * envs.num_envs

    # Only complete rollouts are collected and used for updates.
    num_updates = cfg.total_timesteps // rollout_size

    if num_updates == 0:
        raise ValueError(
            f"total_timesteps={cfg.total_timesteps} is smaller than one rollout "
            f"({T} steps x {N} envs = {rollout_size}); nothing would be collected"
        )

    # Time-major storage reused for every rollout: (T, N, ...).
    rollout_buffer = RolloutBuffer(
        rollout_steps=T,
        num_envs=N,
        observation_space=envs.single_observation_space,
        action_space=envs.single_action_space,
        device=cfg.device,
    )

    for _ in range(num_updates):

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

    # ------------------------------------------------------------------
    # Final evaluation and cleanup
    # ------------------------------------------------------------------

    # Evaluate the final policy even when training ends between checkpoints.
    if evaluator is not None:
        evaluator.evaluate_now(learner.policy, logger)

    # Only finalize resources created inside this function.
    return logger.finish() if owns_logger else logger.history
