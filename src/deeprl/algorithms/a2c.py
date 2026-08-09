from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.distributions import Distribution
from dataclasses import dataclass

from gymnasium.vector import AutoresetMode, VectorEnv

from deeprl.logger import Logger, MetricHistory
from deeprl.stats import TrainingStats
from deeprl.advantages import generalized_advantage_estimate
from deeprl.buffers import RolloutBuffer
from deeprl.rollouts import collect_rollout, evaluate_next_values
from deeprl.utils import explained_variance

if TYPE_CHECKING:
    from deeprl.evaluate import Evaluator
    from torch.utils.tensorboard import SummaryWriter


@dataclass(frozen=True)
class A2CConfig:
    """Hyperparameters for synchronous advantage actor-critic.
    
    Each update collects T = rollout_steps transitions from every environment.
    With N parallel environments, the rollout contains B = T * N transitions
    and A2C uses all B transitions in one full-batch update.
    """
    
    # Run control
    total_timesteps     :   int          = 500_000 # Total transitions across all environments.
    seed                :   int          = 0
    device              :   torch.device = torch.device("cpu")
    log_every           :   int          = 1000
    
    # Rollout and advantage estimation
    rollout_steps       :   int          = 128      # T: transitions collected per environment.
    gamma               :   float        = 0.99     # Reward discount factor.
    gae_lambda          :   float        = 0.95     # Lambda in generalized advantage estimation.
    
    # Optimization
    lr_actor            :   float        = 3e-4
    lr_critic           :   float        = 1e-3
    ent_coef            :   float        = 0.0      # Entropy-bonus coefficient.
    max_grad_norm       :   float        = 0.5      # Global L2 gradient clipping threshold.
    normalize_advantage :   bool         = False


def update(
    policy      :   nn.Module,
    value_fn    :   nn.Module,
    optim_actor :   torch.optim.Optimizer,
    optim_critic:   torch.optim.Optimizer,
    buffer      :   RolloutBuffer,
    advantages  :   torch.Tensor,
    value_targets:  torch.Tensor,
    cfg         :   A2CConfig
) -> TrainingStats:
    """Perform one full-rollout A2C update."""

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    # Restore module training mode after eval-mode rollout collection. The
    # forward passes below run outside no_grad() and build new autograd graphs.
    policy.train()
    value_fn.train()

    # Metrics are reduced across all batches produced by the update.
    metric_sums: dict[str, float] = {
        "losses/policy_loss": 0.0,
        "losses/value_loss": 0.0,
        "diagnostics/entropy": 0.0,
        "grads/policy_norm": 0.0,
        "grads/value_norm": 0.0,
    }
    num_samples = 0
    gradient_steps = 0

    for batch in buffer.get(
        advantages=advantages,
        value_targets=value_targets,
        batch_size=None,
    ):

        # --------------------------------------------------------------
        # 1. Rollout re-evaluation
        # --------------------------------------------------------------

        # batch_size=None yields the complete flattened rollout, B = T * N.
        # Observations: (B, *obs_shape); actions: (B, *action_shape).
        action_distributions: Distribution = policy(batch.observations)

        # Recompute log pi(A|S) under the current policy. Shape: (B,).
        log_probs = action_distributions.log_prob(batch.actions)

        # Recompute V(S) with gradients enabled. Shape: (B,).
        values: torch.Tensor = value_fn(batch.observations)

        # --------------------------------------------------------------
        # 2. Loss computation
        # --------------------------------------------------------------

        # value_loss = 1/2 * mean[(value_target - V(S))^2]
        #
        # value_targets were computed without gradients, so gradients flow
        # only through V(S), giving the critic its semi-gradient update.
        value_loss = 0.5 * (batch.value_targets - values).square().mean()

        # Mean policy entropy across the rollout. This remains differentiable
        # because it contributes to the policy loss when ent_coef is nonzero.
        entropy = action_distributions.entropy().mean()

        # Advantages were computed without gradients and therefore remain
        # fixed weights in the policy-gradient objective.
        batch_advantages = batch.advantages

        # Optional normalization changes the scale, but not the ordering, of
        # advantages within this batch.
        if cfg.normalize_advantage and batch_advantages.numel() > 1:
            batch_advantages = (
                batch_advantages - batch_advantages.mean()
            ) / (batch_advantages.std(unbiased=False) + 1e-8)

        # entropy = mean[H(pi(.|S))]
        # policy_loss = -mean[advantage * log pi(A|S)] - ent_coef * entropy
        # The leading minus converts gradient ascent into optimizer descent.
        policy_loss = -(
            (batch_advantages * log_probs).mean()
            + cfg.ent_coef * entropy
        )

        # --------------------------------------------------------------
        # 3. Parameter updates
        # --------------------------------------------------------------

        # Critic semi-gradient update with global L2 gradient clipping.
        optim_critic.zero_grad(set_to_none=True)
        value_loss.backward()
        value_grad_norm = torch.nn.utils.clip_grad_norm_(
            value_fn.parameters(),
            max_norm=cfg.max_grad_norm,
        )
        optim_critic.step()

        # Policy-gradient update with global L2 gradient clipping.
        optim_actor.zero_grad(set_to_none=True)
        policy_loss.backward()
        policy_grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(),
            max_norm=cfg.max_grad_norm,
        )
        optim_actor.step()

        # --------------------------------------------------------------
        # 4. Batch metric accumulation
        # --------------------------------------------------------------

        # Weight by sample count so a smaller final minibatch remains correct
        # when this update structure is later reused by minibatch algorithms.
        batch_samples = batch.observations.shape[0]
        num_samples += batch_samples
        gradient_steps += 1
        for name, value in (
            ("losses/policy_loss", policy_loss),
            ("losses/value_loss", value_loss),
            ("diagnostics/entropy", entropy),
            ("grads/policy_norm", policy_grad_norm),
            ("grads/value_norm", value_grad_norm),
        ):
            metric_sums[name] += value.detach().item() * batch_samples

    # ------------------------------------------------------------------
    # 5. Update metric reduction
    # ------------------------------------------------------------------

    # Produce one mean for the complete update, not one value per batch.
    metrics = {
        name: total / num_samples
        for name, total in metric_sums.items()
    }

    # Explained variance compares collection-time V(S) against the fixed
    # TD(lambda) value targets over the complete rollout.
    metrics.update(
        {
            "diagnostics/explained_variance": explained_variance(
                buffer.values.flatten(), value_targets.flatten()
            ),
            "charts/policy_learning_rate": optim_actor.param_groups[0]["lr"],
            "charts/value_learning_rate": optim_critic.param_groups[0]["lr"],
        }
    )

    return TrainingStats(metrics=metrics, gradient_steps=gradient_steps)


def a2c(
    envs            :   VectorEnv,
    policy          :   nn.Module,
    value_fn        :   nn.Module,
    cfg             :   A2CConfig,
    writer          :   SummaryWriter | None = None,
    logger          :   Logger | None = None,
    evaluator       :   Evaluator | None = None,
) -> MetricHistory:
    """Train policy and value networks using synchronous A2C.

    Collects T steps from N environments, computes GAE, and performs one
    full-batch update over the resulting T * N transitions.
    """

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    # Seed action sampling and rollout shuffling.
    torch.manual_seed(cfg.seed)
    if cfg.device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)

    # Collection stores true final observations and resets completed
    # environments explicitly after their transitions have been recorded.
    assert envs.metadata.get("autoreset_mode") is AutoresetMode.DISABLED, (
        "A2C requires AutoresetMode.DISABLED to store true final observations "
        "before resetting completed environments."
    )

    # Reuse a caller-owned logger, or create one finalized by this function.
    owns_logger = logger is None
    if logger is None:
        logger = Logger(envs.num_envs, cfg.log_every, writer, cfg.device)

    logger.log_hyperparameters(cfg)

    # Policy, critic, rollout storage and environment tensors share one device.
    policy = policy.to(device=cfg.device)
    value_fn = value_fn.to(device=cfg.device)

    optim_critic = torch.optim.SGD(value_fn.parameters(), lr=cfg.lr_critic)
    optim_actor = torch.optim.SGD(policy.parameters(), lr=cfg.lr_actor)

    # Initial observation batch: (N, *obs_shape).
    observations, _ = envs.reset(seed=cfg.seed)

    # T steps from each of N environments produce B = T * N transitions.
    N = envs.num_envs
    T = cfg.rollout_steps
    batch_size = cfg.rollout_steps * envs.num_envs

    # Only complete rollouts are collected and used for updates.
    num_updates = cfg.total_timesteps // batch_size

    if num_updates == 0:
        raise ValueError(
            f"total_timesteps={cfg.total_timesteps} is smaller than one rollout "
            f"({T} steps x {N} envs = {batch_size}); nothing would be collected"
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
        policy.eval()
        value_fn.eval()

        # Fills the buffer with T x N transitions. The returned observations
        # have shape (N, *obs_shape) and start the next rollout.
        observations = collect_rollout(
            envs=envs,
            policy=policy,
            value_fn=value_fn,
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
            value_fn,
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

        # Re-evaluate the rollout with gradients and update both networks
        # using the complete flattened batch of B = T * N samples.
        training_stats = update(
            policy,
            value_fn,
            optim_actor=optim_actor,
            optim_critic=optim_critic,
            buffer=rollout_buffer,
            advantages=advantages,
            value_targets=value_targets,
            cfg=cfg,
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
            evaluator.maybe_evaluate(policy, logger)

    # ------------------------------------------------------------------
    # Final evaluation and cleanup
    # ------------------------------------------------------------------

    # Evaluate the final policy even when training ends between checkpoints.
    if evaluator is not None:
        evaluator.evaluate_now(policy, logger)

    # Only finalize resources created inside this function.
    return logger.finish() if owns_logger else logger.history
