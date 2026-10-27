"""One-step actor-critic with vectorized environment updates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from gymnasium.vector import AutoresetMode, VectorEnv
from torch.distributions import Distribution

from deeprl.logger import Logger, MetricHistory
from deeprl.stats import TrainingStats
from deeprl.utils import grad_norm

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

    from deeprl.evaluate import Evaluator


@dataclass(frozen=True)
class OneStepACConfig:
    """Training and optimization settings for one-step actor-critic."""

    total_timesteps :   int          = 500_000
    gamma           :   float        = 0.99
    seed            :   int          = 0
    device          :   torch.device = torch.device("cpu")
    log_every       :   int          = 1000
    lr_actor        :   float        = 1e-3
    lr_critic       :   float        = 1e-2
    discount_actor  :   bool         = True

    # PyTorch optimizer classes and options, excluding params and lr.
    actor_optimizer_class   :   type[torch.optim.Optimizer] = torch.optim.Adam
    actor_optimizer_kwargs  :   dict[str, Any] = field(default_factory=dict)
    critic_optimizer_class  :   type[torch.optim.Optimizer] = torch.optim.Adam
    critic_optimizer_kwargs :   dict[str, Any] = field(default_factory=dict)


def one_step_actor_critic(
    envs            :   VectorEnv,
    policy          :   nn.Module,
    value_fn        :   nn.Module,
    cfg             :   OneStepACConfig,
    writer          :   SummaryWriter | None = None,
    logger          :   Logger | None = None,
    evaluator       :   Evaluator | None = None,
) -> MetricHistory:
    """Update the actor and critic once per vector step using one-step TD targets."""

    # DISABLED preserves the true next/final observation for every transition.
    # Finished environments are reset explicitly after the update.
    assert envs.metadata.get("autoreset_mode") is AutoresetMode.DISABLED, (
        "this algorithm assumes AutoresetMode.DISABLED: it resets finished "
        f"sub-environments itself via reset_mask. Got {envs.metadata.get("autoreset_mode")}."
    )


    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    # Use a distinct explained-variance metric for one-step TD targets.
    owns_logger = logger is None
    if logger is None:
        logger = Logger(
            envs.num_envs,
            cfg.log_every,
            writer,
            cfg.device,
            ev_name="diagnostics/td_target_explained_variance",
        )
    logger.log_hyperparameters(cfg)

    policy   = policy.to(device=cfg.device)
    value_fn = value_fn.to(device=cfg.device)

    # Construct optimizers after the networks are moved to their device.
    optim_actor = cfg.actor_optimizer_class(
        policy.parameters(),
        lr=cfg.lr_actor,
        **cfg.actor_optimizer_kwargs,
    )
    optim_critic = cfg.critic_optimizer_class(
        value_fn.parameters(),
        lr=cfg.lr_critic,
        **cfg.critic_optimizer_kwargs,
    )

    # Initial observation batch.
    # Shape: (N, *observation_shape).
    obs, _ = envs.reset(seed=cfg.seed)

    # Per-environment I_t = gamma^t.
    # Each environment can be at a different time within its episode.
    discounts = torch.ones(envs.num_envs, device=cfg.device)

    num_vector_steps = cfg.total_timesteps // envs.num_envs

    for _ in range(num_vector_steps):

        # --------------------------------------------------------------
        # 1. Policy forward pass
        # --------------------------------------------------------------

        # Batched policy distribution for N environments;
        # its parameters depend on policy parameters and require gradients.
        action_dist : Distribution = policy(obs)

        # Sample one action per environment;
        # No gradient through sample().
        # Shape: (N,) discrete or (N, action_dim) continuous.
        actions : torch.Tensor = action_dist.sample()

        # Differentiable w.r.t. policy parameters.
        # Shape: (N,) when each environment's action is treated as one event.
        log_probs : torch.Tensor = action_dist.log_prob(actions)

        # --------------------------------------------------------------
        # 2. Environment transition
        # --------------------------------------------------------------

        # Step all N environments once.
        #
        # AutoresetMode.DISABLED is assumed:
        # - every returned row is a genuine transition;
        # - next_obs contains the actual next or final observations;
        # - completed environments are reset explicitly after the update.
        next_obs, rewards, terminations, truncations, _ = envs.step(actions)

        # Each environment episode finishes through either termination
        # or truncation. Shape: (N,), dtype: bool.
        dones : torch.Tensor = torch.logical_or(terminations, truncations)

        # --------------------------------------------------------------
        # 3. Actor-critic targets and losses
        # --------------------------------------------------------------

        # State-value estimate for each current observation.
        values : torch.Tensor = value_fn(obs)

        with torch.no_grad():

            # Next-state value estimates. Shape: (N,).
            next_values : torch.Tensor = value_fn(next_obs)

            # Bootstrap after ordinary transitions and truncations.
            # Do not bootstrap after genuine MDP terminations.
            bootstrap_mask = torch.logical_not(terminations).to(next_values.dtype)

            # TD targets are treated as constants after computation.
            # No gradient propagates through V(next_obs), producing the
            # standard semi-gradient TD update for the critic.
            td_targets = rewards + cfg.gamma * bootstrap_mask * next_values

        # One TD error for each environment.
        # Gradients flow through values, but not through td_targets.
        # Shape: (N,).
        td_errors : torch.Tensor = td_targets - values

        # Mean critic loss across all N genuine transitions.
        critic_loss = 0.5 * td_errors.square().mean()

        # TD errors are fixed weights for the policy-gradient update.
        actor_weights : torch.Tensor = td_errors.detach()

        # Optionally weight the actor update by the episode discount I_t = gamma^t.
        if cfg.discount_actor:
            actor_weights = discounts * actor_weights

        # Mean actor loss across all N genuine transitions.
        actor_loss = -(actor_weights * log_probs).mean()

        with torch.no_grad():
            entropy = action_dist.entropy().mean()

        # --------------------------------------------------------------
        # 4. Parameter updates
        # --------------------------------------------------------------

        # Measure gradient norms after backward() and before the optimizer step.

        # Critic semi-gradient update.
        optim_critic.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad_norm = grad_norm(value_fn)
        optim_critic.step()

        # Actor policy-gradient update.
        optim_actor.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad_norm = grad_norm(policy)
        optim_actor.step()

        # --------------------------------------------------------------
        # 5. Metrics
        # --------------------------------------------------------------

        # Report complete updates and compute explained variance from a rolling
        # window of value/target pairs, rather than a single vector step.
        logger.log_step(
            rewards,
            dones,
            values=values,
            targets=td_targets,
        )
        logger.log_update(
            TrainingStats(
                metrics={
                    "losses/policy_loss": actor_loss.detach().item(),
                    "losses/value_loss": critic_loss.detach().item(),
                    "diagnostics/entropy": entropy.detach().item(),
                    "diagnostics/mean_abs_td_error": (
                        td_errors.detach().abs().mean().item()
                    ),
                    "grads/policy_norm": actor_grad_norm,
                    "grads/value_norm": critic_grad_norm,
                    "charts/policy_learning_rate": optim_actor.param_groups[0]["lr"],
                    "charts/value_learning_rate": optim_critic.param_groups[0]["lr"],
                },
                gradient_steps=1,
            )
        )
        logger.maybe_dump()
        if evaluator is not None:
            evaluator.maybe_evaluate(policy, logger)

        # --------------------------------------------------------------
        # 6. Update discounts
        # --------------------------------------------------------------

        if cfg.discount_actor:
            discounts = torch.where(
                dones,
                torch.ones_like(discounts),
                cfg.gamma * discounts,
            )

        # --------------------------------------------------------------
        # 7. Partial environment reset
        # --------------------------------------------------------------

        if dones.any().item():

            # Reset only completed environments.
            #
            # The returned observation is the complete batch:
            # - reset observations for completed environments;
            # - unchanged next_obs entries for continuing environments.
            obs, _ = envs.reset(options={"reset_mask": dones})

        else:

            # No environment finished, so every environment continues
            # from the observation returned by envs.step()
            obs = next_obs

    if evaluator is not None:
        evaluator.evaluate_now(policy, logger)

    # Flush the final partial interval only when this function owns the logger.
    return logger.finish() if owns_logger else logger.history
