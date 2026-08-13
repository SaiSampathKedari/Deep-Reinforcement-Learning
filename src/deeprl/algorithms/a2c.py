"""Synchronous advantage actor-critic learner and training entry point."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.distributions import Distribution

from gymnasium.vector import VectorEnv

from deeprl.algorithms.on_policy import (
    OnPolicyConfig,
    OnPolicyLearner,
    train_on_policy,
)
from deeprl.buffers import RolloutBuffer
from deeprl.logger import Logger, MetricHistory
from deeprl.stats import TrainingStats
from deeprl.utils import explained_variance

if TYPE_CHECKING:
    from deeprl.evaluate import Evaluator
    from torch.utils.tensorboard import SummaryWriter


@dataclass(frozen=True)
class A2CConfig(OnPolicyConfig):
    """Optimization hyperparameters specific to synchronous A2C."""

    # Optimization
    lr_actor            :   float        = 3e-4
    lr_critic           :   float        = 1e-3
    ent_coef            :   float        = 0.0      # Entropy-bonus coefficient.
    max_grad_norm       :   float        = 0.5      # Global L2 gradient clipping threshold.
    normalize_advantage :   bool         = False


class A2CLearner(OnPolicyLearner):
    """A2C optimizers and algorithm-specific full-rollout update."""

    def __init__(
        self,
        policy      :   nn.Module,
        value_fn    :   nn.Module,
        cfg         :   A2CConfig,
    ) -> None:
        super().__init__(policy, value_fn, cfg)
        self.cfg: A2CConfig = cfg
        self.optim_actor = torch.optim.SGD(
            self.policy.parameters(),
            lr=self.cfg.lr_actor,
        )
        self.optim_critic = torch.optim.SGD(
            self.value_fn.parameters(),
            lr=self.cfg.lr_critic,
        )

    def update(
        self,
        buffer          :   RolloutBuffer,
        advantages      :   torch.Tensor,
        value_targets   :   torch.Tensor,
    ) -> TrainingStats:
        """Perform one full-rollout A2C update."""

        # ------------------------------------------------------------------
        # 0. Setup
        # ------------------------------------------------------------------

        # Restore module training mode after eval-mode rollout collection. The
        # forward passes below run outside no_grad() and build new autograd graphs.
        self.policy.train()
        self.value_fn.train()

        # Accumulate metrics over the complete update. A2C currently produces
        # one full-rollout batch, B = T * N.
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
            action_distributions: Distribution = self.policy(batch.observations)

            # Recompute log pi(A|S) under the current policy. Shape: (B,).
            log_probs = action_distributions.log_prob(batch.actions)

            # Recompute V(S) with gradients enabled. Shape: (B,).
            values: torch.Tensor = self.value_fn(batch.observations)

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
            if self.cfg.normalize_advantage and batch_advantages.numel() > 1:
                batch_advantages = (
                    batch_advantages - batch_advantages.mean()
                ) / (batch_advantages.std(unbiased=False) + 1e-8)

            # entropy = mean[H(pi(.|S))]
            # policy_loss = -mean[advantage * log pi(A|S)] - ent_coef * entropy
            # The leading minus converts gradient ascent into optimizer descent.
            policy_loss = -(
                (batch_advantages * log_probs).mean()
                + self.cfg.ent_coef * entropy
            )

            # --------------------------------------------------------------
            # 3. Parameter updates
            # --------------------------------------------------------------

            # Critic semi-gradient update with global L2 gradient clipping.
            self.optim_critic.zero_grad(set_to_none=True)
            value_loss.backward()
            value_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.value_fn.parameters(),
                max_norm=self.cfg.max_grad_norm,
            )
            self.optim_critic.step()

            # Policy-gradient update with global L2 gradient clipping.
            self.optim_actor.zero_grad(set_to_none=True)
            policy_loss.backward()
            policy_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                max_norm=self.cfg.max_grad_norm,
            )
            self.optim_actor.step()

            # --------------------------------------------------------------
            # 4. Batch metric accumulation
            # --------------------------------------------------------------

            # Weight batch metrics by their number of rollout samples.
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
                "charts/policy_learning_rate": self.optim_actor.param_groups[0]["lr"],
                "charts/value_learning_rate": self.optim_critic.param_groups[0]["lr"],
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

    Collects T steps from N environments and performs one full-batch A2C
    update over each rollout through the shared on-policy training engine.
    """

    learner = A2CLearner(policy, value_fn, cfg)

    return train_on_policy(
        envs=envs,
        learner=learner,
        writer=writer,
        logger=logger,
        evaluator=evaluator,
    )
