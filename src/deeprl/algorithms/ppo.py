"""Proximal Policy Optimization learner and training entry point."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from gymnasium.vector import VectorEnv
from torch.distributions import Distribution

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
    from torch.utils.tensorboard import SummaryWriter

    from deeprl.evaluate import Evaluator


@dataclass(frozen=True)
class PPOConfig(OnPolicyConfig):
    """Optimization hyperparameters specific to PPO-Clip."""

    # Optimization
    lr_critic           :   float        = 3e-4
    lr_actor            :   float        = 1e-3
    batch_size          :   int          = 64
    update_epochs       :   int          = 10
    max_grad_norm       :   float        = 0.5

    # PPO clipped objective
    clip_range          :   float        = 0.2
    value_clip_range    :   float | None = None
    ent_coef            :   float        = 0.0
    normalize_advantage :   bool         = True

    # Optional policy-KL early stopping
    target_kl           :   float | None = None


class PPOLearner(OnPolicyLearner):
    """PPO-Clip optimizers and multi-epoch minibatch update."""

    def __init__(
        self,
        policy      :   nn.Module,
        value_fn    :   nn.Module,
        cfg         :   PPOConfig,
    ) -> None:
        super().__init__(policy, value_fn, cfg)
        self.cfg: PPOConfig = cfg

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
        """Optimize the policy and critic over one completed rollout."""

        # ------------------------------------------------------------------
        # 0. Setup
        # ------------------------------------------------------------------

        # Restore training behavior after eval-mode rollout collection. The
        # forward passes below build fresh autograd graphs for every minibatch.
        self.policy.train()
        self.value_fn.train()

        # Accumulate sample-weighted metrics over all accepted minibatch updates.
        metric_sums: dict[str, float] = {
            "losses/policy_loss": 0.0,
            "losses/value_loss": 0.0,
            "diagnostics/entropy": 0.0,
            "diagnostics/approximate_kl": 0.0,
            "diagnostics/clip_fraction": 0.0,
            "grads/policy_norm": 0.0,
            "grads/value_norm": 0.0,
        }
        num_samples = 0
        gradient_steps = 0

        # Normalize once over the complete rollout so every epoch uses the same
        # fixed advantage weights and normalization statistics.
        if self.cfg.normalize_advantage and advantages.numel() > 1:
            advantages = (
                advantages - advantages.mean()
            ) / (advantages.std(unbiased=False) + 1e-8)

        # This flag propagates a target-KL stop from the minibatch loop to the
        # surrounding epoch loop.
        stop_early = False

        for _ in range(self.cfg.update_epochs):

            # Each call reshuffles the complete rollout before yielding flat
            # minibatches of M transitions.
            for batch in buffer.get(
                advantages=advantages,
                value_targets=value_targets,
                batch_size=self.cfg.batch_size,
            ):

                # --------------------------------------------------------------
                # 1. Minibatch re-evaluation
                # --------------------------------------------------------------

                # Re-evaluate pi_theta(.|S), log pi_theta(A|S), and V_phi(S)
                # under the latest parameters. Each scalar output has shape (M,).
                action_distributions: Distribution = self.policy(batch.observations)
                log_probs: torch.Tensor = action_distributions.log_prob(batch.actions)
                values: torch.Tensor = self.value_fn(batch.observations)

                # --------------------------------------------------------------
                # 2. Clipped value objective
                # --------------------------------------------------------------

                # value_targets are the fixed TD(lambda) regression labels. With
                # clipping disabled, use the ordinary critic semi-gradient loss.
                if self.cfg.value_clip_range is None:
                    value_loss = 0.5 * (
                        batch.value_targets - values
                    ).square().mean()
                else:
                    # Restrict the candidate prediction relative to V_old(S):
                    # clipped_value = V_old + clip(V - V_old, -epsilon_v, epsilon_v)
                    clipped_values = batch.old_values + (
                        values - batch.old_values
                    ).clamp(
                        min=-self.cfg.value_clip_range,
                        max=self.cfg.value_clip_range,
                    )
                    unclipped_value_loss = (batch.value_targets - values).square()
                    clipped_value_loss = (
                        batch.value_targets - clipped_values
                    ).square()

                    # The pessimistic maximum prevents the critic from benefiting
                    # merely by moving farther than the configured value range.
                    value_loss = 0.5 * torch.maximum(
                        unclipped_value_loss,
                        clipped_value_loss,
                    ).mean()

                # --------------------------------------------------------------
                # 3. PPO clipped policy objective
                # --------------------------------------------------------------

                # Entropy remains differentiable because it contributes to the
                # policy objective when ent_coef is nonzero.
                entropy = action_distributions.entropy().mean()

                # ratio = pi_theta(A|S) / pi_old(A|S). The denominator remains
                # fixed because old_log_probs were stored during collection.
                log_ratios = log_probs - batch.old_log_probs
                ratios = torch.exp(log_ratios)

                # Maximize the smaller of the ordinary and ratio-clipped
                # surrogate terms for each sampled transition.
                unclipped_objective = ratios * batch.advantages
                clipped_objective = ratios.clip(
                    min=1.0 - self.cfg.clip_range,
                    max=1.0 + self.cfg.clip_range,
                ) * batch.advantages
                policy_loss = -(
                    torch.min(unclipped_objective, clipped_objective).mean()
                    + self.cfg.ent_coef * entropy
                )

                # --------------------------------------------------------------
                # 4. Policy diagnostics and KL early stopping
                # --------------------------------------------------------------

                # These diagnostics use the rollout actions and do not participate
                # in either gradient computation.
                with torch.no_grad():
                    approximate_kl = ((ratios - 1.0) - log_ratios).mean()
                    clip_fraction = (
                        (ratios - 1.0).abs() > self.cfg.clip_range
                    ).float().mean()

                # Stop before another optimizer step when the current policy has
                # already moved substantially beyond the requested target KL.
                if (
                    self.cfg.target_kl is not None
                    and approximate_kl > 1.5 * self.cfg.target_kl
                ):
                    stop_early = True
                    break

                # --------------------------------------------------------------
                # 5. Parameter updates
                # --------------------------------------------------------------

                # Critic semi-gradient update with global L2 gradient clipping.
                self.optim_critic.zero_grad(set_to_none=True)
                value_loss.backward()
                value_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.value_fn.parameters(),
                    max_norm=self.cfg.max_grad_norm,
                )
                self.optim_critic.step()

                # PPO policy update with global L2 gradient clipping.
                self.optim_actor.zero_grad(set_to_none=True)
                policy_loss.backward()
                policy_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    max_norm=self.cfg.max_grad_norm,
                )
                self.optim_actor.step()

                # --------------------------------------------------------------
                # 6. Minibatch metric accumulation
                # --------------------------------------------------------------

                # Weight by samples so a smaller final minibatch receives its
                # proportional influence. Count only completed optimizer updates.
                num_batch_samples = batch.observations.shape[0]
                num_samples += num_batch_samples
                gradient_steps += 1

                for name, value in (
                    ("losses/policy_loss", policy_loss),
                    ("losses/value_loss", value_loss),
                    ("diagnostics/entropy", entropy),
                    ("diagnostics/approximate_kl", approximate_kl),
                    ("diagnostics/clip_fraction", clip_fraction),
                    ("grads/policy_norm", policy_grad_norm),
                    ("grads/value_norm", value_grad_norm),
                ):
                    metric_sums[name] += value.detach().item() * num_batch_samples

            if stop_early:
                break

        # ------------------------------------------------------------------
        # 7. Update metric reduction
        # ------------------------------------------------------------------

        # Produce one sample-weighted mean for the complete PPO update phase.
        metrics = {
            name: total / num_samples
            for name, total in metric_sums.items()
        }

        # Explained variance uses collection-time values and fixed TD(lambda)
        # targets over the complete rollout, matching the other on-policy learners.
        metrics.update(
            {
                "diagnostics/explained_variance": explained_variance(
                    buffer.values.flatten(),
                    value_targets.flatten(),
                ),
                "charts/policy_learning_rate": self.optim_actor.param_groups[0]["lr"],
                "charts/value_learning_rate": self.optim_critic.param_groups[0]["lr"],
            }
        )

        return TrainingStats(
            metrics=metrics,
            gradient_steps=gradient_steps,
        )


def ppo(
    envs        :   VectorEnv,
    policy      :   nn.Module,
    value_fn    :   nn.Module,
    cfg         :   PPOConfig,
    writer      :   SummaryWriter | None = None,
    logger      :   Logger | None = None,
    evaluator   :   Evaluator | None = None,
) -> MetricHistory:
    """Construct a PPO learner and run the shared on-policy training loop."""

    learner = PPOLearner(policy, value_fn, cfg)

    return train_on_policy(
        envs=envs,
        learner=learner,
        writer=writer,
        logger=logger,
        evaluator=evaluator,
    )
