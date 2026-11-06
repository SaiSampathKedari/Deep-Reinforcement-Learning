"""Proximal Policy Optimization learner and training entry point."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from gymnasium.vector import VectorEnv
from torch.distributions import Distribution, Independent, Normal

from deeprl import learner_state
from deeprl.algorithms.on_policy import (
    OnPolicyConfig,
    OnPolicyLearner,
)
from deeprl.buffers import RolloutBuffer
from deeprl.logger import Logger, MetricHistory
from deeprl.normalization import RunningNormalizer
from deeprl.stats import TrainingStats
from deeprl.training.train_on_policy import train_on_policy
from deeprl.utils import explained_variance

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

    from deeprl.evaluate import Evaluator
    from deeprl.normalization import RewardScaler
    from deeprl.processors import ActionProcessor, ObservationProcessor

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
    normalize_advantage_per_minibatch: bool = False
    normalize_value_targets: bool = False
    log_ratio_clip      :   float | None = None

    # Optional policy-KL early stopping
    target_kl           :   float | None = None

    # PyTorch optimizer classes and options, excluding params and lr.
    actor_optimizer_class   :   type[torch.optim.Optimizer] = torch.optim.Adam
    actor_optimizer_kwargs  :   dict[str, Any] = field(default_factory=dict)
    critic_optimizer_class  :   type[torch.optim.Optimizer] = torch.optim.Adam
    critic_optimizer_kwargs :   dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate PPO update counts and finite optimization coefficients."""
        super().__post_init__()
        
        for name, value in (
            ("batch_size", self.batch_size),
            ("update_epochs", self.update_epochs),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise ValueError(
                    f"{name} must be a positive integer, got {value!r}"
                )

        for name, value in (
            ("lr_actor", self.lr_actor),
            ("lr_critic", self.lr_critic),
            ("ent_coef", self.ent_coef),
            ("max_grad_norm", self.max_grad_norm),
            ("clip_range", self.clip_range),
            ("value_clip_range", self.value_clip_range),
            ("log_ratio_clip", self.log_ratio_clip),
            ("target_kl", self.target_kl),
        ):
            if value is None and name in {
                "value_clip_range", "log_ratio_clip", "target_kl",
            }:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not isfinite(value)
                or value < 0.0
            ):
                raise ValueError(
                    f"{name} must be a finite number >= 0, got {value!r}"
                )


class PPOLearner(OnPolicyLearner):
    """PPO-Clip optimizers and multi-epoch minibatch update."""

    algorithm_id = "ppo"

    mutable_on_resume = OnPolicyLearner.mutable_on_resume | {
        "lr_actor", "lr_critic", "ent_coef", "target_kl", "batch_size",
    }
    
    def __init__(
        self,
        policy      :   nn.Module,
        value_fn    :   nn.Module,
        cfg         :   PPOConfig,
        *,
        obs_processor: ObservationProcessor | None = None,
        action_processor: ActionProcessor | None = None,
        reward_scaler: RewardScaler | None = None,
    ) -> None:
        if cfg.normalize_value_targets and cfg.value_clip_range is not None:
            raise ValueError("value-target normalization cannot be combined with value clipping")
        super().__init__(
            policy,
            value_fn,
            cfg,
            obs_processor=obs_processor,
            action_processor=action_processor,
            reward_scaler=reward_scaler,
        )
        self.cfg: PPOConfig = cfg
        self.value_target_stats = (
            RunningNormalizer((), clip=None).to(cfg.device)
            if cfg.normalize_value_targets else None
        )
        self.configure_optimizers()
    
    def configure_optimizers(self) -> None:
        """Construct optimizers once, after network device placement."""
        if hasattr(self, "optim_actor") or hasattr(self, "optim_critic"):
            raise RuntimeError("optimizers are already configured")
        self.optim_actor = self.cfg.actor_optimizer_class(
            self.policy.parameters(),
            lr=self.cfg.lr_actor,
            **self.cfg.actor_optimizer_kwargs,
        )
        self.optim_critic = self.cfg.critic_optimizer_class(
            self.value_fn.parameters(),
            lr=self.cfg.lr_critic,
            **self.cfg.critic_optimizer_kwargs,
        )
    
    def state_dict(self) -> dict[str, Any]:
        """Return network, optimizer, and optional value-target states."""
        state = {
            "policy": self.policy.state_dict(),
            "value_fn": self.value_fn.state_dict(),
            "optim_actor": self.optim_actor.state_dict(),
            "optim_critic": self.optim_critic.state_dict(),
        }
        if self.value_target_stats is not None:
            state["value_target_stats"] = self.value_target_stats.state_dict()
        if self.reward_scaler is not None:
            state["reward_scaler"] = {
                "config": self.reward_scaler.config,
                "state_dict": self.reward_scaler.state_dict(),
            }
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Validate checkpoint keys before restoring existing components."""
        self.validate_reward_scaler_state(state)
        expected_keys = {"policy", "value_fn", "optim_actor", "optim_critic"}
        if self.value_target_stats is not None:
            expected_keys.add("value_target_stats")
        if self.reward_scaler is not None:
            expected_keys.add("reward_scaler")
        learner_state.validate_state_keys(state, expected_keys)
        self.policy.load_state_dict(state["policy"])
        self.value_fn.load_state_dict(state["value_fn"])
        self.optim_actor.load_state_dict(state["optim_actor"])
        self.optim_critic.load_state_dict(state["optim_critic"])
        if self.value_target_stats is not None:
            self.value_target_stats.load_state_dict(state["value_target_stats"])
        if self.reward_scaler is not None:
            self.reward_scaler.load_state_dict(state["reward_scaler"]["state_dict"])

    def apply_resume_overrides(self, overrides: dict[str, Any]) -> None:
        """Apply changed learning rates after restoring optimizer state."""
        if "lr_actor" in overrides:
            for group in self.optim_actor.param_groups:
                group["lr"] = overrides["lr_actor"]
        if "lr_critic" in overrides:
            for group in self.optim_critic.param_groups:
                group["lr"] = overrides["lr_critic"]

    def update(
        self,
        buffer          :   RolloutBuffer,
        advantages      :   torch.Tensor,
        value_targets   :   torch.Tensor,
    ) -> TrainingStats:
        """Optimize one rollout using fixed advantages/value targets shaped (T, N).

        Each epoch reshuffles B = T * N transitions into minibatches of M.
        Likelihoods, ratios and advantage weights have shape (M,) within a batch.
        """

        # ------------------------------------------------------------------
        # 0. Setup
        # ------------------------------------------------------------------

        # Restore training behavior after eval-mode rollout collection. The
        # forward passes below build fresh autograd graphs for every minibatch.
        self.policy.train()
        self.value_fn.train()

        # Accumulate sample-weighted metrics on device across accepted updates.
        metric_names = (
            "losses/policy_loss",
            "losses/value_loss",
            "diagnostics/entropy",
            "diagnostics/approximate_kl",
            "diagnostics/clip_fraction",
            "grads/policy_norm",
            "grads/value_norm",
        )
        metric_sums = torch.zeros(
            len(metric_names),
            dtype=advantages.dtype,
            device=self.cfg.device,
        )
        num_samples = 0
        gradient_steps = 0

        # Normalize once over the complete rollout so every epoch uses the same
        # fixed advantage weights and normalization statistics.
        if (
            self.cfg.normalize_advantage
            and not self.cfg.normalize_advantage_per_minibatch
            and advantages.numel() > 1
        ):
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
                action_distributions: Distribution = self.inference_policy(batch.observations)
                log_probs: torch.Tensor = action_distributions.log_prob(batch.actions)
                values: torch.Tensor = self.evaluate_values(batch.observations)
                batch_advantages = batch.advantages
                if (
                    self.cfg.normalize_advantage
                    and self.cfg.normalize_advantage_per_minibatch
                    and batch_advantages.numel() > 1
                ):
                    batch_advantages = (
                        batch_advantages - batch_advantages.mean()
                    ) / (batch_advantages.std(correction=1) + 1e-8)

                # --------------------------------------------------------------
                # 2. Clipped value objective
                # --------------------------------------------------------------

                # value_targets are the fixed TD(lambda) regression labels. With
                # clipping disabled, use the ordinary critic semi-gradient loss.
                if self.cfg.value_clip_range is None:
                    loss_values, loss_targets = values, batch.value_targets
                    if self.value_target_stats is not None:
                        # Scale only the regression loss; GAE keeps raw values.
                        self.value_target_stats.update(batch.value_targets)
                        scale = (self.value_target_stats.var + 1e-8).sqrt().clamp_min(1.0)
                        loss_values = (values / scale).clamp(-10.0, 10.0)
                        loss_targets = (batch.value_targets / scale).clamp(-10.0, 10.0)
                    value_loss = 0.5 * (
                        loss_targets - loss_values
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
                objective_log_ratios = log_ratios
                if self.cfg.log_ratio_clip is not None:
                    objective_log_ratios = log_ratios.clamp(
                        -self.cfg.log_ratio_clip,
                        self.cfg.log_ratio_clip,
                    )
                ratios = torch.exp(objective_log_ratios)

                # Maximize the smaller of the ordinary and ratio-clipped
                # surrogate terms for each sampled transition.
                unclipped_objective = ratios * batch_advantages
                clipped_objective = ratios.clip(
                    min=1.0 - self.cfg.clip_range,
                    max=1.0 + self.cfg.clip_range,
                ) * batch_advantages
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
                    diagnostic_ratios = (
                        log_ratios.exp() if self.cfg.log_ratio_clip is not None else ratios
                    )
                    approximate_kl = ((diagnostic_ratios - 1.0) - log_ratios).mean()
                    clip_fraction = (
                        (diagnostic_ratios - 1.0).abs() > self.cfg.clip_range
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

                minibatch_metrics = torch.stack((
                    policy_loss.detach(),
                    value_loss.detach(),
                    entropy.detach(),
                    approximate_kl.detach(),
                    clip_fraction.detach(),
                    policy_grad_norm.detach(),
                    value_grad_norm.detach(),
                ))
                metric_sums.add_(minibatch_metrics, alpha=num_batch_samples)

            if stop_early:
                break

        # ------------------------------------------------------------------
        # 7. Update metric reduction
        # ------------------------------------------------------------------

        # Transfer metrics once, then report sample-weighted update means.
        metrics = {}
        if num_samples > 0:
            metrics = {
                name: total / num_samples
                for name, total in zip(metric_names, metric_sums.tolist())
            }

        rollout_size = buffer.rollout_steps * buffer.num_envs
        metrics.update({
            "diagnostics/early_stop": float(stop_early),
            "diagnostics/optimizer_steps": float(gradient_steps),
            "diagnostics/effective_epochs": num_samples / rollout_size,
            # Unlike the update average, this includes a rejected minibatch.
            "diagnostics/last_minibatch_kl": approximate_kl.item(),
        })

        # Re-evaluate the final policy on every rollout sample, without another
        # shuffle or optimizer step. This precedes observation-statistics updates.
        assert buffer.observations is not None and buffer.actions is not None
        observations = buffer.observations.flatten(0, 1)
        actions = buffer.actions.flatten(0, 1)
        old_log_probs = buffer.log_probs.flatten()
        final_kl_sum = old_log_probs.new_zeros(())
        gaussian_stats = None
        num_action_components = 0
        modes = [(module, module.training) for module in self.policy.modules()]
        fork_devices = []
        if self.cfg.device.type == "cuda":
            fork_devices = [
                self.cfg.device.index
                if self.cfg.device.index is not None else torch.cuda.current_device()
            ]
        self.policy.eval()
        try:
            with torch.no_grad(), torch.random.fork_rng(devices=fork_devices):
                for start in range(0, rollout_size, self.cfg.batch_size):
                    stop = start + self.cfg.batch_size
                    distribution = self.inference_policy(observations[start:stop])
                    log_ratio = (
                        distribution.log_prob(actions[start:stop]) - old_log_probs[start:stop]
                    )
                    final_kl_sum = final_kl_sum + (log_ratio.exp() - 1 - log_ratio).sum()
                    if (
                        isinstance(distribution, Independent)
                        and isinstance(distribution.base_dist, Normal)
                    ):
                        # Mean/std retain action dimensions: (M, *action_shape).
                        std, mean_abs = distribution.stddev, distribution.mean.abs()
                        current = torch.stack((
                            std.sum(),
                            std.min(),
                            std.max(),
                            mean_abs.sum(),
                            mean_abs.max(),
                        ))
                        if gaussian_stats is None:
                            gaussian_stats = current
                        else:
                            gaussian_stats[0] += current[0]
                            gaussian_stats[1] = torch.minimum(gaussian_stats[1], current[1])
                            gaussian_stats[2] = torch.maximum(gaussian_stats[2], current[2])
                            gaussian_stats[3] += current[3]
                            gaussian_stats[4] = torch.maximum(gaussian_stats[4], current[4])
                        num_action_components += std.numel()
        finally:
            for module, training in modes:
                module.training = training

        metrics["diagnostics/final_approximate_kl"] = (final_kl_sum / rollout_size).item()
        if gaussian_stats is not None:
            gaussian_stats[0] /= num_action_components
            gaussian_stats[3] /= num_action_components
            metrics.update(zip((
                "diagnostics/action_std_mean",
                "diagnostics/action_std_min",
                "diagnostics/action_std_max",
                "diagnostics/action_mean_abs_mean",
                "diagnostics/action_mean_abs_max",
            ), gaussian_stats.tolist()))

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
    checkpoint_dir: str | Path | None = None,
    checkpoint_interval: int | None = None,
    run_config: dict[str, Any] | None = None,
    resume_from: str | Path | None = None,
    policy_manifest: dict[str, Any] | None = None,
    reward_scaler: RewardScaler | None = None,
    obs_processor: ObservationProcessor | None = None,
    action_processor: ActionProcessor | None = None,
) -> MetricHistory:
    """Construct a PPO learner and run the shared on-policy training loop."""

    learner = PPOLearner(
        policy,
        value_fn,
        cfg,
        obs_processor=obs_processor,
        action_processor=action_processor,
        reward_scaler=reward_scaler,
    )

    return train_on_policy(
        envs=envs,
        learner=learner,
        writer=writer,
        logger=logger,
        evaluator=evaluator,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
        run_config=run_config,
        resume_from=resume_from,
        policy_manifest=policy_manifest,
    )
