"""Shared training lifecycle for synchronous on-policy algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from gymnasium.vector import AutoresetMode, VectorEnv

from deeprl.advantages import generalized_advantage_estimate
from deeprl.buffers import RolloutBuffer
from deeprl.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    CHECKPOINT_RESUME_CONTRACT,
    load_checkpoint,
    save_checkpoint,
    validate_training_checkpoint,
)
from deeprl.env import to_tensor_data
from deeprl.lifecycle import config_to_jsonable, get_rng_state, seed_torch, set_rng_state
from deeprl.logger import Logger, MetricHistory
from deeprl.rollouts import collect_rollout, evaluate_next_values
from deeprl.stats import TrainingProgress, TrainingStats

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

    from deeprl.evaluate import Evaluator
    from deeprl.normalization import RewardScaler, RunningNormalizer


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
    normalize_observations_during_collection: bool = False
    
    def __post_init__(self) -> None:
        """Validate shared run and advantage-estimation settings."""
        for name, value, minimum in (
            ("total_timesteps", self.total_timesteps, 1),
            ("rollout_steps", self.rollout_steps, 1),
            ("log_every", self.log_every, 0),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < minimum
            ):
                raise ValueError(
                    f"{name} must be an integer >= {minimum}, got {value!r}"
                )

        for name, value in (
            ("gamma", self.gamma),
            ("gae_lambda", self.gae_lambda),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must be between 0 and 1, got {value!r}"
                )
    


class OnPolicyLearner(ABC):
    """On-policy update, named state, and resume-compatibility contract."""
    
    mutable_on_resume = frozenset({"total_timesteps", "device", "log_every"})
    resume_defaults: dict[str, Any] = {
        "normalize_observations_during_collection": False,
    }
    learning_rate_fields: dict[str, str] = {}

    def __init__(
        self,
        policy      :   nn.Module,
        value_fn    :   nn.Module,
        cfg         :   OnPolicyConfig,
    ) -> None:
        policy_ids = {id(p) for p in policy.parameters() if p.requires_grad}
        value_ids = {id(p) for p in value_fn.parameters() if p.requires_grad}
        if policy_ids & value_ids:
            raise ValueError("policy and value_fn must not share trainable parameters")
        
        self.cfg = cfg
        self.policy = policy.to(self.cfg.device)
        self.value_fn = value_fn.to(self.cfg.device)

    @property
    def modules_by_role(self) -> dict[str, nn.Module]:
        """Return checkpointed modules under their stable state keys."""
        return {
            "policy": self.policy,
            "value_fn": self.value_fn,
        }
    
    @property
    def optimizers_by_role(self) -> dict[str, torch.optim.Optimizer]:
        """Return optimizer state; manual-update learners may return none."""
        return {}
    
    def _state_objects(self) -> dict[str, nn.Module | torch.optim.Optimizer]:
        modules = self.modules_by_role
        optimizers = self.optimizers_by_role
        
        if modules.keys() & optimizers.keys():
            raise ValueError("module and optimizer state keys must be distinct")
        return {**modules, **optimizers}
    
    
    def state_dict(self) -> dict[str, Any]:
        """Return named module and optimizer states using the existing flat layout."""
        return {
            name: component.state_dict()
            for name, component in self._state_objects().items()
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Check component names before restoring modules and optimizers."""
        components = self._state_objects()
        missing = components.keys() - state.keys()
        unexpected = state.keys() - components.keys()
        if missing or unexpected:
            raise ValueError(
                f"checkpoint learner state differs: missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )
        for name, component in components.items():
            component.load_state_dict(state[name])
    
    def validate_resume_config(
        self,
        saved_config: dict[str, Any],
        current_config: dict[str, Any],
    ) -> dict[str, float]:
        """Validate configuration and return changed learning rates by optimizer role."""
        changed = [
            name for name, value in current_config.items()
            if name not in self.mutable_on_resume
            and saved_config.get(name, self.resume_defaults.get(name)) != value
        ]
        if changed:
            raise ValueError(f"checkpoint learner config differs: {', '.join(changed)}")

        return {
            role: current_config[name]
            for name, role in self.learning_rate_fields.items()
            if name in saved_config and saved_config[name] != current_config[name]
        }

    def apply_resume_overrides(self, learning_rates: dict[str, float]) -> None:
        """Apply configured learning-rate changes after restoring optimizer state."""
        optimizers = self.optimizers_by_role
        for role, learning_rate in learning_rates.items():
            for group in optimizers[role].param_groups:
                group["lr"] = learning_rate

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
    obs_normalizer  :   RunningNormalizer | None = None,
    reward_scaler   :   RewardScaler | None = None,
) -> MetricHistory:
    """Run the shared synchronous on-policy training lifecycle.

    Collects T steps from N environments, computes GAE, and delegates the
    algorithm-specific optimization schedule to the learner.

    Checkpoint intervals count completed rollout updates. Resume restores the
    learner and counters, then starts fresh episodes toward total_timesteps.
    Use the same environment and networks; simulator state is not restored.
    Versioned checkpoints declare the continue contract, not exact trajectory
    continuation. Legacy checkpoints remain supported with reduced checks.
    Changed actor/critic learning rates override the saved optimizer rates
    without resetting optimizer state. Each learner declares which additional
    settings may change; PPO allows minibatch size, entropy and target KL.
    New checkpoints record the current configuration.
    run_config is optional JSON-compatible experiment metadata.

    By default, an observation normalizer belongs to the policy and updates
    after each learner update. Collection-time normalization instead requires
    an external normalizer and stores normalized observations in the rollout.
    Reward scaling affects GAE inputs, not reported returns; its running
    statistics are checkpointed for resume.
    """
    cfg = learner.cfg

    if envs.metadata.get("autoreset_mode") is not AutoresetMode.DISABLED:
        raise ValueError("train_on_policy requires AutoresetMode.DISABLED")

    if logger is not None:
        if logger.num_envs != envs.num_envs:
            raise ValueError("logger num_envs must match the environment")

        if resume_from is None and (
            logger.global_step != 0 or logger.gradient_step != 0
        ):
            raise ValueError("a new run requires a fresh logger")
    
    online_normalization = cfg.normalize_observations_during_collection
    if online_normalization:
        if obs_normalizer is None:
            raise ValueError("collection-time normalization requires obs_normalizer")
        if any(
            module is obs_normalizer
            for network in (learner.policy, learner.value_fn)
            for module in network.modules()
        ):
            raise ValueError(
                "collection-time obs_normalizer must not be embedded in policy or value_fn"
            )
        obs_normalizer.to(cfg.device)
    elif obs_normalizer is not None and not any(
        module is obs_normalizer for module in learner.policy.modules()
    ):
        raise ValueError("obs_normalizer must be a submodule of learner.policy")

    evaluation_policy = (
        nn.Sequential(obs_normalizer, learner.policy)
        if online_normalization else learner.policy
    )

    if reward_scaler is not None:
        if reward_scaler.returns.numel() != envs.num_envs:
            raise ValueError("reward_scaler num_envs does not match the environment")
        if reward_scaler.gamma != cfg.gamma:
            raise ValueError("reward_scaler gamma does not match the learner")
        reward_scaler.to(cfg.device)

    if checkpoint_interval is not None and checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive or None")

    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    # T steps from each of N environments produce B = T * N transitions.
    N = envs.num_envs
    T = cfg.rollout_steps
    rollout_size = cfg.rollout_steps * envs.num_envs

    # Only complete rollouts are collected and used for updates.
    num_updates = cfg.total_timesteps // rollout_size

    # Store config values, not executable optimizer classes, so the checkpoint
    # remains readable with torch.load(weights_only=True).
    learner_config = config_to_jsonable(cfg)

    algorithm = f"{type(learner).__module__}.{type(learner).__qualname__}"
    progress = TrainingProgress()
    checkpoint = None
    learning_rate_overrides: dict[str, float] = {}
    if resume_from is not None:
        # RNG states stay on CPU; native state loaders move model and optimizer
        # tensors to the devices of the learner's parameters.
        checkpoint = load_checkpoint(resume_from, map_location="cpu")
        validate_training_checkpoint(checkpoint, required_keys=("num_envs",))
        if checkpoint.get("algorithm", algorithm) != algorithm:
            raise ValueError("checkpoint algorithm does not match the learner")
        if checkpoint.get("num_envs", N) != N:
            raise ValueError("checkpoint num_envs does not match the environment")

        saved_config = checkpoint.get("learner_config")
        if saved_config is not None:
            learning_rate_overrides = learner.validate_resume_config(
                config_to_jsonable(saved_config), learner_config
            )

        progress = TrainingProgress(**checkpoint["progress"])
        if progress.global_step != progress.update * rollout_size:
            raise ValueError("checkpoint progress is inconsistent with the rollout size")
        if num_updates <= progress.update:
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
        run_config = config_to_jsonable(run_config)

    # Seed a new run; a resumed run restores its sampling RNG after setup.
    seed_torch(cfg.seed)
    if checkpoint is not None:
        if online_normalization != ("obs_normalizer" in checkpoint):
            raise ValueError("checkpoint observation normalization does not match this run")
        if (reward_scaler is not None) != ("reward_scaler" in checkpoint):
            raise ValueError("checkpoint reward scaling does not match this run")
        learner.load_state_dict(checkpoint["learner"])
        # Optimizer loading restores saved rates as well as moments and counters.
        learner.apply_resume_overrides(learning_rate_overrides)
        if reward_scaler is not None:
            reward_scaler.load_state_dict(checkpoint["reward_scaler"])
        if online_normalization:
            obs_normalizer.load_state_dict(checkpoint["obs_normalizer"])

    # Reuse a caller-owned logger, or create one finalized by this function.
    owns_logger = logger is None
    if logger is None:
        logger = Logger(envs.num_envs, cfg.log_every, writer, cfg.device)
    if checkpoint is not None:
        logger.resume(progress=progress)
        if evaluator is not None:
            evaluator.resume(progress.global_step)
    logger.log_hyperparameters(cfg)

    # Initial observation batch: (N, *obs_shape). Resume also starts new episodes.
    observations, _ = envs.reset(seed=cfg.seed)
    if online_normalization:
        observations = to_tensor_data(observations, batch_size=N, device=cfg.device)
        obs_normalizer.update(observations)
        observations = obs_normalizer(observations)
    if reward_scaler is not None:
        # Fresh episodes need fresh accumulators; retain the running statistics.
        reward_scaler.returns.zero_()

    # Time-major storage reused for every rollout: (T, N, ...).
    rollout_buffer = RolloutBuffer(
        rollout_steps=T,
        num_envs=N,
        device=cfg.device,
    )

    # Older checkpoints may contain learner/progress only. Restore RNG when
    # available, after environment reset and storage initialization are done.
    if checkpoint is not None and checkpoint.get("rng_state") is not None:
        set_rng_state(checkpoint["rng_state"], device=cfg.device)
    checkpoint = None

    for update in range(progress.update + 1, num_updates + 1):

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
            obs_normalizer=obs_normalizer if online_normalization else None,
        )
        progress = replace(progress, global_step=progress.global_step + rollout_size)

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

        # Scale training targets without changing the rewards used for reporting.
        rewards = rollout_buffer.rewards
        if reward_scaler is not None:
            rewards = reward_scaler(rewards, rollout_buffer.dones)

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
            rewards,
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

        # Keep statistics fixed through collection, bootstrapping and optimization.
        # Merge this rollout only after all learner updates have finished.
        if obs_normalizer is not None and not online_normalization:
            obs_normalizer.update(rollout_buffer.observations)

        # --------------------------------------------------------------
        # 5. Metrics and evaluation
        # --------------------------------------------------------------

        # Record collection and learner statistics before checking whether
        # the completed update has crossed a reporting boundary.
        # Both rewards and episode-boundary masks have shape (T, N).
        logger.log_rollout(
            rewards=rollout_buffer.rewards,
            dones=rollout_buffer.terminations | rollout_buffer.truncations,
            progress=progress,
        )
        progress = replace(
            progress,
            gradient_step=progress.gradient_step + training_stats.gradient_steps,
            update=update,
        )
        logger.log_update(training_stats, progress=progress)
        logger.maybe_dump()

        # Evaluation also runs only at completed update boundaries.
        if evaluator is not None:
            evaluator.maybe_evaluate(evaluation_policy, logger, progress=progress)

        # Checkpoint intervals count completed rollout updates.
        if checkpoint_dir is not None:
            save_due = (
                checkpoint_interval is not None
                and progress.update % checkpoint_interval == 0
            )

            if save_due or progress.update == num_updates:
                checkpoint = {
                    "format_version"    : CHECKPOINT_FORMAT_VERSION,
                    "resume_contract"   : CHECKPOINT_RESUME_CONTRACT,
                    "algorithm"         : algorithm,
                    "learner_config"    : learner_config,
                    "num_envs"          : N,
                    "learner"           : learner.state_dict(),
                    "config"            : run_config,
                    "progress"          : asdict(progress),
                    "rng_state"         : get_rng_state(device=cfg.device),
                }
                if reward_scaler is not None:
                    checkpoint["reward_scaler"] = reward_scaler.state_dict()
                if online_normalization:
                    checkpoint["obs_normalizer"] = obs_normalizer.state_dict()
                validate_training_checkpoint(checkpoint, required_keys=("num_envs",))
                save_checkpoint(checkpoint_dir / "last.pt", checkpoint)

                if save_due:
                    save_checkpoint(
                        checkpoint_dir / f"step_{progress.global_step}.pt",
                        checkpoint
                    )

    # ------------------------------------------------------------------
    # Final evaluation and cleanup
    # ------------------------------------------------------------------

    # Evaluate the final policy even when training ends between checkpoints.
    if evaluator is not None:
        evaluator.evaluate_now(evaluation_policy, logger, progress=progress)

    # Only finalize resources created inside this function.
    return logger.finish() if owns_logger else logger.history
