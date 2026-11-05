"""Training lifecycle for synchronous on-policy algorithms."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch.nn as nn
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.spaces import Dict, Tuple

from deeprl.algorithms.on_policy import OnPolicyLearner
from deeprl.advantages import generalized_advantage_estimate
from deeprl.buffers import RolloutBuffer
from deeprl.checkpoint import (
    build_training_checkpoint,
    get_algorithm_id,
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint_algorithm,
    validate_training_checkpoint,
)
from deeprl.env import to_tensor_data
from deeprl.lifecycle import (
    config_to_jsonable, get_provenance, get_rng_state, seed_torch, set_rng_state,
)
from deeprl.logger import Logger, MetricHistory
from deeprl.rollouts import collect_rollout, evaluate_next_values
from deeprl.stats import TrainingProgress

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

    from deeprl.evaluate import Evaluator
    from deeprl.normalization import RewardScaler, RunningNormalizer


def _validate_setup(
    *,
    envs: VectorEnv,
    learner: OnPolicyLearner,
    logger: Logger | None,
    resume_from: str | Path | None,
    checkpoint_interval: int | None,
    obs_normalizer: RunningNormalizer | None,
    reward_scaler: RewardScaler | None,
) -> None:
    """Reject incompatible run inputs before restoring state or resetting envs."""
    cfg = learner.cfg
    if envs.metadata.get("autoreset_mode") is not AutoresetMode.DISABLED:
        raise ValueError("train_on_policy requires AutoresetMode.DISABLED")
    if logger is not None:
        if logger.num_envs != envs.num_envs:
            raise ValueError("logger num_envs must match the environment")
        if resume_from is None and (logger.global_step != 0 or logger.gradient_step != 0):
            raise ValueError("a new run requires a fresh logger")
    if checkpoint_interval is not None and (
        isinstance(checkpoint_interval, bool)
        or not isinstance(checkpoint_interval, int)
        or checkpoint_interval <= 0
    ):
        raise ValueError("checkpoint_interval must be a positive integer or None")

    if cfg.normalize_observations_during_collection:
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
    elif obs_normalizer is not None and not any(
        module is obs_normalizer for module in learner.policy.modules()
    ):
        raise ValueError("obs_normalizer must be a submodule of learner.policy")

    if obs_normalizer is not None:
        space = envs.single_observation_space
        if isinstance(space, (Dict, Tuple)):
            raise ValueError(
                "obs_normalizer cannot normalize structured observations as a whole; "
                "handle per-key preprocessing inside the supplied networks"
            )
        if space.shape is None or tuple(space.shape) != obs_normalizer.shape:
            raise ValueError("obs_normalizer shape must match the observation space")
    if reward_scaler is not None:
        if reward_scaler.returns.numel() != envs.num_envs:
            raise ValueError("reward_scaler num_envs does not match the environment")
        if reward_scaler.gamma != cfg.gamma:
            raise ValueError("reward_scaler gamma does not match the learner")

    rollout_size = cfg.rollout_steps * envs.num_envs
    if cfg.total_timesteps < rollout_size:
        raise ValueError(
            f"total_timesteps={cfg.total_timesteps} is smaller than one rollout "
            f"({cfg.rollout_steps} steps x {envs.num_envs} envs = {rollout_size}); "
            "nothing would be collected"
        )


def _preprocessing_config(
    obs_normalizer: RunningNormalizer | None,
    reward_scaler: RewardScaler | None,
) -> dict[str, Any]:
    """Describe trainer-managed settings that are not module state tensors."""
    return config_to_jsonable({
        "observation": None if obs_normalizer is None else {
            "shape": obs_normalizer.shape,
            "epsilon": obs_normalizer.epsilon,
            "clip": obs_normalizer.clip,
        },
        "reward": None if reward_scaler is None else {
            "gamma": reward_scaler.gamma,
            "epsilon": reward_scaler.epsilon,
            "clip": reward_scaler.clip,
            "update_per_step": reward_scaler.update_per_step,
        },
    })


def _validate_resume(
    checkpoint: dict[str, Any],
    *,
    learner: OnPolicyLearner,
    learner_config: dict[str, Any],
    num_envs: int,
    preprocessing_config: dict[str, Any],
) -> tuple[TrainingProgress, dict[str, Any]]:
    """Check workflow and learner compatibility without loading any state."""
    validate_training_checkpoint(
        checkpoint, required_keys=("num_envs", "preprocessing_config"),
    )
    validate_checkpoint_algorithm(checkpoint, type(learner))
    if checkpoint["num_envs"] != num_envs:
        raise ValueError("checkpoint num_envs does not match the environment")

    resume_overrides = learner.validate_resume_config(
        config_to_jsonable(checkpoint["learner_config"]), learner_config,
    )
    progress = TrainingProgress(**checkpoint["progress"])
    rollout_size = learner.cfg.rollout_steps * num_envs
    if progress.global_step != progress.update * rollout_size:
        raise ValueError("checkpoint progress is inconsistent with the rollout size")
    if learner.cfg.total_timesteps // rollout_size <= progress.update:
        raise ValueError("total_timesteps leaves no remaining complete rollout")

    if learner.cfg.normalize_observations_during_collection != ("obs_normalizer" in checkpoint):
        raise ValueError("checkpoint observation normalization does not match this run")
    if (preprocessing_config["reward"] is not None) != ("reward_scaler" in checkpoint):
        raise ValueError("checkpoint reward scaling does not match this run")
    if config_to_jsonable(checkpoint["preprocessing_config"]) != preprocessing_config:
        raise ValueError("checkpoint preprocessing configuration does not match this run")
    return progress, resume_overrides


def _restore_training_state(
    checkpoint: dict[str, Any],
    *,
    learner: OnPolicyLearner,
    resume_overrides: dict[str, Any],
    obs_normalizer: RunningNormalizer | None,
    reward_scaler: RewardScaler | None,
) -> None:
    """Restore learning components; the trainer restores RNG after env setup."""
    learner.load_state_dict(checkpoint["learner"])
    learner.apply_resume_overrides(resume_overrides)
    if reward_scaler is not None:
        reward_scaler.load_state_dict(checkpoint["reward_scaler"])
    if learner.cfg.normalize_observations_during_collection:
        obs_normalizer.load_state_dict(checkpoint["obs_normalizer"])


def _checkpoint_state(
    *,
    learner: OnPolicyLearner,
    learner_config: dict[str, Any],
    progress: TrainingProgress,
    num_envs: int,
    run_config: dict[str, Any] | None,
    obs_normalizer: RunningNormalizer | None,
    reward_scaler: RewardScaler | None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble validated run state without scheduling or writing a checkpoint."""
    checkpoint = build_training_checkpoint(
        algorithm=get_algorithm_id(type(learner)),
        learner_state=learner.state_dict(),
        learner_config=learner_config,
        run_config=run_config,
        progress=progress,
        rng_state=get_rng_state(device=learner.cfg.device),
        provenance=provenance,
    )
    checkpoint["num_envs"] = num_envs
    checkpoint["preprocessing_config"] = _preprocessing_config(obs_normalizer, reward_scaler)
    if reward_scaler is not None:
        checkpoint["reward_scaler"] = reward_scaler.state_dict()
    if learner.cfg.normalize_observations_during_collection:
        checkpoint["obs_normalizer"] = obs_normalizer.state_dict()
    validate_training_checkpoint(
        checkpoint, required_keys=("num_envs", "preprocessing_config"),
    )
    return checkpoint


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
    Checkpoints declare the continue contract, not exact trajectory continuation.
    Changed actor/critic learning rates override the saved optimizer rates
    without resetting optimizer state. Each learner declares which additional
    settings may change; PPO allows minibatch size, entropy and target KL.
    New checkpoints record the current configuration.
    run_config is optional JSON-compatible experiment metadata.
    Required run state and trainer-managed preprocessing settings are checked
    before restoration. Incomplete checkpoints are rejected.

    By default, an observation normalizer belongs to the policy and updates
    after each learner update. Collection-time normalization instead requires
    an external normalizer and stores normalized observations in the rollout.
    Reward scaling affects GAE inputs, not reported returns; its running
    statistics are checkpointed for resume.
    """
    cfg = learner.cfg
    _validate_setup(
        envs=envs,
        learner=learner,
        logger=logger,
        resume_from=resume_from,
        checkpoint_interval=checkpoint_interval,
        obs_normalizer=obs_normalizer,
        reward_scaler=reward_scaler,
    )

    N, T = envs.num_envs, cfg.rollout_steps
    rollout_size = T * N
    num_updates = cfg.total_timesteps // rollout_size
    online_normalization = cfg.normalize_observations_during_collection
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None

    # Validate plain configuration and resume compatibility before restoring state.
    learner_config = config_to_jsonable(cfg)
    preprocessing_config = _preprocessing_config(obs_normalizer, reward_scaler)
    progress = TrainingProgress()
    checkpoint = None
    resume_overrides: dict[str, Any] = {}
    if resume_from is not None:
        checkpoint = load_checkpoint(resume_from, map_location="cpu")
        progress, resume_overrides = _validate_resume(
            checkpoint,
            learner=learner,
            learner_config=learner_config,
            num_envs=N,
            preprocessing_config=preprocessing_config,
        )
        if run_config is None:
            run_config = checkpoint["config"]
    if run_config is not None:
        run_config = config_to_jsonable(run_config)
    provenance = get_provenance() if checkpoint_dir is not None else None

    # Native state loaders place tensors on the already-constructed components.
    seed_torch(cfg.seed)
    if online_normalization:
        obs_normalizer.to(cfg.device)
    if reward_scaler is not None:
        reward_scaler.to(cfg.device)
    if checkpoint is not None:
        _restore_training_state(
            checkpoint,
            learner=learner,
            resume_overrides=resume_overrides,
            obs_normalizer=obs_normalizer,
            reward_scaler=reward_scaler,
        )
    evaluation_policy = (
        nn.Sequential(obs_normalizer, learner.policy)
        if online_normalization else learner.policy
    )

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

    # Restore RNG after environment reset and storage initialization.
    if checkpoint is not None:
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
                checkpoint = _checkpoint_state(
                    learner=learner,
                    learner_config=learner_config,
                    progress=progress,
                    num_envs=N,
                    run_config=run_config,
                    obs_normalizer=obs_normalizer,
                    reward_scaler=reward_scaler,
                    provenance=provenance,
                )
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
