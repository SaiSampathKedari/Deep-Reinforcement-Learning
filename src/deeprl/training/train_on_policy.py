"""Training lifecycle for synchronous on-policy algorithms."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gymnasium.vector import AutoresetMode, VectorEnv

from deeprl.algorithms.on_policy import OnPolicyLearner
from deeprl.buffers import RolloutBuffer
from deeprl.checkpoint import (
    build_training_checkpoint,
    get_algorithm_id,
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint_algorithm,
    validate_training_checkpoint,
)
from deeprl.lifecycle import (
    config_to_jsonable, get_provenance, get_rng_state, seed_torch, set_rng_state,
)
from deeprl.logger import Logger, MetricHistory
from deeprl.collectors import collect_rollout
from deeprl.data_spec import DataSpec
from deeprl.policy_io import describe_policy, validate_policy_manifest
from deeprl.stats import TrainingProgress

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

    from deeprl.evaluate import Evaluator
    from deeprl.processors import ActionProcessor, ObservationProcessor


def _validate_setup(
    *,
    envs: VectorEnv,
    learner: OnPolicyLearner,
    logger: Logger | None,
    resume_from: str | Path | None,
    checkpoint_interval: int | None,
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

    if learner.reward_scaler is not None:
        if learner.reward_scaler.returns.numel() != envs.num_envs:
            raise ValueError("reward_scaler num_envs does not match the environment")

    rollout_size = cfg.rollout_steps * envs.num_envs
    if cfg.total_timesteps < rollout_size:
        raise ValueError(
            f"total_timesteps={cfg.total_timesteps} is smaller than one rollout "
            f"({cfg.rollout_steps} steps x {envs.num_envs} envs = {rollout_size}); "
            "nothing would be collected"
        )


def _preprocessing_config(
    obs_processor: ObservationProcessor | None,
    action_processor: ActionProcessor | None,
) -> dict[str, Any]:
    """Describe input/output processing settings, separate from training targets."""
    observation = None
    if obs_processor is not None:
        observation = {"spec": obs_processor.spec.to_dict(), "config": obs_processor.config}
    return config_to_jsonable({
        "observation": observation,
        "action": None if action_processor is None else action_processor.config,
    })


def _validate_resume(
    checkpoint: dict[str, Any],
    *,
    learner: OnPolicyLearner,
    learner_config: dict[str, Any],
    num_envs: int,
    preprocessing_config: dict[str, Any],
    policy_manifest: dict[str, Any] | None = None,
) -> tuple[TrainingProgress, dict[str, Any]]:
    """Check workflow and learner compatibility without loading any state."""
    validate_training_checkpoint(
        checkpoint,
        required_keys=("num_envs", "preprocessing_config"),
    )
    validate_checkpoint_algorithm(checkpoint, type(learner))
    if checkpoint["num_envs"] != num_envs:
        raise ValueError("checkpoint num_envs does not match the environment")

    # Learning settings and counters must describe the same rollout schedule.
    resume_overrides = learner.validate_resume_config(
        config_to_jsonable(checkpoint["learner_config"]),
        learner_config,
    )
    progress = TrainingProgress(**checkpoint["progress"])
    rollout_size = learner.cfg.rollout_steps * num_envs
    if progress.global_step != progress.update * rollout_size:
        raise ValueError("checkpoint progress is inconsistent with the rollout size")
    if learner.cfg.total_timesteps // rollout_size <= progress.update:
        raise ValueError("total_timesteps leaves no remaining complete rollout")

    # Restored weights must see the same input/output processing as before.
    if (learner.obs_processor is not None) != ("observation_processor" in checkpoint):
        raise ValueError("checkpoint observation processor does not match this run")
    learner.validate_reward_scaler_state(checkpoint["learner"])
    if config_to_jsonable(checkpoint["preprocessing_config"]) != preprocessing_config:
        raise ValueError("checkpoint preprocessing configuration does not match this run")
    saved_manifest = checkpoint.get("policy_manifest")
    if (saved_manifest is None) != (policy_manifest is None):
        raise ValueError("checkpoint policy manifest does not match this run")
    if saved_manifest is not None:
        saved_manifest = validate_policy_manifest(saved_manifest)
        current_manifest = _validate_policy_description(learner, policy_manifest)
        if any(
            saved_manifest[key] != current_manifest[key]
            for key in saved_manifest
            if key != "provenance"
        ):
            raise ValueError("checkpoint policy configuration does not match this run")
    return progress, resume_overrides


def _validate_policy_description(
    learner: OnPolicyLearner,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Check a declared inference description against the actual processing path."""
    manifest = validate_policy_manifest(manifest)
    actual = describe_policy(
        learner.inference_policy,
        spec=DataSpec.from_dict(manifest["spec"]),
        model_id=manifest["model"]["id"],
        model_config=manifest["model"]["config"],
        decision_id=manifest["decision"]["id"],
        provenance=manifest["provenance"],
    )
    if actual != manifest:
        raise ValueError("policy manifest does not match the learner's inference configuration")
    return manifest


def _restore_training_state(
    checkpoint: dict[str, Any],
    *,
    learner: OnPolicyLearner,
    resume_overrides: dict[str, Any],
) -> None:
    """Restore learning components; the trainer restores RNG after env setup."""
    learner.load_state_dict(checkpoint["learner"])
    learner.apply_resume_overrides(resume_overrides)
    if learner.obs_processor is not None:
        learner.obs_processor.load_state_dict(checkpoint["observation_processor"])


def _checkpoint_state(
    *,
    learner: OnPolicyLearner,
    learner_config: dict[str, Any],
    progress: TrainingProgress,
    num_envs: int,
    run_config: dict[str, Any] | None,
    provenance: dict[str, Any] | None = None,
    policy_manifest: dict[str, Any] | None = None,
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
    checkpoint["preprocessing_config"] = _preprocessing_config(
        learner.obs_processor,
        learner.inference_policy.action_processor,
    )
    if learner.obs_processor is not None:
        checkpoint["observation_processor"] = learner.obs_processor.state_dict()
    if policy_manifest is not None:
        checkpoint["policy_manifest"] = _validate_policy_description(learner, policy_manifest)
    validate_training_checkpoint(
        checkpoint,
        required_keys=("num_envs", "preprocessing_config"),
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
    policy_manifest :   dict[str, Any] | None = None,
) -> MetricHistory:
    """Run the shared synchronous on-policy training lifecycle.

    Collects T steps from N environments and delegates target preparation
    and the algorithm-specific optimization schedule to the learner.

    Checkpoint intervals count completed rollout updates. Resume restores the
    learner and counters, then starts fresh episodes toward total_timesteps.
    Use the same environment and networks; simulator state is not restored.
    Resume does not reproduce an uninterrupted simulator trajectory.
    Changed actor/critic learning rates override the saved optimizer rates
    without resetting optimizer state. Each learner declares which additional
    settings may change; PPO allows minibatch size, entropy and target KL.
    New checkpoints record the current configuration.
    run_config is optional JSON-compatible experiment metadata.
    Required run state and trainer-managed preprocessing settings are checked
    before restoration. Incomplete checkpoints are rejected.

    Rollout observations stay raw. The learner's observation processor shares
    actor/critic statistics, frozen through collection, bootstrap and optimization.
    Statistics update once afterwards from the rollout's current observations.
    Fresh processors start as identity; reset never fits or replaces statistics.
    Reward scaling affects GAE inputs, not reported returns; its settings and
    running statistics are checkpointed as learner state, not inference state.

    A supplied policy_manifest records inference construction/settings alongside
    learner weights, without duplicating them. Simple training-only callers may
    omit it; their checkpoints then have no inference reconstruction descriptor.
    """
    # ------------------------------------------------------------------
    # 0. Validate, restore and initialize the run
    # ------------------------------------------------------------------

    cfg = learner.cfg
    _validate_setup(
        envs=envs,
        learner=learner,
        logger=logger,
        resume_from=resume_from,
        checkpoint_interval=checkpoint_interval,
    )

    N, T = envs.num_envs, cfg.rollout_steps
    rollout_size = T * N
    num_updates = cfg.total_timesteps // rollout_size
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None

    # Validate plain configuration and resume compatibility before restoring state.
    learner_config = config_to_jsonable(cfg)
    preprocessing_config = _preprocessing_config(
        learner.obs_processor,
        learner.inference_policy.action_processor,
    )
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
            policy_manifest=policy_manifest,
        )
        if run_config is None:
            run_config = checkpoint["config"]
    if run_config is not None:
        run_config = config_to_jsonable(run_config)
    if policy_manifest is not None:
        policy_manifest = _validate_policy_description(learner, policy_manifest)
    provenance = get_provenance() if checkpoint_dir is not None else None

    # Native state loaders place tensors on the already-constructed components.
    seed_torch(cfg.seed)
    if checkpoint is not None:
        _restore_training_state(
            checkpoint,
            learner=learner,
            resume_overrides=resume_overrides,
        )
    evaluation_policy = learner.inference_policy

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
    if learner.reward_scaler is not None:
        # Fresh episodes need fresh accumulators; retain the running statistics.
        learner.reward_scaler.returns.zero_()

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
            policy=learner.inference_policy,
            value_fn=learner.evaluate_values,
            rollout_buffer=rollout_buffer,
            observations=observations,
        )
        progress = replace(
            progress,
            global_step=progress.global_step + rollout_size,
        )

        # --------------------------------------------------------------
        # 2. Target preparation
        # --------------------------------------------------------------

        # Bootstrap true successors and compute fixed GAE targets without
        # changing the stored observations, rewards or processing statistics.
        advantages, value_targets = learner.prepare_targets(rollout_buffer)

        # --------------------------------------------------------------
        # 3. Parameter updates
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
        if learner.obs_processor is not None:
            learner.obs_processor.update(rollout_buffer.observations)

        # --------------------------------------------------------------
        # 4. Metrics and evaluation
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

        # --------------------------------------------------------------
        # 5. Save completed-update checkpoints
        # --------------------------------------------------------------

        # Intervals count rollout updates; no partially updated state is saved.
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
                    provenance=provenance,
                    policy_manifest=policy_manifest,
                )
                save_checkpoint(checkpoint_dir / "last.pt", checkpoint)

                if save_due:
                    save_checkpoint(
                        checkpoint_dir / f"step_{progress.global_step}.pt",
                        checkpoint
                    )

    # ------------------------------------------------------------------
    # 6. Final evaluation and cleanup
    # ------------------------------------------------------------------

    # Evaluate the final policy even when training ends between checkpoints.
    if evaluator is not None:
        evaluator.evaluate_now(evaluation_policy, logger, progress=progress)

    # Only finalize resources created inside this function.
    return logger.finish() if owns_logger else logger.history
