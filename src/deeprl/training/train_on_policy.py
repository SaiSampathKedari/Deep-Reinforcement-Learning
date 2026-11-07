"""The training loop for on-policy algorithms (PPO, A2C, NPG, TRPO).

The idea
--------
On-policy training repeats four things until the step budget is used up::

    1. play    run the current policy in N environments for T steps
               (one rollout = T x N transitions)
    2. learn   score every action in the rollout (advantages) and update
               the networks with it
    3. forget  throw the rollout away
    4. record  log, evaluate, save a checkpoint

Why throw the data away? On-policy methods ask "how good is *this*
policy?", and answer it from data that this policy produced. After one
cycle the policy has moved on, so the old rollout increasingly describes
a policy that no longer exists. Within one cycle the learner may still
use the rollout several times: PPO runs several epochs of minibatches
over it, and its clipping keeps the policy close to the one that
collected the data. So each trainer cycle collects a fresh rollout,
optimizes on it, then discards it. The price: on-policy methods need
many environment steps.

Who does what
-------------
* This trainer owns the **schedule**: when to collect, learn, log,
  evaluate and save, and the step counters (``TrainingProgress``).
* The learner owns the **math**: targets, losses, optimizers.
* ``collect_rollout`` owns **stepping the environments**.
* The caller owns the environments, the networks and, optionally, the
  logger; this trainer never closes what it did not create.

The parallel file
-----------------
``train_off_policy.py`` has the same numbered steps 0-9. Where they differ::

    on-policy (this file)                   off-policy (train_off_policy.py)
    --------------------------------------  ----------------------------------------
    RolloutBuffer, refilled every update     ReplayBuffer, keeps old data (FIFO)
    a cycle = one rollout of T x N steps     a cycle = train_freq vector steps,
      then one learner update                  then gradient_steps minibatch updates
    the learner owns the inference policy    the caller passes a behaviour policy
                                               (explores) and an inference policy
    observation statistics update AFTER      observation statistics update DURING
      the learner update                       collection, step by step
    no target networks                       target networks, refreshed every
                                               target_update_interval updates
    resume continues whole rollouts          resume refills the empty replay first
"""

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
from deeprl.collectors import collect_rollout
from deeprl.data_spec import DataSpec
from deeprl.env import to_tensor_data
from deeprl.inference import InferencePolicy
from deeprl.lifecycle import (
    config_to_jsonable,
    get_provenance,
    get_rng_state,
    seed_torch,
    set_rng_state,
)
from deeprl.logger import Logger, MetricHistory
from deeprl.policy_io import describe_policy, validate_policy_manifest
from deeprl.stats import TrainingProgress

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

    from deeprl.evaluate import Evaluator


def _validate_setup(
    *,
    envs: VectorEnv,
    learner: OnPolicyLearner,
    logger: Logger | None,
    resume_from: str | Path | None,
    checkpoint_interval: int | None,
) -> None:
    """Fail fast on a bad setup, before anything is loaded or reset.

    A mistake caught here costs nothing. The same mistake found an hour into
    training costs an hour.

    Raises:
        ValueError: If any of the checks below fails.
    """
    cfg = learner.cfg
    num_envs = envs.num_envs

    # 1. The environments must not reset themselves. collect_rollout stores
    #    the true final observation of each episode and then resets that
    #    environment itself; it relies on nothing resetting behind its back.
    if envs.metadata.get("autoreset_mode") is not AutoresetMode.DISABLED:
        raise ValueError("train_on_policy requires AutoresetMode.DISABLED")

    # 2. A logger from the caller must track the same N environments, and a
    #    new run needs a fresh one. A used logger would continue the step
    #    counts of some other run.
    if logger is not None:
        if logger.num_envs != num_envs:
            raise ValueError("logger num_envs must match the environment")
        if resume_from is None and (logger.global_step != 0 or logger.gradient_step != 0):
            raise ValueError("a new run requires a fresh logger")

    # 3. "Save every k updates" only makes sense for a whole number k > 0.
    if checkpoint_interval is not None and (
        isinstance(checkpoint_interval, bool)
        or not isinstance(checkpoint_interval, int)
        or checkpoint_interval <= 0
    ):
        raise ValueError("checkpoint_interval must be a positive integer or None")

    # 4. The reward scaler keeps one running return per environment, so it
    #    needs exactly N slots.
    if (
        learner.reward_scaler is not None
        and learner.reward_scaler.returns.numel() != num_envs
    ):
        raise ValueError("reward_scaler num_envs does not match the environment")

    # 5. Training runs whole rollouts only. If not even one fits in the
    #    budget, the loop would run zero times and silently do nothing.
    rollout_size = cfg.rollout_steps * num_envs
    if cfg.total_timesteps < rollout_size:
        raise ValueError(
            f"total_timesteps={cfg.total_timesteps} is smaller than one rollout "
            f"({cfg.rollout_steps} steps x {num_envs} envs = {rollout_size}); "
            "nothing would be collected"
        )


def _preprocessing_config(
    inference_policy: InferencePolicy,
) -> dict[str, Any]:
    """Write down how observations and actions are processed, as a JSON-compatible dictionary.

    Why: trained weights only make sense together with the processing they
    were trained with. A network trained on normalized inputs is useless on
    raw ones; a policy trained with actions rescaled to [-0.4, 0.4] acts
    wrongly with another scale. So this description goes into every
    checkpoint, and a resumed run must match it exactly.

    Returns:
        ``{"observation": {"spec": ..., "config": ...} or None,
        "action": {...} or None}``. These are settings only; the learned
        observation statistics are saved separately.
    """
    obs_processor = inference_policy.observation_processor
    action_processor = inference_policy.action_processor

    # Observation side: which features exist (spec) and how they are
    # normalized (config). None when there is no observation processor.
    observation = None
    if obs_processor is not None:
        observation = {"spec": obs_processor.spec.to_dict(), "config": obs_processor.config}

    # Action side: the action processor's bounds and clipping, or None.
    return config_to_jsonable({
        "observation": observation,
        "action": None if action_processor is None else action_processor.config,
    })


def _validate_policy_description(
    inference_policy: InferencePolicy,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Check that a policy manifest matches what the live policy does.

    The caller writes the manifest (``describe_policy``) and the trainer
    stores it in checkpoints, so a policy folder can later be exported from
    any checkpoint. If the manifest disagreed with what the policy actually
    does (say it lists ``normalize=["state"]`` but the processor normalizes
    nothing), every exported policy would be rebuilt wrongly. So: describe
    the live policy again and require an exact match.

    What this can check: everything visible on the live policy, such as
    the actor's features, the observation and action processing, the
    action selector and the default mode. What it cannot check: the model
    id and the model config. They are copied from the manifest into the
    new description, so they remain the caller's own declarations and
    must be right to begin with.

    Returns:
        The checked manifest.

    Raises:
        ValueError: If the manifest is invalid or does not match.
    """
    # 1. The manifest itself must be well-formed.
    manifest = validate_policy_manifest(manifest)

    # 2. Describe the live policy, using the manifest's own names and model
    #    config (those cannot be read from the policy object).
    actual = describe_policy(
        inference_policy,
        spec=DataSpec.from_dict(manifest["spec"]),
        model_id=manifest["model"]["id"],
        model_config=manifest["model"]["config"],
        decision_id=manifest["decision"]["id"],
        provenance=manifest["provenance"],
    )

    # 3. Both descriptions must be identical.
    if actual != manifest:
        raise ValueError("policy manifest does not match the inference configuration")
    return manifest


def _validate_resume(
    checkpoint: dict[str, Any],
    *,
    learner: OnPolicyLearner,
    learner_config: dict[str, Any],
    num_envs: int,
    preprocessing_config: dict[str, Any],
    policy_manifest: dict[str, Any] | None = None,
) -> tuple[TrainingProgress, dict[str, Any]]:
    """Check that this run can continue from the checkpoint, before loading anything.

    The fields below are compared first and the state is loaded later, so
    a checkpoint from a different setup fails with a clear message
    instead of half-loading. Only these supported compatibility fields
    are checked here; tensor contents are checked later, while loading,
    and errors there can still happen (see ``learner_state.py``).

    Returns:
        ``(progress, resume_overrides)``: the restored step counters, and the
        allowed settings that changed (for example a new learning rate).

    Raises:
        ValueError: If anything below does not match.
    """
    # 1. Same kind of checkpoint, same algorithm, same number of
    #    environments. N matters: the reward scaler has one slot per
    #    environment, and the counters below are in steps of N.
    validate_training_checkpoint(
        checkpoint,
        required_keys=("num_envs", "preprocessing_config"),
    )
    validate_checkpoint_algorithm(checkpoint, type(learner))
    if checkpoint["num_envs"] != num_envs:
        raise ValueError("checkpoint num_envs does not match the environment")

    # 2. Learner settings: only the fields the learner allows (such as the
    #    learning rate) may differ. The differences come back as overrides.
    resume_overrides = learner.validate_resume_config(
        config_to_jsonable(checkpoint["learner_config"]),
        learner_config,
    )

    # 3. The counters must fit this rollout size. On-policy training only
    #    ever completes whole rollouts, so after k updates exactly
    #    k * T * N steps were collected. Anything else means the checkpoint
    #    came from a different T or N.
    progress = TrainingProgress(**checkpoint["progress"])
    rollout_size = learner.cfg.rollout_steps * num_envs
    if progress.global_step != progress.update * rollout_size:
        raise ValueError("checkpoint progress is inconsistent with the rollout size")

    # 4. There must be work left. total_timesteps is the budget for the whole
    #    run, not extra steps, so to train longer, raise it.
    if learner.cfg.total_timesteps // rollout_size <= progress.update:
        raise ValueError("total_timesteps leaves no remaining complete rollout")

    # 5. The restored weights must see the same processing as before:
    #    observation statistics saved if and only if this run has a
    #    processor, the same reward scaling, and identical processing
    #    settings.
    if (learner.obs_processor is not None) != ("observation_processor" in checkpoint):
        raise ValueError("checkpoint observation processor does not match this run")
    learner.validate_reward_scaler_state(checkpoint["learner"])
    if config_to_jsonable(checkpoint["preprocessing_config"]) != preprocessing_config:
        raise ValueError("checkpoint preprocessing configuration does not match this run")

    # 6. Policy manifest: present in both or in neither, and identical except
    #    for provenance (provenance records this run's code, so it may change).
    saved_manifest = checkpoint.get("policy_manifest")
    if (saved_manifest is None) != (policy_manifest is None):
        raise ValueError("checkpoint policy manifest does not match this run")
    if saved_manifest is not None:
        saved_manifest = validate_policy_manifest(saved_manifest)
        current_manifest = _validate_policy_description(learner.inference_policy, policy_manifest)
        if any(
            saved_manifest[key] != current_manifest[key]
            for key in saved_manifest
            if key != "provenance"
        ):
            raise ValueError("checkpoint policy configuration does not match this run")
    return progress, resume_overrides


def _restore_training_state(
    checkpoint: dict[str, Any],
    *,
    learner: OnPolicyLearner,
    resume_overrides: dict[str, Any],
) -> None:
    """Load the saved learning state into the already-built learner.

    The random-number state is restored later, by the trainer itself, after
    the environments are reset (see step 3 of ``train_on_policy``).
    """
    # 1. Networks, optimizers and the reward scaler.
    learner.load_state_dict(checkpoint["learner"])

    # 2. Allowed setting changes, such as a new learning rate. This must come
    #    after step 1: loading the optimizer state would otherwise put the
    #    old learning rate back.
    learner.apply_resume_overrides(resume_overrides)

    # 3. The observation statistics (mean, var, count).
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
    """Collect everything needed to resume into one dictionary (no file is written).

    Saved: the learner's state and config, the counters, the random-number
    state, the run config and provenance, plus what the trainer owns: N, the
    processing settings, the observation statistics and the policy manifest.

    Not saved:

    * the environments' state: a resumed run starts new episodes;
    * the rollout buffer: checkpoints are only taken after a complete
      update, when the rollout has been used and is no longer needed.

    Returns:
        The checked checkpoint dictionary.
    """
    # 1. The common part: learner state, configs, counters, random numbers.
    checkpoint = build_training_checkpoint(
        algorithm=get_algorithm_id(type(learner)),
        learner_state=learner.state_dict(),
        learner_config=learner_config,
        run_config=run_config,
        progress=progress,
        rng_state=get_rng_state(device=learner.cfg.device),
        provenance=provenance,
    )

    # 2. The trainer's part: N, processing settings and statistics.
    checkpoint["num_envs"] = num_envs
    checkpoint["preprocessing_config"] = _preprocessing_config(learner.inference_policy)
    if learner.obs_processor is not None:
        checkpoint["observation_processor"] = learner.obs_processor.state_dict()

    # 3. The policy manifest, checked again against the live policy so the
    #    saved copy can never drift from what the policy really does.
    if policy_manifest is not None:
        checkpoint["policy_manifest"] = _validate_policy_description(
            learner.inference_policy, policy_manifest,
        )

    # 4. Check the whole dictionary before handing it over for saving.
    validate_training_checkpoint(
        checkpoint,
        required_keys=("num_envs", "preprocessing_config"),
    )
    return checkpoint


def train_on_policy(
    *,
    envs               : VectorEnv,
    learner            : OnPolicyLearner,
    writer             : SummaryWriter | None = None,
    logger             : Logger | None = None,
    evaluator          : Evaluator | None = None,
    checkpoint_dir     : str | Path | None = None,
    checkpoint_interval: int | None = None,
    run_config         : dict[str, Any] | None = None,
    resume_from        : str | Path | None = None,
    policy_manifest    : dict[str, Any] | None = None,
) -> MetricHistory:
    """Train an on-policy learner: collect a rollout, learn from it, repeat.

    Each update collects T steps from N environments (one rollout), lets the
    learner compute targets and update its networks, then logs, evaluates
    and checkpoints. ``total_timesteps // (T * N)`` updates run in total;
    leftover steps that do not fill a whole rollout are not collected.

    Observations and processing:
        The rollout stores raw observations. The learner's observation
        processor (shared by actor and critic) keeps its statistics frozen
        from collection through the last gradient step, and only then adds
        the rollout's observations (see step 5 for why). A new processor
        starts as identity; resetting the environments never fits or
        replaces statistics. Reward scaling changes only the rewards used
        for advantages, never the logged returns.

    Resuming (``resume_from``):
        Restores the learner (networks, optimizers, reward scaler), the
        observation statistics, the counters and PyTorch's random-number
        state, then starts fresh episodes. The simulator itself is not
        restored, so the run is not identical to an uninterrupted one. Use
        the same environment and networks. ``total_timesteps`` is the budget
        for the whole run. A changed actor or critic learning rate replaces
        the saved one without resetting the optimizer state; each learner
        lists which other settings may change (PPO also allows
        ``batch_size``, ``ent_coef`` and ``target_kl``). New checkpoints
        record the current settings. Incomplete checkpoints are rejected.

    Checkpoints:
        Taken only after a complete update, never in the middle of one.
        ``last.pt`` holds the latest saved completed update;
        ``step_<global_step>.pt`` files are kept every
        ``checkpoint_interval`` updates. An interrupted update is not saved.

    Args:
        envs: N vectorized environments returning tensors, created with
            ``AutoresetMode.DISABLED`` (for example by ``make_vec_env``).
        learner: The algorithm (``PPOLearner``, ``A2CLearner``, ...), already
            built with its networks and processors.
        writer: TensorBoard writer for the internal logger. Ignored when
            ``logger`` is given.
        logger: Your own ``Logger`` (for example with a metrics file or a
            progress bar). You keep ownership and close it yourself.
            ``None`` creates an internal logger, closed by this function.
        evaluator: Optional ``Evaluator`` for scheduled evaluations.
        checkpoint_dir: Folder for checkpoints. ``None`` saves nothing.
        checkpoint_interval: Keep a ``step_<global_step>.pt`` every this
            many updates (``last.pt`` is written at the same times). ``None``
            writes only ``last.pt``, at the end.
        run_config: Optional experiment settings (environment id, network
            sizes, ...) stored in checkpoints as a JSON-compatible
            dictionary. When resuming
            without one, the checkpoint's own is reused.
        resume_from: A checkpoint to continue from.
        policy_manifest: Optional ``describe_policy`` output. Stored in
            checkpoints so a policy folder can be exported from any of them
            (``export_policy_from_checkpoint``, which also needs the actor's
            weights under ``learner["policy"]``; on-policy learners save
            them there). Training-only callers can leave it out.

    Returns:
        The logger's in-memory metric history.

    Raises:
        ValueError: If the setup or the checkpoint does not fit this run.
    """
    # ------------------------------------------------------------------
    # 0. Validate the run
    # ------------------------------------------------------------------
    # Cheap checks first, before loading or resetting anything expensive.
    cfg = learner.cfg
    num_envs = envs.num_envs
    inference_policy = learner.inference_policy
    obs_processor = inference_policy.observation_processor
    _validate_setup(
        envs=envs,
        learner=learner,
        logger=logger,
        resume_from=resume_from,
        checkpoint_interval=checkpoint_interval,
    )

    # ------------------------------------------------------------------
    # 1. Describe this run, and check the checkpoint when resuming
    # ------------------------------------------------------------------
    # JSON-compatible descriptions of the learner settings and the processing.
    # They are saved in checkpoints and compared when resuming.
    learner_config = config_to_jsonable(cfg)
    preprocessing_config = _preprocessing_config(inference_policy)
    if policy_manifest is not None:
        policy_manifest = _validate_policy_description(inference_policy, policy_manifest)

    # A new run starts at zero. A resumed run loads the checkpoint (to the
    # CPU; each part moves to its own device when restored) and checks that
    # its structure, counters, settings and processing fit this run, but
    # restores nothing yet. Tensor contents are only checked later, while
    # loading.
    progress = TrainingProgress()
    checkpoint = None
    resume_overrides: dict[str, Any] = {}
    if resume_from is not None:
        checkpoint = load_checkpoint(resume_from, map_location="cpu")
        progress, resume_overrides = _validate_resume(
            checkpoint,
            learner=learner,
            learner_config=learner_config,
            num_envs=num_envs,
            preprocessing_config=preprocessing_config,
            policy_manifest=policy_manifest,
        )
        # Keep describing the same experiment if the caller gave no settings.
        if run_config is None:
            run_config = checkpoint["config"]
    if run_config is not None:
        run_config = config_to_jsonable(run_config)

    # Record which code produced the run (versions, git commit), once, and
    # only when checkpoints will be written.
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
    provenance = get_provenance() if checkpoint_dir is not None else None

    # ------------------------------------------------------------------
    # 2. Create the rollout storage, the schedule and the logger
    # ------------------------------------------------------------------
    # One buffer shaped (T, N, ...) is allocated once and refilled for every
    # rollout; reusing it avoids allocating memory every update.
    rollout_buffer = RolloutBuffer(
        rollout_steps=cfg.rollout_steps,
        num_envs=num_envs,
        device=cfg.device,
    )

    # Only whole rollouts are collected: this many updates fit the budget.
    rollout_size = cfg.rollout_steps * num_envs
    num_updates = cfg.total_timesteps // rollout_size

    # Use the caller's logger, or create one. A logger created here must be
    # closed here too, which is why the rest runs inside try/finally.
    owns_logger = logger is None
    if logger is None:
        logger = Logger(num_envs, cfg.log_every, writer, cfg.device)

    try:
        # --------------------------------------------------------------
        # 3. Restore the saved state and start fresh episodes
        # --------------------------------------------------------------
        # Seed PyTorch for action sampling and minibatch shuffling. (The
        # networks already exist; their initial weights were seeded by the
        # caller, before building them.)
        seed_torch(cfg.seed)

        # On resume: load the learner and the statistics, and let the logger
        # and the evaluator continue their schedules from the restored step.
        if checkpoint is not None:
            _restore_training_state(
                checkpoint,
                learner=learner,
                resume_overrides=resume_overrides,
            )
            logger.resume(progress=progress)
            if evaluator is not None:
                evaluator.resume(progress.global_step)
        logger.log_hyperparameters(cfg)

        # Every run, new or resumed, starts fresh episodes: the simulator's
        # state is never saved. Observations stay raw: (N, *obs_shape), or a
        # TensorDict with batch size N.
        observations, _ = envs.reset(seed=cfg.seed)
        observations = to_tensor_data(observations, batch_size=num_envs)

        # The reward scaler tracks each environment's running discounted
        # return. New episodes start that sum from zero; the learned spread
        # (its statistics) is kept.
        if learner.reward_scaler is not None:
            learner.reward_scaler.returns.zero_()

        # Restore the random-number state last. The setup above may draw
        # random numbers; restoring afterwards means PyTorch's default
        # generators (the CPU one, and the selected CUDA device's) continue
        # from where they were saved. Other generators (Python, NumPy, the
        # simulator) are not restored, so the resumed run is not guaranteed
        # to match an uninterrupted one.
        if checkpoint is not None:
            set_rng_state(checkpoint["rng_state"], device=cfg.device)

        # The checkpoint holds a full copy of the weights and optimizer
        # state; drop it to free the memory.
        checkpoint = None

        # A resumed run continues counting from its saved update number.
        for update in range(progress.update + 1, num_updates + 1):
            # ----------------------------------------------------------
            # 4. Play: collect one rollout with the current policy
            # ----------------------------------------------------------
            # Evaluation mode turns off training-only layer behaviour (such
            # as dropout), so the recorded actions, log-probabilities and
            # values come from the plain networks. Each learner sets the
            # modes it needs in update: PPO and A2C switch both networks to
            # training mode; NPG and TRPO keep the actor in evaluation mode
            # (gradients still flow in evaluation mode).
            learner.policy.eval()
            learner.value_fn.eval()

            # Fill the buffer with exactly T x N transitions. The returned
            # observations, (N, *obs_shape), are where the environments are
            # now: the next rollout continues from there, so episodes simply
            # run across rollout boundaries.
            observations = collect_rollout(
                envs=envs,
                policy=inference_policy,
                value_fn=learner.evaluate_values,
                rollout_buffer=rollout_buffer,
                observations=observations,
            )
            progress = replace(
                progress,
                global_step=progress.global_step + rollout_size,
            )

            # ----------------------------------------------------------
            # 5. Learn: score the rollout, then update the networks
            # ----------------------------------------------------------
            # Score every action once, up front, with the networks that
            # collected the rollout: advantages (how much better than
            # expected each action turned out) and value targets (what the
            # critic should have predicted). They stay fixed during the
            # update; recomputing them as the critic changes would make the
            # learner chase a moving target.
            advantages, value_targets = learner.prepare_targets(rollout_buffer)

            # The algorithm decides how to use the rollout: PPO runs several
            # epochs of minibatches, A2C takes one gradient step on the whole
            # rollout, NPG and TRPO attempt one natural-gradient policy step
            # per batch they draw from it (TRPO rejects a step that fails its
            # checks).
            training_stats = learner.update(
                buffer=rollout_buffer,
                advantages=advantages,
                value_targets=value_targets,
            )

            # Only now add this rollout to the observation statistics. During
            # the update, PPO compares the new policy with the log-
            # probabilities recorded during collection. Both must see the
            # same normalization, or the comparison is off before any
            # learning happens. So the statistics stay frozen from collection
            # through the last gradient step and are updated once, here.
            # Current observations (not next ones) are used, so each
            # observation counts once.
            if obs_processor is not None:
                obs_processor.update(rollout_buffer.observations)

            # ----------------------------------------------------------
            # 6. Record: log the rollout and the update, evaluate if due
            # ----------------------------------------------------------
            # Training episodes that ended in this rollout. Rewards and
            # episode ends are both (T, N); the rewards are what the
            # environment gave, before any reward scaling.
            logger.log_rollout(
                rewards=rollout_buffer.rewards,
                dones=rollout_buffer.terminations | rollout_buffer.truncations,
                progress=progress,
            )

            # Count the gradient steps and the finished update, then log the
            # learner's statistics with the updated counters.
            progress = replace(
                progress,
                gradient_step=progress.gradient_step + training_stats.gradient_steps,
                update=update,
            )
            logger.log_update(training_stats, progress=progress)

            # Write a report and run an evaluation, each only when due.
            logger.maybe_dump()
            if evaluator is not None:
                evaluator.maybe_evaluate(inference_policy, logger, progress=progress)

            # ----------------------------------------------------------
            # 7. Save: checkpoint after a complete update
            # ----------------------------------------------------------
            # Saving happens here, between updates, so a checkpoint never
            # holds a half-updated learner. last.pt holds the latest saved
            # completed update (resume from it); step_<global_step>.pt files
            # are kept every checkpoint_interval updates as history. The
            # final update always writes last.pt, so a finished run can be
            # resumed or exported.
            if checkpoint_dir is not None:
                save_due = (
                    checkpoint_interval is not None
                    and progress.update % checkpoint_interval == 0
                )
                if save_due or progress.update == num_updates:
                    checkpoint_state = _checkpoint_state(
                        learner=learner,
                        learner_config=learner_config,
                        progress=progress,
                        num_envs=num_envs,
                        run_config=run_config,
                        provenance=provenance,
                        policy_manifest=policy_manifest,
                    )
                    save_checkpoint(checkpoint_dir / "last.pt", checkpoint_state)
                    if save_due:
                        save_checkpoint(
                            checkpoint_dir / f"step_{progress.global_step}.pt",
                            checkpoint_state,
                        )

        # --------------------------------------------------------------
        # 8. Evaluate the final policy
        # --------------------------------------------------------------
        # Only after training finished successfully. Skipped if the last
        # scheduled evaluation already happened at this exact step.
        if evaluator is not None:
            evaluator.evaluate_now(inference_policy, logger, progress=progress)
    finally:
        # --------------------------------------------------------------
        # 9. Close the logger, even after an error
        # --------------------------------------------------------------
        # Close the logger this function created, even if training crashed
        # or was interrupted (Ctrl+C), so the metrics written so far are
        # flushed. A logger from the caller is left for the caller.
        if owns_logger:
            logger.finish()

    return logger.history
