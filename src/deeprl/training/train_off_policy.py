"""The training loop for off-policy algorithms (DQN, DDPG, TD3, SAC).

The idea
--------
Off-policy training repeats three things until the step budget is used up::

    1. play    step all N environments train_freq times with the behaviour
               policy (which explores), and store every transition in a
               large replay buffer
    2. learn   draw gradient_steps random minibatches from the replay buffer
               and update the learner on each one
    3. record  log, evaluate, save a checkpoint

Why keep old data? Off-policy methods learn a value function (Q) from the
Bellman equation, which holds for any transition, no matter which policy
collected it. So old experience stays useful, and reusing it many times
makes these methods sample-efficient. Drawing minibatches at random from a
large buffer also breaks the strong correlation between consecutive steps,
which helps keep the updates stable (it does not guarantee it). This is
the "experience replay" of DQN (Mnih et al., 2015).

Two policies
------------
* The **behaviour policy** acts while collecting data. It explores:
  epsilon-greedy for DQN, added noise for DDPG and TD3, sampling for SAC.
* The **inference policy** is the plain policy that is evaluated, saved
  and deployed. It skips the behaviour policy's extra exploration (no
  epsilon, no added noise). It is not necessarily deterministic: the
  evaluator asks for deterministic or stochastic actions, as configured.

Both use the same network and the same observation processor, so they see
inputs the same way. The caller builds both (on ``learner.cfg.device``);
the trainer never moves trained modules.

Who does what
-------------
* This trainer owns the **schedule**: when to collect, learn, log,
  evaluate and save, and the step counters (``TrainingProgress``).
* The learner owns the **math**: losses, reward transforms, target
  networks, delayed updates, optimizers.
* ``collect_transitions`` owns **stepping the environments**.
* The caller owns the environments, the networks, both policies and,
  optionally, the logger; this trainer never closes what it did not create.

The parallel file
-----------------
``train_on_policy.py`` has the same numbered steps 0-9. Where they differ::

    on-policy (train_on_policy.py)          off-policy (this file)
    --------------------------------------  ----------------------------------------
    RolloutBuffer, refilled every update     ReplayBuffer, keeps old data (FIFO)
    a cycle = one rollout of T x N steps     a cycle = train_freq vector steps,
      then one learner update                  then gradient_steps minibatch updates
    the learner owns the inference policy    the caller passes a behaviour policy
                                               (explores) and an inference policy
    observation statistics update AFTER      observation statistics update DURING
      the learner update                       collection, step by step
    resume continues whole rollouts          resume refills the empty replay first
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from gymnasium.vector import AutoresetMode, VectorEnv, VectorWrapper
from gymnasium.wrappers.vector import NumpyToTorch

from deeprl.algorithms.off_policy import OffPolicyLearner
from deeprl.buffers import ReplayBuffer
from deeprl.checkpoint import (
    build_training_checkpoint,
    get_algorithm_id,
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint_algorithm,
    validate_training_checkpoint,
)
from deeprl.collectors import collect_transitions
from deeprl.data_spec import DataSpec, TensorData
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
from deeprl.stats import TrainingProgress, TrainingStats

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

    from deeprl.evaluate import Evaluator


def _validate_setup(
    *,
    envs: VectorEnv,
    learner: OffPolicyLearner,
    logger: Logger | None,
    resume_from: str | Path | None,
    checkpoint_interval: int | None,
) -> None:
    """Fail fast on a bad setup, before anything is loaded or reset.

    A mistake caught here costs nothing. The same mistake found an hour into
    training costs an hour.

    Raises:
        ValueError: If a setting does not fit.
        TypeError: If the environments do not return tensors.
    """
    cfg = learner.cfg
    num_envs = envs.num_envs

    # 1. The environments must not reset themselves. collect_transitions
    #    stores the true final observation of each episode and then resets
    #    that environment itself; it relies on nothing resetting behind its
    #    back.
    if envs.metadata.get("autoreset_mode") is not AutoresetMode.DISABLED:
        raise ValueError("train_off_policy requires AutoresetMode.DISABLED")

    # 2. The environments must return tensors, which the replay buffer
    #    stores. Walk down the wrapper chain looking for NumpyToTorch.
    wrapped = envs
    while isinstance(wrapped, VectorWrapper) and not isinstance(wrapped, NumpyToTorch):
        wrapped = wrapped.env
    if not isinstance(wrapped, NumpyToTorch):
        raise TypeError("train_off_policy requires a NumpyToTorch vector wrapper")

    # 3. A logger from the caller must track the same N environments, and a
    #    new run needs a fresh one. A used logger would continue the step
    #    counts of some other run.
    if logger is not None:
        if logger.num_envs != num_envs:
            raise ValueError("logger num_envs must match the environment")
        if resume_from is None and (logger.global_step != 0 or logger.gradient_step != 0):
            raise ValueError("a new run requires a fresh logger")

    # 4. "Save every k cycles" only makes sense for a whole number k > 0.
    if checkpoint_interval is not None and (
        type(checkpoint_interval) is not int or checkpoint_interval <= 0
    ):
        raise ValueError("checkpoint_interval must be a positive integer or None")

    # 5. One vector step adds N transitions at once, so the replay buffer
    #    must hold at least N, and the budget must allow at least one step.
    if cfg.buffer_size < num_envs:
        raise ValueError("buffer_size must hold at least one vector step (num_envs transitions)")
    if cfg.total_timesteps < num_envs:
        raise ValueError("total_timesteps must allow at least one vector step")


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
    stores it in checkpoints next to the learner state. If the manifest
    disagreed with what the policy actually does (say it lists
    ``normalize=["state"]`` but the processor normalizes nothing), a
    policy rebuilt from it would be wrong. So: describe
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
    learner: OffPolicyLearner,
    learner_config: dict[str, Any],
    inference_policy: InferencePolicy,
    num_envs: int,
    preprocessing_config: dict[str, Any],
    policy_manifest: dict[str, Any] | None,
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
    #    environments (the counters below are in steps of N).
    validate_training_checkpoint(
        checkpoint,
        required_keys=("num_envs", "preprocessing_config"),
    )
    validate_checkpoint_algorithm(checkpoint, type(learner))
    if checkpoint["num_envs"] != num_envs:
        raise ValueError("checkpoint num_envs does not match the environment")

    # 2. Learner settings: only the fields the learner allows (such as the
    #    total budget or the devices) may differ. The differences come back
    #    as overrides.
    saved_config = config_to_jsonable(checkpoint["learner_config"])
    resume_overrides = learner.validate_resume_config(saved_config, learner_config)
    progress = TrainingProgress(**checkpoint["progress"])

    # 3. The counters must be possible. Unlike on-policy training, the last
    #    cycle of a run may be shorter than train_freq, so the step count is
    #    not simply update * train_freq * N. What must hold:
    #    - steps come in whole vector steps (a multiple of N);
    #    - at least one cycle finished (checkpoints are taken after cycles);
    #    - each cycle had at least one vector step, so there are no more
    #      cycles than vector steps;
    #    - the saved run did not go past its own (rounded-down) budget.
    saved_total = saved_config["total_timesteps"]
    if type(saved_total) is not int or saved_total <= 0:
        raise ValueError("checkpoint total_timesteps must be a positive integer")
    if (
        progress.global_step % num_envs != 0
        or progress.update == 0
        or progress.update > progress.global_step // num_envs
        or progress.global_step > saved_total // num_envs * num_envs
    ):
        raise ValueError("checkpoint progress is inconsistent with completed collection cycles")

    # 4. There must be work left. total_timesteps is the budget for the whole
    #    run, not extra steps, so to train longer, raise it.
    if learner.cfg.total_timesteps // num_envs * num_envs <= progress.global_step:
        raise ValueError("total_timesteps leaves no remaining complete vector step")

    # 5. The restored weights must see the same processing as before:
    #    observation statistics saved if and only if this run has a
    #    processor, and identical processing settings.
    if (inference_policy.observation_processor is not None) != ("observation_processor" in checkpoint):
        raise ValueError("checkpoint observation processor does not match this run")
    if config_to_jsonable(checkpoint["preprocessing_config"]) != preprocessing_config:
        raise ValueError("checkpoint preprocessing configuration does not match this run")

    # 6. Policy manifest: present in both or in neither, and identical except
    #    for provenance (provenance records this run's code, so it may
    #    change). The current manifest was already checked against the live
    #    policy in step 1 of train_off_policy.
    saved_manifest = checkpoint.get("policy_manifest")
    if (saved_manifest is None) != (policy_manifest is None):
        raise ValueError("checkpoint policy manifest does not match this run")
    if saved_manifest is not None:
        saved_manifest = validate_policy_manifest(saved_manifest)
        if any(
            saved_manifest[key] != policy_manifest[key]
            for key in saved_manifest if key != "provenance"
        ):
            raise ValueError("checkpoint policy configuration does not match this run")
    return progress, resume_overrides


def _restore_training_state(
    checkpoint: dict[str, Any],
    *,
    learner: OffPolicyLearner,
    inference_policy: InferencePolicy,
    resume_overrides: dict[str, Any],
) -> None:
    """Load the saved learning state into the already-built learner.

    The random-number state is restored later, by the trainer itself, after
    the environments are reset (see step 3 of ``train_off_policy``).
    """
    # 1. Networks, target networks, optimizers and any other learner state.
    learner.load_state_dict(checkpoint["learner"])

    # 2. Allowed setting changes. This must come after step 1, so loading
    #    the saved state cannot put the old values back.
    learner.apply_resume_overrides(resume_overrides)

    # 3. The observation statistics (mean, var, count), shared by both
    #    policies and the learner.
    obs_processor = inference_policy.observation_processor
    if obs_processor is not None:
        obs_processor.load_state_dict(checkpoint["observation_processor"])


def _checkpoint_state(
    *,
    learner: OffPolicyLearner,
    learner_config: dict[str, Any],
    inference_policy: InferencePolicy,
    progress: TrainingProgress,
    num_envs: int,
    run_config: dict[str, Any] | None,
    provenance: dict[str, Any] | None,
    policy_manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    """Collect everything needed to resume into one dictionary (no file is written).

    Saved: the learner's state and config, the counters, the random-number
    state, the run config and provenance, plus what the trainer owns: N, the
    processing settings, the observation statistics and the policy manifest.

    Not saved:

    * the replay buffer: it can be gigabytes (a million transitions), so a
      resumed run refills it instead (see ``learning_starts``);
    * the environments' state: a resumed run starts new episodes.

    Returns:
        The checked checkpoint dictionary.
    """
    # 1. The common part: learner state, configs, counters, random numbers.
    checkpoint = build_training_checkpoint(
        algorithm=get_algorithm_id(type(learner)),
        learner_state=learner.state_dict(),
        learner_config=learner_config,
        progress=progress,
        rng_state=get_rng_state(device=learner.cfg.device),
        run_config=run_config,
        provenance=provenance,
    )

    # 2. The trainer's part: N, processing settings and statistics.
    checkpoint["num_envs"] = num_envs
    checkpoint["preprocessing_config"] = _preprocessing_config(inference_policy)
    obs_processor = inference_policy.observation_processor
    if obs_processor is not None:
        checkpoint["observation_processor"] = obs_processor.state_dict()

    # 3. The policy manifest, checked again against the live policy so the
    #    saved copy can never drift from what the policy really does.
    if policy_manifest is not None:
        checkpoint["policy_manifest"] = _validate_policy_description(
            inference_policy, policy_manifest,
        )

    # 4. Check the whole dictionary before handing it over for saving.
    validate_training_checkpoint(
        checkpoint,
        required_keys=("num_envs", "preprocessing_config"),
    )
    return checkpoint


def train_off_policy(
    *,
    envs               : VectorEnv,
    learner            : OffPolicyLearner,
    behaviour_policy   : Callable[[TensorData, int], TensorData],
    inference_policy   : InferencePolicy,
    writer             : SummaryWriter | None = None,
    logger             : Logger | None = None,
    evaluator          : Evaluator | None = None,
    checkpoint_dir     : str | Path | None = None,
    checkpoint_interval: int | None = None,
    run_config         : dict[str, Any] | None = None,
    resume_from        : str | Path | None = None,
    policy_manifest    : dict[str, Any] | None = None,
) -> MetricHistory:
    """Train an off-policy learner: collect a little, learn from replay, repeat.

    One cycle collects ``train_freq`` vector steps (N transitions each) into
    the replay buffer, then runs ``gradient_steps`` minibatch updates. The
    two together set the replay ratio: ``gradient_steps / (train_freq * N)``
    minibatch updates per collected transition. That counts update calls.
    Each call draws ``batch_size`` transitions at random, so the number of
    sampled transitions per collected one is ``batch_size`` times larger,
    and individual transitions are drawn different numbers of times (some
    often, some never). Learning only starts after ``learning_starts`` fresh
    transitions have been collected by this call.

    The budget is rounded down to whole vector steps (not whole cycles): the
    last cycle may be shorter than ``train_freq`` and still runs its
    updates. Minibatches are drawn uniformly with replacement, so
    ``batch_size`` may even be larger than the number of stored transitions.

    The behaviour policy:
        Called as ``behaviour_policy(observations, global_step)`` with raw
        observations and the number of transitions collected so far (for
        schedules such as a decaying epsilon). It does all exploration,
        including purely random actions during warm-up if wanted. It returns
        actions as the learner and replay use them (for example in
        [-1, 1]); only the command sent to the environment goes through
        ``inference_policy.action_processor``.

    Observations and processing:
        Replay stores raw observations. The shared observation processor
        adds each observation that was used to choose a training action to
        its statistics, once, right after that step (see step 4 for why
        this is fine here). Final observations of episodes, and the last
        observation of the run, are never acted on, so they are not
        counted. The
        statistics stay fixed during each learning phase, so the learner
        processes current and next observations with the same snapshot.
        Losses, reward transforms, delayed updates and target networks
        belong to the learner. Work tied to environment steps goes in
        ``learner.on_environment_step``.

    Resuming (``resume_from``) is a warm restart:
        Restored: the learner (networks, target networks, optimizers), the
        observation statistics, the counters and PyTorch's random-number
        state. Not restored: the replay buffer, which starts empty and is
        refilled with ``learning_starts`` fresh transitions before learning
        continues, and the episodes, which start fresh. Schedules that use
        the step count (exploration, target updates) continue where they
        were; epsilon does not jump back to its start value.
        ``total_timesteps`` is the budget for the whole run, not extra
        steps. Private random generators or exploration state must be saved
        by their owner in the learner state. The run is not identical to an
        uninterrupted one.

    Checkpoints:
        Taken only after a complete cycle (warm-up cycles count too), never
        in the middle of one. ``last.pt`` holds the latest saved completed
        cycle;
        ``step_<global_step>.pt`` files are kept every
        ``checkpoint_interval`` cycles; the final cycle always writes
        ``last.pt``. An interrupted cycle is not saved.

    Args:
        envs: N vectorized environments wrapped in ``NumpyToTorch`` and
            created with ``AutoresetMode.DISABLED`` (for example by
            ``make_vec_env``).
        learner: The algorithm, already built with its networks.
        behaviour_policy: The exploring policy used for collection (see
            above).
        inference_policy: The plain policy used for evaluation and saving.
            Its observation processor is the shared one, and its action
            processor converts actions into environment commands.
        writer: TensorBoard writer for the internal logger. Ignored when
            ``logger`` is given.
        logger: Your own ``Logger``. You keep ownership and close it
            yourself. ``None`` creates an internal logger, closed by this
            function.
        evaluator: Optional ``Evaluator`` for scheduled evaluations.
        checkpoint_dir: Folder for checkpoints. ``None`` saves nothing.
        checkpoint_interval: Keep a ``step_<global_step>.pt`` every this
            many cycles (``last.pt`` is written at the same times). ``None``
            writes only ``last.pt``, at the end.
        run_config: Optional experiment settings stored in checkpoints as a
            JSON-compatible dictionary. When resuming without one, the checkpoint's own is
            reused.
        resume_from: A checkpoint to continue from.
        policy_manifest: Optional ``describe_policy`` output, stored in
            checkpoints. It describes the inference policy; it does not
            dictate the names of the networks in the learner's state.
            Note: ``export_policy_from_checkpoint`` currently also needs
            the actor's weights under ``learner["policy"]``, which
            off-policy learners are not required to provide, so not every
            off-policy checkpoint can be exported yet.

    Returns:
        The logger's in-memory metric history.

    Raises:
        ValueError: If the setup or the checkpoint does not fit this run, or
            the learner reports an invalid gradient step count.
        TypeError: If the environments do not return tensors.
    """
    # ------------------------------------------------------------------
    # 0. Validate the run
    # ------------------------------------------------------------------
    # Cheap checks first, before loading or resetting anything expensive.
    cfg = learner.cfg
    num_envs = envs.num_envs
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
            inference_policy=inference_policy,
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
    # 2. Create the replay buffer, the schedule and the logger
    # ------------------------------------------------------------------
    # One flat buffer of transitions, (capacity, ...). It is stored on
    # replay_device (often the CPU, to save GPU memory) and each sampled
    # minibatch is moved to cfg.device. A resumed run starts it empty.
    replay_buffer = ReplayBuffer(
        capacity=cfg.buffer_size,
        storage_device=cfg.replay_device,
        sample_device=cfg.device,
    )

    # Each vector step adds N transitions, so the budget is rounded down to
    # whole vector steps.
    total_timesteps = cfg.total_timesteps // num_envs * num_envs

    # Learning waits for learning_starts fresh transitions from this call.
    # Why: minibatches drawn from a nearly empty buffer would train on the
    # same few transitions again and again. On resume the buffer is empty
    # again, so it is counted from the restored step, not from zero.
    learning_start_step = progress.global_step + cfg.learning_starts

    # Use the caller's logger, or create one. A logger created here must be
    # closed here too, which is why the rest runs inside try/finally.
    owns_logger = logger is None
    if logger is None:
        logger = Logger(num_envs, cfg.log_every, writer, cfg.device)

    try:
        # --------------------------------------------------------------
        # 3. Restore the saved state and start fresh episodes
        # --------------------------------------------------------------
        # Seed PyTorch for exploration and minibatch sampling. (The networks
        # already exist; their initial weights were seeded by the caller,
        # before building them.)
        seed_torch(cfg.seed)

        # On resume: load the learner and the statistics, and let the logger
        # and the evaluator continue their schedules from the restored step.
        if checkpoint is not None:
            _restore_training_state(
                checkpoint,
                learner=learner,
                inference_policy=inference_policy,
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

        # Cycles continue until the (rounded-down) budget is used up.
        while progress.global_step < total_timesteps:
            # ----------------------------------------------------------
            # 4. Play: collect up to train_freq vector steps into replay
            # ----------------------------------------------------------
            # Usually train_freq steps; fewer in the last cycle, so the
            # budget is never exceeded.
            num_steps = min(
                cfg.train_freq,
                (total_timesteps - progress.global_step) // num_envs,
            )

            # Evaluation mode turns off training-only layer behaviour (such
            # as dropout) while acting. The learner sets the modes it needs
            # during its updates.
            inference_policy.eval()

            # One vector step at a time, so that after every step the
            # trainer can update the statistics, call the learner's per-step
            # hook and log.
            for _ in range(num_steps):
                # Copy the current observations before stepping. They are
                # needed for the statistics after the step, but some vector
                # environments reuse their output memory, so the step could
                # overwrite them. Only needed when there are statistics.
                if obs_processor is not None:
                    fresh_observations = (
                        observations.clone() if isinstance(observations, torch.Tensor)
                        else observations.apply(torch.clone)
                    )

                # Act with the behaviour policy (it explores), step all N
                # environments, store N transitions in replay, and reset the
                # finished episodes. global_step tells exploration schedules
                # how far training is. Returns the next observations, and
                # rewards and episode ends shaped (1, N).
                observations, rewards, dones = collect_transitions(
                    envs=envs,
                    behaviour_policy=behaviour_policy,
                    replay_buffer=replay_buffer,
                    observations=observations,
                    num_steps=1,
                    global_step=progress.global_step,
                    action_processor=inference_policy.action_processor,
                )
                progress = replace(progress, global_step=progress.global_step + num_envs)

                # Add the observations just used to choose actions to the
                # statistics, once each. (Final observations of episodes are
                # never acted on, so they are not counted.) Updating during
                # collection is fine here, unlike in on-policy training:
                # replay keeps raw observations and no log-probabilities, and
                # the learner normalizes every sampled minibatch with the
                # current statistics at update time, so there is no "old
                # versus new normalization" to keep apart.
                if obs_processor is not None:
                    obs_processor.update(fresh_observations)

                # Work that runs on an environment-step clock, for example
                # DQN copying its network into the target network every C
                # environment steps. It is bookkeeping, not learning, so it
                # runs without gradients.
                with torch.no_grad():
                    learner.on_environment_step(progress=progress)

                # Training episodes that ended in this step. [0] removes the
                # time axis of size 1, leaving shape (N,).
                logger.log_step(rewards[0], dones[0], progress=progress)

            # ----------------------------------------------------------
            # 5. Learn: update the learner on random replay minibatches
            # ----------------------------------------------------------
            # Nothing is learned during warm-up (see learning_start_step).
            # After that, gradient_steps minibatches per cycle. The learner
            # owns everything inside an update: targets, losses, optimizer
            # steps, delayed actor updates, soft target-network updates.
            if progress.global_step >= learning_start_step:
                for _ in range(cfg.gradient_steps):
                    # batch_size random transitions; every field has leading
                    # dimension B = cfg.batch_size.
                    batch = replay_buffer.sample(cfg.batch_size)
                    training_stats = learner.update(batch, progress=progress)

                    # One minibatch is one update: 1 if the learner used it,
                    # 0 if it skipped it. This keeps gradient_step meaning
                    # the same thing for every algorithm.
                    if (
                        type(training_stats.gradient_steps) is not int
                        or training_stats.gradient_steps not in (0, 1)
                    ):
                        raise ValueError("one replay minibatch must report gradient_steps as 0 or 1")
                    progress = replace(
                        progress,
                        gradient_step=progress.gradient_step + training_stats.gradient_steps,
                    )
                    logger.log_update(training_stats, progress=progress)

            # ----------------------------------------------------------
            # 6. Record: count the cycle, log, evaluate if due
            # ----------------------------------------------------------
            # One more completed cycle (collection plus learning).
            progress = replace(progress, update=progress.update + 1)

            # Pass the new counters to the logger even when warm-up skipped
            # all updates. The empty statistics add no metrics.
            logger.log_update(TrainingStats({}, 0), progress=progress)

            # Write a report and run an evaluation, each only when due. The
            # evaluation uses the inference policy, without the behaviour
            # policy's exploration.
            logger.maybe_dump()
            if evaluator is not None:
                evaluator.maybe_evaluate(inference_policy, logger, progress=progress)

            # ----------------------------------------------------------
            # 7. Save: checkpoint after a complete cycle
            # ----------------------------------------------------------
            # Saving happens here, between cycles, so a checkpoint never
            # holds a half-finished cycle. The interval counts cycles,
            # warm-up cycles included. last.pt holds the latest saved
            # completed cycle (resume from it); step_<global_step>.pt files
            # are kept as history. The final cycle always writes last.pt.
            # The replay buffer is not saved (see _checkpoint_state).
            if checkpoint_dir is not None:
                save_due = (
                    checkpoint_interval is not None
                    and progress.update % checkpoint_interval == 0
                )
                if save_due or progress.global_step == total_timesteps:
                    checkpoint_state = _checkpoint_state(
                        learner=learner,
                        learner_config=learner_config,
                        inference_policy=inference_policy,
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
