"""Collect experience from vectorized environments into training buffers.

The idea
--------
The collectors are the only code that talks to the environments during
training. Each step they ask the policy for one action per environment, step
all N environments together, and write the result into a buffer as
transitions::

    (S_t, A_t, R_{t+1}, S_{t+1}, terminated, truncated)

The learners never see the environments. They read these transitions from the
buffer, so everything they compute (values, GAE, Bellman targets) depends on
the collector storing the right ``S_{t+1}``.

Two collectors
--------------
* ``collect_rollout`` (on-policy: PPO, A2C, NPG, TRPO) fills a
  ``RolloutBuffer`` with exactly T steps from N environments, together with
  the value estimate and log-probability of every action. The buffer is
  emptied and refilled for every rollout.
* ``collect_transitions`` (off-policy: DQN, DDPG, TD3, SAC) adds T steps from N
  environments to a ``ReplayBuffer``. Transitions already there are kept until
  the buffer is full; after that, each new transition overwrites the oldest
  one (first in, first out).

Episode ends: the one rule that matters
---------------------------------------
When an episode ends, the stored ``S_{t+1}`` must be the state the episode
**actually ended in**, not the first state of the next episode. Why:

* after a **truncation** (a time limit) the episode was cut off, not finished,
  so the targets bootstrap from it: ``R_{t+1} + gamma * V(S_{t+1})``;
* after a **termination** (a real end, such as a fall) the flag switches
  bootstrapping off, so ``S_{t+1}`` is not used.

Storing the next episode's start instead would make every time-limit target
use the value of a state from a different episode.

Environments differ in *when* they restart a finished episode. Gymnasium calls
this the autoreset mode. Two modes are supported, and both store exactly the
same transitions::

    DISABLED   step() returns the true last state, and the environment waits.
               The collector stores it, then restarts only the finished rows
               with reset(options={"reset_mask": dones}).
               Used by: Gymnasium environments made with make_vec_env.

    SAME_STEP  step() has already restarted the finished rows, so its
               observations for them are fresh starts. The true last states
               are in info["final_obs"]. The collector stores those, and acts
               next on the fresh starts; it never calls reset().
               Used by: batched GPU simulators such as Isaac Lab
               (with env_cfg.compute_final_obs=True).

``NEXT_STEP``, Gymnasium's default, is rejected: the step after an episode end
ignores its action and returns the reset observation, which would be stored
as a fake transition.

Which environments can be used
------------------------------
Any object that follows the Gymnasium vector API and provides ``metadata``
(with ``"autoreset_mode"``), ``num_envs``, ``single_action_space``, ``reset()``
and ``step()``, returning torch tensors. It does not have to inherit
``gymnasium.vector.VectorEnv``: Isaac Lab's environment, for example, is a
batched ``gym.Env``.

Other rules
-----------
* **Data is stored raw.** Observations and rewards go into the buffer exactly
  as the environment returned them. Normalization happens later, when a
  network reads them.
* **The sampled action is stored, the converted command is sent.** If the
  policy has an action processor (for example rescaling [-1, 1] to the joint
  limits), only the command sent to the environment is converted. The buffer
  keeps the original sample, which training needs.
* **Copy before keeping.** Some environments reuse the same memory for every
  step's output, so the collectors copy the observations they will store, and
  the buffers copy everything they receive.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch
import torch.nn as nn
from gymnasium.vector import AsyncVectorEnv, AutoresetMode, SyncVectorEnv, VectorEnv
from tensordict import TensorDictBase
from torch.distributions import Distribution

from deeprl.buffers import ReplayBuffer, RolloutBuffer, validate_batch
from deeprl.data_spec import TensorData
from deeprl.env import from_tensor_data, to_tensor_data
from deeprl.inference import InferencePolicy
from deeprl.processors import ActionProcessor


def supported_autoreset_mode(envs: VectorEnv, *, caller: str) -> AutoresetMode:
    """Return the environments' reset mode, if the collectors support it.

    The collectors handle episode ends differently in the two supported modes
    (see the module docstring), so they need to know which one the
    environments use. The trainers call this in their setup checks as well,
    so a wrong mode fails before anything is reset.

    Args:
        envs: The vector environments. Their ``metadata["autoreset_mode"]``
            says when finished episodes are restarted.
        caller: Name used in error messages, for example
            ``"train_on_policy"``.

    Returns:
        ``AutoresetMode.DISABLED`` or ``AutoresetMode.SAME_STEP``.

    Raises:
        ValueError: For ``NEXT_STEP``, a missing or unknown mode, or one of
            Gymnasium's own ``SyncVectorEnv``/``AsyncVectorEnv`` running in
            ``SAME_STEP`` mode. Those store each final observation as a
            separate Python object, which ``NumpyToTorch`` cannot convert;
            create them with ``AutoresetMode.DISABLED`` (``make_vec_env``
            does this).
    """
    mode = envs.metadata.get("autoreset_mode")

    # 1. DISABLED: the collector restarts finished rows itself, after storing
    #    their true last states. Nothing more to check.
    if mode is AutoresetMode.DISABLED:
        return mode

    # 2. SAME_STEP: accepted for environments that report a full batch of
    #    final observations (Isaac Lab, torch-native vector envs). Gymnasium's
    #    own vector envs in this mode report them one object per environment
    #    instead, which cannot become tensors, so they are refused. Checking
    #    the base environment's own mode (not just the metadata) still accepts
    #    a wrapper that implements SAME_STEP on top of a DISABLED base.
    if mode is AutoresetMode.SAME_STEP:
        if isinstance(envs, VectorEnv):
            base = envs.unwrapped
            if (
                isinstance(base, (SyncVectorEnv, AsyncVectorEnv))
                and base.autoreset_mode is AutoresetMode.SAME_STEP
            ):
                raise ValueError(
                    f"{caller} requires AutoresetMode.DISABLED for Gymnasium "
                    "SyncVectorEnv/AsyncVectorEnv; their SAME_STEP final observations "
                    "cannot pass through NumpyToTorch"
                )
        return mode

    # 3. Anything else (NEXT_STEP, a missing key, a typo) is refused.
    raise ValueError(
        f"{caller} requires AutoresetMode.DISABLED or AutoresetMode.SAME_STEP, "
        f"got {mode!r}. NEXT_STEP is not supported: its next step ignores its action "
        "and returns a reset observation, creating a false transition."
    )


def _require_tensor_outputs(
    rewards: object,
    terminations: object,
    truncations: object,
    *,
    num_envs: int,
) -> None:
    """Check the step outputs the collectors use before anything is stored.

    The episode-end mask ``dones = terminations | truncations`` decides which
    rows get their final observations and which rows are restarted, and it is
    computed before the buffer runs its own checks. So check here that:

    * rewards and both flags are torch tensors (Gymnasium environments return
      NumPy arrays unless they are wrapped in ``NumpyToTorch``);
    * both flags are booleans with one value per environment, shape ``(N,)``.

    Args:
        rewards: Rewards returned by ``step()``.
        terminations: Termination flags returned by ``step()``.
        truncations: Truncation flags returned by ``step()``.
        num_envs: N, the number of environments.

    Raises:
        TypeError: If an output is not a tensor.
        ValueError: If a flag is not boolean or not shaped ``(N,)``.
    """
    # 1. Tensors only: the buffers and the mask arithmetic below need them,
    #    and the message says how to fix the most common cause.
    for name, value in (
        ("rewards", rewards),
        ("terminations", terminations),
        ("truncations", truncations),
    ):
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"environment step() must return torch tensors ({name}); "
                "wrap Gymnasium vector envs in NumpyToTorch (make_vec_env does this)"
            )

    # 2. The flags must be booleans with one value per environment. Why:
    #    "not terminated" (~terminations) in the learners only means that for
    #    booleans, and the mask must line up with the N rows it selects.
    for name, flags in (("terminations", terminations), ("truncations", truncations)):
        if flags.dtype != torch.bool:
            raise ValueError(f"{name} must have bool dtype, got {flags.dtype}")
        if flags.shape != (num_envs,):
            raise ValueError(f"{name} must have shape ({num_envs},), got {tuple(flags.shape)}")


def _final_observations(
    next_observations: TensorData,
    infos: object,
    dones: torch.Tensor,
    num_envs: int,
) -> TensorData:
    """Put the true last states back into the rows the environment restarted.

    Used only for ``SAME_STEP`` environments, and only on steps where an
    episode ended. The environment has already restarted those rows, so
    ``next_observations`` holds fresh starts there. The transition needs the
    state the old episode actually ended in: after a time limit, the targets
    bootstrap from it. The environment saved those states in
    ``info["final_obs"]`` before restarting.

    Rows that did not end keep their ``next_observations`` unchanged. Neither
    input is modified (the simulator may reuse its buffers, and
    ``next_observations`` is also what the policy acts on next); a new tensor
    or TensorDict is returned.

    Args:
        next_observations: What ``step()`` returned, converted to a tensor or
            TensorDict with batch size N. Fresh starts in the ended rows.
        infos: The ``info`` dictionary returned by ``step()``.
        dones: Episode-end flags (terminated or truncated), boolean, shaped
            ``(N,)``.
        num_envs: N, the number of environments.

    Returns:
        The true next observations of this step, with the same structure,
        shapes and dtypes as ``next_observations``.

    Raises:
        ValueError: If ``info["final_obs"]`` is missing (Isaac Lab: set
            ``env_cfg.compute_final_obs=True``), is Gymnasium's
            per-environment object array, cannot be converted, or does not
            match the observations' structure, shapes and dtypes.
    """
    # 1. The final observations must be there. Never fall back to the fresh
    #    starts: that would silently bootstrap time limits from the wrong
    #    episode.
    if not isinstance(infos, Mapping) or infos.get("final_obs") is None:
        raise ValueError(
            "SAME_STEP episode ended without info['final_obs']; "
            "for Isaac Lab set env_cfg.compute_final_obs=True"
        )

    # 2. Gymnasium's own SAME_STEP format (one Python object per environment)
    #    is not supported. This test needs no NumPy import: NumPy object
    #    arrays have dtype kind "O", and torch dtypes have no kind at all.
    final = infos["final_obs"]
    if getattr(getattr(final, "dtype", None), "kind", None) == "O":
        raise ValueError(
            "Gymnasium object-array final_obs is unsupported; "
            "use AutoresetMode.DISABLED for Gymnasium vector envs"
        )
    # 3. Convert to the same form as the observations: a tensor, or a
    #    TensorDict for dict observations such as Isaac Lab's groups.
    try:
        final = to_tensor_data(final, batch_size=num_envs)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("final_obs could not be converted to tensor data") from exc

    # 4. Check the structure with the buffer's own schema check (batch size,
    #    keys, shapes, dtypes). Why here, before the buffer: torch.where would
    #    silently broadcast a wrongly shaped tensor, and a TensorDict missing
    #    a key would silently keep the fresh value for that key.
    validate_batch(
        final, batch_size=num_envs, name="final_obs", reference=next_observations
    )

    # 5. Take the ended rows from the final observations and every other row
    #    from next_observations. The mask is moved to the observations'
    #    device first, because simulators may keep their flags on the CPU
    #    while the observations live on the GPU.
    if isinstance(next_observations, TensorDictBase):
        result = next_observations.clone()
        result[dones] = final[dones]
        return result
    mask = dones.to(next_observations.device).reshape(
        num_envs, *([1] * (next_observations.ndim - 1))
    )
    return torch.where(mask, final, next_observations)


@torch.no_grad()
def collect_rollout(
    envs            :   VectorEnv,
    policy          :   nn.Module,
    value_fn        :   Callable[[TensorData], torch.Tensor],
    rollout_buffer  :   RolloutBuffer,
    observations    :   TensorData | dict | tuple,
) -> TensorData:
    """Collect one T-step rollout from N vectorized environments.

    Each step: evaluate the current observations, sample one action per
    environment, step all environments, store the transition, and continue
    from where each environment is now. Episodes that ended are restarted by
    the collector (``DISABLED``) or were already restarted by the environment
    (``SAME_STEP``); either way the stored next observation is the state the
    episode ended in (see the module docstring). When this returns, the
    buffer is full.

    What is stored:

    * observations and rewards exactly as the environment returned them;
    * the sampled actions and their log-probabilities (needed for PPO's ratio);
    * the value estimate of each observation;
    * the true next observation, including the final one of an episode;
    * separate termination and truncation flags.

    Args:
        envs: N batched environments returning torch tensors, in
            ``DISABLED`` or ``SAME_STEP`` reset mode. For Gymnasium tasks use
            ``make_vec_env``. Any object with ``metadata``, ``num_envs``,
            ``single_action_space``, ``reset()`` and ``step()`` works; it
            does not have to inherit ``VectorEnv``.
        policy: Returns one action distribution per environment. An
            ``InferencePolicy`` also applies its observation processor, and
            its action processor converts the commands sent to the environment.
        value_fn: Maps raw observations to value estimates shaped ``(N,)``.
        rollout_buffer: Storage for T steps from N environments. It is
            emptied and refilled.
        observations: Current observations of all N environments, usually
            where the previous rollout ended.

    Returns:
        The observations to start the next rollout from, shaped
        ``(N, *obs_shape)``. Dict and tuple observations are returned as
        nested TensorDicts with batch size ``(N,)``. Environments whose
        episode ended are already at their next episode's first state.

    Raises:
        ValueError: If the reset mode is not supported, a step returns
            malformed flags, or a ``SAME_STEP`` environment ends an episode
            without valid ``info["final_obs"]``.
        TypeError: If a step does not return torch tensors.
    """

    # --------------------------------------------------------------
    # 0. Setup
    # --------------------------------------------------------------

    # Restart writing at the first row of the preallocated rollout storage.
    rollout_buffer.reset()

    # Read the reset mode once. It decides two things in every step below:
    # where the true final observation comes from, and whether this
    # collector restarts finished episodes itself.
    same_step = (
        supported_autoreset_mode(envs, caller="collect_rollout")
        is AutoresetMode.SAME_STEP
    )
    observations = to_tensor_data(observations, batch_size=envs.num_envs)

    for _ in range(rollout_buffer.rollout_steps):

        # Copy the current observations. Some environments reuse the same
        # memory for every step's output; without a copy, the observation
        # stored below would silently change on the next step.
        # Leafwise cloning supports unsigned depth tensors in structured states.
        observations = (
            observations.apply(torch.clone)
            if isinstance(observations, TensorDictBase)
            else observations.clone()
        )

        # --------------------------------------------------------------
        # 1. Current-state evaluation
        # --------------------------------------------------------------

        # Value estimate of every current observation. Shape: (N,).
        values = value_fn(observations)

        # One batched action distribution for all N environments.
        action_distributions: Distribution = policy(observations)

        # Sample one action per environment.
        # Shape: (N,) for discrete actions, (N, *action_shape) otherwise.
        actions = to_tensor_data(
            action_distributions.sample(),
            batch_size=envs.num_envs,
        )

        # Log-probability of each sampled action under the current policy,
        # log pi(A|S). PPO compares it with the updated policy later. Shape: (N,).
        # Structured distributions consume TensorDict actions here and during updates.
        log_probs: torch.Tensor = action_distributions.log_prob(actions)

        # --------------------------------------------------------------
        # 2. Environment transition
        # --------------------------------------------------------------

        # Convert the sampled actions into environment commands, if the policy
        # has an action processor. The sampled actions stay unchanged for storage.
        commands = actions
        if isinstance(policy, InferencePolicy) and policy.action_processor is not None:
            commands = policy.action_processor(actions)

        # Step every environment once.
        #   next_observations: (N, *obs_shape)
        #   rewards, terminations, truncations: (N,)
        # In DISABLED mode next_observations holds the true next state of
        # every row. In SAME_STEP mode the rows whose episode just ended
        # already hold a fresh start, and their true last states are in
        # infos["final_obs"].
        env_actions = from_tensor_data(commands, envs.single_action_space)
        next_observations, rewards, terminations, truncations, infos = envs.step(env_actions)

        # Check the outputs before anything uses them (tensors; boolean flags
        # shaped (N,)).
        _require_tensor_outputs(
            rewards, terminations, truncations,
            num_envs=envs.num_envs,
        )
        next_observations = to_tensor_data(
            next_observations,
            batch_size=envs.num_envs,
        )

        # An episode ends with either a termination (a real end, such as a
        # fall) or a truncation (a time limit). Shape: (N,), dtype: bool.
        # any_done is read once: on a GPU simulator it costs a GPU-to-CPU
        # synchronization.
        dones = torch.logical_or(terminations, truncations)
        any_done = bool(dones.any().item())

        # The next state to store. In SAME_STEP mode, put the true last
        # states back into the rows the environment already restarted.
        transition_next = next_observations
        if same_step and any_done:
            transition_next = _final_observations(
                next_observations, infos, dones, envs.num_envs
            )

        # --------------------------------------------------------------
        # 3. Transition storage
        # --------------------------------------------------------------

        # Store (S_t, A_t, R_{t+1}, S_{t+1}, terminated, truncated) with the
        # true S_{t+1}, in both reset modes. The buffer copies the data into
        # its own time-major storage, row by row.
        rollout_buffer.add(
            observation=observations,
            action=actions,
            reward=rewards,
            next_observation=transition_next,
            termination=terminations,
            truncation=truncations,
            value=values,
            log_prob=log_probs,
        )

        # --------------------------------------------------------------
        # 4. Next observations for acting
        # --------------------------------------------------------------

        # What the policy acts on next:
        # * SAME_STEP: exactly what step() returned. The finished rows are
        #   already fresh starts; never call reset() here (in Isaac Lab, for
        #   example, it would restart every environment).
        # * DISABLED, some episode ended: restart only the finished rows,
        #   now that their true last states are stored.
        # * DISABLED, no episode ended: continue from the next observations.
        if same_step:
            observations = next_observations

        elif any_done:

            # Reset only the finished environments. The returned batch holds
            # fresh observations for the reset environments and the unchanged
            # next observations for all others.
            observations, _ = envs.reset(options={"reset_mask": dones})
            observations = to_tensor_data(
                observations,
                batch_size=envs.num_envs,
            )

        else:

            # No episode ended: every environment continues from its next
            # observation.
            observations = next_observations

    return observations


@torch.no_grad()
def collect_transitions(
    envs            :   VectorEnv,
    behaviour_policy:   Callable[[TensorData, int], TensorData],
    replay_buffer   :   ReplayBuffer,
    observations    :   TensorData | dict | tuple,
    *,
    num_steps       :   int,
    global_step     :   int,
    action_processor:   ActionProcessor | None = None,
) -> tuple[TensorData, torch.Tensor, torch.Tensor]:
    """Add T steps from N environments (T * N transitions) to a replay buffer.

    Transitions already in the buffer are kept until it is full; after that,
    each new transition overwrites the oldest one. Each step: choose actions
    with the behaviour policy, step all environments, store the transitions,
    and continue from where each environment is now. Episodes that ended are
    restarted by the collector (``DISABLED``) or were already restarted by the
    environment (``SAME_STEP``); either way the stored next observation is the
    state the episode ended in (see the module docstring).

    The behaviour policy is the policy used while collecting data, including
    exploration (for example epsilon-greedy). It receives the raw observations
    and the number of transitions collected so far, which exploration
    schedules use. Its actions are stored unchanged; only the commands sent to
    the environment go through ``action_processor``.

    Collection never fits or applies observation statistics: the replay buffer
    and the returned observations stay raw. If the behaviour policy needs
    normalized inputs, it processes them itself.

    The caller is responsible for the rest of the setup: ``num_steps`` is
    positive, ``global_step`` is non-negative, and the replay capacity is at
    least N. ``train_off_policy`` checks all of this before training starts.

    Args:
        envs: N batched environments returning torch tensors, in
            ``DISABLED`` or ``SAME_STEP`` reset mode. For Gymnasium tasks use
            ``make_vec_env`` (``DISABLED`` and ``NumpyToTorch``). Any object
            with ``metadata``, ``num_envs``, ``single_action_space``,
            ``reset()`` and ``step()`` works; it does not have to inherit
            ``VectorEnv``.
        behaviour_policy: ``(observations, global_step) -> actions``, returning
            one action per environment in the coordinates the critic uses.
        replay_buffer: Storage that the new transitions are added to.
        observations: Current observations of all N environments.
        num_steps: Number of vector steps T to collect.
        global_step: Transitions collected before this call, used by
            exploration schedules.
        action_processor: Optional conversion from actions to environment
            commands, for example rescaling to the joint limits.

    Returns:
        A tuple of three items:

        * the observations to start the next collection from;
        * the raw rewards, shaped ``(T, N)``;
        * the episode-end flags (terminated or truncated), shaped ``(T, N)``.

        The rewards and flags keep the time and environment axes for logging;
        the replay buffer itself stores the T * N transitions as one flat list.

    Raises:
        ValueError: If the reset mode is not supported, a step returns
            malformed flags, or a ``SAME_STEP`` environment ends an episode
            without valid ``info["final_obs"]``.
        TypeError: If a step does not return torch tensors.
    """

    # --------------------------------------------------------------
    # 0. Setup
    # --------------------------------------------------------------
    N = envs.num_envs

    # Read the reset mode once. It decides two things in every step below:
    # where the true final observation comes from, and whether this
    # collector restarts finished episodes itself.
    same_step = (
        supported_autoreset_mode(envs, caller="collect_transitions")
        is AutoresetMode.SAME_STEP
    )
    observations = to_tensor_data(observations, batch_size=N)

    # Per-step rewards and episode ends, returned for logging.
    reward_steps: list[torch.Tensor] = []
    done_steps: list[torch.Tensor] = []

    for step in range(num_steps):
        # Copy the current observations so the environment cannot overwrite
        # them before they are stored (see collect_rollout).
        # Leafwise cloning supports unsigned depth tensors in structured states.
        observations = (
            observations.apply(torch.clone)
            if isinstance(observations, TensorDictBase)
            else observations.clone()
        )

        # --------------------------------------------------------------
        # 1. Behaviour-policy action selection
        # --------------------------------------------------------------

        # global_step + step * N is the number of transitions collected so
        # far, so exploration schedules (such as a decaying epsilon) advance
        # correctly within this call.
        actions = to_tensor_data(
            behaviour_policy(observations, global_step + step * N),
            batch_size=N,
        )

        # --------------------------------------------------------------
        # 2. Environment transition
        # --------------------------------------------------------------

        # Convert actions into environment commands; the actions themselves
        # are stored unchanged below.
        commands = actions if action_processor is None else action_processor(actions)

        # Step every environment once (shapes as in collect_rollout). In
        # SAME_STEP mode the rows whose episode just ended already hold a
        # fresh start, and their true last states are in infos["final_obs"].
        env_actions = from_tensor_data(commands, envs.single_action_space)
        next_observations, rewards, terminations, truncations, infos = envs.step(env_actions)

        # Check the outputs before anything uses them (tensors; boolean flags
        # shaped (N,)).
        _require_tensor_outputs(
            rewards, terminations, truncations,
            num_envs=N,
        )
        next_observations = to_tensor_data(next_observations, batch_size=N)

        # Episode ends (terminated or truncated), shape (N,). any_done is read
        # once: on a GPU simulator it costs a GPU-to-CPU synchronization.
        dones = torch.logical_or(terminations, truncations)
        any_done = bool(dones.any().item())

        # The next state to store. In SAME_STEP mode, put the true last
        # states back into the rows the environment already restarted.
        transition_next = next_observations
        if same_step and any_done:
            transition_next = _final_observations(
                next_observations, infos, dones, N
            )

        # --------------------------------------------------------------
        # 3. Transition storage
        # --------------------------------------------------------------

        # Store N transitions (S_t, A_t, R_{t+1}, S_{t+1}, terminated,
        # truncated) with the true S_{t+1}, in both reset modes.
        replay_buffer.add(
            observation=observations,
            action=actions,
            reward=rewards,
            next_observation=transition_next,
            termination=terminations,
            truncation=truncations,
        )
        # Keep copies for logging: the replay buffer may overwrite old rows,
        # and the environment may reuse these tensors on the next step.
        reward_steps.append(rewards.clone())
        done_steps.append(dones.clone())

        # --------------------------------------------------------------
        # 4. Next observations for acting
        # --------------------------------------------------------------

        # What the behaviour policy acts on next (as in collect_rollout):
        # SAME_STEP continues from what step() returned, because the finished
        # rows are already fresh starts; DISABLED restarts only the finished
        # rows, now that their true last states are stored. Fresh
        # observations stay raw; any normalization happens inside the
        # behaviour policy.
        if same_step:
            observations = next_observations
        elif any_done:
            observations, _ = envs.reset(options={"reset_mask": dones})
            observations = to_tensor_data(observations, batch_size=N)
        else:
            observations = next_observations

    return observations, torch.stack(reward_steps), torch.stack(done_steps)
