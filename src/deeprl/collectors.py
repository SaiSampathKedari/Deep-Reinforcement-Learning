"""Collect experience from vectorized environments into training buffers.

This module has two collectors:

* ``collect_rollout`` (on-policy: PPO, A2C, NPG, TRPO) fills a
  ``RolloutBuffer`` with exactly T steps from N environments, together with
  the value estimate and log-probability of every action. The buffer is
  emptied and refilled for every rollout.
* ``collect_transitions`` (off-policy: DQN, DDPG, TD3, SAC) adds T steps from N
  environments to a ``ReplayBuffer``. Transitions already there are kept until
  the buffer is full; after that, each new transition overwrites the oldest
  one (first in, first out).

Both follow the same rules:

* **Environments never reset themselves.** Our collectors require
  environments created with ``AutoresetMode.DISABLED``. When an episode
  ends, the collector first stores the true final observation as the next
  observation, and only then resets that environment. Gymnasium's automatic
  reset modes also keep the final observation, but each in a different place
  (in ``info`` for ``SAME_STEP``; with an extra reset step for
  ``NEXT_STEP``). Disabling them gives one simple rule for every
  environment.
* **Data is stored raw.** Observations and rewards go into the buffer exactly
  as the environment returned them. Normalization happens later, when a
  network reads them.
* **The sampled action is stored, the converted command is sent.** If the
  policy has an action processor (for example rescaling [-1, 1] to the joint
  limits), only the command sent to the environment is converted. The buffer
  keeps the original sample, which training needs.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
from gymnasium.vector import VectorEnv
from tensordict import TensorDictBase
from torch.distributions import Distribution

from deeprl.buffers import ReplayBuffer, RolloutBuffer
from deeprl.data_spec import TensorData
from deeprl.env import from_tensor_data, to_tensor_data
from deeprl.inference import InferencePolicy
from deeprl.processors import ActionProcessor


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
    environment, step all environments, store the transition, and reset the
    environments whose episode ended. When this returns, the buffer is full.

    What is stored:

    * observations and rewards exactly as the environment returned them;
    * the sampled actions and their log-probabilities (needed for PPO's ratio);
    * the value estimate of each observation;
    * the true next observation, including the final one of an episode;
    * separate termination and truncation flags.

    Args:
        envs: N vectorized environments returning Torch tensors
            (``NumpyToTorch``) and created with ``AutoresetMode.DISABLED``.
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
        nested TensorDicts with batch size ``(N,)``.
    """

    # --------------------------------------------------------------
    # 0. Setup
    # --------------------------------------------------------------

    # Restart writing at the first row of the preallocated rollout storage.
    rollout_buffer.reset()
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

        # Step every environment once. Because environments never reset
        # themselves, next_observations holds the true next state, or the
        # final state if the episode just ended.
        #
        # next_observations: (N, *obs_shape)
        # rewards, terminations, truncations: (N,)
        env_actions = from_tensor_data(commands, envs.single_action_space)
        next_observations, rewards, terminations, truncations, _ = envs.step(env_actions)
        next_observations = to_tensor_data(
            next_observations,
            batch_size=envs.num_envs,
        )

        # --------------------------------------------------------------
        # 3. Transition storage
        # --------------------------------------------------------------

        # Store the step before any environment is reset, so final
        # observations are kept. The buffer copies the data into its own
        # time-major storage, row by row.
        rollout_buffer.add(
            observation=observations,
            action=actions,
            reward=rewards,
            next_observation=next_observations,
            termination=terminations,
            truncation=truncations,
            value=values,
            log_prob=log_probs,
        )

        # --------------------------------------------------------------
        # 4. Partial environment reset
        # --------------------------------------------------------------

        # An episode ends with either a termination (a true end, such as a
        # fall) or a truncation (a time limit). Shape: (N,), dtype: bool.
        dones = torch.logical_or(terminations, truncations)

        if dones.any().item():

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
    and reset the environments whose episode ended.

    The behaviour policy is the policy used while collecting data, including
    exploration (for example epsilon-greedy). It receives the raw observations
    and the number of transitions collected so far, which exploration
    schedules use. Its actions are stored unchanged; only the commands sent to
    the environment go through ``action_processor``.

    Collection never fits or applies observation statistics: the replay buffer
    and the returned observations stay raw. If the behaviour policy needs
    normalized inputs, it processes them itself.

    The caller is responsible for the setup: environments with
    ``NumpyToTorch`` and ``AutoresetMode.DISABLED``, ``num_steps`` a positive
    integer, ``global_step`` a non-negative integer, and a replay capacity of
    at least N.

    Args:
        envs: N vectorized environments returning Torch tensors.
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
    """

    # --------------------------------------------------------------
    # 0. Setup
    # --------------------------------------------------------------
    N = envs.num_envs
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
        env_actions = from_tensor_data(commands, envs.single_action_space)
        next_observations, rewards, terminations, truncations, _ = envs.step(env_actions)
        next_observations = to_tensor_data(next_observations, batch_size=N)

        # --------------------------------------------------------------
        # 3. Transition storage
        # --------------------------------------------------------------

        # Store the step before resetting any environment, so the true final
        # observation of every finished episode is kept.
        replay_buffer.add(
            observation=observations,
            action=actions,
            reward=rewards,
            next_observation=next_observations,
            termination=terminations,
            truncation=truncations,
        )
        dones = torch.logical_or(terminations, truncations)

        # Keep copies for logging: the replay buffer may overwrite old rows,
        # and the environment may reuse these tensors on the next step.
        reward_steps.append(rewards.clone())
        done_steps.append(dones.clone())

        # --------------------------------------------------------------
        # 4. Partial environment reset
        # --------------------------------------------------------------

        # Reset only the finished environments. Their fresh observations stay
        # raw; any normalization happens inside the behaviour policy.
        if dones.any().item():
            observations, _ = envs.reset(options={"reset_mask": dones})
            observations = to_tensor_data(observations, batch_size=N)
        else:
            observations = next_observations

    return observations, torch.stack(reward_steps), torch.stack(done_steps)
