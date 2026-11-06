"""On-policy rollout and off-policy transition collection with explicit resets."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
from gymnasium.vector import VectorEnv
from torch.distributions import Distribution

from deeprl.buffers import ReplayBuffer, RolloutBuffer
from deeprl.data_spec import TensorData
from deeprl.env import from_tensor_data, to_tensor_data
from deeprl.inference import InferencePolicy


@torch.no_grad()
def collect_rollout(
    envs            :   VectorEnv,
    policy          :   nn.Module,
    value_fn        :   Callable[[TensorData], torch.Tensor],
    rollout_buffer  :   RolloutBuffer,
    observations    :   TensorData | dict | tuple,
) -> TensorData:
    """Collect one T-step rollout from N vectorized environments.

    The buffer is full on return. The returned observations have shape
    (N, *obs_shape) and are the starting states for the next rollout.
    Dict/Tuple observations are represented by nested TensorDicts with batch
    size (N,). Network outputs are stored without autograd graphs. Collection
    requires NumpyToTorch and AutoresetMode.DISABLED.

    Observations are stored and returned unchanged. The supplied inference policy
    and value callable process model inputs without modifying storage or moments.
    Rewards are stored as returned by the environment.
    Actions and log-probabilities describe the original policy samples; an
    inference policy's output processor converts only the environment commands.
    """

    # Restart writing at the first row of the preallocated rollout storage.
    rollout_buffer.reset()
    observations = to_tensor_data(observations, batch_size=envs.num_envs)

    for _ in range(rollout_buffer.rollout_steps):

        # Preserve current observations before the environment reuses its storage.
        observations = observations.clone()

        # --------------------------------------------------------------
        # 1. Current-state evaluation
        # --------------------------------------------------------------

        # Current observations: (N, *obs_shape).
        # Collection-time values: (N,).
        values = value_fn(observations)

        # One batched action distribution for all N environments.
        action_distributions: Distribution = policy(observations)

        # Actions: (N,) for discrete actions or (N, *action_shape) otherwise.
        actions = to_tensor_data(
            action_distributions.sample(),
            batch_size=envs.num_envs,
        )

        # Collection-time log pi(A|S). Shape: (N,).
        # Structured distributions consume TensorDict actions here and during updates.
        log_probs: torch.Tensor = action_distributions.log_prob(actions)

        # --------------------------------------------------------------
        # 2. Environment transition
        # --------------------------------------------------------------

        # Step every environment once. Under disabled autoreset,
        # next_observations contains the true next or final states.
        #
        # next_observations: (N, *obs_shape)
        # rewards, terminations, truncations: (N,)
        commands = actions
        if isinstance(policy, InferencePolicy) and policy.action_processor is not None:
            commands = policy.action_processor(actions)
        env_actions = from_tensor_data(commands, envs.single_action_space)
        next_observations, rewards, terminations, truncations, _ = envs.step(env_actions)
        next_observations = to_tensor_data(
            next_observations,
            batch_size=envs.num_envs,
        )

        # --------------------------------------------------------------
        # 3. Transition storage
        # --------------------------------------------------------------

        # Store the true next observations before any completed environment
        # is reset. Tensor assignment copies this vector step into buffer-owned
        # time-major storage.
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

        # An episode ends after either a termination or truncation.
        # Shape: (N,), dtype: bool.
        dones = torch.logical_or(terminations, truncations)

        if dones.any().item():

            # Reset only completed environments. The returned batch contains
            # reset observations for completed environments and unchanged next
            # observations for continuing environments.
            observations, _ = envs.reset(options={"reset_mask": dones})
            observations = to_tensor_data(
                observations,
                batch_size=envs.num_envs,
            )

        else:

            # Every environment continues from its returned next observation.
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
) -> tuple[TensorData, torch.Tensor, torch.Tensor]:
    """Add T * N transitions to replay without clearing existing data.

    Return raw starting observations for the next collection, raw rewards
    (T, N), and episode boundaries (T, N). Replay flattens these T * N
    transitions; the returned time/environment axes are for logging.
    The behaviour policy takes batched observations and the cumulative
    transition count, and returns one action per environment.

    The caller validates setup: NumpyToTorch, AutoresetMode.DISABLED, positive
    integer num_steps, nonnegative integer global_step, and replay capacity
    at least N. These are collection preconditions, as in collect_rollout.

    The behaviour policy processes its raw inputs; collection never fits or
    applies observation statistics. Replay and returned observations stay raw.
    """
    N = envs.num_envs
    observations = to_tensor_data(observations, batch_size=N)
    reward_steps: list[torch.Tensor] = []
    done_steps: list[torch.Tensor] = []

    for step in range(num_steps):
        # Preserve current observations before the environment reuses its storage.
        observations = observations.clone()

        # --------------------------------------------------------------
        # 1. Behaviour-policy action selection
        # --------------------------------------------------------------
        actions = to_tensor_data(
            behaviour_policy(observations, global_step + step * N),
            batch_size=N,
        )

        # --------------------------------------------------------------
        # 2. Environment transition
        # --------------------------------------------------------------
        env_actions = from_tensor_data(actions, envs.single_action_space)
        next_observations, rewards, terminations, truncations, _ = envs.step(env_actions)
        next_observations = to_tensor_data(next_observations, batch_size=N)

        # --------------------------------------------------------------
        # 3. Transition storage
        # --------------------------------------------------------------
        # Copy true final observations into replay before resetting any env.
        replay_buffer.add(
            observation=observations,
            action=actions,
            reward=rewards,
            next_observation=next_observations,
            termination=terminations,
            truncation=truncations,
        )
        dones = torch.logical_or(terminations, truncations)

        # Keep logging data even when replay wraps or the env reuses its tensors.
        reward_steps.append(rewards.clone())
        done_steps.append(dones.clone())

        # --------------------------------------------------------------
        # 4. Partial environment reset
        # --------------------------------------------------------------
        # Reset states stay raw; normalization happens before the next action.
        if dones.any().item():
            observations, _ = envs.reset(options={"reset_mask": dones})
            observations = to_tensor_data(observations, batch_size=N)
        else:
            observations = next_observations

    return observations, torch.stack(reward_steps), torch.stack(done_steps)


