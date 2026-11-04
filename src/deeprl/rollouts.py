"""Rollout collection and next-state value evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from gymnasium.vector import VectorEnv
from torch.distributions import Distribution

from deeprl.buffers import RolloutBuffer, TensorData
from deeprl.env import from_tensor_data, to_tensor_data

if TYPE_CHECKING:
    from deeprl.normalization import RunningNormalizer


@torch.no_grad()
def collect_rollout(
    envs            :   VectorEnv,
    policy          :   nn.Module,
    value_fn        :   nn.Module,
    buffer          :   RolloutBuffer,
    observations    :   TensorData | dict | tuple,
    obs_normalizer  :   RunningNormalizer | None = None,
) -> TensorData:
    """Collect one T-step rollout from N vectorized environments.

    The buffer is full on return. The returned observations have shape
    (N, *obs_shape) and are the starting states for the next rollout.
    Dict/Tuple observations are represented by nested TensorDicts with batch
    size (N,). Network outputs are stored without autograd graphs. Collection
    requires NumpyToTorch and AutoresetMode.DISABLED.

    With an external obs_normalizer, incoming observations must already be
    normalized. New observations update its statistics before normalization;
    the buffer and returned observations contain the normalized policy inputs.
    """

    # Restart writing at the first row of the preallocated rollout storage.
    buffer.reset()
    observations = to_tensor_data(observations, batch_size=envs.num_envs)

    for _ in range(buffer.rollout_steps):
        
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
        actions = to_tensor_data(action_distributions.sample(), batch_size=envs.num_envs)

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
        env_actions = from_tensor_data(actions, envs.single_action_space)
        next_observations, rewards, terminations, truncations, _ = envs.step(env_actions)
        next_observations = to_tensor_data(next_observations, batch_size=envs.num_envs)
        if obs_normalizer is not None:
            obs_normalizer.update(next_observations)
            next_observations = obs_normalizer(next_observations)

        # --------------------------------------------------------------
        # 3. Transition storage
        # --------------------------------------------------------------

        # Store the true next observations before any completed environment
        # is reset. Tensor assignment copies this vector step into buffer-owned
        # time-major storage.
        buffer.add(
            observation=observations,
            action=actions,
            reward=rewards,
            next_observation=next_observations,
            termination=terminations,
            truncation=truncations,
            value=values,
            log_prob=log_probs
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
            observations, _ = envs.reset(options={"reset_mask":dones})
            observations = to_tensor_data(observations, batch_size=envs.num_envs)
            if obs_normalizer is not None:
                # Only reset entries are new samples; continuing observations
                # were already included by the preceding environment step.
                obs_normalizer.update(observations[dones])
                observations = obs_normalizer(observations)

        else:

            # Every environment continues from its returned next observation.
            observations = next_observations

    return observations


@torch.no_grad()
def evaluate_next_values(
    value_fn            :   nn.Module,
    next_observations   :   TensorData,
    terminations        :   torch.Tensor,
) -> torch.Tensor:
    """Evaluate V(S_{t+1}) for every nonterminated rollout transition."""

    # Ordinary and truncated transitions require V(S_{t+1}); true
    # terminations do not. Shape: (T, N).
    nonterminal_mask = ~terminations

    # Terminal entries remain zero. Shape: (T, N).
    next_values = torch.zeros(
        terminations.shape,
        dtype=torch.float32,
        device=terminations.device
    )

    # Boolean indexing gathers all selected next observations into one batch:
    # (K, *obs_shape) -> (K,), where K is the number of nonterminal transitions.
    next_values[nonterminal_mask] = value_fn(
        next_observations[nonterminal_mask]
    )

    return next_values
