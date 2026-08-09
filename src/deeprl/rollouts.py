"""Rollout collection and next-state value evaluation."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Distribution

from gymnasium.vector import VectorEnv

from deeprl.buffers import RolloutBuffer


@torch.no_grad()
def collect_rollout(
    envs            :   VectorEnv,
    policy          :   nn.Module,
    value_fn        :   nn.Module,
    buffer          :   RolloutBuffer,
    observations    :   torch.Tensor,
) -> torch.Tensor:
    """Collect one T-step rollout from N vectorized environments.

    The buffer is full on return. The returned observations have shape
    (N, *obs_shape) and are the starting states for the next rollout.
    Network outputs are stored without autograd graphs. Collection requires
    AutoresetMode.DISABLED.
    """

    # Restart writing at the first row of the preallocated rollout storage.
    buffer.reset()

    for _ in range(buffer.rollout_steps):

        # --------------------------------------------------------------
        # 1. Current-state evaluation
        # --------------------------------------------------------------

        # Current observations: (N, *obs_shape).
        # Collection-time values: (N,).
        values = value_fn(observations)

        # One batched action distribution for all N environments.
        action_distributions: Distribution = policy(observations)

        # Actions: (N,) for discrete actions or (N, *action_shape) otherwise.
        actions: torch.Tensor = action_distributions.sample()

        # Collection-time log pi(A|S). Shape: (N,).
        log_probs: torch.Tensor = action_distributions.log_prob(actions)

        # --------------------------------------------------------------
        # 2. Environment transition
        # --------------------------------------------------------------

        # Step every environment once. Under disabled autoreset,
        # next_observations contains the true next or final states.
        #
        # next_observations: (N, *obs_shape)
        # rewards, terminations, truncations: (N,)
        next_observations, rewards, terminations, truncations, _ = envs.step(actions)

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

        else:

            # Every environment continues from its returned next observation.
            observations = next_observations

    return observations


@torch.no_grad()
def evaluate_next_values(
    value_fn            :   nn.Module,
    next_observations   :   torch.Tensor,
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
        device=next_observations.device
    )

    # Boolean indexing gathers all selected next observations into one batch:
    # (K, *obs_shape) -> (K,), where K is the number of nonterminal transitions.
    next_values[nonterminal_mask] = value_fn(
        next_observations[nonterminal_mask]
    )

    return next_values
