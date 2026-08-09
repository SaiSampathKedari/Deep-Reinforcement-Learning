"""Advantage estimators.

Pure functions over a stored rollout: no networks, no config, no state. Shapes
are (T, N) throughout -- T rollout steps across N parallel environments.
"""

from __future__ import annotations
import torch


@torch.no_grad()
def generalized_advantage_estimate(
    rewards      : torch.Tensor,   # (T, N)
    values       : torch.Tensor,   # (T, N)  V(s_t)
    next_values  : torch.Tensor,   # (T, N)  V(actual s_{t+1}), the true pre-reset state
    terminations : torch.Tensor,   # (T, N)  bool
    truncations  : torch.Tensor,   # (T, N)  bool
    *,
    gamma        : float,
    gae_lambda   : float,
) -> tuple[torch.Tensor, torch.Tensor]:

    T = rewards.shape[0]
    dtype = values.dtype

    # Bootstrap after ordinary transitions and truncations.
    # Do not bootstrap after genuine MDP terminations.
    bootstrap_mask = (~terminations).to(dtype=dtype)

    # Stop the trace at any episode boundary: s_{t+1} starts a new episode,
    # so A_{t+1} does not belong in A_t.
    trace_mask = (~(terminations | truncations)).to(dtype=dtype)

    # One TD error per environment per step. Shape: (T, N).
    td_targets = rewards + gamma * bootstrap_mask * next_values
    td_errors = td_targets - values

    # Constant across the recursion, so it is built once as (T, N) rather than
    # recomputed from three tensors on every iteration of the loop below.
    decay = (gamma * gae_lambda) * trace_mask

    # A_t depends on A_{t+1}, so the rollout is walked backwards.
    # last_gae_lam carries one running value per environment. Shape: (N,).
    advantages = torch.zeros_like(values)
    last_gae_lam = torch.zeros_like(values[0])

    for t in reversed(range(T)):
        last_gae_lam = td_errors[t] + decay[t] * last_gae_lam
        advantages[t] = last_gae_lam

    value_targets = advantages + values

    return advantages, value_targets
