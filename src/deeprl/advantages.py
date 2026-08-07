"""Advantage estimators.

Pure functions over a stored rollout: no networks, no config, no state. Shapes
are (T, N) throughout -- T rollout steps across N parallel environments.
"""

from __future__ import annotations
import torch


@torch.no_grad()
def gae(
    rewards     : torch.Tensor,   # (T, N)
    values      : torch.Tensor,   # (T, N)  V(s_t)
    next_values : torch.Tensor,   # (T, N)  V(s_{t+1}), the true final observation
    terminations: torch.Tensor,   # (T, N)  bool
    truncations : torch.Tensor,   # (T, N)  bool
    *,
    gamma       : float,
    gae_lambda  : float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation (Schulman et al. 2016).

        A_t = delta_t + gamma * lambda * A_{t+1}

    lambda = 0 recovers the one-step TD advantage; lambda = 1 the discounted
    sum of TD errors to the end of the rollout.

    Returns:
        advantages:    (T, N)
        value_targets: (T, N), advantages + values -- the critic's regression target.
    """
    T, N = rewards.shape
    dtype = values.dtype

    # Bootstrap after ordinary transitions and truncations.
    # Do not bootstrap after genuine MDP terminations.
    bootstrap_mask = (~terminations).to(dtype=dtype)

    # Stop the trace at an episode boundary: s_{t+1} starts a new episode,
    # so A_{t+1} does not belong in A_t.
    continuation_mask = (~(terminations | truncations)).to(dtype=dtype)

    # Constant across the recursion, so it is built once as (T, N) rather than
    # recomputed from three tensors on every iteration of the loop below.
    decay = (gamma * gae_lambda) * continuation_mask

    # One TD error per environment per step. Shape: (T, N).
    td_targets = rewards + gamma * bootstrap_mask * next_values
    td_errors = td_targets - values

    # A_t depends on A_{t+1}, so the rollout is walked backwards.
    # gae carries one running value per environment. Shape: (N,).
    advantages = torch.empty_like(td_errors)
    gae = torch.zeros(N, dtype=dtype, device=values.device)

    for t in reversed(range(T)):
        gae = td_errors[t] + decay[t] * gae
        advantages[t] = gae
    value_targets = advantages + values
    
    return advantages, value_targets
    