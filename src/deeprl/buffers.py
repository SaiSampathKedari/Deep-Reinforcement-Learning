from __future__ import annotations

from collections.abc import Iterator
from typing import NamedTuple

import torch
import gymnasium as gym


class RolloutBatch(NamedTuple):
    """Flat data consume by an on-policy update.
    
    B is either T * N or one minbatch of that rollout
    """
    observations    :   torch.Tensor    # (B, *obs_shape)
    actions         :   torch.Tensor    # (B, *action_shape)
    old_log_probs   :   torch.Tensor    # (B,)
    old_values      :   torch.Tensor    # (B,)
    advantages      :   torch.Tensor    # (B,)
    value_targets   :   torch.Tensor    # (B,)

def _space_dtype(space : gym.Space) -> torch.dtype:
    return torch.as_tensor(space.sample()).dtype

class RolloutBuffer:
    def __init__(
        self,
        rollout_steps       :   int,
        num_envs            :   int,
        observation_space   :   gym.Space,
        action_space        :   gym.Space,
        device              :   torch.device
    ):
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self.device = device
        
        observation_dtype = _space_dtype(observation_space)
        action_dtype = _space_dtype(action_space)
        
        T, N = self.rollout_steps, self.num_envs
        
        self.observations       = torch.zeros(T, N, dtype=observation_dtype, device=self.device )
        self.actions            = torch.zeros(T, N, dtype=action_dtype, device=self.device)
        self.rewards            = torch.zeros(T, N, dtype=torch.float32, device=self.device)
        self.terminations       = torch.zeros(T, N, dtype=torch.bool, device=self.device)
        self.truncations        = torch.zeros(T,N, dtype=torch.bool, device=self.device)
        self.next_observations  = torch.zeros(T, N, dtype=observation_dtype, device=self.device)
        self.values             = torch.zeros(T, N, dtype=torch.float32, device=self.device)
        self.log_probs          = torch.zeros(T, N, dtype=torch.float32, device=self.device)
        
        self.pos = 0
        self.full = False
        
    
    def reset(self) -> None:
        self.pos = 0
        self.full = False
        
    def add(
        self,
        *,
        observation         :   torch.Tensor,   # (N, *obs_shape)
        action              :   torch.Tensor,   # (N, *action_shape)
        reward              :   torch.Tensor,   # (N,) float32
        next_observation    :   torch.Tensor,   # (N, *obs_shape)
        termination         :   torch.Tensor,   # (N,) bool
        truncation          :   torch.Tensor,   # (N,) bool
        value               :   torch.Tensor,   # (N,) float32
        log_prob            :   torch.Tensor    # (N,) float32
    ) -> None:
        """Store one vector-environment step.

        Every argument represents N environments. Data is copied into owned
        storage, so later environment resets cannot mutate stored observations.
        """
        assert self.pos < self.rollout_steps
        
        self.observations[self.pos] = observation
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_observations[self.pos] = next_observation
        self.terminations[self.pos] = termination
        self.truncations[self.pos] = truncation
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        
        self.pos += 1
        self.full = self.pos == self.rollout_steps  
        
    
        