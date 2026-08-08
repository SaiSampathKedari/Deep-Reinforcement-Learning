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
        observation_shape = observation_space.shape
        
        action_dtype = _space_dtype(action_space)
        action_shape = action_space.shape
        

        
        T, N = self.rollout_steps, self.num_envs
        
        self.observations       = torch.zeros((T, N, *observation_shape), dtype=observation_dtype, device=self.device )
        self.actions            = torch.zeros((T, N, *action_shape), dtype=action_dtype, device=self.device)
        self.rewards            = torch.zeros(T, N, dtype=torch.float32, device=self.device)
        self.terminations       = torch.zeros(T, N, dtype=torch.bool, device=self.device)
        self.truncations        = torch.zeros(T, N, dtype=torch.bool, device=self.device)
        self.next_observations  = torch.zeros((T, N, *observation_shape), dtype=observation_dtype, device=self.device)
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
    
    
    def get(
        self,
        advantages      :   torch.Tensor, # (T, N)
        value_targets   :   torch.Tensor, # (T, N)
        batch_size      :   int | None = None
    ) -> Iterator[RolloutBatch]:
        assert self.full
        
        rollout_size = self.rollout_steps * self.num_envs
        
        # flatten (T, N, ...) -> (B, ...).
        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        old_log_probs = self.log_probs.flatten(0, 1)
        old_values = self.values.flatten(0, 1)
        advantages = advantages.flatten(0, 1)
        value_targets = value_targets.flatten(0,1)
        
        # A2C, NPG, and TRPO use the complete rollout.
        if batch_size is None:
            batch_size = rollout_size
        
        # PPO needs shuffled minibatches. Shuffling does not change a full-batch
        # A2C update because the loss takes the mean over the entire batch.
        indices = torch.randperm(
            rollout_size,
            device=self.device
        )
        
        for start in range(0, rollout_size, batch_size):
            batch_indices = indices[start:start+batch_size]
            
            yield RolloutBatch(
                observations=observations[batch_indices],
                actions=actions[batch_indices],
                old_log_probs=old_log_probs[batch_indices],
                old_values=old_values[batch_indices],
                advantages=advantages[batch_indices],
                value_targets=value_targets[batch_indices],
            )