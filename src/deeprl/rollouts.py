from __future__ import annotations


import torch
import torch.nn as nn
from torch.distributions import Distribution

from gymnasium.vector import VectorEnv

from deeprl.buffers import RolloutBuffer
from deeprl.logger import Logger

@torch.no_grad()
def collect_rollout(
    envs            :   VectorEnv,
    policy          :   nn.Module,
    value_fn        :   nn.Module,
    buffer          :   RolloutBuffer,
    observations    :   torch.Tensor,
    logger          :   Logger
) -> torch.Tensor:
    """Collect one T-step rollout from N vectorized environments.
    
    Returns the observations from which the next rollout should begin.
    """
    buffer.reset()
    
    for _ in range(buffer.rollout_steps):
        
        values = value_fn(observations)
        
        action_distributions :  Distribution = policy(observations)
        actions : torch.Tensor = action_distributions.sample()
        log_probs : torch.Tensor = action_distributions.log_prob(actions)
        
        next_observations, rewards, terminations, truncations, _ = envs.step(actions)
        
        # Store next_observations before resetting finished environments.
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
        
        dones = torch.logical_or(terminations, truncations)
        
        if dones.any().item():
            observations, _ = envs.reset(options={"reset_mask":dones})
        else:
            observations = next_observations
    
    return observations


@torch.no_grad()
def evaluate_next_values(
    value_fn            :   nn.Module,
    next_observations   :   torch.Tensor,
    terminations        :   torch.Tensor,
) -> torch.Tensor:
    """Compute bootstrap values V(s') for every rollout transition."""
    
    bootstrap_mask = ~terminations
    
    next_values = torch.zeros(
        terminations.shape,
        dtype=torch.float32,
        device=next_observations.device
    )
    
    next_values[bootstrap_mask] = value_fn(
        next_observations[bootstrap_mask]
    )

    return next_values
    
    
    