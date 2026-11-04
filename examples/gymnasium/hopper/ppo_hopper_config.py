"""Baseline PPO training settings for Hopper."""

import torch

from deeprl.algorithms.ppo import PPOConfig

ENV_ID = "Hopper-v5"
NUM_ENVS = 8
POLICY_HIDDEN_SIZES = (64, 64)
VALUE_HIDDEN_SIZES = (64, 64)
LOG_STD_INIT = -1.0  # Initial standard deviation: exp(-1) = 0.368 in native units.

# Set either to None to disable it. These settings are saved with each run.
OBS_NORMALIZATION = {"epsilon": 1e-8, "clip": 10.0}
REWARD_SCALING = {"epsilon": 1e-8}

# One rollout contains 256 x 8 = 2,048 transitions.
PPO_CONFIG = PPOConfig(
    total_timesteps=2_048_000,
    seed=0,
    device=torch.device("cpu"),
    rollout_steps=256,
    gamma=0.99,
    gae_lambda=0.95,
    lr_actor=3e-4,
    lr_critic=3e-4,
    batch_size=64,
    update_epochs=10,
    clip_range=0.2,
    normalize_advantage=True,
    ent_coef=0.0,
    max_grad_norm=0.5,
    actor_optimizer_class=torch.optim.Adam,
    actor_optimizer_kwargs={"eps": 1e-5},
    critic_optimizer_class=torch.optim.Adam,
    critic_optimizer_kwargs={"eps": 1e-5},
    log_every=2_048,
)

# Evaluation interval counts training transitions.
EVAL_EVERY = 40_960
EVAL_EPISODES = 5
EVAL_SEED = 10_000

# Checkpoint interval counts completed rollout updates.
CHECKPOINT_INTERVAL = 20
