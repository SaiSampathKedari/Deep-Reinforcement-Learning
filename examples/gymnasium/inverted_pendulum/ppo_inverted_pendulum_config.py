"""PPO training settings for InvertedPendulum."""

import torch

from deeprl.algorithms.ppo import PPOConfig

ENV_ID = "InvertedPendulum-v5"
NUM_ENVS = 8
LOG_STD_INIT = 0.0  # Initial standard deviation: exp(0) = 1.

# One rollout contains 256 x 8 = 2,048 transitions.
PPO_CONFIG = PPOConfig(
    total_timesteps=204_800,
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
    critic_optimizer_class=torch.optim.Adam,
    log_every=2_048,
)

# Evaluation interval counts training transitions.
EVAL_EVERY = 20_480
EVAL_EPISODES = 10
EVAL_SEED = 10_000

# Checkpoint interval counts completed rollout updates.
CHECKPOINT_INTERVAL = 10