"""PPO training settings for CartPole."""

import torch

from deeprl.algorithms.ppo import PPOConfig

# Environment and network dimensions.
ENV_ID = "CartPole-v1"
NUM_ENVS = 8
POLICY_HIDDEN_SIZES = (64, 64)
VALUE_HIDDEN_SIZES = (64, 64)

# One rollout contains 128 x 8 = 1,024 transitions.
PPO_CONFIG = PPOConfig(
    total_timesteps=102_400,
    seed=0,
    device=torch.device("cpu"),
    rollout_steps=128,
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
    log_every=1_024,
)

# Separate evaluation episodes; interval counts training transitions.
EVAL_EVERY = 10_240
EVAL_EPISODES = 10
EVAL_SEED = 10_000

# Save every 10 completed rollout updates, and at the end of training.
CHECKPOINT_INTERVAL = 10
