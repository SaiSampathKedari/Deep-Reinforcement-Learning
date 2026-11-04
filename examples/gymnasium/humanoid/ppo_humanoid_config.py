"""Humanoid PPO settings from SLM-Lab's archived MuJoCo benchmark.

See README.md for source links and the deliberate differences in this port.
"""

import torch

from deeprl.algorithms.ppo import PPOConfig

ENV_ID = "Humanoid-v5"
NUM_ENVS = 16
POLICY_HIDDEN_SIZES = (256, 256)
VALUE_HIDDEN_SIZES = (256, 256)
LOG_STD_INIT = 0.0
LOG_STD_BOUNDS = (-5.0, 2.0)
ACTION_RANGE = (-1.0, 1.0)  # Rescaled to the native [-0.4, 0.4] control coordinates.
REFERENCE_SPEC = (
    "https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/data/"
    "ppo_mujoco_arc_humanoid_2026_02_12_185439/ppo_mujoco_arc_humanoid_spec.json"
)

# Set either to None to disable it. These settings are saved with each run.
OBS_NORMALIZATION = {"epsilon": 1e-8, "clip": 10.0}
REWARD_SCALING = {"epsilon": 1e-8, "update_per_step": True, "clip": 10.0}

# One rollout contains 2,048 x 16 = 32,768 transitions.
PPO_CONFIG = PPOConfig(
    total_timesteps=12_288_000,
    seed=0,
    device=torch.device("cpu"),
    rollout_steps=2048,
    gamma=0.99,
    gae_lambda=0.95,
    lr_actor=3e-4,
    lr_critic=3e-4,
    batch_size=64,
    update_epochs=10,
    clip_range=0.2,
    normalize_advantage=True,
    normalize_advantage_per_minibatch=True,
    normalize_value_targets=True,
    normalize_observations_during_collection=True,
    log_ratio_clip=20.0,
    ent_coef=0.0,
    target_kl=None,
    max_grad_norm=0.5,
    actor_optimizer_class=torch.optim.AdamW,
    actor_optimizer_kwargs={"eps": 1e-8, "weight_decay": 0.01, "betas": (0.9, 0.999)},
    critic_optimizer_class=torch.optim.AdamW,
    critic_optimizer_kwargs={"eps": 1e-8, "weight_decay": 0.01, "betas": (0.9, 0.999)},
    log_every=32_768,
)

# Evaluation interval counts training transitions.
EVAL_EVERY = 98_304
EVAL_EPISODES = 5
EVAL_SEED = 10_000

# Checkpoint interval counts completed rollout updates.
CHECKPOINT_INTERVAL = 3
