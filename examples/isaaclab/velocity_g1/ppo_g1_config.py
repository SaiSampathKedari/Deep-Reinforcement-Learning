"""PPO settings for Isaac Lab's rough-terrain Unitree G1 task.

The full recipe follows Isaac Lab v3.0.0's G1 RSL-RL example where our PPO
has equivalent options. Start with the small smoke run before selecting it.
"""

from pathlib import Path

import torch

from deeprl.algorithms.ppo import PPOConfig

TASK_ID = "Isaac-Velocity-Rough-G1"
DEVICE = torch.device("cuda:0")
SMOKE_TEST = True
NUM_ENVS = 64 if SMOKE_TEST else 4096
NUM_UPDATES = 10 if SMOKE_TEST else 5000
ENV_OVERRIDES: tuple[str, ...] = ()
RANDOM_INITIAL_EPISODE_LENGTHS = True

POLICY_HIDDEN_SIZES = (512, 256, 128)
VALUE_HIDDEN_SIZES = (512, 256, 128)
LOG_STD_INIT = 0.0  # exp(0) = 1 initial action standard deviation.
LOG_STD_BOUNDS = None

ROLLOUT_STEPS = 24
ROLLOUT_SIZE = ROLLOUT_STEPS * NUM_ENVS
PPO_CONFIG = PPOConfig(
    total_timesteps=NUM_UPDATES * ROLLOUT_SIZE,
    seed=0,
    device=DEVICE,
    rollout_steps=ROLLOUT_STEPS,
    gamma=0.99,
    gae_lambda=0.95,
    lr_actor=1e-3,
    lr_critic=1e-3,
    batch_size=ROLLOUT_SIZE // 4,
    update_epochs=5,
    clip_range=0.2,
    value_clip_range=0.2,
    ent_coef=0.008,
    normalize_advantage=True,
    max_grad_norm=1.0,
    lr_schedule="adaptive",
    desired_kl=0.01,
    target_kl=None,
    log_every=ROLLOUT_SIZE,
)

# Checkpoints are saved after completed rollout updates, not mid-rollout.
CHECKPOINT_INTERVAL = 10 if SMOKE_TEST else 50
OUTPUT_ROOT = Path("outputs/isaaclab/velocity_g1/ppo") / (
    "smoke" if SMOKE_TEST else "full"
)
RESUME_FROM: Path | None = None

# Playback uses an exported policy folder from a finished run.
PLAY_POLICY_DIR: Path | None = None
PLAY_NUM_EPISODES = 10
PLAY_NUM_ENVS = 1
PLAY_SEED = 1000
PLAY_RECORD_VIDEO = False
PLAY_VIDEO_LENGTH = 200
PLAY_OUTPUT_ROOT = Path("outputs/isaaclab/velocity_g1/play")
