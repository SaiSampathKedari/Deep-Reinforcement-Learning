import torch
import gymnasium as gym
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.wrappers.vector.numpy_to_torch import NumpyToTorch
from gymnasium.envs.registration import VectorizeMode


def make_vec_env(
    env_id              : str,
    num_envs            : int,
    device              : torch.device = torch.device("cpu"),
    vectorization_mode  : VectorizeMode = VectorizeMode.SYNC,
) -> VectorEnv:

    if num_envs <= 0:
        raise ValueError(
            f"num_envs must be positive, got {num_envs}"
        )

    envs = gym.make_vec(
        env_id,
        num_envs=num_envs,
        vectorization_mode= vectorization_mode,
        vector_kwargs={
            "autoreset_mode": AutoresetMode.DISABLED,
        },
    )

    return NumpyToTorch(envs, device=device)