import gymnasium as gym
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.wrappers.vector.numpy_to_torch import NumpyToTorch


def make_vec_env(
    env_id: str,
    num_envs: int,
    device: str = "cpu",
) -> VectorEnv:

    if num_envs <= 0:
        raise ValueError(
            f"num_envs must be positive, got {num_envs}"
        )

    envs = gym.make_vec(
        env_id,
        num_envs=num_envs,
        vectorization_mode="sync",
        vector_kwargs={
            "autoreset_mode": AutoresetMode.DISABLED,
        },
    )

    return NumpyToTorch(envs, device=device)