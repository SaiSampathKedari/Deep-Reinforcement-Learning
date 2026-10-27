import gymnasium as gym
import torch
from gymnasium.envs.registration import VectorizeMode
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.wrappers.vector.numpy_to_torch import NumpyToTorch
from tensordict import TensorDict, TensorDictBase

from deeprl.buffers import TensorData


def to_tensor_data(
    data: object,
    *,
    batch_size: int | None = None,
    device: torch.device | None = None,
) -> TensorData:
    """Convert fixed-shape Gym data without flattening or casting its leaves.

    ``batch_size=N`` preserves an existing vector batch. Omitting it adds one
    batch axis for a scalar environment. Tuple entries use string index keys.
    """
    if isinstance(data, (dict, tuple)):
        items = data.items() if isinstance(data, dict) else (
            (str(index), value) for index, value in enumerate(data)
        )
        return TensorDict(
            {
                key: to_tensor_data(value, batch_size=batch_size, device=device)
                for key, value in items
            },
            batch_size=[1 if batch_size is None else batch_size],
            device=device,
        )

    if isinstance(data, TensorDictBase):
        tensor = data if device is None else data.to(device)
    else:
        tensor = torch.as_tensor(data, device=device)
    return tensor.unsqueeze(0) if batch_size is None else tensor


def from_tensor_data(
    data: TensorData,
    space: gym.Space,
    *,
    numpy: bool = False,
) -> object:
    """Restore Gym containers, with tensor leaves or scalar-env NumPy leaves.

    With ``numpy=True``, remove the policy's batch axis before calling this
    function. Discrete leaves then become Python integers.
    """
    if isinstance(space, gym.spaces.Dict):
        return {
            key: from_tensor_data(data[key], subspace, numpy=numpy)
            for key, subspace in space.spaces.items()
        }
    if isinstance(space, gym.spaces.Tuple):
        return tuple(
            from_tensor_data(data[str(index)], subspace, numpy=numpy)
            for index, subspace in enumerate(space.spaces)
        )
    if not numpy:
        return data
    if isinstance(space, gym.spaces.Discrete):
        return int(data.item())
    return data.detach().cpu().numpy()


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
