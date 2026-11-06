"""The boundary between Gymnasium environments and the library's tensors.

Gymnasium environments speak NumPy: observations are arrays, dicts or tuples,
and actions must come back in the same form. The rest of the library speaks
Torch: plain tensors, or ``TensorDict`` for structured data. This module
converts between the two and creates vectorized environments.

* ``to_tensor_data``: Gymnasium data -> tensors (environment to library).
* ``from_tensor_data``: tensors -> Gymnasium data (library to environment).
* ``make_vec_env``: N environments that return tensors and never reset
  themselves.

The conversions keep the data as it is: shapes, feature structure and dtypes
are not changed (no flattening, no casting to float). Any processing happens
later, in the ``ObservationProcessor``.

Example:
    A Dict observation from 8 environments::

        {"joints": array (8, 12), "camera": array (8, 64, 64, 3) uint8}

    becomes a ``TensorDict`` with ``batch_size=[8]`` and the same two keys,
    with the camera still ``uint8``.
"""

import gymnasium as gym
import torch
from gymnasium.envs.registration import VectorizeMode
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.wrappers.vector.numpy_to_torch import NumpyToTorch
from tensordict import TensorDict, TensorDictBase

from deeprl.data_spec import TensorData


def to_tensor_data(
    data: object,
    *,
    batch_size: int | None = None,
    device: torch.device | None = None,
) -> TensorData:
    """Convert Gymnasium data into a tensor or a ``TensorDict``.

    * An array or tensor becomes a tensor with the same shape and dtype.
    * A dict becomes a ``TensorDict`` with the same keys.
    * A tuple becomes a ``TensorDict`` with the keys ``"0"``, ``"1"``, and
      so on, one per entry.

    Nested dicts and tuples are converted the same way, level by level.

    Every result has a batch axis in front:

    * Vector environments already return a batch of N, so pass
      ``batch_size=N`` and the shapes are kept as they are.
    * A single environment returns one sample without a batch axis. Leave
      ``batch_size`` out and a batch axis of size 1 is added.

    Args:
        data: An observation or action: an array, a tensor, a dict, a tuple
            or a ``TensorDict``.
        batch_size: N if ``data`` already has a batch axis of size N, or
            ``None`` to add a batch axis of size 1.
        device: Device to put the result on. ``None`` keeps the current
            device.

    Returns:
        A tensor shaped ``(B, *feature_shape)``, or a ``TensorDict`` with
        ``batch_size=[B]``, where B is ``batch_size`` or 1. The result may
        share memory with ``data``; copy it before keeping it if the
        environment could overwrite ``data``.
    """
    # 1. Structured data (dict or tuple): convert each entry, then collect the
    #    entries in a TensorDict. Tuple entries get the keys "0", "1", ...
    #    because TensorDict keys must be strings.
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

    # 2. A single tensor or array. A TensorDict is already converted, so it is
    #    only moved. Anything else becomes a tensor with its original dtype.
    if isinstance(data, TensorDictBase):
        tensor = data if device is None else data.to(device)
    else:
        tensor = torch.as_tensor(data, device=device)

    # 3. Add a batch axis of size 1 for a single environment's data.
    return tensor.unsqueeze(0) if batch_size is None else tensor


def from_tensor_data(
    data: TensorData,
    space: gym.Space,
    *,
    numpy: bool = False,
) -> object:
    """Convert tensors back into the form a Gymnasium space expects.

    This is the reverse of ``to_tensor_data``, mostly used for actions. The
    ``space`` gives the target structure: a ``Dict`` space gets a dict back,
    a ``Tuple`` space gets a tuple back, in the space's order.

    Args:
        data: A tensor or a ``TensorDict``, for example the policy's actions.
        space: The space to match, for example ``envs.single_action_space``.
        numpy: ``False`` keeps tensors, for environments wrapped in
            ``NumpyToTorch`` (as ``make_vec_env`` does). ``True`` makes
            NumPy arrays, and Python integers for ``Discrete`` spaces, for a
            plain single Gymnasium environment. With ``True``, remove the
            batch axis first (take ``actions[0]``).

    Returns:
        A dict, a tuple, a tensor, a NumPy array or an integer, matching
        ``space``.
    """
    # 1. Dict space: rebuild the dict, one entry per key of the space.
    if isinstance(space, gym.spaces.Dict):
        return {
            key: from_tensor_data(data[key], subspace, numpy=numpy)
            for key, subspace in space.spaces.items()
        }

    # 2. Tuple space: rebuild the tuple from the keys "0", "1", ...
    if isinstance(space, gym.spaces.Tuple):
        return tuple(
            from_tensor_data(data[str(index)], subspace, numpy=numpy)
            for index, subspace in enumerate(space.spaces)
        )

    # 3. A single leaf, for an environment that accepts tensors: unchanged.
    if not numpy:
        return data

    # 4. A single leaf, for a NumPy environment: a Discrete action becomes a
    #    Python integer; anything else becomes a NumPy array on the CPU.
    if isinstance(space, gym.spaces.Discrete):
        return int(data.item())
    return data.detach().cpu().numpy()


def make_vec_env(
    env_id              : str,
    num_envs            : int,
    device              : torch.device = torch.device("cpu"),
    vectorization_mode  : VectorizeMode = VectorizeMode.SYNC,
) -> VectorEnv:
    """Create N copies of an environment that return tensors and never reset themselves.

    * **Torch outputs**: observations, rewards and flags come back as tensors
      on ``device``, and actions can be sent as tensors.
    * **No automatic resets**: when an episode ends, the environment waits.
      The collector stores the true final observation first and then resets
      only the finished environments (see ``collectors.py``).

    Observations and actions keep their structure. Rewards and the
    termination and truncation flags have shape ``(N,)``.

    Args:
        env_id: A registered Gymnasium id, for example ``"Hopper-v5"``.
        num_envs: N, the number of environment copies.
        device: Device for the returned tensors.
        vectorization_mode: ``SYNC`` steps the copies one after another in
            this process. ``ASYNC`` runs each copy in its own process, which
            helps when one environment step is slow.

    Returns:
        A vector environment with N copies.

    Raises:
        ValueError: If ``num_envs`` is not positive.
    """
    # 1. At least one environment is needed.
    if num_envs <= 0:
        raise ValueError(
            f"num_envs must be positive, got {num_envs}"
        )

    # 2. Create N copies with automatic resets turned off.
    envs = gym.make_vec(
        env_id,
        num_envs=num_envs,
        vectorization_mode=vectorization_mode,
        vector_kwargs={
            "autoreset_mode": AutoresetMode.DISABLED,
        },
    )

    # 3. Convert NumPy outputs to tensors on ``device``, and tensor actions
    #    back to NumPy, at every step.
    return NumpyToTorch(envs, device=device)
