"""On-policy rollout storage and off-policy replay storage."""

from __future__ import annotations

from collections.abc import Iterator
from typing import NamedTuple

import torch
from tensordict import LazyStackedTensorDict, TensorDictBase

type TensorData = torch.Tensor | TensorDictBase


class RolloutBatch(NamedTuple):
    """One flat batch consumed by an on-policy update.

    B is T * N or a minibatch. Old policy outputs and GAE targets stay fixed
    throughout the update.
    """
    observations    :   TensorData      # each leaf: (B, ...)
    actions         :   TensorData      # each leaf: (B, ...)
    old_log_probs   :   torch.Tensor    # (B,) log pi_old(A|S)
    old_values      :   torch.Tensor    # (B,) V_old(S)
    advantages      :   torch.Tensor    # (B,)
    value_targets   :   torch.Tensor    # (B,)


@torch.no_grad()
def _allocate_storage(
    batch         : TensorData,
    *,
    leading_shape : tuple[int, ...],
    device        : torch.device,
) -> TensorData:
    """Allocate zeroed storage using the feature structure of a batch."""

    if isinstance(batch, torch.Tensor):
        return torch.zeros(
            (*leading_shape, *batch.shape[1:]),
            dtype=batch.dtype,
            device=device,
        )

    return batch[0].to(device).expand(*leading_shape).clone().zero_()


def _validate_batch(
    batch       : TensorData,
    *,
    batch_size  : int,
    name        : str,
    reference   : TensorData | None = None,
) -> None:
    """Check the batch axis and preserve the first insertion's leaf schema."""
    # Lazy stacks can hide keys that are not shared by every item.
    if isinstance(batch, LazyStackedTensorDict):
        raise TypeError(f"{name} must use a homogeneous dense TensorDict, not a lazy stack")
    if not isinstance(batch, (torch.Tensor, TensorDictBase)):
        raise TypeError(f"{name} must be a Tensor or TensorDict")
    if len(batch.shape) == 0 or batch.shape[0] != batch_size:
        raise ValueError(
            f"{name} must have leading dimension {batch_size}, "
            f"got shape {tuple(batch.shape)}"
        )
    if reference is not None and (
        isinstance(batch, torch.Tensor) != isinstance(reference, torch.Tensor)
    ):
        raise TypeError(f"{name} must keep its Tensor or TensorDict structure")

    if isinstance(batch, torch.Tensor):
        if isinstance(reference, torch.Tensor):
            if batch.shape != reference.shape:
                raise ValueError(
                    f"{name} must have shape {tuple(reference.shape)}, "
                    f"got {tuple(batch.shape)}"
                )
            if batch.dtype != reference.dtype:
                raise ValueError(
                    f"{name} must have dtype {reference.dtype}, got {batch.dtype}"
                )
        return

    if batch.batch_size != (batch_size,):
        raise ValueError(f"{name} must have TensorDict batch_size ({batch_size},)")
    if isinstance(reference, TensorDictBase) and set(batch.keys()) != set(reference.keys()):
        raise ValueError(f"{name} must keep the same TensorDict keys")
    for key, value in batch.items():
        _validate_batch(
            value,
            batch_size=batch_size,
            name=f"{name}.{key}",
            reference=reference[key] if reference is not None else None,
        )


class RolloutBuffer:
    """Own one time-major rollout of T steps from N environments.

    The buffer stores transitions and collection-time policy outputs. It does
    not call environments or networks and does not compute training targets.
    Dense TensorDicts preserve nested fields. Leaf shapes and dtypes are fixed
    by the first step, including across resets.
    """

    def __init__(
        self,
        rollout_steps       :   int,
        num_envs            :   int,
        device              :   torch.device
    ) -> None:
        if rollout_steps <= 0:
            raise ValueError(f"rollout_steps must be positive, got {rollout_steps}")
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")

        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self.device = device

        T, N = self.rollout_steps, self.num_envs

        # Observation and action structures are learned from the first step.
        self.observations: TensorData | None = None
        self.actions: TensorData | None = None
        self.next_observations: TensorData | None = None

        # Scalar transition data has a known time-major shape.
        self.rewards = torch.zeros((T, N), dtype=torch.float32, device=device)
        self.terminations = torch.zeros((T, N), dtype=torch.bool, device=device)
        self.truncations = torch.zeros((T, N), dtype=torch.bool, device=device)
        self.values = torch.zeros((T, N), dtype=torch.float32, device=device)
        self.log_probs = torch.zeros((T, N), dtype=torch.float32, device=device)

        # pos identifies the next unwritten time row.
        self.pos = 0
        self.full = False

    def reset(self) -> None:
        """Reset the write position without reallocating rollout storage."""
        self.pos = 0
        self.full = False

    @property
    def dones(self) -> torch.Tensor:
        """(T, N) episode boundaries. Derived, so it cannot disagree."""
        return torch.logical_or(self.terminations, self.truncations)

    @torch.no_grad()
    def add(
        self,
        *,
        observation         :   TensorData,     # each leaf: (N, ...)
        action              :   TensorData,     # each leaf: (N, ...)
        reward              :   torch.Tensor,   # (N,) float32
        next_observation    :   TensorData,     # each leaf: (N, ...)
        termination         :   torch.Tensor,   # (N,) bool
        truncation          :   torch.Tensor,   # (N,) bool
        value               :   torch.Tensor,   # (N,) float32
        log_prob            :   torch.Tensor    # (N,) float32
    ) -> None:
        """Store one vector-environment step at the current time index."""

        if self.full:
            raise RuntimeError(
                "rollout buffer is full; call reset() before collecting again"
            )

        N = self.num_envs

        # Validate all fields before writing, including nested keys and shapes.
        for name, batch, storage in (
            ("observation", observation, self.observations),
            ("action", action, self.actions),
            ("next_observation", next_observation, self.next_observations),
        ):
            _validate_batch(
                batch,
                batch_size=N,
                name=name,
                reference=storage[0] if storage is not None else None,
            )

        # Scalar transition fields must have exactly one value per environment.
        scalar_fields = (
            ("reward", reward, self.rewards),
            ("termination", termination, self.terminations),
            ("truncation", truncation, self.truncations),
            ("value", value, self.values),
            ("log_prob", log_prob, self.log_probs),
        )
        for name, tensor, _ in scalar_fields:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a Tensor")
            if tensor.shape != (N,):
                raise ValueError(
                    f"{name} must have shape ({N},), "
                    f"got {tuple(tensor.shape)}"
                )

        # Storage structure and dtypes are inferred from the first rollout step.
        if self.observations is None:
            leading_shape = (self.rollout_steps, self.num_envs)

            self.observations = _allocate_storage(
                observation, leading_shape=leading_shape, device=self.device,
            )
            self.actions = _allocate_storage(
                action, leading_shape=leading_shape, device=self.device,
            )
            self.next_observations = _allocate_storage(
                next_observation, leading_shape=leading_shape, device=self.device,
            )
        assert self.actions is not None
        assert self.next_observations is not None

        # Copy the structured transition data into time row pos.
        self.observations[self.pos].copy_(observation.detach().to(self.device))
        self.actions[self.pos].copy_(action.detach().to(self.device))
        self.next_observations[self.pos].copy_(
            next_observation.detach().to(self.device)
        )
        for _, tensor, storage in scalar_fields:
            storage[self.pos].copy_(tensor.detach())

        self.pos += 1
        self.full = self.pos == self.rollout_steps

    def get(
        self,
        *,
        advantages      :   torch.Tensor,   # (T, N)
        value_targets   :   torch.Tensor,   # (T, N)
        batch_size      :   int | None = None,
    ) -> Iterator[RolloutBatch]:
        """Yield aligned flat batches from one completed rollout.

        GAE outputs are supplied after collection. None yields a full batch;
        an integer yields shuffled minibatches, including any final remainder.
        """

        observations = self.observations
        actions = self.actions

        if not self.full or observations is None or actions is None:
            raise RuntimeError("cannot get batches from an incomplete rollout")

        expected = (self.rollout_steps, self.num_envs)
        if advantages.shape != expected:
            raise ValueError(
                f"advantages must have shape {expected}, got {tuple(advantages.shape)}"
            )
        if value_targets.shape != expected:
            raise ValueError(
                f"value_targets must have shape {expected}, "
                f"got {tuple(value_targets.shape)}"
            )

        rollout_size = self.rollout_steps * self.num_envs

        if batch_size is None:
            batch_size = rollout_size
        elif batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        # Flatten (T, N, ...) -> (T * N, ...) identically for every field.
        observations = observations.flatten(0, 1)
        actions = actions.flatten(0, 1)
        old_log_probs = self.log_probs.flatten(0, 1)
        old_values = self.values.flatten(0, 1)
        advantages = advantages.flatten(0, 1)
        value_targets = value_targets.flatten(0, 1)

        # One permutation keeps observations, actions, and targets aligned.
        indices = torch.randperm(rollout_size, device=self.device)

        for start in range(0, rollout_size, batch_size):
            batch_indices = indices[start : start + batch_size]

            yield RolloutBatch(
                observations=observations[batch_indices],
                actions=actions[batch_indices],
                old_log_probs=old_log_probs[batch_indices],
                old_values=old_values[batch_indices],
                advantages=advantages[batch_indices],
                value_targets=value_targets[batch_indices],
            )


class ReplayBatch(NamedTuple):
    """One flat batch sampled for an off-policy update.

    Each algorithm computes its own Bellman targets from these transitions.
    """

    observations        :   TensorData      # each leaf: (B, ...)
    actions             :   TensorData      # each leaf: (B, ...)
    rewards             :   torch.Tensor    # (B,) float32
    next_observations   :   TensorData      # each leaf: (B, ...)
    terminations        :   torch.Tensor    # (B,) bool
    truncations         :   torch.Tensor    # (B,) bool


class ReplayBuffer:
    """Own fixed-capacity circular storage for off-policy transitions.

    The buffer stores complete transitions and samples them uniformly. It does
    not interact with environments, networks, or algorithm-specific targets.
    Dense TensorDicts preserve nested fields. Leaf shapes and dtypes are fixed
    by the first batch, including across resets.
    """

    def __init__(
        self,
        capacity            :   int,
        sample_device       :   torch.device,
        storage_device      :   torch.device | None = None
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")

        self.capacity = capacity
        self.sample_device = sample_device

        # Large replay storage defaults to CPU; sampled batches are transferred
        # to sample_device immediately before an update.
        self.storage_device = torch.device("cpu") if storage_device is None else storage_device

        C = self.capacity

        # Infer structured fields from the first batch, then allocate C rows.
        self.observations: TensorData | None = None
        self.actions: TensorData | None = None
        self.next_observations: TensorData | None = None

        # Each scalar field has one entry per transition, not per environment.
        self.rewards = torch.zeros(C, dtype=torch.float32, device=self.storage_device)
        self.terminations = torch.zeros(C, dtype=torch.bool, device=self.storage_device)
        self.truncations = torch.zeros(C, dtype=torch.bool, device=self.storage_device)

        # pos is the next write location; size is the readable population.
        self.pos = 0
        self.size = 0

    def reset(self) -> None:
        """Reset replay metadata without reallocating storage."""
        self.pos = 0
        self.size = 0

    @property
    def full(self) -> bool:
        return self.size == self.capacity

    def __len__(self) -> int:
        """Return the number of transitions available for sampling."""
        return self.size

    @torch.no_grad()
    def add(
        self,
        *,
        observation     :   TensorData,     # each leaf: (K, ...)
        action          :   TensorData,     # each leaf: (K, ...)
        reward          :   torch.Tensor,   # (K,) float32
        next_observation:   TensorData,     # each leaf: (K, ...)
        termination     :   torch.Tensor,   # (K,) bool
        truncation      :   torch.Tensor,   # (K,) bool
    ) -> None:
        """Store K aligned transitions; a single transition needs a batch axis."""

        if not isinstance(observation, (torch.Tensor, TensorDictBase)):
            raise TypeError("observation must be a Tensor or TensorDict")
        if len(observation.shape) == 0:
            raise ValueError("observation must have a leading batch dimension")
        num_transitions = observation.shape[0]

        if num_transitions == 0:
            raise ValueError("cannot add an empty transition batch")

        if num_transitions > self.capacity:
            raise ValueError(
                f"cannot add {num_transitions} transitions to a replay buffer "
                f"with capacity {self.capacity}"
            )

        # Validate before allocation or writes, including nested keys and shapes.
        for name, batch, storage in (
            ("observation", observation, self.observations),
            ("action", action, self.actions),
            ("next_observation", next_observation, self.next_observations),
        ):
            _validate_batch(
                batch,
                batch_size=num_transitions,
                name=name,
                reference=storage[:num_transitions] if storage is not None else None,
            )

        scalar_fields = (
            ("reward", reward, self.rewards),
            ("termination", termination, self.terminations),
            ("truncation", truncation, self.truncations),
        )
        for name, tensor, _ in scalar_fields:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a Tensor")
            if tensor.shape != (num_transitions,):
                raise ValueError(
                    f"{name} must have shape ({num_transitions},), "
                    f"got {tuple(tensor.shape)}"
                )

        # Allocate C transition rows using the first batch's feature structure.
        if self.observations is None:
            leading_shape = (self.capacity,)
            self.observations = _allocate_storage(
                observation,
                leading_shape=leading_shape,
                device=self.storage_device,
            )
            self.actions = _allocate_storage(
                action,
                leading_shape=leading_shape,
                device=self.storage_device,
            )
            self.next_observations = _allocate_storage(
                next_observation,
                leading_shape=leading_shape,
                device=self.storage_device,
            )
        assert self.actions is not None
        assert self.next_observations is not None

        indices = (
            self.pos + torch.arange(num_transitions, device=self.storage_device)
        ) % self.capacity

        # Indexed assignment writes into storage; copy_ on an indexed tensor
        # would only modify the temporary gathered copy.
        self.observations[indices] = observation.detach().to(self.storage_device)
        self.actions[indices] = action.detach().to(self.storage_device)
        self.next_observations[indices] = next_observation.detach().to(self.storage_device)
        for _, tensor, storage in scalar_fields:
            storage[indices] = tensor.detach().to(
                dtype=storage.dtype,
                device=storage.device,
            )

        self.pos = (self.pos + num_transitions) % self.capacity
        self.size = min(self.size + num_transitions, self.capacity)

    def sample(
        self,
        batch_size  :   int,
    ) -> ReplayBatch:
        """Sample uniformly with replacement from the stored transitions.

        The requested batch may exceed the current buffer size; in that case,
        transitions can appear more than once in the returned batch.
        """

        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if self.size == 0:
            raise RuntimeError("cannot sample from an empty replay buffer")

        assert self.observations is not None
        assert self.actions is not None
        assert self.next_observations is not None

        # Every valid transition is in [0, size) before the first wrap. Once the
        # buffer wraps, size equals capacity and every storage location is valid.
        indices = torch.randint(
            low=0,
            high=self.size,
            size=(batch_size,),
            device=self.storage_device,
        )

        # Gather on the storage device, then transfer only the sampled data to
        # the learner's device.
        return ReplayBatch(
            observations=self.observations[indices].to(device=self.sample_device),
            actions=self.actions[indices].to(device=self.sample_device),
            rewards=self.rewards[indices].to(device=self.sample_device),
            next_observations=self.next_observations[indices].to(
                device=self.sample_device
            ),
            terminations=self.terminations[indices].to(device=self.sample_device),
            truncations=self.truncations[indices].to(device=self.sample_device),
        )
