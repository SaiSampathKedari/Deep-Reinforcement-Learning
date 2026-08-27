"""On-policy rollout storage and off-policy replay storage."""

from __future__ import annotations

from collections.abc import Iterator
from typing import NamedTuple

import gymnasium as gym
import torch


class RolloutBatch(NamedTuple):
    """One flat batch consumed by an on-policy update.

    B is either the complete T * N rollout or one minibatch. Stored policy
    outputs describe the policy that collected the rollout; advantages and
    value targets are fixed estimator outputs computed after collection.
    """
    observations    :   torch.Tensor    # (B, *obs_shape)
    actions         :   torch.Tensor    # (B, *action_shape)
    old_log_probs   :   torch.Tensor    # (B,) log pi_old(A|S)
    old_values      :   torch.Tensor    # (B,) V_old(S)
    advantages      :   torch.Tensor    # (B,)
    value_targets   :   torch.Tensor    # (B,)


def _space_dtype(space : gym.Space) -> torch.dtype:
    """Return the Torch dtype produced by a Gymnasium space."""
    return torch.as_tensor(space.sample()).dtype


class RolloutBuffer:
    """Own one time-major rollout of T steps from N environments.

    The buffer stores transitions and collection-time policy outputs. It does
    not call environments or networks and does not compute training targets.
    """

    def __init__(
        self,
        rollout_steps       :   int,
        num_envs            :   int,
        observation_space   :   gym.Space,
        action_space        :   gym.Space,
        device              :   torch.device
    ):
        # --------------------------------------------------------------
        # Rollout geometry and dtypes
        # --------------------------------------------------------------

        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self.device = device

        # Observation and action tensors preserve their Gymnasium space dtypes.
        observation_dtype = _space_dtype(observation_space)
        observation_shape = observation_space.shape

        action_dtype = _space_dtype(action_space)
        action_shape = action_space.shape

        T, N = self.rollout_steps, self.num_envs

        # --------------------------------------------------------------
        # Time-major storage
        # --------------------------------------------------------------

        # Every row stores one vector-environment step containing N aligned
        # transitions. Scalar transition data therefore has shape (T, N).
        self.observations       = torch.zeros((T, N, *observation_shape), dtype=observation_dtype, device=self.device )
        self.actions            = torch.zeros((T, N, *action_shape), dtype=action_dtype, device=self.device)
        self.rewards            = torch.zeros(T, N, dtype=torch.float32, device=self.device)
        self.next_observations  = torch.zeros((T, N, *observation_shape), dtype=observation_dtype, device=self.device)
        self.terminations       = torch.zeros(T, N, dtype=torch.bool, device=self.device)
        self.truncations        = torch.zeros(T, N, dtype=torch.bool, device=self.device)
        self.values             = torch.zeros(T, N, dtype=torch.float32, device=self.device)
        self.log_probs          = torch.zeros(T, N, dtype=torch.float32, device=self.device)

        # pos identifies the next unwritten time row. Storage is reusable and
        # becomes readable only after all T rows have been written.
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

    def add(
        self,
        *,
        observation         :   torch.Tensor,   # (N, *obs_shape)
        action              :   torch.Tensor,   # (N, *action_shape)
        reward              :   torch.Tensor,   # (N,) float32
        next_observation    :   torch.Tensor,   # (N, *obs_shape)
        termination         :   torch.Tensor,   # (N,) bool
        truncation          :   torch.Tensor,   # (N,) bool
        value               :   torch.Tensor,   # (N,) float32
        log_prob            :   torch.Tensor    # (N,) float32
    ) -> None:
        """Store one vector-environment step.

        Every argument represents N environments. Data is copied into owned
        storage, so later environment resets cannot mutate stored observations.
        """
        # One add() call must fit in the next unwritten time row.
        assert self.pos < self.rollout_steps

        # Validate each input against one storage row, then copy all fields at
        # the same time index so the transition remains aligned.
        for name, tensor, storage in (
            ("observation", observation, self.observations),
            ("action", action, self.actions),
            ("reward", reward, self.rewards),
            ("next_observation", next_observation, self.next_observations),
            ("termination", termination, self.terminations),
            ("truncation", truncation, self.truncations),
            ("value", value, self.values),
            ("log_prob", log_prob, self.log_probs),
        ):
            if tensor.shape != storage.shape[1:]:
                raise ValueError(
                    f"{name} must have shape {tuple(storage.shape[1:])}, "
                    f"got {tuple(tensor.shape)}"
                )
            storage[self.pos] = tensor

        # Advance to the next time row; the rollout is complete after T writes.
        self.pos += 1
        self.full = self.pos == self.rollout_steps


    def get(
        self,
        *,
        advantages      :   torch.Tensor, # (T, N)
        value_targets   :   torch.Tensor, # (T, N)
        batch_size      :   int | None = None
    ) -> Iterator[RolloutBatch]:
        """Yield aligned flat batches from one completed rollout.

        Advantages and value targets are supplied here because they are
        computed only after collection. ``batch_size=None`` yields one full
        A2C batch; an integer yields shuffled minibatches for algorithms such
        as PPO.
        """

        # --------------------------------------------------------------
        # 1. Validate the completed rollout and estimator outputs
        # --------------------------------------------------------------

        assert self.full

        expected = (self.rollout_steps, self.num_envs)
        if advantages.shape != expected:
            raise ValueError(
                f"advantages must have shape {expected}, got {tuple(advantages.shape)}"
            )
        if value_targets.shape != expected:
            raise ValueError(
                f"value_targets must have shape {expected}, got {tuple(value_targets.shape)}"
            )

        rollout_size = self.rollout_steps * self.num_envs

        # --------------------------------------------------------------
        # 2. Flatten time and environment dimensions
        # --------------------------------------------------------------

        # Every field uses the same mapping: (T, N, ...) -> (B, ...), where
        # B = T * N. Corresponding transition fields therefore stay aligned.
        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        old_log_probs = self.log_probs.flatten(0, 1)
        old_values = self.values.flatten(0, 1)
        advantages = advantages.flatten(0, 1)
        value_targets = value_targets.flatten(0,1)

        # --------------------------------------------------------------
        # 3. Construct the batch order
        # --------------------------------------------------------------

        # A2C, NPG, and TRPO use the complete rollout.
        if batch_size is None:
            batch_size = rollout_size

        # One shared permutation preserves alignment across every field. For a
        # full-batch A2C update, permutation does not change a mean loss.
        indices = torch.randperm(
            rollout_size,
            device=self.device
        )

        # --------------------------------------------------------------
        # 4. Yield full batches or minibatches
        # --------------------------------------------------------------

        for start in range(0, rollout_size, batch_size):
            batch_indices = indices[start:start+batch_size]

            # Index every flattened field with the same sample indices.
            yield RolloutBatch(
                observations=observations[batch_indices],
                actions=actions[batch_indices],
                old_log_probs=old_log_probs[batch_indices],
                old_values=old_values[batch_indices],
                advantages=advantages[batch_indices],
                value_targets=value_targets[batch_indices],
            )


class ReplayBatch(NamedTuple):
    """One flat batch consumed by an off-policy update.

    B transitions are sampled independently from replay storage. Targets are
    intentionally absent because each algorithm computes its own Bellman target.
    """

    observations        :   torch.Tensor    # (B, *obs_shape)
    actions             :   torch.Tensor    # (B, *action_shape)
    rewards             :   torch.Tensor    # (B,) float32
    next_observations   :   torch.Tensor    # (B, *obs_shape)
    terminations        :   torch.Tensor    # (B,) bool
    truncations         :   torch.Tensor    # (B,) bool


class ReplayBuffer:
    """Own fixed-capacity circular storage for off-policy transitions.

    The buffer stores complete transitions and samples them uniformly. It does
    not interact with environments, networks, or algorithm-specific targets.
    """

    def __init__(
        self,
        capacity            :   int,
        observation_space   :   gym.Space,
        action_space        :   gym.Space,
        sample_device       :   torch.device,
        storage_device      :   torch.device | None = None
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")

        # --------------------------------------------------------------
        # Replay geometry and devices
        # --------------------------------------------------------------

        self.capacity = capacity
        self.sample_device = sample_device

        # Large replay storage defaults to CPU; sampled batches are transferred
        # to sample_device immediately before an update.
        self.storage_device = (
            torch.device("cpu")
            if storage_device is None
            else storage_device
        )

        observation_dtype = _space_dtype(observation_space)
        observation_shape = observation_space.shape

        action_dtype = _space_dtype(action_space)
        action_shape = action_space.shape

        C = self.capacity

        # --------------------------------------------------------------
        # Flat transition storage
        # --------------------------------------------------------------

        # Each row stores one complete transition. C is the maximum number of
        # transitions retained before new data begins overwriting old data.
        self.observations = torch.zeros(
            (C, *observation_shape),
            dtype=observation_dtype,
            device=self.storage_device,
        )
        self.actions = torch.zeros(
            (C, *action_shape),
            dtype=action_dtype,
            device=self.storage_device,
        )
        self.rewards = torch.zeros(
            C,
            dtype=torch.float32,
            device=self.storage_device,
        )
        self.next_observations = torch.zeros(
            (C, *observation_shape),
            dtype=observation_dtype,
            device=self.storage_device,
        )
        self.terminations = torch.zeros(
            C,
            dtype=torch.bool,
            device=self.storage_device,
        )
        self.truncations = torch.zeros(
            C,
            dtype=torch.bool,
            device=self.storage_device,
        )

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
        observation     :   torch.Tensor,   # (M, *obs_shape)
        action          :   torch.Tensor,   # (M, *action_shape)
        reward          :   torch.Tensor,   # (M,) float32
        next_observation:   torch.Tensor,   # (M, *obs_shape)
        termination     :   torch.Tensor,   # (M,) bool
        truncation      :   torch.Tensor,   # (M,) bool
    ) -> None:
        """Store M aligned transitions in circular replay storage."""

        # --------------------------------------------------------------
        # 1. Circular write locations
        # --------------------------------------------------------------

        num_transitions = observation.shape[0]

        if num_transitions > self.capacity:
            raise ValueError(
                f"cannot add {num_transitions} transitions to a replay buffer "
                f"with capacity {self.capacity}"
            )

        # Modulo indexing wraps a write across the end of storage.
        indices = (
            self.pos + torch.arange(
                num_transitions,
                device=self.storage_device,
            )
        ) % self.capacity

        fields = (
            ("observation", observation, self.observations),
            ("action", action, self.actions),
            ("reward", reward, self.rewards),
            ("next_observation", next_observation, self.next_observations),
            ("termination", termination, self.terminations),
            ("truncation", truncation, self.truncations),
        )

        # --------------------------------------------------------------
        # 2. Validate and copy the transition batch
        # --------------------------------------------------------------

        # Validate the complete transition batch before changing any storage.
        for name, tensor, storage in fields:
            expected_shape = (num_transitions, *storage.shape[1:])
            if tensor.shape != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}, "
                    f"got {tuple(tensor.shape)}"
                )

        # Copy every field with the same indices to preserve alignment.
        for _, tensor, storage in fields:
            storage[indices] = tensor.detach().to(
                dtype=storage.dtype,
                device=storage.device,
            )

        # --------------------------------------------------------------
        # 3. Advance replay metadata
        # --------------------------------------------------------------

        self.pos = (self.pos + num_transitions) % self.capacity

        self.size = min(
            self.size + num_transitions,
            self.capacity
        )

    def sample(
        self,
        batch_size  :   int,
    ) -> ReplayBatch:
        """Sample uniformly with replacement from the stored transitions.

        The requested batch may exceed the current buffer size; in that case,
        transitions can appear more than once in the returned batch.
        """

        if self.size == 0:
            raise RuntimeError("cannot sample from an empty replay buffer")

        # --------------------------------------------------------------
        # 1. Uniform transition sampling
        # --------------------------------------------------------------

        # Every valid transition is in [0, size) before the first wrap. Once the
        # buffer wraps, size equals capacity and every storage location is valid.
        indices = torch.randint(
            low=0,
            high=self.size,
            size=(batch_size,),
            device=self.storage_device,
        )

        # --------------------------------------------------------------
        # 2. Assemble the training batch
        # --------------------------------------------------------------

        # Gather on the storage device, then transfer only the sampled data to
        # the learner's device.
        return ReplayBatch(
            observations=self.observations[indices].to(
                device=self.sample_device
            ),
            actions=self.actions[indices].to(device=self.sample_device),
            rewards=self.rewards[indices].to(device=self.sample_device),
            next_observations=self.next_observations[indices].to(
                device=self.sample_device
            ),
            terminations=self.terminations[indices].to(
                device=self.sample_device
            ),
            truncations=self.truncations[indices].to(device=self.sample_device),
        )
