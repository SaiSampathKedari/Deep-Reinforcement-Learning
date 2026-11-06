"""Plain JSON contracts for fixed-shape Tensor and TensorDict features.

Validation never casts, reshapes or changes data. This initial contract covers
feature semantics, not dataset record fields, temporal metadata or action chunks.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

import torch
from tensordict import LazyStackedTensorDict, TensorDictBase

type TensorData = torch.Tensor | TensorDictBase


def _validate_shape(shape: tuple[int, ...], name: str) -> None:
    """Require fixed, nonnegative dimensions without coercing the declaration."""
    if not isinstance(shape, tuple) or any(
        type(size) is not int or size < 0 for size in shape
    ):
        raise ValueError(f"{name} must be a tuple of nonnegative integers")


@dataclass(frozen=True)
class FeatureSpec:
    """One named feature; shape excludes all leading batch dimensions.

    Dtypes use canonical PyTorch names, such as ``float32`` or ``uint8``.
    Slash-delimited names address nested TensorDict keys. Unknown semantics
    remain unspecified; no units or frames are inferred. Groups are actor,
    critic and privileged. Privileged features are critic-only and cannot be
    assigned to the actor; an empty group list is shared.
    """

    name: str
    shape: tuple[int, ...]
    dtype: str
    modality: str = "state"
    units: str | None = None
    frame: str | None = None
    ordering: tuple[str, ...] = ()
    groups: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or any(not part for part in self.name.split("/")):
            raise ValueError("feature name must contain nonempty slash-delimited keys")
        _validate_shape(self.shape, f"{self.name}.shape")
        dtype = getattr(torch, self.dtype, None) if isinstance(self.dtype, str) else None
        if not isinstance(dtype, torch.dtype) or str(dtype) != f"torch.{self.dtype}":
            raise ValueError(f"{self.name}.dtype must be a canonical PyTorch dtype name")
        for name in ("modality", "units", "frame"):
            value = getattr(self, name)
            if value is None and name != "modality":
                continue
            if not isinstance(value, str) or not value:
                raise ValueError(f"{self.name}.{name} must be a nonempty string")
        for name in ("ordering", "groups"):
            value = getattr(self, name)
            if not isinstance(value, tuple) or any(
                not isinstance(item, str) or not item for item in value
            ):
                raise ValueError(f"{self.name}.{name} must be a tuple of nonempty strings")
            if len(set(value)) != len(value):
                raise ValueError(f"{self.name}.{name} must not contain duplicates")
        if set(self.groups) - {"actor", "critic", "privileged"}:
            raise ValueError(f"{self.name}.groups must use actor, critic or privileged")
        if "actor" in self.groups and "privileged" in self.groups:
            raise ValueError(f"{self.name}.groups cannot combine actor and privileged")
        if self.ordering and (not self.shape or len(self.ordering) != self.shape[-1]):
            raise ValueError(f"{self.name}.ordering must name each element of the last axis")

    def to_dict(self) -> dict[str, Any]:
        """Return independent JSON-compatible feature data."""
        return {
            field.name: (
                list(value)
                if isinstance(value := getattr(self, field.name), tuple)
                else value
            )
            for field in fields(self)
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FeatureSpec:
        """Read current feature fields directly, rejecting unknown fields."""
        if not isinstance(data, Mapping):
            raise ValueError("feature spec must be a mapping")
        allowed = {field.name for field in fields(cls)}
        if set(data) - allowed or {"name", "shape", "dtype"} - set(data):
            raise ValueError("feature spec has missing or unknown fields")
        values = dict(data)
        for name in ("shape", "ordering", "groups"):
            if name in values:
                if not isinstance(values[name], (list, tuple)):
                    raise ValueError(f"feature {name} must be an array")
                values[name] = tuple(values[name])
        return cls(**values)


@dataclass(frozen=True)
class DataSpec:
    """Ordered observation/action feature declarations, with no fitted state."""

    observations: tuple[FeatureSpec, ...]
    actions: tuple[FeatureSpec, ...]

    def __post_init__(self) -> None:
        for name in ("observations", "actions"):
            features = getattr(self, name)
            if not isinstance(features, tuple) or any(
                not isinstance(feature, FeatureSpec) for feature in features
            ):
                raise TypeError(f"{name} must be a tuple of FeatureSpec values")
            names = {feature.name for feature in features}
            if len(names) != len(features):
                raise ValueError(f"{name} contains duplicate feature names")
            for path in names:
                parts = path.split("/")
                if any("/".join(parts[:index]) in names for index in range(1, len(parts))):
                    raise ValueError(f"{name} contains conflicting feature paths: {path}")
        if not self.observations and not self.actions:
            raise ValueError("data spec must declare at least one feature")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible data without a format version or registry."""
        return {
            "observations": [feature.to_dict() for feature in self.observations],
            "actions": [feature.to_dict() for feature in self.actions],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DataSpec:
        """Read exactly the two current feature lists."""
        if not isinstance(data, Mapping) or set(data) != {"observations", "actions"}:
            raise ValueError("data spec must contain exactly observations and actions")
        if any(not isinstance(data[name], (list, tuple)) for name in data):
            raise ValueError("data spec observations and actions must be arrays")
        return cls(
            observations=tuple(
                FeatureSpec.from_dict(item) for item in data["observations"]
            ),
            actions=tuple(FeatureSpec.from_dict(item) for item in data["actions"]),
        )

    def validate_observations(
        self,
        data: TensorData,
        *,
        batch_shape: tuple[int, ...],
    ) -> None:
        """Check observation keys, storage shapes and dtypes without conversion."""
        _validate_data(data, self.observations, batch_shape, "observations")

    def validate_actions(
        self,
        data: TensorData,
        *,
        batch_shape: tuple[int, ...],
    ) -> None:
        """Check action keys, storage shapes and dtypes without conversion."""
        _validate_data(data, self.actions, batch_shape, "actions")

    def validate_compatible(self, other: DataSpec) -> None:
        """Reject structural or semantic differences; mappings must be explicit."""
        if not isinstance(other, DataSpec):
            raise TypeError("other must be a DataSpec")
        for name in ("observations", "actions"):
            expected, actual = getattr(self, name), getattr(other, name)
            if len(expected) != len(actual):
                raise ValueError(f"incompatible {name}: feature counts differ")
            for left, right in zip(expected, actual):
                for field in fields(FeatureSpec):
                    if getattr(left, field.name) != getattr(right, field.name):
                        raise ValueError(
                            f"incompatible {name}.{left.name}.{field.name}: "
                            f"{getattr(left, field.name)!r} != {getattr(right, field.name)!r}"
                        )


def _tensor_leaves(
    data: TensorDictBase,
    batch_shape: tuple[int, ...],
    prefix: tuple[str, ...] = (),
) -> dict[tuple[str, ...], torch.Tensor]:
    """Collect dense tensor leaves while checking every nested batch shape."""
    if isinstance(data, LazyStackedTensorDict):
        raise TypeError("data must use a homogeneous dense TensorDict, not a lazy stack")
    if tuple(data.batch_size) != batch_shape:
        raise ValueError(f"TensorDict {'/'.join(prefix)} must have batch_shape {batch_shape}")
    if prefix and not len(data.keys()):
        raise ValueError(f"TensorDict keys include an undeclared empty branch: {'/'.join(prefix)}")
    leaves = {}
    for key, value in data.items():
        path = (*prefix, key)
        if isinstance(value, TensorDictBase):
            leaves.update(_tensor_leaves(value, batch_shape, path))
        elif isinstance(value, torch.Tensor):
            leaves[path] = value
        else:
            raise TypeError(f"{'/'.join(path)} must be a Tensor feature")
    return leaves


def _validate_data(
    data: TensorData,
    features: tuple[FeatureSpec, ...],
    batch_shape: tuple[int, ...],
    name: str,
) -> None:
    """Match storage to its declared keys and batch-plus-feature shapes."""
    _validate_shape(batch_shape, "batch_shape")
    if isinstance(data, torch.Tensor):
        if len(features) != 1:
            raise ValueError(f"{name}: a Tensor requires a single declared feature")
        leaves = {tuple(features[0].name.split("/")): data}
    elif isinstance(data, TensorDictBase):
        leaves = _tensor_leaves(data, batch_shape)
        expected = {tuple(feature.name.split("/")) for feature in features}
        if set(leaves) != expected:
            raise ValueError(
                f"{name} keys differ: missing={sorted(expected - set(leaves))}, "
                f"unexpected={sorted(set(leaves) - expected)}"
            )
    else:
        raise TypeError(f"{name} must be a Tensor or TensorDict")
    for feature in features:
        value = leaves[tuple(feature.name.split("/"))]
        expected_shape = (*batch_shape, *feature.shape)
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"{name}.{feature.name} shape must be {expected_shape}, got {tuple(value.shape)}"
            )
        if value.dtype != getattr(torch, feature.dtype):
            raise ValueError(
                f"{name}.{feature.name} dtype must be {feature.dtype}, got {value.dtype}"
            )
