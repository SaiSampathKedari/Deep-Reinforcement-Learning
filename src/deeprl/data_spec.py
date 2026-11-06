"""Data specifications: written descriptions of observation and action data.

A tensor is only numbers. A data specification records what those numbers
mean, so that every part of the library agrees on the data, and so that
mistakes fail loudly instead of silently.

This module has two classes:

* ``FeatureSpec`` describes one named part of the data, for example a robot's
  joint positions or one camera image: its shape and dtype, and optionally its
  units, coordinate frame, element names and which networks may read it.
* ``DataSpec`` groups all observation features and all action features of a
  task.

Specifications are saved with the data they describe, in two places:

* a policy folder stores its spec as JSON, inside ``manifest.json``;
* a training checkpoint stores it as a plain dictionary inside the
  checkpoint file (with the observation processor settings).

They are used to check real data. They never change the data.

Example:
    >>> spec = DataSpec(
    ...     observations=(FeatureSpec("state", (11,), "float32"),),
    ...     actions=(FeatureSpec("action", (3,), "float32"),),
    ... )
    >>> spec.validate_observations(torch.zeros(8, 11), batch_shape=(8,))
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

import torch
from tensordict import LazyStackedTensorDict, TensorDictBase

# A single tensor, or a TensorDict of named tensors for data with several parts
# (for example, a camera image together with joint positions).
type TensorData = torch.Tensor | TensorDictBase


def _validate_shape(shape: tuple[int, ...], name: str) -> None:
    """Check that a shape is a tuple of non-negative integers.

    Args:
        shape: The shape to check, for example ``(3, 224, 224)``.
        name: Label used in the error message.

    Raises:
        ValueError: If ``shape`` is not a tuple, or contains anything other
            than integers that are zero or larger.
    """
    # Lists such as [3, 224] are rejected on purpose: shapes are always tuples.
    # ``type(size) is not int`` also rejects booleans and floats as sizes.
    if not isinstance(shape, tuple) or any(
        type(size) is not int or size < 0 for size in shape
    ):
        raise ValueError(f"{name} must be a tuple of nonnegative integers")


@dataclass(frozen=True)
class FeatureSpec:
    """Description of one observation or action feature.

    A feature is one named part of the data, such as ``state``, ``joint_pos``
    or ``images/wrist``. The description says what the numbers are; it holds
    no data itself. All fields are checked when the object is created, and
    the object cannot be changed afterwards (``frozen=True``).

    Attributes:
        name: Key of the feature. A ``/`` refers to a nested key, so
            ``"images/wrist"`` means ``data["images"]["wrist"]``.
        shape: Size of one sample, without batch dimensions. For example
            ``(11,)`` for a state vector or ``()`` for one discrete action.
        dtype: Full PyTorch dtype name, such as ``"float32"`` or ``"uint8"``.
        modality: Kind of data, such as ``"state"`` or ``"image"``.
        units: Physical units, such as ``"rad"``, or ``None`` if unknown.
        frame: Coordinate frame, such as ``"base"``, or ``None`` if unknown.
        ordering: One name for each element of the last axis, such as joint
            names. Empty if the elements are not named.
        groups: Networks allowed to read the feature: ``"actor"``,
            ``"critic"`` or ``"privileged"``. An empty tuple means both actor
            and critic (shared). ``("actor",)`` or ``("critic",)`` limits the
            feature to that network. ``"privileged"`` marks information that
            is available during training but not to the deployed actor, for
            example the true friction from a simulator or an object pose from
            a motion-capture system. Only the critic may read it.

    Example:
        >>> FeatureSpec("joint_pos", (7,), "float32", units="rad")
    """

    # Required fields.
    name: str                          # Key, for example "state" or "images/wrist".
    shape: tuple[int, ...]             # One sample, no batch axes, for example (11,).
    dtype: str                         # Full PyTorch dtype name, for example "float32".

    # Optional fields that add meaning.
    modality: str = "state"            # Kind of data: "state", "image", ...
    units: str | None = None           # For example "rad"; None means unknown.
    frame: str | None = None           # For example "base"; None means unknown.
    ordering: tuple[str, ...] = ()     # One name per element of the last axis.
    groups: tuple[str, ...] = ()       # "actor", "critic", "privileged"; empty = both.

    def __post_init__(self) -> None:
        """Check every field when the object is created.

        A dataclass calls this method automatically right after ``__init__``.
        The first invalid field raises an error, so an invalid description can
        never be used later.

        Raises:
            ValueError: If any field is invalid.
        """
        # 1. The name must be text, and every part between "/" must be non-empty.
        #    "images/wrist" is valid; "" and "images//wrist" are not.
        if not isinstance(self.name, str) or any(not part for part in self.name.split("/")):
            raise ValueError("feature name must contain nonempty slash-delimited keys")

        # 2. The shape must be a tuple of non-negative integers.
        _validate_shape(self.shape, f"{self.name}.shape")

        # 3. The dtype must be the full name of a real PyTorch dtype.
        #    getattr(torch, "float32") returns torch.float32. Short aliases such
        #    as "float" also exist in torch, but str(torch.float) is
        #    "torch.float32", so the second comparison rejects them.
        dtype = getattr(torch, self.dtype, None) if isinstance(self.dtype, str) else None
        if not isinstance(dtype, torch.dtype) or str(dtype) != f"torch.{self.dtype}":
            raise ValueError(f"{self.name}.dtype must be a canonical PyTorch dtype name")

        # 4. Text fields must not be empty. modality is always required;
        #    units and frame may be None, which means "unknown".
        for name in ("modality", "units", "frame"):
            value = getattr(self, name)
            if value is None and name != "modality":
                continue
            if not isinstance(value, str) or not value:
                raise ValueError(f"{self.name}.{name} must be a nonempty string")

        # 5. ordering and groups must be tuples of non-empty, unique strings.
        for name in ("ordering", "groups"):
            value = getattr(self, name)
            if not isinstance(value, tuple) or any(
                not isinstance(item, str) or not item for item in value
            ):
                raise ValueError(f"{self.name}.{name} must be a tuple of nonempty strings")
            if len(set(value)) != len(value):
                raise ValueError(f"{self.name}.{name} must not contain duplicates")

        # 6. Only three group names exist. A typo such as "critc" would
        #    otherwise hide the feature from both networks without any error.
        if set(self.groups) - {"actor", "critic", "privileged"}:
            raise ValueError(f"{self.name}.groups must use actor, critic or privileged")

        # 7. Privileged data is available during training but not to the
        #    deployed actor (for example, simulator friction or motion-capture
        #    poses), so the actor must never be allowed to read it.
        if "actor" in self.groups and "privileged" in self.groups:
            raise ValueError(f"{self.name}.groups cannot combine actor and privileged")

        # 8. ordering names the elements of the last axis, so it needs exactly
        #    one name per element: 7 joint names for a shape of (7,).
        if self.ordering and (not self.shape or len(self.ordering) != self.shape[-1]):
            raise ValueError(f"{self.name}.ordering must name each element of the last axis")

    def to_dict(self) -> dict[str, Any]:
        """Return the feature as a JSON-ready dictionary.

        Tuples become lists because JSON has no tuple type. The result is a
        new dictionary, so changing it does not change this object.

        Example:
            >>> FeatureSpec("state", (11,), "float32").to_dict()["shape"]
            [11]
        """
        # Visit every dataclass field in declaration order. Tuple values
        # (shape, ordering, groups) become lists; other values are kept as is.
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
        """Create a feature from a dictionary made by :meth:`to_dict`.

        This is used when reading a specification back from JSON.

        Args:
            data: Field names and values. ``name``, ``shape`` and ``dtype``
                are required; the other fields are optional.

        Returns:
            A checked ``FeatureSpec``.

        Raises:
            ValueError: If a required field is missing, an unknown field is
                present, or a value is invalid.
        """
        # 1. The input must be a dictionary.
        if not isinstance(data, Mapping):
            raise ValueError("feature spec must be a mapping")

        # 2. Reject unknown fields (for example a misspelled "untis") and missing
        #    required fields, instead of silently ignoring or defaulting them.
        allowed = {field.name for field in fields(cls)}
        if set(data) - allowed or {"name", "shape", "dtype"} - set(data):
            raise ValueError("feature spec has missing or unknown fields")

        # 3. JSON stores tuples as lists, so convert them back. Working on a
        #    copy leaves the caller's dictionary unchanged.
        values = dict(data)
        for name in ("shape", "ordering", "groups"):
            if name in values:
                if not isinstance(values[name], (list, tuple)):
                    raise ValueError(f"feature {name} must be an array")
                values[name] = tuple(values[name])

        # 4. Create the object; this runs __post_init__, which checks every value.
        return cls(**values)


@dataclass(frozen=True)
class DataSpec:
    """Description of all observation and action features of a task.

    Attributes:
        observations: Features that the networks receive as input.
        actions: Features that the policy produces.

    Feature names must be unique, and one name cannot contain another: for
    example, ``"images"`` and ``"images/wrist"`` cannot both exist. Either
    list may be empty, but not both.

    Example:
        >>> DataSpec(
        ...     observations=(FeatureSpec("state", (4,), "float32"),),
        ...     actions=(FeatureSpec("action", (), "int64"),),
        ... )
    """

    observations: tuple[FeatureSpec, ...]
    actions: tuple[FeatureSpec, ...]

    def __post_init__(self) -> None:
        """Check the feature lists and their names when the object is created.

        Raises:
            TypeError: If a list is not a tuple of ``FeatureSpec`` objects.
            ValueError: If names repeat or conflict, or both lists are empty.
        """
        # Steps 1-3 run once for the observations and once for the actions.
        for name in ("observations", "actions"):
            features = getattr(self, name)

            # 1. The list must be a tuple that contains only FeatureSpec objects.
            if not isinstance(features, tuple) or any(
                not isinstance(feature, FeatureSpec) for feature in features
            ):
                raise TypeError(f"{name} must be a tuple of FeatureSpec values")

            # 2. Names must be unique within the list.
            names = {feature.name for feature in features}
            if len(names) != len(features):
                raise ValueError(f"{name} contains duplicate feature names")

            # 3. One name may not sit inside another. For "a/b/c" we check the
            #    parents "a" and "a/b": a feature with either name would have to
            #    be a tensor and a dictionary at the same time.
            for path in names:
                parts = path.split("/")
                if any("/".join(parts[:index]) in names for index in range(1, len(parts))):
                    raise ValueError(f"{name} contains conflicting feature paths: {path}")

        # 4. A specification with no features at all describes nothing.
        if not self.observations and not self.actions:
            raise ValueError("data spec must declare at least one feature")

    def to_dict(self) -> dict[str, Any]:
        """Return the specification as a JSON-ready dictionary.

        The result has two keys, ``observations`` and ``actions``, each a list
        of feature dictionaries made by :meth:`FeatureSpec.to_dict`.
        """
        return {
            "observations": [feature.to_dict() for feature in self.observations],
            "actions": [feature.to_dict() for feature in self.actions],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DataSpec:
        """Create a specification from a dictionary made by :meth:`to_dict`.

        Args:
            data: A dictionary with exactly the keys ``observations`` and
                ``actions``, each holding a list of feature dictionaries.

        Returns:
            A checked ``DataSpec``.

        Raises:
            ValueError: If the dictionary does not contain exactly
                ``observations`` and ``actions``, or a feature is invalid.
        """
        # 1. Exactly these two keys, each holding a list; anything else is a
        #    malformed specification.
        if not isinstance(data, Mapping) or set(data) != {"observations", "actions"}:
            raise ValueError("data spec must contain exactly observations and actions")
        if any(not isinstance(data[name], (list, tuple)) for name in data):
            raise ValueError("data spec observations and actions must be arrays")

        # 2. Rebuild every feature; each one is checked as it is created.
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
        """Check that observation data matches this specification.

        The data is only checked, never converted.

        Args:
            data: A tensor (allowed only when one feature is declared), or a
                TensorDict with exactly the declared keys.
            batch_shape: Leading batch dimensions of every feature, for
                example ``(N,)`` for one step from N environments or
                ``(T, N)`` for a rollout.

        Raises:
            ValueError: If the keys, shapes or dtypes do not match.
        """
        _validate_data(data, self.observations, batch_shape, "observations")

    def validate_actions(
        self,
        data: TensorData,
        *,
        batch_shape: tuple[int, ...],
    ) -> None:
        """Check that action data matches this specification.

        Same rules as :meth:`validate_observations`, applied to the actions.
        """
        _validate_data(data, self.actions, batch_shape, "actions")

    def validate_compatible(self, other: DataSpec) -> None:
        """Check that another specification describes exactly the same data.

        Every field of every feature must be equal, in the same order. Use this
        to compare a saved policy's specification with the current environment
        or robot.

        Args:
            other: The specification to compare with.

        Raises:
            TypeError: If ``other`` is not a ``DataSpec``.
            ValueError: Naming the first feature and field that differ.
        """
        # 1. Only another DataSpec can be compared.
        if not isinstance(other, DataSpec):
            raise TypeError("other must be a DataSpec")

        # Steps 2-3 run once for the observations and once for the actions.
        for name in ("observations", "actions"):
            expected, actual = getattr(self, name), getattr(other, name)

            # 2. Both specifications must list the same number of features.
            if len(expected) != len(actual):
                raise ValueError(f"incompatible {name}: feature counts differ")

            # 3. Compare features position by position, field by field. The same
            #    features listed in a different order also count as a difference.
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
    """Return every tensor in a TensorDict, keyed by its path.

    Nested TensorDicts are visited recursively, so ``{"images": {"wrist": x}}``
    becomes ``{("images", "wrist"): x}``.

    Args:
        data: The TensorDict to walk through.
        batch_shape: Batch shape that this TensorDict and every nested
            TensorDict must have.
        prefix: Path of ``data`` inside the outer TensorDict. Empty at the top
            level; the recursive calls fill it in.

    Raises:
        TypeError: If ``data`` is a lazy stack or holds a non-tensor value.
        ValueError: If a batch shape differs or a nested TensorDict is empty.
    """
    # 1. A lazy stack can hold items with different keys, so one fixed set of
    #    keys cannot describe it.
    if isinstance(data, LazyStackedTensorDict):
        raise TypeError("data must use a homogeneous dense TensorDict, not a lazy stack")

    # 2. Every level must have the same batch shape, for example (N,).
    if tuple(data.batch_size) != batch_shape:
        raise ValueError(f"TensorDict {'/'.join(prefix)} must have batch_shape {batch_shape}")

    # 3. An empty nested TensorDict contains no features, so no spec can match it.
    if prefix and not len(data.keys()):
        raise ValueError(f"TensorDict keys include an undeclared empty branch: {'/'.join(prefix)}")

    # 4. Collect each tensor under its full path, walking into nested TensorDicts.
    leaves = {}
    for key, value in data.items():
        path = (*prefix, key)
        if isinstance(value, TensorDictBase):
            # Nested TensorDict: collect its tensors under this path.
            leaves.update(_tensor_leaves(value, batch_shape, path))
        elif isinstance(value, torch.Tensor):
            leaves[path] = value
        else:
            # Strings and other non-tensor values cannot be checked or stored.
            raise TypeError(f"{'/'.join(path)} must be a Tensor feature")
    return leaves


def _validate_data(
    data: TensorData,
    features: tuple[FeatureSpec, ...],
    batch_shape: tuple[int, ...],
    name: str,
) -> None:
    """Check data against a list of features: keys, shapes and dtypes.

    Shared by :meth:`DataSpec.validate_observations` and
    :meth:`DataSpec.validate_actions`. The data is never changed.

    Args:
        data: A tensor (only when exactly one feature is declared) or a
            TensorDict.
        features: The declared features.
        batch_shape: Leading batch dimensions, for example ``(N,)``.
        name: ``"observations"`` or ``"actions"``, used in error messages.

    Raises:
        TypeError: If ``data`` is neither a tensor nor a TensorDict.
        ValueError: If the keys, shapes or dtypes do not match.
    """
    # 1. The batch shape itself must be valid, for example (N,) or (T, N).
    _validate_shape(batch_shape, "batch_shape")

    # 2. Collect the data's tensors, keyed by their path.
    if isinstance(data, torch.Tensor):
        # A plain tensor has no keys, so it can only stand for a single feature.
        if len(features) != 1:
            raise ValueError(f"{name}: a Tensor requires a single declared feature")
        leaves = {tuple(features[0].name.split("/")): data}
    elif isinstance(data, TensorDictBase):
        leaves = _tensor_leaves(data, batch_shape)

        # The keys must be exactly the declared names: none missing, none extra.
        expected = {tuple(feature.name.split("/")) for feature in features}
        if set(leaves) != expected:
            raise ValueError(
                f"{name} keys differ: missing={sorted(expected - set(leaves))}, "
                f"unexpected={sorted(set(leaves) - expected)}"
            )
    else:
        raise TypeError(f"{name} must be a Tensor or TensorDict")

    # 3. Check each feature's shape and dtype.
    for feature in features:
        value = leaves[tuple(feature.name.split("/"))]

        # The full shape is the batch shape followed by one sample's shape,
        # for example (N,) + (11,) = (N, 11).
        expected_shape = (*batch_shape, *feature.shape)
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"{name}.{feature.name} shape must be {expected_shape}, got {tuple(value.shape)}"
            )

        # The dtype must match exactly; nothing is converted.
        if value.dtype != getattr(torch, feature.dtype):
            raise ValueError(
                f"{name}.{feature.name} dtype must be {feature.dtype}, got {value.dtype}"
            )
