"""Training checkpoints: what they contain, how they are checked, saved and loaded.

A training checkpoint is everything needed to continue a run later. It is a
plain dictionary of state dicts and simple values, not serialized live model
or learner objects. ``torch.save`` still uses pickle to write it; loading
uses restricted unpickling (see ``load_checkpoint``)::

    {
        "algorithm":      "ppo",            # which learner made it
        "learner":        {...},            # learner.state_dict(): networks, optimizers, ...
        "learner_config": {...},            # the learner's config, as plain values
        "config":         {...} or None,    # the experiment's settings (env id, network sizes, ...)
        "progress":       {"global_step": ..., "gradient_step": ..., "update": ...},
        "rng_state":      {"torch": ..., "cuda": ... or None},
        "provenance":     {...},            # optional: library versions and git commit
    }

The training workflow adds its own fields, for example ``num_envs``,
``preprocessing_config``, ``observation_processor`` and ``policy_manifest``.

Saving::

    checkpoint = build_training_checkpoint(...)   # the common fields
    checkpoint["num_envs"] = ...                  # workflow fields
    validate_training_checkpoint(checkpoint, required_keys=(...))
    save_checkpoint(path, checkpoint)

Resuming::

    checkpoint = load_checkpoint(path)
    validate_training_checkpoint(checkpoint, required_keys=(...))
    validate_checkpoint_algorithm(checkpoint, type(learner))
    # then each owner checks and restores its own part

This module checks only the overall structure. Each part's owner (the
learner, the observation processor, the training workflow) checks its own
contents before restoring them.

Resuming restores the learning state and starts fresh environment episodes.
It does not continue the exact simulator trajectory of an uninterrupted run.
"""

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from deeprl.stats import TrainingProgress


def get_algorithm_id(learner_type: type) -> str:
    """Return the name a learner class writes into its checkpoints.

    A learner can declare a short, stable name with a class attribute, for
    example ``algorithm_id = "ppo"``. Without one, the class's full import
    path is used, for example ``"mypackage.learners.MyLearner"``.

    The name is not inherited. A subclass of ``PPOLearner`` gets its own name
    unless it sets ``algorithm_id = "ppo"`` itself, because a subclass may
    change what its saved state means.

    Args:
        learner_type: The learner class, for example ``type(learner)``.

    Returns:
        The algorithm name.

    Raises:
        ValueError: If the declared ``algorithm_id`` is not a non-empty string.
    """
    # 1. Look only at the class itself (its own __dict__), not its parents.
    #    Fall back to the class's full import path.
    identity = learner_type.__dict__.get(
        "algorithm_id", f"{learner_type.__module__}.{learner_type.__qualname__}"
    )

    # 2. The name must be usable text.
    if not isinstance(identity, str) or not identity.strip():
        raise ValueError("algorithm_id must be a non-empty string")
    return identity


def validate_checkpoint_algorithm(
    checkpoint: Mapping[str, Any],
    learner_type: type,
) -> None:
    """Check that a checkpoint was made by the same algorithm as the learner.

    Prevents, for example, loading an A2C checkpoint into a PPO learner.

    Args:
        checkpoint: A loaded training checkpoint.
        learner_type: The class of the learner that will restore it.

    Raises:
        ValueError: If the saved algorithm name differs.
    """
    if checkpoint.get("algorithm") != get_algorithm_id(learner_type):
        raise ValueError("checkpoint algorithm does not match the learner")


def build_training_checkpoint(
    *,
    algorithm: str,
    learner_state: dict[str, Any],
    learner_config: dict[str, Any],
    progress: TrainingProgress,
    rng_state: dict[str, Any],
    run_config: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Put the common checkpoint fields together in one dictionary.

    This only arranges what it is given. It does not collect state, copy
    tensors or write files. The caller then adds its workflow fields and
    checks them with ``validate_training_checkpoint`` before saving.

    Args:
        algorithm: The learner's name, from ``get_algorithm_id``.
        learner_state: ``learner.state_dict()``.
        learner_config: The learner's config as plain values (see
            ``lifecycle.config_to_jsonable``).
        progress: The step counters at the time of saving.
        rng_state: Random-number state, from ``lifecycle.get_rng_state``.
        run_config: The experiment's settings, or ``None``.
        provenance: Versions and git commit, from
            ``lifecycle.get_provenance``, or ``None`` to leave it out.

    Returns:
        The checkpoint dictionary, already checked.
    """
    # 1. Arrange the common fields. Progress becomes a plain dictionary.
    checkpoint = {
        "algorithm": algorithm,
        "learner": learner_state,
        "learner_config": learner_config,
        "config": run_config,
        "progress": asdict(progress),
        "rng_state": rng_state,
    }

    # 2. Provenance is optional and only stored when given.
    if provenance is not None:
        checkpoint["provenance"] = provenance

    # 3. Check the structure now, so a bad checkpoint fails when it is made,
    #    not later when someone tries to resume from it.
    validate_training_checkpoint(checkpoint)
    return checkpoint


def validate_training_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    required_keys: Sequence[str] = (),
) -> None:
    """Check a checkpoint's overall structure before anything is restored.

    Checks that the required fields exist and have the right kind of value.
    It does not check the contents of the parts: algorithm settings, tensor
    values and environment settings are checked by their owners (the learner
    and the training workflow).

    Args:
        checkpoint: A training checkpoint.
        required_keys: Extra fields the workflow requires, for example
            ``("num_envs", "preprocessing_config")``.

    Raises:
        ValueError: If a field is missing or has the wrong kind of value.
    """
    # 1. Require the common fields and any workflow-specific fields.
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must be a mapping")
    required = {
        "learner", "progress", "algorithm", "learner_config", "config",
        "rng_state", *required_keys,
    }
    missing = required - checkpoint.keys()
    if missing:
        raise ValueError(f"checkpoint is missing required state: {', '.join(sorted(missing))}")

    # 2. The learner state must be a dictionary. Progress must have exactly
    #    the three counters, and creating a TrainingProgress checks that each
    #    is a non-negative integer.
    if not isinstance(checkpoint["learner"], Mapping):
        raise ValueError("checkpoint learner must be a state mapping")
    progress = checkpoint["progress"]
    if not isinstance(progress, Mapping) or set(progress) != {
        "global_step", "gradient_step", "update",
    }:
        raise ValueError("checkpoint progress must contain global_step, gradient_step, and update")
    TrainingProgress(**progress)

    # 3. Check the descriptive fields: the algorithm name, the configs, and
    #    the optional workflow fields when they are present.
    if (
        not isinstance(checkpoint["algorithm"], str) or not checkpoint["algorithm"].strip()
    ):
        raise ValueError("checkpoint algorithm must be a non-empty string")
    if not isinstance(checkpoint["learner_config"], Mapping):
        raise ValueError("checkpoint learner_config must be a mapping")
    if checkpoint["config"] is not None and not isinstance(checkpoint["config"], Mapping):
        raise ValueError("checkpoint config must be a mapping or None")
    if "provenance" in checkpoint and not isinstance(checkpoint["provenance"], Mapping):
        raise ValueError("checkpoint provenance must be a mapping")

    # Number of environments: a positive integer (True is not accepted).
    if "num_envs" in checkpoint:
        num_envs = checkpoint["num_envs"]
        if type(num_envs) is not int or num_envs <= 0:
            raise ValueError("checkpoint num_envs must be a positive integer")

    # Observation statistics: a state dict.
    if "observation_processor" in checkpoint and not isinstance(
        checkpoint["observation_processor"], Mapping
    ):
        raise ValueError("checkpoint observation_processor must be a state mapping")

    # Processing settings: exactly an "observation" and an "action" entry,
    # each a dictionary, or None when that processing is turned off.
    if "preprocessing_config" in checkpoint:
        preprocessing = checkpoint["preprocessing_config"]
        if not isinstance(preprocessing, Mapping) or set(preprocessing) != {
            "observation", "action",
        }:
            raise ValueError("checkpoint preprocessing_config must contain observation and action")
        for name, settings in preprocessing.items():
            if settings is not None and not isinstance(settings, Mapping):
                raise ValueError(f"checkpoint preprocessing_config {name} must be a mapping or None")

    # 4. Random-number state. "torch" (the CPU generator) is always needed.
    #    "cuda" is always present, and None for CPU-only runs. A CUDA run
    #    must have it.
    rng_state = checkpoint["rng_state"]
    if not isinstance(rng_state, Mapping):
        raise ValueError("checkpoint rng_state must be a mapping")
    if "cuda" not in rng_state:
        raise ValueError("checkpoint rng_state is missing cuda")
    if (
        str(checkpoint["learner_config"].get("device", "cpu")).split(":", 1)[0] == "cuda"
        and rng_state["cuda"] is None
    ):
        raise ValueError("checkpoint rng_state cuda is required for CUDA training")

    # Each saved generator state must look like what PyTorch produces: a
    # non-empty, one-dimensional uint8 tensor.
    for name in ("torch", "cuda"):
        state = rng_state.get(name)
        if name == "cuda" and state is None:
            continue
        if (
            not isinstance(state, torch.Tensor)
            or state.dtype != torch.uint8
            or state.ndim != 1
            or state.numel() == 0
        ):
            raise ValueError(f"checkpoint rng_state {name} must be a non-empty 1D uint8 tensor")


def save_checkpoint(
    path: str | Path,
    checkpoint: dict[str, Any],
) -> None:
    """Save a checkpoint by writing a temporary file, then replacing the old one.

    The checkpoint is first written to a temporary file next to ``path``.
    Only after that write succeeds is it renamed to ``path``, which replaces
    the old file in one step. If the program stops during the write, the
    previous checkpoint at ``path`` is still intact.

    Limits: this assumes one writer at a time (two runs saving to the same
    path share one temporary name), and it does not force the data onto the
    disk, so a power loss right after saving can still lose it.

    Args:
        path: Where to save, for example ``runs/hopper/last.pt``. Missing
            folders are created.
        checkpoint: The checkpoint dictionary.
    """
    # 1. Create the folder if needed.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 2. Write the complete file under a temporary name.
    temporary_path = path.with_name(f"{path.name}.tmp")
    torch.save(checkpoint, temporary_path)

    # 3. Replace the old file with the new one in a single rename.
    temporary_path.replace(path)


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint's data without restoring anything.

    Only tensors and plain Python values (dicts, lists, numbers, strings) can
    be loaded (``weights_only=True``). PyTorch then refuses to rebuild
    arbitrary Python objects, which is far safer than a full unpickle. Still,
    only load checkpoints from sources you trust.

    Args:
        path: The checkpoint file.
        map_location: Device to load tensors onto. CPU by default; each owner
            moves its part to the right device when restoring.

    Returns:
        The checkpoint dictionary.

    Raises:
        ValueError: If the file does not hold a dictionary.
    """
    # 1. Load with restricted unpickling.
    checkpoint = torch.load(
        path,
        map_location=map_location,
        weights_only=True,
    )

    # 2. A checkpoint is always a dictionary.
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must be a mapping")
    return checkpoint
