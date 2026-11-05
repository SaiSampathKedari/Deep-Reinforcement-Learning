"""Atomic checkpoint I/O and the current training-state contract.

Training checkpoints contain learner state, configuration, progress and RNG
state. Workflows declare additional required fields. Component state and
preprocessing settings are validated by their owners before restoration.

The resume contract is "continue": restore learning state and start fresh
environment episodes, not an exact uninterrupted simulator trajectory.
"""

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from deeprl.stats import TrainingProgress


CHECKPOINT_RESUME_CONTRACT = "continue"


def get_algorithm_id(learner_type: type) -> str:
    """Use an explicitly declared stable ID, or the concrete class's full name.

    IDs are not inherited: a custom subclass is a distinct algorithm unless it
    explicitly opts into the same checkpoint contract as its parent.
    """
    identity = learner_type.__dict__.get(
        "algorithm_id", f"{learner_type.__module__}.{learner_type.__qualname__}"
    )
    if not isinstance(identity, str) or not identity.strip():
        raise ValueError("algorithm_id must be a non-empty string")
    return identity


def validate_checkpoint_algorithm(
    checkpoint: Mapping[str, Any], learner_type: type,
) -> None:
    """Require the learner's current algorithm identity."""
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
    """Build the common envelope from native state and plain configuration.

    This does not capture state, clone model tensors, or write files. The caller
    adds workflow-specific state and validates its required keys before saving.
    """
    checkpoint = {
        "resume_contract": CHECKPOINT_RESUME_CONTRACT,
        "algorithm": algorithm,
        "learner": learner_state,
        "learner_config": learner_config,
        "config": run_config,
        "progress": asdict(progress),
        "rng_state": rng_state,
    }
    if provenance is not None:
        checkpoint["provenance"] = provenance
    validate_training_checkpoint(checkpoint)
    return checkpoint


def validate_training_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    required_keys: Sequence[str] = (),
) -> None:
    """Validate run-state structure before owners restore their components.

    Algorithm-specific configuration rules, tensor contents, and environment
    settings remain the responsibility of the learner and training workflow.
    """
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must be a mapping")
    required = {
        "learner", "progress", "algorithm", "learner_config", "config",
        "rng_state", "resume_contract", *required_keys,
    }
    missing = required - checkpoint.keys()
    if missing:
        raise ValueError(f"checkpoint is missing required state: {', '.join(sorted(missing))}")

    if not isinstance(checkpoint["learner"], Mapping):
        raise ValueError("checkpoint learner must be a state mapping")
    progress = checkpoint["progress"]
    if not isinstance(progress, Mapping) or set(progress) != {
        "global_step", "gradient_step", "update",
    }:
        raise ValueError("checkpoint progress must contain global_step, gradient_step, and update")
    TrainingProgress(**progress)

    if checkpoint["resume_contract"] != CHECKPOINT_RESUME_CONTRACT:
        raise ValueError("unsupported checkpoint resume_contract; only continue is supported")
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
    if "num_envs" in checkpoint:
        num_envs = checkpoint["num_envs"]
        if type(num_envs) is not int or num_envs <= 0:
            raise ValueError("checkpoint num_envs must be a positive integer")
    for name in ("obs_normalizer", "reward_scaler"):
        if name in checkpoint and not isinstance(checkpoint[name], Mapping):
            raise ValueError(f"checkpoint {name} must be a state mapping")

    if "preprocessing_config" in checkpoint:
        preprocessing = checkpoint["preprocessing_config"]
        if not isinstance(preprocessing, Mapping) or set(preprocessing) != {
            "observation", "reward",
        }:
            raise ValueError("checkpoint preprocessing_config must contain observation and reward")
        for name, settings in preprocessing.items():
            if settings is not None and not isinstance(settings, Mapping):
                raise ValueError(f"checkpoint preprocessing_config {name} must be a mapping or None")

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
    """Atomically replace a checkpoint containing state dicts and plain metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_name(f"{path.name}.tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Read a state mapping with restricted loading; do not restore objects."""
    checkpoint = torch.load(
        path,
        map_location=map_location,
        weights_only=True,
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must be a mapping")
    return checkpoint
