"""Checkpoint I/O and the versioned training-state contract.

Version 1 keeps the flat payload: learner, progress, algorithm, learner_config,
config, rng_state, format_version, and resume_contract. Workflows declare any
additional required keys; on-policy training requires num_envs. Optional
obs_normalizer and reward_scaler entries hold component state dictionaries.

The only supported resume contract is "continue": restore supported learning
state, not an exact uninterrupted trajectory. The on-policy trainer starts
fresh environment episodes. Missing format_version denotes the legacy layout;
validation does not invent missing state or relabel it as version 1.
"""

import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from deeprl.stats import TrainingProgress


CHECKPOINT_FORMAT_VERSION = 1
CHECKPOINT_RESUME_CONTRACT = "continue"


def _format_version(checkpoint: Mapping[str, Any]) -> int:
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must be a mapping")
    if "format_version" not in checkpoint:
        return 0
    version = checkpoint["format_version"]
    if type(version) is not int or version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"unsupported checkpoint format_version: {version!r}")
    return version


def validate_training_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    required_keys: Sequence[str] = (),
) -> None:
    """Validate run-state structure before owners restore their components.

    Legacy checkpoints require learner and progress, but may lack RNG or
    compatibility metadata. Version 1 requires those fields. Algorithm-specific
    configuration rules, tensor contents, and environment compatibility remain
    the responsibility of the learner and training workflow.
    """
    version = _format_version(checkpoint)
    required = {"learner", "progress"}
    if version == CHECKPOINT_FORMAT_VERSION:
        required.update({
            "algorithm", "learner_config", "config", "rng_state",
            "resume_contract", *required_keys,
        })
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

    if "resume_contract" in checkpoint and (
        checkpoint["resume_contract"] != CHECKPOINT_RESUME_CONTRACT
    ):
        raise ValueError("unsupported checkpoint resume_contract; only continue is supported")
    if "algorithm" in checkpoint and (
        not isinstance(checkpoint["algorithm"], str) or not checkpoint["algorithm"].strip()
    ):
        raise ValueError("checkpoint algorithm must be a non-empty string")
    if version or checkpoint.get("learner_config") is not None:
        if not isinstance(checkpoint.get("learner_config"), Mapping):
            raise ValueError("checkpoint learner_config must be a mapping")
    if checkpoint.get("config") is not None and not isinstance(checkpoint["config"], Mapping):
        raise ValueError("checkpoint config must be a mapping or None")
    if "num_envs" in checkpoint:
        num_envs = checkpoint["num_envs"]
        if type(num_envs) is not int or num_envs <= 0:
            raise ValueError("checkpoint num_envs must be a positive integer")
    for name in ("obs_normalizer", "reward_scaler"):
        if name in checkpoint and not isinstance(checkpoint[name], Mapping):
            raise ValueError(f"checkpoint {name} must be a state mapping")

    rng_state = checkpoint.get("rng_state")
    if version or rng_state is not None:
        if not isinstance(rng_state, Mapping):
            raise ValueError("checkpoint rng_state must be a mapping")
        if version and "cuda" not in rng_state:
            raise ValueError("checkpoint rng_state is missing cuda")
        if (
            version
            and str(checkpoint["learner_config"].get("device", "cpu")).split(":", 1)[0] == "cuda"
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

    if version == 0:
        unchecked = sorted({
            name for name in ("algorithm", "learner_config", "rng_state", *required_keys)
            if checkpoint.get(name) is None
        })
        if unchecked:
            warnings.warn(
                f"legacy checkpoint lacks {', '.join(unchecked)}; continuation cannot "
                "restore or validate that state. Missing RNG uses the configured seed.",
                UserWarning,
                stacklevel=2,
            )


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
    """Read restricted state and reject unknown formats; do not restore objects."""
    checkpoint = torch.load(
        path,
        map_location=map_location,
        weights_only=True,
    )
    _format_version(checkpoint)
    return checkpoint
