"""Helpers used around a training run: records, settings and random numbers.

* ``get_provenance``: record which code produced a run (library versions and
  git commit).
* ``config_to_jsonable``: turn a config into plain values that can be saved
  and compared.
* ``validate_run_config``: check that a resumed run uses the same experiment
  settings, such as the environment and network sizes.
* ``seed_torch``, ``get_rng_state``, ``set_rng_state``: make runs
  reproducible and let a resumed run continue the same random numbers.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch

from deeprl import __version__


def get_provenance() -> dict[str, str | bool | None]:
    """Return a record of the code that produced a run.

    Saved in checkpoints and policy folders, so a result can be traced back
    to its code later::

        {
            "deeprl_version": "0.1.0",
            "torch_version": "2.8.0",
            "git_commit": "1a78900...",   # None if unknown
            "git_dirty": False,           # True if there were uncommitted changes
        }

    Git is asked only when this package runs from a source checkout (the
    ``src/deeprl`` folder of a git repository). It is never asked about the
    caller's own folder or about the repository an installed package happens
    to sit in. If git is missing, fails or takes longer than 2 seconds, the
    git fields stay ``None``.

    Returns:
        The provenance record.
    """
    # 1. Versions are always known; git details start as unknown.
    provenance: dict[str, str | bool | None] = {
        "deeprl_version": str(__version__),
        "torch_version": str(torch.__version__),
        "git_commit": None,
        "git_dirty": None,
    }
    try:
        # 2. Find this package's repository: <root>/src/deeprl. Stop if the
        #    package is not in that layout or the root is not a git repository.
        package_dir = Path(__file__).resolve().parent
        root = package_dir.parent.parent
        if package_dir.parent.name != "src" or not (root / ".git").exists():
            return provenance

        # 3. Ask git for the current commit and for uncommitted changes. A
        #    non-empty "git status --porcelain" means there are changes.
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        # Git is not installed, failed or timed out: keep the fields unknown.
        return provenance

    # 4. Fill in the git details.
    provenance.update(git_commit=commit, git_dirty=bool(status))
    return provenance


def _config_json_default(value: Any) -> Any:
    """Describe a config value that JSON cannot store directly.

    ``json.dumps`` calls this for each value it does not know. The value is
    replaced by a plain description, never kept as a live object.

    Args:
        value: The value JSON could not store.

    Returns:
        A plain replacement:

        * a class becomes its full name, for example ``torch.nn.Tanh`` becomes
          ``"torch.nn.modules.activation.Tanh"``;
        * a device or a path becomes a string, for example ``"cuda:0"``;
        * a dataclass or any other mapping becomes a dictionary.

    Raises:
        TypeError: For any other kind of value.
    """
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, (torch.device, Path)):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def config_to_jsonable(config: Any) -> dict[str, Any]:
    """Copy a config into plain values: dicts, lists, strings, numbers, booleans and None.

    Plain values can be saved in checkpoints and JSON files, logged, and
    compared with ``==``. For example, a saved and a current config can be
    compared field by field when resuming.

    Conversions: classes become their full names, devices and paths become
    strings, and tuples become lists. NaN and infinity are rejected.

    Args:
        config: A dataclass instance (for example a ``PPOConfig``) or a
            dictionary.

    Returns:
        A plain dictionary copy.

    Raises:
        TypeError: If ``config`` is not a dataclass instance or a mapping, or
            it holds a value that cannot be described.
        ValueError: If it holds NaN or infinity.

    Example:
        >>> from deeprl.algorithms.ppo import PPOConfig
        >>> config_to_jsonable(PPOConfig(device=torch.device("cuda")))["device"]
        'cuda'
    """
    # 1. Accept a dataclass instance or a mapping (a dataclass class itself is
    #    not a config).
    if not isinstance(config, Mapping) and not (
        is_dataclass(config) and not isinstance(config, type)
    ):
        raise TypeError("configuration must be a dataclass instance or mapping")

    # 2. Write to JSON text and read it back. This makes a plain copy and
    #    fails loudly on anything JSON cannot represent.
    return json.loads(
        json.dumps(config, default=_config_json_default, allow_nan=False)
    )


def validate_run_config(
    saved_config: Mapping[str, Any],
    current_config: Mapping[str, Any],
    *,
    keys: Sequence[str],
) -> None:
    """Check that a resumed run uses the same experiment settings.

    These are settings outside the learner config that the saved state
    depends on, for example the environment id or the network sizes. The
    caller chooses which fields must match, for example::

        validate_run_config(saved, current, keys=("env_id", "policy_network"))

    Each chosen field must be present in both configs. A setting that is
    turned off must still be written, as ``None``.

    Args:
        saved_config: The experiment config stored in the checkpoint.
        current_config: The experiment config of the resuming run.
        keys: The fields that must match.

    Raises:
        ValueError: If a config is not a mapping, or a chosen field is
            missing or different.
    """
    # 1. Both configs must be dictionaries.
    if not isinstance(saved_config, Mapping):
        raise ValueError("Checkpoint has no valid experiment configuration.")
    if not isinstance(current_config, Mapping):
        raise ValueError("Current experiment configuration must be a mapping.")

    # 2. Convert both to plain values, so that, for example, a tuple and the
    #    list it was saved as compare equal.
    saved = config_to_jsonable(saved_config)
    current = config_to_jsonable(current_config)

    # 3. Every chosen field must exist in both and have the same value.
    for name in keys:
        if name not in current:
            raise ValueError(f"Current experiment configuration is missing {name}.")
        if name not in saved:
            raise ValueError(f"Checkpoint experiment configuration is missing {name}.")
        if saved[name] != current[name]:
            raise ValueError(f"Checkpoint {name} does not match this experiment.")


def seed_torch(seed: int) -> None:
    """Seed PyTorch's default random-number generators.

    Call this before building the networks, so the initial weights are the
    same in every run with the same seed.

    Environments are seeded separately, with ``envs.reset(seed=...)``. This
    does not seed Python's ``random``, NumPy, Gymnasium action spaces, or
    ``torch.Generator`` objects created explicitly.

    Args:
        seed: The seed.
    """
    torch.manual_seed(seed)


def get_rng_state(
    *,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor | None]:
    """Return the current state of PyTorch's default random-number generators.

    Saved in checkpoints, so a resumed run continues the same random numbers
    (for example, for minibatch shuffling and action sampling).

    Args:
        device: The training device. For a CUDA device, that device's
            generator is saved too. A CPU run never touches CUDA.

    Returns:
        ``{"torch": <CPU state>, "cuda": <CUDA state or None>}``. Each state is
        a one-dimensional ``uint8`` tensor. Only one CUDA device is covered;
        this is not state for multi-GPU training.
    """
    device = torch.device(device)
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def set_rng_state(
    state: Mapping[str, Any],
    *,
    device: str | torch.device = "cpu",
) -> None:
    """Restore PyTorch's default random-number generators from a checkpoint.

    Call this last, after all setup (building networks, resetting
    environments, creating buffers). Setup that draws random numbers after
    the restore would use up part of the restored sequence, and the resumed
    run would differ.

    Args:
        state: The saved state, from ``get_rng_state``.
        device: The training device of the resumed run. A CPU run ignores any
            saved CUDA state. A CUDA run restores the saved CUDA state onto
            this device, which may differ from the original one (for example
            ``cuda:1`` instead of ``cuda:0``).
    """
    device = torch.device(device)

    # 1. Restore the CPU generator. PyTorch requires the state on the CPU.
    torch.set_rng_state(state["torch"].cpu())

    # 2. Restore the CUDA generator, for CUDA runs that saved one.
    if device.type == "cuda" and state.get("cuda") is not None:
        torch.cuda.set_rng_state(state["cuda"].cpu(), device=device)
