"""Inference-only policy directories with explicit, caller-owned model factories.

JSON is descriptive data, never an import instruction. Tensor files are loaded
with PyTorch's restricted loader. Publishing requires a new destination so a
reader cannot observe weights from one save and statistics from another.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from deeprl.checkpoint import load_checkpoint, validate_training_checkpoint
from deeprl.data_spec import DataSpec, TensorData
from deeprl.inference import (
    InferencePolicy, deterministic_action, distribution_action, greedy_action,
)
from deeprl.lifecycle import get_provenance
from deeprl.processors import ActionProcessor, ObservationProcessor


def _json_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    """Copy plain configuration while rejecting non-finite JSON numbers."""
    if not isinstance(value, Mapping):
        raise ValueError("policy configuration must be a mapping")
    return json.loads(json.dumps(dict(value), allow_nan=False))


def _actor_spec(spec: DataSpec) -> DataSpec:
    """Drop critic-only observations without changing action declarations."""
    return DataSpec(
        tuple(
            feature for feature in spec.observations
            if not feature.groups or "actor" in feature.groups
        ),
        spec.actions,
    )


def _selector_id(selector: Callable) -> str | None:
    """Identify built-in decisions; custom callables need caller-supplied IDs."""
    if selector is distribution_action:
        return "distribution"
    if selector is greedy_action:
        return "greedy"
    if selector is deterministic_action:
        return "deterministic"
    return None


def validate_policy_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate current fields and transform semantics; return an independent copy."""
    data = _json_copy(manifest)
    if set(data) != {
        "spec", "model", "decision", "observation_processor", "action_processor",
        "inference", "provenance",
    }:
        raise ValueError("policy manifest has missing or unknown fields")
    spec = DataSpec.from_dict(data["spec"])
    if not spec.observations or not spec.actions:
        raise ValueError("policy spec must declare observations and actions")
    # Inference descriptions cannot require features reserved for the critic.
    spec.validate_compatible(_actor_spec(spec))
    for field, keys in (("model", {"id", "config"}), ("decision", {"id"})):
        value = data[field]
        if not isinstance(value, dict) or set(value) != keys:
            raise ValueError(f"policy {field} must contain {sorted(keys)}")
        if not isinstance(value["id"], str) or not value["id"].strip():
            raise ValueError(f"policy {field} id must be a nonempty string")
    if not isinstance(data["model"]["config"], dict):
        raise ValueError("policy model config must be a mapping")
    inference = data["inference"]
    if not isinstance(inference, dict) or set(inference) != {"deterministic"}:
        raise ValueError("policy inference must contain deterministic")
    if type(inference["deterministic"]) is not bool:
        raise ValueError("policy deterministic setting must be boolean")
    if not isinstance(data["provenance"], dict):
        raise ValueError("policy provenance must be a mapping")

    observation = data["observation_processor"]
    if observation is not None:
        if not isinstance(observation, dict) or set(observation) != {
            "normalize", "epsilon", "clip",
        }:
            raise ValueError("policy observation_processor settings are incomplete or unknown")
        ObservationProcessor(spec, **observation)
    action = data["action_processor"]
    if action is not None:
        if not isinstance(action, dict) or set(action) != {
            "source_bounds", "target_bounds", "clip",
        }:
            raise ValueError("policy action_processor settings are incomplete or unknown")
        processor = ActionProcessor(**action)
        if processor.source_low is not None:
            if (
                len(spec.actions) != 1
                or not getattr(torch, spec.actions[0].dtype).is_floating_point
            ):
                raise ValueError("bounded action processing requires one floating-point action feature")
            shape = spec.actions[0].shape
            try:
                if torch.broadcast_shapes(processor.source_low.shape, shape) != shape:
                    raise ValueError("action bounds do not match the action feature shape")
            except RuntimeError as error:
                raise ValueError("action bounds do not match the action feature shape") from error
    return data


def describe_policy(
    policy: InferencePolicy,
    *,
    spec: DataSpec,
    model_id: str,
    model_config: Mapping[str, Any],
    decision_id: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe the actor's actual processing and construction, without weights."""
    known_id = _selector_id(policy.action_selector)
    if known_id is None and decision_id in ("distribution", "greedy", "deterministic"):
        raise ValueError("custom action selectors cannot use a built-in decision id")
    if decision_id is None:
        decision_id = known_id
    elif known_id is not None and decision_id != known_id:
        raise ValueError("decision id does not match the built-in action selector")
    if decision_id is None:
        raise ValueError("a custom action selector requires an explicit decision_id")

    # Keep the actor's transform settings, excluding privileged-feature fitting.
    actor_spec = _actor_spec(spec)
    if policy.observation_processor is None and actor_spec.observations != spec.observations:
        raise ValueError(
            "non-actor observations require an observation processor to enforce actor selection"
        )
    spec = actor_spec
    observation = None
    if policy.observation_processor is not None:
        processor = policy.observation_processor
        DataSpec(_actor_spec(processor.spec).observations, ()).validate_compatible(
            DataSpec(spec.observations, ()),
        )
        observation = processor.config
        names = {feature.name for feature in spec.observations}
        observation["normalize"] = [name for name in processor.normalize if name in names]
    return validate_policy_manifest(
        {
            "spec": spec.to_dict(),
            "model": {"id": model_id, "config": dict(model_config)},
            "decision": {"id": decision_id},
            "observation_processor": observation,
            "action_processor": (
                None if policy.action_processor is None else policy.action_processor.config
            ),
            "inference": {"deterministic": policy.deterministic},
            "provenance": get_provenance() if provenance is None else dict(provenance),
        }
    )


def _processor_state(
    source: ObservationProcessor | None,
    manifest: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Project fitted moments onto the actor-only feature order in the manifest."""
    if source is None:
        return {}
    # Export only actor features; indices can change when privileged inputs drop.
    exported = ObservationProcessor(
        DataSpec.from_dict(manifest["spec"]),
        **manifest["observation_processor"],
    )
    for index, name in enumerate(exported.normalize):
        old_index = source.normalize.index(name)
        exported.normalizers[str(index)].load_state_dict(
            source.normalizers[str(old_index)].state_dict()
        )
    return exported.state_dict()


def _tensor_state(state: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Require named tensors and detach an independent CPU copy for persistence."""
    if not isinstance(state, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise ValueError("policy state must contain only named tensors")
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def save_policy(
    directory: str | Path,
    policy: InferencePolicy,
    *,
    spec: DataSpec,
    model_id: str,
    model_config: Mapping[str, Any],
    decision_id: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> None:
    """Publish a complete inference snapshot to a new directory, without overwrite."""

    # 1. Validate the destination and snapshot the inference components.
    directory = Path(directory)
    if directory.exists() or directory.is_symlink():
        raise FileExistsError(f"policy destination already exists: {directory}")
    manifest = describe_policy(
        policy,
        spec=spec,
        model_id=model_id,
        model_config=model_config,
        decision_id=decision_id,
        provenance=provenance,
    )
    model_state = _tensor_state(policy.model.state_dict())
    processor_state = _tensor_state(_processor_state(policy.observation_processor, manifest))
    _write_policy(directory, manifest, model_state, processor_state)


def _write_policy(
    directory: Path,
    manifest: dict[str, Any],
    model_state: dict[str, torch.Tensor],
    processor_state: dict[str, torch.Tensor],
) -> None:
    """Publish an inference snapshot only after all three files are complete."""
    if directory.exists() or directory.is_symlink():
        raise FileExistsError(f"policy destination already exists: {directory}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{directory.name}-",
            dir=directory.parent,
        )
    )
    try:
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        torch.save(model_state, temporary / "model.pt")
        torch.save(processor_state, temporary / "processor.pt")
        temporary.rename(directory)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def export_policy_from_checkpoint(
    checkpoint_path: str | Path,
    directory: str | Path,
) -> None:
    """Export an on-policy training snapshot without constructing its learner.

    The checkpoint must contain a policy_manifest, learner['policy'] weights,
    and the declared preprocessing state. Only actor-visible statistics are
    exported; optimizers, critics and reward scalers stay in the checkpoint.
    The destination must be new. Loading the result uses load_policy() and the
    task's trusted model factory, just like an export from a finished run.
    """
    # 1. Read the saved inference description and weights without rebuilding models.
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    validate_training_checkpoint(
        checkpoint,
        required_keys=("policy_manifest", "preprocessing_config"),
    )
    manifest = validate_policy_manifest(checkpoint["policy_manifest"])
    if "policy" not in checkpoint["learner"]:
        raise ValueError("checkpoint is missing learner policy weights")
    model_state = _tensor_state(checkpoint["learner"]["policy"])
    preprocessing = checkpoint["preprocessing_config"]
    if preprocessing["action"] != manifest["action_processor"]:
        raise ValueError("checkpoint action processing does not match the policy manifest")

    # 2. Restore the shared statistics, then select only the actor's features.
    settings = preprocessing["observation"]
    source = None
    if settings is None:
        if manifest["observation_processor"] is not None or "observation_processor" in checkpoint:
            raise ValueError("checkpoint observation processing does not match the policy manifest")
    else:
        if set(settings) != {"spec", "config"} or "observation_processor" not in checkpoint:
            raise ValueError("checkpoint is missing observation processor settings or state")
        spec = DataSpec.from_dict(settings["spec"])
        source = ObservationProcessor(spec, **settings["config"])
        actor_spec = DataSpec.from_dict(manifest["spec"])
        DataSpec(_actor_spec(spec).observations, ()).validate_compatible(
            DataSpec(actor_spec.observations, ()),
        )
        actor_names = {feature.name for feature in actor_spec.observations}
        actor_config = source.config
        actor_config["normalize"] = [name for name in source.normalize if name in actor_names]
        if actor_config != manifest["observation_processor"]:
            raise ValueError("checkpoint observation processing does not match the policy manifest")
        source.load_state_dict(_tensor_state(checkpoint["observation_processor"]))
        for normalizer in source.normalizers.values():
            if not all(torch.isfinite(value).all() for value in normalizer.buffers()) or (
                (normalizer.var < 0).any() or normalizer.count < 0
            ):
                raise ValueError("invalid observation processor statistics")
    processor_state = _tensor_state(_processor_state(source, manifest))

    # 3. Publish the same inference-only directory used by completed runs.
    manifest["provenance"]["training_step"] = checkpoint["progress"]["global_step"]
    _write_policy(Path(directory), manifest, model_state, processor_state)


def load_policy(
    directory: str | Path,
    model_factory: Callable[[dict[str, Any]], nn.Module],
    *,
    model_id: str,
    device: str | torch.device = "cpu",
    action_selector: Callable[[Any, bool], TensorData] | None = None,
    decision_id: str | None = None,
    expected_spec: DataSpec | None = None,
) -> InferencePolicy:
    """Restore an evaluation-mode policy using trusted code, never JSON imports.

    The factory receives the saved constructor config and builds the model on
    CPU, with the saved parameter and buffer dtypes. It must not construct a
    learner. Custom selectors require a matching explicit identity; built-in
    selectors need no caller-supplied code.
    """

    # 1. Validate the saved contract before constructing or loading any model.
    directory = Path(directory)
    manifest = validate_policy_manifest(
        json.loads(
            (directory / "manifest.json").read_text(encoding="utf-8"),
        )
    )
    if manifest["model"]["id"] != model_id:
        raise ValueError("policy model id does not match the supplied factory")
    spec = DataSpec.from_dict(manifest["spec"])
    if expected_spec is not None:
        _actor_spec(expected_spec).validate_compatible(spec)

    # 2. Resolve a built-in decision or verify the caller's explicit custom one.
    saved_decision = manifest["decision"]["id"]
    if action_selector is None:
        if saved_decision == "distribution":
            action_selector = distribution_action
        elif saved_decision == "greedy":
            action_selector = greedy_action
        elif saved_decision == "deterministic":
            action_selector = deterministic_action
        else:
            raise ValueError("custom decision requires a supplied action_selector and decision_id")
    supplied_decision = _selector_id(action_selector)
    if supplied_decision is None and saved_decision in (
        "distribution", "greedy", "deterministic",
    ):
        raise ValueError("custom action selectors cannot override a built-in decision id")
    if decision_id is not None:
        if supplied_decision is not None and supplied_decision != decision_id:
            raise ValueError("decision id does not match the supplied action selector")
        supplied_decision = decision_id
    if supplied_decision != saved_decision:
        raise ValueError("policy decision id does not match the supplied action selector")

    # 3. Restore tensor state with trusted architecture code and no learner.
    model_state = _tensor_state(
        torch.load(
            directory / "model.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    processor_state = _tensor_state(
        torch.load(
            directory / "processor.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    # Random initialization in the CPU factory must not advance the caller's RNG.
    with torch.random.fork_rng(devices=[]):
        model = model_factory(_json_copy(manifest["model"]["config"]))
    if not isinstance(model, nn.Module):
        raise TypeError("model_factory must return an nn.Module")
    for name, tensor in model.state_dict().items():
        if name in model_state and isinstance(tensor, torch.Tensor) and (
            model_state[name].dtype != tensor.dtype
        ):
            raise ValueError(
                f"policy model tensor {name!r} dtype mismatch: saved {model_state[name].dtype}, "
                f"factory {tensor.dtype}"
            )
    model.load_state_dict(model_state)

    # 4. Restore saved preprocessing without fitting and compose the action path.
    observation = None
    if manifest["observation_processor"] is not None:
        observation = ObservationProcessor(spec, **manifest["observation_processor"])
        observation.load_state_dict(processor_state)
        for normalizer in observation.normalizers.values():
            if not all(torch.isfinite(value).all() for value in normalizer.buffers()) or (
                (normalizer.var < 0).any() or normalizer.count < 0
            ):
                raise ValueError("invalid observation processor statistics")
    elif processor_state:
        raise ValueError("processor tensors provided without an observation processor")
    action = (
        None
        if manifest["action_processor"] is None
        else ActionProcessor(**manifest["action_processor"])
    )
    policy = InferencePolicy(
        model,
        observation_processor=observation,
        action_processor=action,
        action_selector=action_selector,
        deterministic=manifest["inference"]["deterministic"],
    ).to(device).eval()
    policy.manifest = manifest
    return policy
