"""Save a trained policy as a folder, and load it back without the training code.

A policy folder holds the inference settings, the actor's weights and the
observation statistics, and nothing that is only needed for training (no
critic, optimizers or reward scaler). It does not hold code: loading also
needs the caller's model factory (the code that builds the network), its
dependencies, and any custom action selector::

    policy/
        manifest.json   # what the policy is: data spec, model, action rule, processing settings
        model.pt        # the actor's weights
        processor.pt    # observation statistics (mean, var, count) for the actor's features

A manifest looks like this::

    {
        "spec":                  {...},   # the actor's observations and the actions (a DataSpec)
        "model":                 {"id": "examples.hopper.build_policy", "config": {"obs_dim": 11, ...}},
        "decision":              {"id": "distribution"},   # the action selector
        "observation_processor": {"normalize": ["state"], "epsilon": 1e-8, "clip": 10.0},  # or None
        "action_processor":      {"source_bounds": ..., "target_bounds": ..., "clip": true},  # or None
        "inference":             {"deterministic": false},   # default mode of act()
        "provenance":            {...},   # versions, git commit, training step, ...
    }

Three entry points:

* ``save_policy``: write a folder from a live ``InferencePolicy``, for
  example at the end of training.
* ``export_policy_from_checkpoint``: write the same folder from a training
  checkpoint, without building the learner.
* ``load_policy``: read a folder back into an ``InferencePolicy``, ready to act.

Rules that keep this safe and correct:

* **Actor features only.** Privileged observations that only the critic
  sees are left out of the spec and the statistics, because the deployed
  policy does not have them.
* **JSON describes, it never imports.** The loader never imports or runs
  code named by the manifest. The caller supplies the code that builds the
  model (``model_factory``). The caller's ``model_id`` must match the saved
  id, which catches using the wrong factory by mistake. It is only a string
  comparison, though: the caller is responsible for passing the factory
  that really matches that id.
* **Tensor files are loaded with restricted unpickling.** With
  ``weights_only=True``, PyTorch only rebuilds tensors and plain values, not
  arbitrary Python objects. This is far safer than a full unpickle, but still
  only load files from sources you trust.
* **A folder is written once and never overwritten.** All three files are
  written to a temporary folder, which is then renamed into place in one
  step. A reader never sees weights from one save and statistics from
  another.

Example:
    Illustrative; ``learner``, ``spec``, ``build_policy`` and ``observation``
    come from the training or evaluation script.

    >>> save_policy(
    ...     "runs/hopper/policy",
    ...     learner.inference_policy,
    ...     spec=spec,
    ...     model_id="examples.hopper.build_policy",
    ...     model_config={"obs_dim": 11, "action_dim": 3, "hidden_sizes": [64, 64]},
    ... )
    >>> policy = load_policy(
    ...     "runs/hopper/policy", build_policy, model_id="examples.hopper.build_policy"
    ... )
    >>> command = policy.act(observation)
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
    """Return a plain JSON copy of a dictionary.

    Writing to JSON text and reading it back makes an independent copy and
    rejects anything JSON cannot store, including NaN and infinity.

    Raises:
        ValueError: If ``value`` is not a mapping, or holds NaN or infinity.
    """
    if not isinstance(value, Mapping):
        raise ValueError("policy configuration must be a mapping")
    return json.loads(json.dumps(dict(value), allow_nan=False))


def _actor_spec(spec: DataSpec) -> DataSpec:
    """Return the part of a spec the actor sees.

    Keeps observations with no groups (seen by everyone) and observations in
    the ``"actor"`` group. Drops critic-only (privileged) observations. The
    actions are kept unchanged.
    """
    return DataSpec(
        tuple(
            feature for feature in spec.observations
            if not feature.groups or "actor" in feature.groups
        ),
        spec.actions,
    )


def _selector_id(selector: Callable) -> str | None:
    """Return the saved name of a built-in action selector.

    Returns:
        ``"distribution"``, ``"greedy"`` or ``"deterministic"`` for the three
        built-in selectors, or ``None`` for a custom one. A custom selector
        needs a name from the caller (``decision_id``).
    """
    if selector is distribution_action:
        return "distribution"
    if selector is greedy_action:
        return "greedy"
    if selector is deterministic_action:
        return "deterministic"
    return None


def validate_policy_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Check a policy manifest and return an independent copy of it.

    Checks the structure (exactly the seven fields) and the meaning: the
    spec is valid and actor-only, and the processing settings can actually
    be built for that spec.

    Args:
        manifest: A manifest, from ``describe_policy`` or ``manifest.json``.

    Returns:
        A plain copy of the manifest.

    Raises:
        ValueError: If a field is missing, unknown or invalid.
    """
    # 1. Make a plain copy and require exactly the seven fields.
    data = _json_copy(manifest)
    if set(data) != {
        "spec", "model", "decision", "observation_processor", "action_processor",
        "inference", "provenance",
    }:
        raise ValueError("policy manifest has missing or unknown fields")

    # 2. The spec must be valid, with observations and actions, and hold only
    #    features the actor sees. A deployed policy cannot require a feature
    #    that only the critic had during training.
    spec = DataSpec.from_dict(data["spec"])
    if not spec.observations or not spec.actions:
        raise ValueError("policy spec must declare observations and actions")
    spec.validate_compatible(_actor_spec(spec))

    # 3. "model" is {"id", "config"} and "decision" is {"id"}. Each id is a
    #    non-empty string, and the model config is a dictionary.
    for field, keys in (("model", {"id", "config"}), ("decision", {"id"})):
        value = data[field]
        if not isinstance(value, dict) or set(value) != keys:
            raise ValueError(f"policy {field} must contain {sorted(keys)}")
        if not isinstance(value["id"], str) or not value["id"].strip():
            raise ValueError(f"policy {field} id must be a nonempty string")
    if not isinstance(data["model"]["config"], dict):
        raise ValueError("policy model config must be a mapping")

    # 4. "inference" holds only the default mode, a true boolean.
    inference = data["inference"]
    if not isinstance(inference, dict) or set(inference) != {"deterministic"}:
        raise ValueError("policy inference must contain deterministic")
    if type(inference["deterministic"]) is not bool:
        raise ValueError("policy deterministic setting must be boolean")

    # 5. Provenance is free-form, but must be a dictionary.
    if not isinstance(data["provenance"], dict):
        raise ValueError("policy provenance must be a mapping")

    # 6. Observation processing: None, or exactly its three settings. Building
    #    a processor checks them against the spec (for example, that each
    #    normalized feature exists and is floating-point).
    observation = data["observation_processor"]
    if observation is not None:
        if not isinstance(observation, dict) or set(observation) != {
            "normalize", "epsilon", "clip",
        }:
            raise ValueError("policy observation_processor settings are incomplete or unknown")
        ObservationProcessor(spec, **observation)

    # 7. Action processing: None, or exactly its three settings. Building a
    #    processor checks the settings themselves.
    action = data["action_processor"]
    if action is not None:
        if not isinstance(action, dict) or set(action) != {
            "source_bounds", "target_bounds", "clip",
        }:
            raise ValueError("policy action_processor settings are incomplete or unknown")
        processor = ActionProcessor(**action)

        # 8. Bounds only make sense for one continuous action feature, and
        #    they must fit its shape (one bound per action dimension, or one
        #    for all).
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
    """Build the manifest for a live policy, without its weights.

    The manifest describes what the policy really does: the actor's
    features, how the model is built, how actions are chosen, and its
    processing settings. Training checkpoints store it too, so a policy
    folder can be exported later (see ``export_policy_from_checkpoint``).

    Args:
        policy: The policy to describe, usually ``learner.inference_policy``.
        spec: The full training spec. Critic-only observations are left out
            of the manifest.
        model_id: A name for the code that builds the model, for example
            ``"examples.hopper.build_policy"``. Loading checks it.
        model_config: The arguments that code needs, as plain JSON values,
            for example ``{"obs_dim": 11, "action_dim": 3}``.
        decision_id: Only needed for a custom action selector: a name for it.
            Built-in selectors name themselves.
        provenance: A record of the code and run. ``None`` uses
            ``get_provenance()``.

    Returns:
        The checked manifest.

    Raises:
        ValueError: If the decision id is missing or wrong, the processing
            does not match the spec, or the manifest is invalid.
    """
    # 1. Name the action selector. Built-in selectors have fixed names. A
    #    custom selector needs an explicit name, and may not borrow a built-in
    #    one. A given name must agree with a built-in selector.
    known_id = _selector_id(policy.action_selector)
    if known_id is None and decision_id in ("distribution", "greedy", "deterministic"):
        raise ValueError("custom action selectors cannot use a built-in decision id")
    if decision_id is None:
        decision_id = known_id
    elif known_id is not None and decision_id != known_id:
        raise ValueError("decision id does not match the built-in action selector")
    if decision_id is None:
        raise ValueError("a custom action selector requires an explicit decision_id")

    # 2. Keep only the actor's features. If the spec has critic-only
    #    features, an observation processor is required: it is what removes
    #    them before the model sees the observations.
    actor_spec = _actor_spec(spec)
    if policy.observation_processor is None and actor_spec.observations != spec.observations:
        raise ValueError(
            "non-actor observations require an observation processor to enforce actor selection"
        )
    spec = actor_spec

    # 3. Describe the observation processing, limited to the actor. The
    #    processor's actor features must be the spec's actor features, and
    #    only actor features stay in the "normalize" list.
    observation = None
    if policy.observation_processor is not None:
        processor = policy.observation_processor
        DataSpec(_actor_spec(processor.spec).observations, ()).validate_compatible(
            DataSpec(spec.observations, ()),
        )
        observation = processor.config
        names = {feature.name for feature in spec.observations}
        observation["normalize"] = [name for name in processor.normalize if name in names]

    # 4. Put the manifest together and check it.
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
    """Return the observation statistics of the actor's features only.

    The processor stores each feature's statistics under its position in
    the ``normalize`` list ("0", "1", ...). Dropping critic-only features can
    shift those positions. For example, with ``normalize=["base_velocity",
    "joints"]`` where ``base_velocity`` is critic-only, the statistics of
    ``joints`` move from "1" to "0".

    Args:
        source: The training processor (all features), or ``None``.
        manifest: The checked manifest, whose processor settings list the
            actor's features.

    Returns:
        The state dict of an actor-only processor, or ``{}`` without a
        processor.
    """
    if source is None:
        return {}

    # 1. Build an empty processor with the manifest's actor-only settings.
    exported = ObservationProcessor(
        DataSpec.from_dict(manifest["spec"]),
        **manifest["observation_processor"],
    )

    # 2. Copy each feature's statistics from its old position to its new one.
    for index, name in enumerate(exported.normalize):
        old_index = source.normalize.index(name)
        exported.normalizers[str(index)].load_state_dict(
            source.normalizers[str(old_index)].state_dict()
        )
    return exported.state_dict()


def _tensor_state(state: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Check a state dict and return independent CPU copies of its tensors.

    Copies keep the saved values fixed even if training continues, and CPU
    tensors load on any machine.

    Raises:
        ValueError: If any key is not a string or any value is not a tensor.
    """
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
    """Save a live policy as a new policy folder.

    Writes ``manifest.json``, ``model.pt`` and ``processor.pt``. The folder
    must not exist yet; an existing policy is never overwritten.

    Args:
        directory: The new folder, for example ``runs/hopper/policy``.
        policy: The policy to save, usually ``learner.inference_policy``.
        spec: The full training spec (see ``describe_policy``).
        model_id: A name for the code that builds the model.
        model_config: The arguments that code needs, as plain JSON values.
        decision_id: Only needed for a custom action selector.
        provenance: A record of the code and run. ``None`` uses
            ``get_provenance()``.

    Raises:
        FileExistsError: If ``directory`` already exists.
        ValueError: If the policy cannot be described (see
            ``describe_policy``).
    """

    # 1. The destination must be new. Describe the policy.
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

    # 2. Copy the actor's weights and the actor's observation statistics.
    model_state = _tensor_state(policy.model.state_dict())
    processor_state = _tensor_state(_processor_state(policy.observation_processor, manifest))

    # 3. Write the folder.
    _write_policy(directory, manifest, model_state, processor_state)


def _write_policy(
    directory: Path,
    manifest: dict[str, Any],
    model_state: dict[str, torch.Tensor],
    processor_state: dict[str, torch.Tensor],
) -> None:
    """Write a policy folder so it appears complete or not at all.

    Raises:
        FileExistsError: If ``directory`` already exists.
    """
    # 1. Check again that the destination is new, right before writing.
    if directory.exists() or directory.is_symlink():
        raise FileExistsError(f"policy destination already exists: {directory}")

    # 2. Create a hidden temporary folder next to the destination. Being in
    #    the same parent folder makes the final rename a single step.
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{directory.name}-",
            dir=directory.parent,
        )
    )
    try:
        # 3. Write the three files into the temporary folder.
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        torch.save(model_state, temporary / "model.pt")
        torch.save(processor_state, temporary / "processor.pt")

        # 4. Rename the complete folder to its final name.
        temporary.rename(directory)
    finally:
        # 5. If anything failed, remove the temporary folder.
        if temporary.exists():
            shutil.rmtree(temporary)


def export_policy_from_checkpoint(
    checkpoint_path: str | Path,
    directory: str | Path,
) -> None:
    """Write a policy folder from an on-policy training checkpoint.

    Useful for exporting a policy from the middle of a run, or from a run
    that stopped early. The learner is not built: the checkpoint already
    holds the manifest, the actor's weights and the observation statistics.

    Only what the actor needs is exported. Optimizers, the critic and the
    reward scaler stay in the checkpoint. The result is the same kind of
    folder as ``save_policy`` writes, and is loaded with ``load_policy``.

    Args:
        checkpoint_path: The training checkpoint. It must contain
            ``policy_manifest``, the actor's weights (``learner["policy"]``)
            and the processing settings and statistics.
        directory: The new policy folder. It must not exist yet.

    Raises:
        ValueError: If the checkpoint is incomplete, its parts disagree with
            the manifest, or the statistics are invalid.
        FileExistsError: If ``directory`` already exists.
    """
    # 1. Load and check the checkpoint, the manifest and the actor's weights.
    #    The saved action processing must match the manifest.
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

    # 2. Observation processing.
    settings = preprocessing["observation"]
    source = None
    if settings is None:
        # 2a. Processing was off: the manifest must agree, and there must be
        #     no saved statistics.
        if manifest["observation_processor"] is not None or "observation_processor" in checkpoint:
            raise ValueError("checkpoint observation processing does not match the policy manifest")
    else:
        # 2b. Processing was on: rebuild the training processor (all features).
        if set(settings) != {"spec", "config"} or "observation_processor" not in checkpoint:
            raise ValueError("checkpoint is missing observation processor settings or state")
        spec = DataSpec.from_dict(settings["spec"])
        source = ObservationProcessor(spec, **settings["config"])

        # 2c. Its actor part must match the manifest: the same actor features
        #     and the same settings.
        actor_spec = DataSpec.from_dict(manifest["spec"])
        DataSpec(_actor_spec(spec).observations, ()).validate_compatible(
            DataSpec(actor_spec.observations, ()),
        )
        actor_names = {feature.name for feature in actor_spec.observations}
        actor_config = source.config
        actor_config["normalize"] = [name for name in source.normalize if name in actor_names]
        if actor_config != manifest["observation_processor"]:
            raise ValueError("checkpoint observation processing does not match the policy manifest")

        # 2d. Load the saved statistics and reject broken ones (NaN, infinity,
        #     negative variance or count).
        source.load_state_dict(_tensor_state(checkpoint["observation_processor"]))
        for normalizer in source.normalizers.values():
            if not all(torch.isfinite(value).all() for value in normalizer.buffers()) or (
                (normalizer.var < 0).any() or normalizer.count < 0
            ):
                raise ValueError("invalid observation processor statistics")

    # 2e. Keep the statistics of the actor's features only.
    processor_state = _tensor_state(_processor_state(source, manifest))

    # 3. Record when the checkpoint was taken, then write the same kind of
    #    folder as save_policy.
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
    """Load a policy folder into an ``InferencePolicy`` in evaluation mode.

    The model is built by the caller's own code (``model_factory``), never
    by code named in the JSON. The saved statistics are restored as they
    are; nothing is fitted.

    Args:
        directory: The policy folder.
        model_factory: Trusted code that builds the actor network from the
            saved config, for example ``build_policy(config) -> nn.Module``.
            It builds only the actor, never a learner, on the CPU, with the
            same parameter dtypes as the saved weights.
        model_id: The identifier of ``model_factory``. It must equal the
            saved identifier. This compares strings only; the caller remains
            responsible for passing the trusted factory that matches it.
        device: Device to put the policy on.
        action_selector: Only for a custom decision: the selector to use.
            Built-in decisions need nothing here.
        decision_id: Only for a custom decision: its name, which must equal
            the saved one.
        expected_spec: What the environment provides. If given, the saved
            spec must match its actor part.

    Returns:
        The policy in evaluation mode, with the manifest attached as
        ``policy.manifest``.

    Raises:
        ValueError: If the folder is invalid, or the factory, selector or spec
            does not match it.
        TypeError: If ``model_factory`` does not return an ``nn.Module``.
    """

    # 1. Read and check the manifest before building or loading anything.
    #    The model id must match the caller's factory, and the saved spec
    #    must match what the environment provides.
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

    # 2. Choose the action selector. Without one from the caller, use the
    #    built-in selector the manifest names. A custom selector must come
    #    with the matching custom name, and may not replace a built-in one.
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

    # 3. Load the weights and statistics (tensors only, restricted loading).
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

    # 4. Build the model with the caller's factory. fork_rng saves and
    #    restores the CPU random-number state, so the random initial weights
    #    do not change the caller's random numbers.
    with torch.random.fork_rng(devices=[]):
        model = model_factory(_json_copy(manifest["model"]["config"]))
    if not isinstance(model, nn.Module):
        raise TypeError("model_factory must return an nn.Module")

    # 5. Saved and built tensors must have the same dtype. load_state_dict
    #    would otherwise convert silently, for example float64 to float32.
    for name, tensor in model.state_dict().items():
        if name in model_state and isinstance(tensor, torch.Tensor) and (
            model_state[name].dtype != tensor.dtype
        ):
            raise ValueError(
                f"policy model tensor {name!r} dtype mismatch: saved {model_state[name].dtype}, "
                f"factory {tensor.dtype}"
            )
    model.load_state_dict(model_state)

    # 6. Rebuild the observation processor with the saved statistics (no
    #    fitting), and reject broken ones. Statistics without a processor
    #    mean the folder is inconsistent.
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

    # 7. Rebuild the action processor from its settings.
    action = (
        None
        if manifest["action_processor"] is None
        else ActionProcessor(**manifest["action_processor"])
    )

    # 8. Put the policy together, move it to the device and switch to
    #    evaluation mode. Keep the manifest for reference.
    policy = InferencePolicy(
        model,
        observation_processor=observation,
        action_processor=action,
        action_selector=action_selector,
        deterministic=manifest["inference"]["deterministic"],
    ).to(device).eval()
    policy.manifest = manifest
    return policy
