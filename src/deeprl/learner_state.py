"""Checks used when a learner resumes training from a checkpoint.

A learner saves its state as a dictionary of named parts, for example::

    {"policy": ..., "value_fn": ..., "optim_actor": ..., "optim_critic": ...}

The two checks here run before anything is restored, so the most common
mistakes fail early, while the learner is still untouched:

* ``validate_resume_config``: the new run's settings may differ from the
  saved ones only in fields that are safe to change, such as the learning
  rate or the total number of steps.
* ``validate_state_keys``: the checkpoint holds exactly the parts this
  learner expects, no more and no fewer.

They check names and settings only, not tensor contents. An error found
later, while loading tensors (for example, a shape mismatch in an optimizer
state), can still happen after another part was already restored. Resuming
is not all-or-nothing, so after such an error, build a new learner instead of
using the partly restored one.

Resume order, as run by the trainer::

    overrides = learner.validate_resume_config(saved_config, current_config)
    learner.load_state_dict(checkpoint["learner"])   # calls validate_state_keys first
    learner.apply_resume_overrides(overrides)        # for example, set the new learning rate
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any


def validate_state_keys(
    state: Mapping[str, Any],
    expected_keys: Collection[str],
) -> None:
    """Check that a saved learner state has exactly the expected parts.

    Each learner calls this at the start of ``load_state_dict``, before it
    restores anything.

    Args:
        state: The learner state from a checkpoint.
        expected_keys: The part names this learner saves, for example
            ``{"policy", "value_fn", "optim_actor", "optim_critic"}``.

    Raises:
        ValueError: If ``state`` is not a dictionary, or a part is missing or
            unexpected. The message lists both.

    Example:
        >>> validate_state_keys({"policy": {}}, {"policy", "value_fn"})
        Traceback (most recent call last):
        ValueError: checkpoint learner state differs: missing=['value_fn'], unexpected=[]
    """
    # 1. The state must be a dictionary.
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint learner state must be a mapping")

    # 2. Compare the names: "missing" are expected but not saved; "unexpected"
    #    are saved but not expected (for example, a reward scaler saved by a
    #    run that had one, loaded into a learner without one).
    expected = set(expected_keys)
    missing = expected - state.keys()
    unexpected = state.keys() - expected

    # 3. Any difference means the checkpoint was made by a different setup.
    if missing or unexpected:
        raise ValueError(
            f"checkpoint learner state differs: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )


def validate_resume_config(
    saved_config: Mapping[str, Any],
    current_config: Mapping[str, Any],
    *,
    mutable_on_resume: Collection[str],
) -> dict[str, Any]:
    """Check a resumed run's settings and return the allowed changes.

    Some settings may change when training resumes, such as the learning
    rate or the total number of steps. Others may not: changing ``gamma``
    halfway would make the saved networks and optimizers mean something
    else.

    Args:
        saved_config: The learner config stored in the checkpoint, as a plain
            dictionary.
        current_config: The learner config of the run that is resuming, in
            the same form.
        mutable_on_resume: Names of the fields allowed to change. Each learner
            defines its own set.

    Returns:
        The fields whose values changed, with their new values. Every one of
        them is in ``mutable_on_resume``. Empty if nothing changed.

    Raises:
        ValueError: If the two configs have different field names, or a field
            outside ``mutable_on_resume`` changed.

    Example:
        >>> validate_resume_config(
        ...     {"lr_actor": 3e-4, "gamma": 0.99},
        ...     {"lr_actor": 1e-4, "gamma": 0.99},
        ...     mutable_on_resume={"lr_actor"},
        ... )
        {'lr_actor': 0.0001}
    """
    # 1. Both configs must have the same fields. A field added or removed
    #    means the code or the algorithm changed since the checkpoint.
    missing = current_config.keys() - saved_config.keys()
    unexpected = saved_config.keys() - current_config.keys()
    if missing or unexpected:
        raise ValueError(
            f"checkpoint learner config differs: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )

    # 2. Fields that are not allowed to change must keep their saved values.
    changed = [
        name for name, value in current_config.items()
        if name not in mutable_on_resume
        and saved_config[name] != value
    ]
    if changed:
        raise ValueError(f"checkpoint learner config differs: {', '.join(changed)}")

    # 3. Return every field that changed. Step 2 already rejected the others,
    #    so these are all allowed changes.
    return {
        name: current_config[name]
        for name in current_config
        if saved_config[name] != current_config[name]
    }
