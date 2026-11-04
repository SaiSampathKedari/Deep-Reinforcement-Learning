"""Compatibility imports; use on_policy_base and training.train_on_policy."""

from deeprl.algorithms.on_policy_base import OnPolicyConfig, OnPolicyLearner
from deeprl.training.train_on_policy import train_on_policy

__all__ = ["OnPolicyConfig", "OnPolicyLearner", "train_on_policy"]
