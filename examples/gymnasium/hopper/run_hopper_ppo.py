"""Train PPO on Hopper using the shared on-policy training loop."""

import argparse
import json
from contextlib import ExitStack
from dataclasses import replace
from functools import partial
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import AutoresetMode
from gymnasium.wrappers.vector import NumpyToTorch

from deeprl.training.train_on_policy import train_on_policy
from deeprl.algorithms.ppo import PPOLearner
from deeprl.checkpoint import load_checkpoint
from deeprl.data_spec import DataSpec, FeatureSpec
from deeprl.evaluate import Evaluator
from deeprl.lifecycle import (
    config_to_jsonable,
    get_provenance,
    seed_torch,
    validate_run_config,
)
from deeprl.logger import Logger
from deeprl.normalization import RewardScaler
from deeprl.policies import DiagonalGaussianPolicy
from deeprl.policy_io import describe_policy, save_policy
from deeprl.processors import ActionProcessor, ObservationProcessor
from deeprl.value import ValueFunction
from examples.gymnasium.hopper import (
    ppo_hopper_config as settings,
)
from examples.gymnasium.hopper.hopper_networks import (
    PolicyNetwork,
    ValueNetwork,
    build_policy,
)


def main() -> None:
    """Train with resolved settings and export the final inference policy."""
    # 0. Resolve config defaults and command-line overrides.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=settings.PPO_CONFIG.seed)
    parser.add_argument(
        "--device",
        type=torch.device,
        default=settings.PPO_CONFIG.device,
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=settings.PPO_CONFIG.total_timesteps,
    )
    parser.add_argument(
        "--vectorization-mode",
        choices=("sync", "async"),
        default="sync",
        help="Run CPU environments sequentially or in separate processes.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()

    cfg = replace(
        settings.PPO_CONFIG,
        seed=args.seed,
        device=args.device,
        total_timesteps=args.total_timesteps,
    )

    # 1. Validate resume settings and record the new run.
    run_config = {
        "env_id": settings.ENV_ID,
        "vectorization_mode": args.vectorization_mode,
        "policy_network": f"{PolicyNetwork.__module__}.{PolicyNetwork.__qualname__}",
        "value_network": f"{ValueNetwork.__module__}.{ValueNetwork.__qualname__}",
        "policy_hidden_sizes": list(settings.POLICY_HIDDEN_SIZES),
        "value_hidden_sizes": list(settings.VALUE_HIDDEN_SIZES),
        "log_std_init": settings.LOG_STD_INIT,
        "clip_actions": True,
        "obs_normalization": settings.OBS_NORMALIZATION,
        "reward_scaling": settings.REWARD_SCALING,
        "eval_every": settings.EVAL_EVERY,
        "eval_episodes": settings.EVAL_EPISODES,
        "eval_seed": settings.EVAL_SEED,
        "checkpoint_interval": settings.CHECKPOINT_INTERVAL,
        "resume_from": str(args.resume_from.resolve()) if args.resume_from else None,
    }

    # The shared trainer checks learner settings and restores training state.
    if args.resume_from:
        saved_config = load_checkpoint(args.resume_from)["config"]
        validate_run_config(
            saved_config,
            run_config,
            keys=(
                "env_id",
                "policy_network",
                "value_network",
                "policy_hidden_sizes",
                "value_hidden_sizes",
                "log_std_init",
                "clip_actions",
                "obs_normalization",
                "reward_scaling",
            ),
        )

    # Record resolved settings using plain names for Python classes and devices.
    learner_config = config_to_jsonable(cfg)

    # Require a fresh directory so an earlier run cannot be overwritten.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "experiment": run_config,
        "learner": learner_config,
        "num_envs": settings.NUM_ENVS,
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    seed_torch(cfg.seed)  # Seed before initializing network weights.

    # 2. Create environments and logging resources with guaranteed cleanup.
    with ExitStack() as stack:
        # Environments receive native commands; the policy converts its samples.
        vector_kwargs: dict[str, object] = {"autoreset_mode": AutoresetMode.DISABLED}
        if args.vectorization_mode == "async":
            # Spawn workers without inheriting the parent's CUDA runtime.
            vector_kwargs["context"] = "spawn"
        envs = gym.make_vec(
            settings.ENV_ID,
            num_envs=settings.NUM_ENVS,
            vectorization_mode=args.vectorization_mode,
            vector_kwargs=vector_kwargs,
            wrappers=(partial(gym.wrappers.DtypeObservation, dtype=np.float32),),
        )
        stack.callback(envs.close)
        envs = NumpyToTorch(envs, device=cfg.device)

        eval_env = stack.enter_context(
            gym.wrappers.DtypeObservation(
                gym.make(settings.ENV_ID),
                dtype=np.float32,
            )
        )
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            writer = None
        else:
            writer = stack.enter_context(
                SummaryWriter(log_dir=str(args.output_dir / "events"))
            )

        # 3. Construct the models, processors, and learner.
        # Vector batches: (N, obs_dim) observations and (N, action_dim) continuous actions.
        obs_dim = envs.single_observation_space.shape[0]
        action_dim = envs.single_action_space.shape[0]
        action_processor = ActionProcessor(
            source_bounds=(envs.single_action_space.low, envs.single_action_space.high),
            clip=True,
        )

        mean_network = PolicyNetwork(
            obs_dim,
            action_dim,
            settings.POLICY_HIDDEN_SIZES,
        )
        value_network = ValueNetwork(obs_dim, settings.VALUE_HIDDEN_SIZES)

        # Both models share the state contract declared by the environment adapter.
        obs_processor = ObservationProcessor(
            DataSpec(
                observations=(
                    FeatureSpec(
                        "state",
                        tuple(envs.single_observation_space.shape),
                        envs.single_observation_space.dtype.name,
                    ),
                ),
                actions=(),
            ),
            normalize=() if settings.OBS_NORMALIZATION is None else ("state",),
            **(settings.OBS_NORMALIZATION or {}),
        )

        policy = DiagonalGaussianPolicy(
            mean_network,
            action_dim=action_dim,
            log_std_init=settings.LOG_STD_INIT,
        )
        value_fn = ValueFunction(value_network)

        # Reward scaling affects training targets, not environment rewards.
        reward_scaler = None
        if settings.REWARD_SCALING is not None:
            reward_scaler = RewardScaler(
                settings.NUM_ENVS,
                gamma=cfg.gamma,
                **settings.REWARD_SCALING,
            )
        learner = PPOLearner(
            policy,
            value_fn,
            cfg,
            obs_processor=obs_processor,
            action_processor=action_processor,
            reward_scaler=reward_scaler,
        )

        # 4. Describe the actor's inference interface independently of training state.
        learner.inference_policy.deterministic = True
        policy_spec = DataSpec(
            observations=obs_processor.spec.observations,
            actions=(
                FeatureSpec("action", tuple(envs.single_action_space.shape), "float32"),
            ),
        )
        model_id = f"{build_policy.__module__}.{build_policy.__name__}"
        model_config = {
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "hidden_sizes": list(settings.POLICY_HIDDEN_SIZES),
            "log_std_init": settings.LOG_STD_INIT,
        }
        provenance = {**get_provenance(), "env_id": settings.ENV_ID}
        policy_manifest = describe_policy(
            learner.inference_policy,
            spec=policy_spec,
            model_id=model_id,
            model_config=model_config,
            provenance=provenance,
        )

        # 5. Configure evaluation and training metrics.
        evaluator = Evaluator(
            eval_env,
            eval_every=settings.EVAL_EVERY,
            num_episodes=settings.EVAL_EPISODES,
            device=cfg.device,
            seed=settings.EVAL_SEED,
        )
        logger = Logger(
            settings.NUM_ENVS,
            cfg.log_every,
            writer,
            cfg.device,
            metrics_path=args.output_dir / "metrics.jsonl",
            total_timesteps=cfg.total_timesteps,
            show_progress=True,
        )
        stack.callback(logger.finish)  # Flush metrics before closing the writer.

        # 6. Collect rollouts, update the learner, and save training checkpoints.
        train_on_policy(
            envs=envs,
            learner=learner,
            logger=logger,
            evaluator=evaluator,
            checkpoint_dir=args.output_dir / "checkpoints",
            checkpoint_interval=settings.CHECKPOINT_INTERVAL,
            run_config=run_config,
            resume_from=args.resume_from,
            policy_manifest=policy_manifest,
        )

        # 7. Export the completed actor and frozen preprocessing for inference.
        rollout_size = cfg.rollout_steps * settings.NUM_ENVS
        save_policy(
            args.output_dir / "policy",
            learner.inference_policy,
            spec=policy_spec,
            model_id=model_id,
            model_config=model_config,
            provenance={
                **provenance,
                "training_step": cfg.total_timesteps // rollout_size * rollout_size,
            },
        )

    print(f"Run saved to {args.output_dir}")


if __name__ == "__main__":
    main()
