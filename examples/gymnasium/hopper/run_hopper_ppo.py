"""Train PPO on Hopper using the shared on-policy training loop."""

import argparse
import json
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import torch
from gymnasium.vector import AutoresetMode
from gymnasium.wrappers.vector import NumpyToTorch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from deeprl.algorithms.on_policy import train_on_policy
from deeprl.algorithms.ppo import PPOLearner
from deeprl.checkpoint import load_checkpoint
from deeprl.evaluate import Evaluator
from deeprl.lifecycle import config_to_jsonable, seed_torch, validate_run_config
from deeprl.logger import Logger
from deeprl.normalization import RewardScaler, RunningNormalizer
from deeprl.policies import DiagonalGaussianPolicy
from deeprl.value import ValueFunction
from examples.gymnasium.hopper import (
    ppo_hopper_config as settings,
)
from examples.gymnasium.hopper.hopper_networks import (
    PolicyNetwork,
    ValueNetwork,
)


def main() -> None:
    # Config supplies defaults; command-line arguments override selected values.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=settings.PPO_CONFIG.seed)
    parser.add_argument(
        "--device", type=torch.device, default=settings.PPO_CONFIG.device
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=settings.PPO_CONFIG.total_timesteps,
    )
    parser.add_argument(
        "--vectorization-mode", choices=("sync", "async"), default="sync",
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

    # Preserve settings needed to reconstruct and identify this experiment.
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

    # Close environments and flush logging even if training raises an exception.
    with ExitStack() as stack:
        # Clip executed actions only; the buffer retains raw Gaussian samples.
        vector_kwargs: dict[str, object] = {"autoreset_mode": AutoresetMode.DISABLED}
        if args.vectorization_mode == "async":
            # Spawn workers without inheriting the parent's CUDA runtime.
            vector_kwargs["context"] = "spawn"
        envs = gym.make_vec(
            settings.ENV_ID,
            num_envs=settings.NUM_ENVS,
            vectorization_mode=args.vectorization_mode,
            vector_kwargs=vector_kwargs,
            wrappers=(gym.wrappers.ClipAction,),
        )
        stack.callback(envs.close)
        envs = NumpyToTorch(envs, device=cfg.device)

        eval_env = stack.enter_context(
            gym.wrappers.ClipAction(gym.make(settings.ENV_ID))
        )
        writer = stack.enter_context(
            SummaryWriter(log_dir=str(args.output_dir / "events"))
        )

        obs_dim = envs.single_observation_space.shape[0]
        action_dim = envs.single_action_space.shape[0]

        mean_network: nn.Module = PolicyNetwork(
            obs_dim, action_dim, settings.POLICY_HIDDEN_SIZES
        )
        value_network: nn.Module = ValueNetwork(obs_dim, settings.VALUE_HIDDEN_SIZES)

        # Both networks use the same moments; only the shared trainer updates them.
        obs_normalizer = None
        if settings.OBS_NORMALIZATION is not None:
            obs_normalizer = RunningNormalizer((obs_dim,), **settings.OBS_NORMALIZATION)
            mean_network = nn.Sequential(obs_normalizer, mean_network)
            value_network = nn.Sequential(obs_normalizer, value_network)

        policy = DiagonalGaussianPolicy(
            mean_network,
            action_dim=action_dim,
            log_std_init=settings.LOG_STD_INIT,
        )
        value_fn = ValueFunction(value_network)
        learner = PPOLearner(policy, value_fn, cfg)

        # Reward scaling affects training targets, not environment rewards.
        reward_scaler = None
        if settings.REWARD_SCALING is not None:
            reward_scaler = RewardScaler(
                settings.NUM_ENVS, gamma=cfg.gamma, **settings.REWARD_SCALING
            )

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

        train_on_policy(
            envs=envs,
            learner=learner,
            logger=logger,
            evaluator=evaluator,
            checkpoint_dir=args.output_dir / "checkpoints",
            checkpoint_interval=settings.CHECKPOINT_INTERVAL,
            run_config=run_config,
            resume_from=args.resume_from,
            obs_normalizer=obs_normalizer,
            reward_scaler=reward_scaler,
        )

    print(f"Run saved to {args.output_dir}")


if __name__ == "__main__":
    main()
