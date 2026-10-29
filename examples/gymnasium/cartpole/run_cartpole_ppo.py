"""Train PPO on CartPole using the shared on-policy training loop."""

import argparse
import json
from contextlib import ExitStack
from dataclasses import asdict, replace
from pathlib import Path

import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter

from deeprl.algorithms.on_policy import train_on_policy
from deeprl.algorithms.ppo import PPOLearner
from deeprl.checkpoint import load_checkpoint
from deeprl.env import make_vec_env
from deeprl.evaluate import Evaluator
from deeprl.logger import Logger
from deeprl.policies import CategoricalPolicy
from deeprl.value import ValueFunction
from examples.gymnasium.cartpole import ppo_cartpole_config as settings
from examples.gymnasium.cartpole.cartpole_networks import PolicyNetwork, ValueNetwork


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
        "policy_hidden_sizes": list(settings.POLICY_HIDDEN_SIZES),
        "value_hidden_sizes": list(settings.VALUE_HIDDEN_SIZES),
        "eval_every": settings.EVAL_EVERY,
        "eval_episodes": settings.EVAL_EPISODES,
        "eval_seed": settings.EVAL_SEED,
        "checkpoint_interval": settings.CHECKPOINT_INTERVAL,
        "resume_from": str(args.resume_from.resolve()) if args.resume_from else None,
    }

    # The shared trainer checks learner settings and restores training state.
    if args.resume_from:
        saved_config = load_checkpoint(args.resume_from)["config"]
        for name in ("env_id", "policy_hidden_sizes", "value_hidden_sizes"):
            if saved_config[name] != run_config[name]:
                raise ValueError(f"Checkpoint {name} does not match this experiment.")

    # Record resolved settings using plain names for Python classes and devices.
    learner_config = asdict(cfg)
    for name, value in learner_config.items():
        if isinstance(value, type):
            learner_config[name] = f"{value.__module__}.{value.__qualname__}"
        elif isinstance(value, torch.device):
            learner_config[name] = str(value)

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

    torch.manual_seed(cfg.seed)  # Seed before initializing network weights.

    # Close environments and flush logging even if training raises an exception.
    with ExitStack() as stack:
        envs = make_vec_env(settings.ENV_ID, settings.NUM_ENVS, device=cfg.device)
        stack.callback(envs.close)

        eval_env = stack.enter_context(gym.make(settings.ENV_ID))
        writer = stack.enter_context(
            SummaryWriter(log_dir=str(args.output_dir / "events"))
        )

        obs_dim = envs.single_observation_space.shape[0]
        num_actions = int(envs.single_action_space.n)

        policy = CategoricalPolicy(
            PolicyNetwork(obs_dim, num_actions, settings.POLICY_HIDDEN_SIZES)
        )
        value_fn = ValueFunction(ValueNetwork(obs_dim, settings.VALUE_HIDDEN_SIZES))
        learner = PPOLearner(policy, value_fn, cfg)

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
        )

    print(f"Run saved to {args.output_dir}")


if __name__ == "__main__":
    main()
