"""Train the pedagogical one-step actor-critic on CartPole.

Saved policy/value weights are inference-only, not resumable training checkpoints.
"""

import argparse
import json
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import torch
from gymnasium.wrappers import RecordVideo

from deeprl.algorithms.one_step_actor_critic import OneStepACConfig, one_step_actor_critic
from deeprl.checkpoint import save_checkpoint
from deeprl.env import make_vec_env
from deeprl.evaluate import Evaluator, evaluate_policy
from deeprl.lifecycle import config_to_jsonable, seed_torch
from deeprl.logger import Logger
from deeprl.policies import CategoricalPolicy
from deeprl.value import ValueFunction
from examples.gymnasium.cartpole.cartpole_networks import PolicyNetwork, ValueNetwork


ENV_ID = "CartPole-v1"
NUM_ENVS = 4
HIDDEN_SIZES = (64, 64)
EVAL_EVERY = 10_000
EVAL_EPISODES = 10
VIDEO_EPISODES = 4
DEFAULT_CONFIG = OneStepACConfig(
    total_timesteps=60_000,
    seed=0,
    device=torch.device("cpu"),
    log_every=2_000,
    lr_actor=3e-4,
    lr_critic=3e-3,
    discount_actor=False,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed)
    parser.add_argument("--device", type=torch.device, default=DEFAULT_CONFIG.device)
    parser.add_argument(
        "--total-timesteps", type=int, default=DEFAULT_CONFIG.total_timesteps,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    if args.total_timesteps < NUM_ENVS:
        parser.error(f"--total-timesteps must be at least {NUM_ENVS}")

    cfg = replace(
        DEFAULT_CONFIG,
        seed=args.seed,
        device=args.device,
        total_timesteps=args.total_timesteps,
    )
    run_config = {
        "env_id": ENV_ID,
        "policy_hidden_sizes": list(HIDDEN_SIZES),
        "value_hidden_sizes": list(HIDDEN_SIZES),
        "eval_every": EVAL_EVERY,
        "eval_episodes": EVAL_EPISODES,
        "eval_seed": cfg.seed + 10_000,
        "video_episodes": 0 if args.no_video else VIDEO_EPISODES,
        "video_seed": cfg.seed + 20_000,
    }
    learner_config = config_to_jsonable(cfg)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "config.json").write_text(
        json.dumps({
            "experiment": run_config,
            "learner": learner_config,
            "num_envs": NUM_ENVS,
        }, indent=2) + "\n",
        encoding="utf-8",
    )

    seed_torch(cfg.seed)  # Initialize networks reproducibly before training.
    with ExitStack() as stack:
        envs = make_vec_env(ENV_ID, NUM_ENVS, device=cfg.device)
        stack.callback(envs.close)
        eval_env = stack.enter_context(gym.make(ENV_ID))

        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            writer = None
        else:
            writer = stack.enter_context(
                SummaryWriter(log_dir=str(args.output_dir / "events"))
            )

        obs_dim = envs.single_observation_space.shape[0]
        num_actions = int(envs.single_action_space.n)
        policy = CategoricalPolicy(PolicyNetwork(obs_dim, num_actions, HIDDEN_SIZES))
        value_fn = ValueFunction(ValueNetwork(obs_dim, HIDDEN_SIZES))
        evaluator = Evaluator(
            eval_env,
            eval_every=EVAL_EVERY,
            num_episodes=EVAL_EPISODES,
            device=cfg.device,
            seed=run_config["eval_seed"],
        )
        logger = Logger(
            NUM_ENVS, cfg.log_every, writer, cfg.device,
            metrics_path=args.output_dir / "metrics.jsonl",
            total_timesteps=cfg.total_timesteps,
            show_progress=True,
            ev_name="diagnostics/td_target_explained_variance",
        )
        stack.callback(logger.finish)  # Flush before the writer is closed.
        one_step_actor_critic(
            envs=envs, policy=policy, value_fn=value_fn, cfg=cfg,
            logger=logger, evaluator=evaluator,
        )

    # Keep evaluation-compatible weights without claiming resumable run state.
    # The one-step API does not expose optimizer, RNG, or continuation state.
    weights_path = args.output_dir / "weights" / "policy.pt"
    save_checkpoint(weights_path, {
        "artifact_type": "inference_weights",
        "resume_supported": False,
        "algorithm": "one_step_actor_critic",
        "learner": {"policy": policy.state_dict(), "value_fn": value_fn.state_dict()},
        "progress": {"global_step": logger.global_step},
        "learner_config": learner_config,
        "config": run_config,
        "num_envs": NUM_ENVS,
    })

    if not args.no_video:
        video_env = gym.make(ENV_ID, render_mode="rgb_array")
        try:
            video_env = RecordVideo(
                video_env,
                video_folder=str(args.output_dir / "videos"),
                name_prefix="policy",
                episode_trigger=lambda episode: True,
                disable_logger=True,
            )
            evaluate_policy(
                video_env, policy,
                num_episodes=VIDEO_EPISODES,
                device=cfg.device,
                deterministic=True,
                seed=run_config["video_seed"],
            )
        finally:
            video_env.close()

    print(f"Run saved to {args.output_dir}")
    print(f"Inference-only weights (no resume): {weights_path}")


if __name__ == "__main__":
    main()
