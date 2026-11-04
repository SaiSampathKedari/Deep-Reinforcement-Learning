"""Evaluate a saved Hopper policy, with optional rendering or video."""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import mujoco
import torch
from gymnasium.envs.mujoco.hopper_v5 import DEFAULT_CAMERA_CONFIG
from torch import nn

from deeprl.checkpoint import load_checkpoint
from deeprl.evaluate import evaluate_policy
from deeprl.normalization import RunningNormalizer
from deeprl.policies import DiagonalGaussianPolicy
from examples.gymnasium.hopper.hopper_networks import (
    PolicyNetwork,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument(
        "--mode", choices=("deterministic", "stochastic"), default="deterministic"
    )
    parser.add_argument("--device", type=torch.device, default=torch.device("cpu"))
    parser.add_argument("--output", type=Path, help="Evaluation JSON output path.")

    display = parser.add_mutually_exclusive_group()
    display.add_argument("--render", action="store_true")
    display.add_argument("--video-dir", type=Path)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")

    # Reconstruct the saved policy, independent of the current training config.
    checkpoint = load_checkpoint(args.checkpoint)
    settings = checkpoint["config"]
    if settings["env_id"] != "Hopper-v5":
        raise ValueError("Checkpoint env_id does not match Hopper-v5.")
    if settings["policy_network"] != (
        f"{PolicyNetwork.__module__}.{PolicyNetwork.__qualname__}"
    ):
        raise ValueError("Checkpoint policy_network does not match this evaluator.")
    if args.video_dir and args.video_dir.exists():
        raise FileExistsError("Choose a new --video-dir to preserve existing videos.")

    name = f"{args.checkpoint.stem}_{args.mode}_seed{args.seed}_episodes{args.episodes}"
    render_mode = "rgb_array" if args.video_dir else ("human" if args.render else None)

    # Follow the robot in the window; use the model's tracking camera for videos.
    env = gym.make(
        settings["env_id"],
        render_mode=render_mode,
        camera_name="track",
        default_camera_config={
            **DEFAULT_CAMERA_CONFIG,
            "type": mujoco.mjtCamera.mjCAMERA_TRACKING,
        },
    )
    try:
        # Apply the same action clipping used during training.
        if settings["clip_actions"]:
            env = gym.wrappers.ClipAction(env)
        if args.video_dir:
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=str(args.video_dir),
                name_prefix=name,
                episode_trigger=lambda episode: True,
            )

        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        mean_network: nn.Module = PolicyNetwork(
            obs_dim, action_dim, tuple(settings["policy_hidden_sizes"])
        )
        # Older checkpoints used raw observations and have no normalizer settings.
        obs_normalization = settings.get("obs_normalization")
        if obs_normalization is not None:
            mean_network = nn.Sequential(
                RunningNormalizer((obs_dim,), **obs_normalization), mean_network
            )
        policy = DiagonalGaussianPolicy(
            mean_network,
            action_dim=action_dim,
            log_std_init=settings["log_std_init"],
        ).to(args.device)
        # Restore network weights, learned log std and frozen observation statistics.
        policy.load_state_dict(checkpoint["learner"]["policy"])

        returns, lengths = evaluate_policy(
            env,
            policy,
            num_episodes=args.episodes,
            device=args.device,
            deterministic=args.mode == "deterministic",
            seed=args.seed,
        )
    finally:
        env.close()  # Also finalizes the last recorded episode.

    # Episode scores are undiscounted; this std is across evaluation episodes.
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "env_id": settings["env_id"],
        "training_step": checkpoint["progress"]["global_step"],
        "mode": args.mode,
        "seed": args.seed,
        "num_episodes": args.episodes,
        "episode_seeds": list(range(args.seed, args.seed + args.episodes)),
        "episode_returns": returns.tolist(),
        "episode_lengths": lengths.tolist(),
        "return_type": "undiscounted",
        "mean_return": returns.mean().item(),
        "std_return": returns.std(unbiased=False).item(),
        "video_dir": str(args.video_dir.resolve()) if args.video_dir else None,
    }

    output = (
        args.output or args.checkpoint.parent.parent / "evaluations" / f"{name}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(
        f"{args.mode}: mean return {result['mean_return']:.2f}, "
        f"episode std {result['std_return']:.2f}"
    )
    print(f"Results: {output}")
    if args.video_dir:
        print(f"Videos: {args.video_dir}")


if __name__ == "__main__":
    main()
