"""Evaluate an exported InvertedPendulum policy, with optional rendering or video."""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from deeprl.data_spec import DataSpec, FeatureSpec
from deeprl.evaluate import evaluate_policy
from deeprl.policy_io import load_policy
from examples.gymnasium.inverted_pendulum.inverted_pendulum_networks import build_policy


def main() -> None:
    """Load a frozen actor, evaluate seeded episodes, and save raw returns."""
    # 0. Read evaluation and display options.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy_dir", type=Path)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument(
        "--mode",
        choices=("deterministic", "stochastic"),
        help="Override the inference mode saved with the policy.",
    )
    parser.add_argument("--device", type=torch.device, default=torch.device("cpu"))
    parser.add_argument("--output", type=Path, help="Evaluation JSON output path.")
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--render", action="store_true")
    display.add_argument("--video-dir", type=Path)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.video_dir and args.video_dir.exists():
        raise FileExistsError("Choose a new --video-dir to preserve existing videos.")

    # 1. Create the environment with the declared observation dtype.
    render_mode = "rgb_array" if args.video_dir else ("human" if args.render else None)
    env = gym.make("InvertedPendulum-v5", render_mode=render_mode)
    try:
        env = gym.wrappers.DtypeObservation(env, dtype=np.float32)
        # 2. Restore the actor and preprocessing against unbatched feature shapes.
        spec = DataSpec(
            observations=(
                FeatureSpec(
                    "state",
                    tuple(env.observation_space.shape),
                    env.observation_space.dtype.name,
                ),
            ),
            actions=(
                FeatureSpec("action", tuple(env.action_space.shape), "float32"),
            ),
        )
        policy = load_policy(
            args.policy_dir,
            build_policy,
            model_id=f"{build_policy.__module__}.{build_policy.__name__}",
            expected_spec=spec,
            device=args.device,
        )
        deterministic = (
            policy.deterministic if args.mode is None else args.mode == "deterministic"
        )
        mode = "deterministic" if deterministic else "stochastic"
        name = f"{args.policy_dir.name}_{mode}_seed{args.seed}_episodes{args.episodes}"
        # 3. Evaluate complete episodes, optionally recording every episode.
        if args.video_dir:
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=str(args.video_dir),
                name_prefix=name,
                episode_trigger=lambda episode: True,
            )
        returns, lengths = evaluate_policy(
            env,
            policy,
            num_episodes=args.episodes,
            device=args.device,
            deterministic=deterministic,
            seed=args.seed,
        )
    finally:
        env.close()  # Also finalizes the last recorded episode.

    # 4. Save undiscounted scores and the resolved evaluation settings.
    result = {
        "policy_dir": str(args.policy_dir.resolve()),
        "env_id": "InvertedPendulum-v5",
        "training_step": policy.manifest["provenance"].get("training_step"),
        "mode": mode,
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
    output = args.output or args.policy_dir.parent / "evaluations" / f"{name}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"{mode}: mean return {result['mean_return']:.2f}, "
        f"episode std {result['std_return']:.2f}"
    )
    print(f"Results: {output}")
    if args.video_dir:
        print(f"Videos: {args.video_dir}")


if __name__ == "__main__":
    main()
