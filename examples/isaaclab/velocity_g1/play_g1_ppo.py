"""Evaluate an exported G1 policy on Isaac Lab's rough-terrain task."""

import json
from contextlib import ExitStack
from datetime import datetime

import gymnasium as gym
import torch
from gymnasium.vector import AutoresetMode

from deeprl.collectors import supported_autoreset_mode
from deeprl.data_spec import DataSpec, FeatureSpec
from deeprl.env import from_tensor_data, to_tensor_data
from deeprl.policy_io import load_policy
from examples.isaaclab.velocity_g1 import ppo_g1_config as settings
from examples.isaaclab.velocity_g1.g1_networks import build_policy


@torch.no_grad()
def run_episodes(
    envs,
    policy,
    *,
    num_episodes: int,
    seed: int,
    device: torch.device,
) -> tuple[list[float], list[int]]:
    """Run deterministic episodes in a SAME_STEP vector environment.

    Isaac Lab resets finished rows inside ``step``. The returned observation
    for a finished row is already the start of its next episode.
    """
    if num_episodes < 1:
        raise ValueError("num_episodes must be positive")

    observations, _ = envs.reset(seed=seed)
    observations = to_tensor_data(observations, batch_size=envs.num_envs, device=device)
    episode_returns = torch.zeros(envs.num_envs, device=device)
    episode_lengths = torch.zeros(envs.num_envs, dtype=torch.int64, device=device)
    completed_returns: list[float] = []
    completed_lengths: list[int] = []

    while len(completed_returns) < num_episodes:
        commands = policy.act(observations, deterministic=True)
        next_observations, rewards, terminated, truncated, _ = envs.step(
            from_tensor_data(commands, envs.single_action_space)
        )
        episode_returns += rewards
        episode_lengths += 1

        done = terminated | truncated
        for index in done.nonzero(as_tuple=True)[0].tolist():
            completed_returns.append(float(episode_returns[index].item()))
            completed_lengths.append(int(episode_lengths[index].item()))
        episode_returns[done] = 0
        episode_lengths[done] = 0
        observations = to_tensor_data(
            next_observations,
            batch_size=envs.num_envs,
            device=device,
        )

    return completed_returns[:num_episodes], completed_lengths[:num_episodes]


def main() -> None:
    """Load the exported actor and save raw episode returns and lengths."""
    if settings.PLAY_POLICY_DIR is None:
        raise ValueError(
            "Set PLAY_POLICY_DIR in ppo_g1_config.py to an exported policy folder"
        )

    # Import Isaac Lab only when running the example, so the rest of deeprl
    # remains usable without the simulator installed.
    import isaaclab_tasks  # noqa: F401
    from isaaclab.app import launch_simulation
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    env_cfg = parse_env_cfg(
        settings.TASK_ID,
        device=str(settings.DEVICE),
        num_envs=settings.PLAY_NUM_ENVS,
        overrides=list(settings.ENV_OVERRIDES),
    )
    env_cfg.play_mode()
    env_cfg.scene.num_envs = settings.PLAY_NUM_ENVS
    env_cfg.seed = settings.PLAY_SEED

    output_dir = settings.PLAY_OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir.mkdir(parents=True, exist_ok=False)
    if settings.PLAY_RECORD_VIDEO:
        from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
        from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

        env_cfg.sim.visualizer_cfgs = [NewtonGLVisualizerCfg(headless=True)]
        env_cfg.video_recorders = [
            VideoRecorderCfg(
                source="visualizer:newton_gl",
                output_dir=str(output_dir / "videos"),
                video_length=settings.PLAY_VIDEO_LENGTH,
                video_interval=0,
            )
        ]

    with launch_simulation(env_cfg):
        with ExitStack() as stack:
            env = gym.make(settings.TASK_ID, cfg=env_cfg)
            stack.callback(env.close)
            envs = env.unwrapped
            if (
                supported_autoreset_mode(envs, caller="play_g1_ppo")
                is not AutoresetMode.SAME_STEP
            ):
                raise RuntimeError("G1 playback requires SAME_STEP resets")

            observation_spaces = envs.single_observation_space.spaces
            if set(observation_spaces) != {"policy"}:
                raise ValueError(
                    "G1 must expose exactly one 'policy' observation group"
                )
            observation_space = observation_spaces["policy"]
            action_space = envs.single_action_space
            if (
                observation_space.dtype.name != "float32"
                or action_space.dtype.name != "float32"
            ):
                raise ValueError("G1 policy observations and actions must be float32")
            spec = DataSpec(
                observations=(
                    FeatureSpec("policy", observation_space.shape, "float32"),
                ),
                actions=(FeatureSpec("action", action_space.shape, "float32"),),
            )
            policy = load_policy(
                settings.PLAY_POLICY_DIR,
                build_policy,
                model_id=f"{build_policy.__module__}.{build_policy.__name__}",
                expected_spec=spec,
                device=settings.DEVICE,
            )
            if policy.manifest["provenance"].get("task_id") != settings.TASK_ID:
                raise ValueError("Exported policy was not trained for this G1 task")

            returns, lengths = run_episodes(
                envs,
                policy,
                num_episodes=settings.PLAY_NUM_EPISODES,
                seed=settings.PLAY_SEED,
                device=settings.DEVICE,
            )

    results = {
        "task_id": settings.TASK_ID,
        "policy_dir": str(settings.PLAY_POLICY_DIR),
        "seed": settings.PLAY_SEED,
        "episode_returns": returns,
        "episode_lengths": lengths,
        "mean_return": sum(returns) / len(returns),
        "mean_length": sum(lengths) / len(lengths),
        "record_video": settings.PLAY_RECORD_VIDEO,
        "video_length": settings.PLAY_VIDEO_LENGTH
        if settings.PLAY_RECORD_VIDEO
        else None,
        "video_dir": str(output_dir / "videos") if settings.PLAY_RECORD_VIDEO else None,
    }
    (output_dir / "evaluation.json").write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Mean return: {results['mean_return']:.2f} over {len(returns)} episodes")
    print(f"Evaluation saved to {output_dir}")


if __name__ == "__main__":
    main()
