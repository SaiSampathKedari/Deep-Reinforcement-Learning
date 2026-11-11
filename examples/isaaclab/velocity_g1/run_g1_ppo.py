"""Train PPO on Isaac Lab's rough-terrain Unitree G1 task."""

import json
from collections.abc import Mapping
from contextlib import ExitStack
from datetime import datetime
from importlib.metadata import version
from numbers import Real
from typing import Any

import gymnasium as gym
import torch
from gymnasium.vector import AutoresetMode
from torch.utils.tensorboard import SummaryWriter

from deeprl.algorithms.ppo import PPOLearner
from deeprl.checkpoint import load_checkpoint
from deeprl.collectors import supported_autoreset_mode
from deeprl.data_spec import DataSpec, FeatureSpec
from deeprl.lifecycle import (
    config_to_jsonable,
    get_provenance,
    seed_torch,
    validate_run_config,
)
from deeprl.logger import Logger
from deeprl.policy_io import describe_policy, save_policy
from deeprl.processors import ObservationProcessor
from deeprl.training.train_on_policy import train_on_policy
from deeprl.value import ValueFunction
from examples.isaaclab.velocity_g1 import ppo_g1_config as settings
from examples.isaaclab.velocity_g1.g1_networks import ValueNetwork, build_policy


def _metric_scalar(value: Any, device: torch.device) -> torch.Tensor | None:
    """Turn a numeric Isaac Lab metric into one detached scalar on the sim GPU."""
    if isinstance(value, torch.Tensor):
        if value.numel() == 0 or value.dtype == torch.bool:
            return None
        return value.detach().to(device=device, dtype=torch.float32).mean()
    if isinstance(value, Real) and not isinstance(value, bool):
        return torch.tensor(float(value), device=device)
    return None


class G1TrainingEnv:
    """Add reference-style episode starts and rollout-level task metrics."""

    def __init__(
        self,
        env: Any,
        logger: Logger,
        *,
        rollout_steps: int,
        initial_step: int = 0,
        random_initial_episode_lengths: bool = True,
    ) -> None:
        self.env = env
        self.logger = logger
        self.metadata = env.metadata
        self.num_envs = env.num_envs
        self.single_action_space = env.single_action_space
        self.device = torch.device(env.device)
        self.rollout_steps = rollout_steps
        self.global_step = initial_step
        self.random_initial_episode_lengths = random_initial_episode_lengths
        self._steps_in_rollout = 0
        self._metric_sums: dict[str, torch.Tensor] = {}
        self._metric_weights: dict[str, torch.Tensor] = {}

    def reset(self, *, seed: int | None = None) -> tuple[Any, Any]:
        observations, info = self.env.reset(seed=seed)
        if self.random_initial_episode_lengths:
            lengths = self.env.episode_length_buf
            lengths.copy_(
                torch.randint_like(lengths, high=int(self.env.max_episode_length))
            )
        return observations, info

    def step(self, actions: torch.Tensor) -> tuple[Any, Any, Any, Any, Any]:
        observations, rewards, terminations, truncations, info = self.env.step(actions)
        self.global_step += self.num_envs
        self._steps_in_rollout += 1

        # Isaac Lab refreshes info["log"] only when episodes reset. Aggregate
        # those fresh metrics on-device and transfer them once per rollout.
        dones = terminations | truncations
        if dones.any().item() and isinstance(info, Mapping):
            log = info.get("log")
            if isinstance(log, Mapping):
                weight = dones.sum().to(dtype=torch.float32)
                for name, value in log.items():
                    scalar = _metric_scalar(value, self.device)
                    if scalar is None:
                        continue
                    name = f"isaaclab/{name}"
                    weighted_value = scalar * weight
                    if name in self._metric_sums:
                        self._metric_sums[name] += weighted_value
                        self._metric_weights[name] += weight
                    else:
                        self._metric_sums[name] = weighted_value
                        self._metric_weights[name] = weight.clone()

        if self._steps_in_rollout == self.rollout_steps:
            if self._metric_sums:
                names = list(self._metric_sums)
                values = torch.stack(
                    [
                        self._metric_sums[name] / self._metric_weights[name]
                        for name in names
                    ]
                ).tolist()
                for name, value in zip(names, values, strict=True):
                    self.logger.record(name, value, step=self.global_step)
                self._metric_sums.clear()
                self._metric_weights.clear()
            self._steps_in_rollout = 0

        return observations, rewards, terminations, truncations, info


def main() -> None:
    """Launch Isaac Lab, train on one GPU, and export the final actor."""
    # Isaac Lab registers its Gymnasium tasks before their configs are parsed.
    import isaaclab_tasks  # noqa: F401
    from isaaclab.app import launch_simulation
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    cfg = settings.PPO_CONFIG
    env_cfg = parse_env_cfg(
        settings.TASK_ID,
        device=str(cfg.device),
        num_envs=settings.NUM_ENVS,
        overrides=list(settings.ENV_OVERRIDES),
    )
    if not hasattr(env_cfg, "compute_final_obs"):
        raise RuntimeError(
            "This example requires released Isaac Lab v3.0.0 with compute_final_obs"
        )
    env_cfg.compute_final_obs = True
    env_cfg.seed = cfg.seed
    physics_cfg = env_cfg.sim.physics
    solver_cfg = getattr(physics_cfg, "solver_cfg", None)
    simulator_versions = {
        "isaaclab": version("isaaclab"),
        "newton": version("newton"),
        "torch": torch.__version__,
        "physics_backend": type(physics_cfg).__name__,
        "physics_solver": type(solver_cfg).__name__ if solver_cfg is not None else None,
    }

    # The launcher starts the runtime required by the task's physics backend.
    # The simulator, models and rollout storage use the configured CUDA device.
    with launch_simulation(env_cfg):
        with ExitStack() as stack:
            env = gym.make(settings.TASK_ID, cfg=env_cfg)
            stack.callback(env.close)
            envs = env.unwrapped
            if (
                supported_autoreset_mode(envs, caller="run_g1_ppo")
                is not AutoresetMode.SAME_STEP
            ):
                raise RuntimeError("G1 must provide SAME_STEP resets and final_obs")

            # Isaac Lab owns the action scale. Infer sizes from this task's
            # spaces, rather than assuming a fixed G1 joint count.
            observation_spaces = envs.single_observation_space.spaces
            if set(observation_spaces) != {"policy"}:
                raise ValueError(
                    "G1 must expose exactly one 'policy' observation group"
                )
            observation_space = observation_spaces["policy"]
            action_space = envs.single_action_space
            obs_dim = observation_space.shape[0]
            action_dim = action_space.shape[0]
            if (
                observation_space.dtype.name != "float32"
                or action_space.dtype.name != "float32"
            ):
                raise ValueError("G1 policy observations and actions must be float32")

            spec = DataSpec(
                observations=(FeatureSpec("policy", (obs_dim,), "float32"),),
                actions=(FeatureSpec("action", (action_dim,), "float32"),),
            )
            model_config = {
                "obs_dim": obs_dim,
                "action_dim": action_dim,
                "hidden_sizes": list(settings.POLICY_HIDDEN_SIZES),
                "log_std_init": settings.LOG_STD_INIT,
                "log_std_bounds": settings.LOG_STD_BOUNDS,
            }
            run_config = {
                "task_id": settings.TASK_ID,
                "num_envs": settings.NUM_ENVS,
                "env_overrides": list(settings.ENV_OVERRIDES),
                "random_initial_episode_lengths": settings.RANDOM_INITIAL_EPISODE_LENGTHS,
                "simulator_versions": simulator_versions,
                "policy_model": model_config,
                "value_hidden_sizes": list(settings.VALUE_HIDDEN_SIZES),
            }
            initial_step = 0
            if settings.RESUME_FROM is not None:
                checkpoint = load_checkpoint(settings.RESUME_FROM)
                validate_run_config(
                    checkpoint["config"], run_config, keys=tuple(run_config)
                )
                initial_step = int(checkpoint["progress"]["global_step"])
                del checkpoint

            output_dir = settings.OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir.mkdir(parents=True, exist_ok=False)
            (output_dir / "config.json").write_text(
                json.dumps(
                    {"experiment": run_config, "learner": config_to_jsonable(cfg)},
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            writer = stack.enter_context(
                SummaryWriter(log_dir=str(output_dir / "events"))
            )

            seed_torch(cfg.seed)
            policy = build_policy(model_config)
            value_fn = ValueFunction(ValueNetwork(obs_dim, settings.VALUE_HIDDEN_SIZES))
            obs_processor = ObservationProcessor(spec, normalize=())
            learner = PPOLearner(
                policy,
                value_fn,
                cfg,
                obs_processor=obs_processor,
            )

            # Deterministic is the exported inference default; collection still
            # samples from the Gaussian distribution for PPO exploration.
            learner.inference_policy.deterministic = True
            model_id = f"{build_policy.__module__}.{build_policy.__name__}"
            provenance = {
                **get_provenance(),
                "task_id": settings.TASK_ID,
                **simulator_versions,
            }
            policy_manifest = describe_policy(
                learner.inference_policy,
                spec=spec,
                model_id=model_id,
                model_config=model_config,
                provenance=provenance,
            )
            logger = Logger(
                settings.NUM_ENVS,
                cfg.log_every,
                writer,
                cfg.device,
                metrics_path=output_dir / "metrics.jsonl",
                total_timesteps=cfg.total_timesteps,
                show_progress=True,
            )
            stack.callback(logger.finish)

            training_env = G1TrainingEnv(
                envs,
                logger,
                rollout_steps=cfg.rollout_steps,
                initial_step=initial_step,
                random_initial_episode_lengths=settings.RANDOM_INITIAL_EPISODE_LENGTHS,
            )
            train_on_policy(
                envs=training_env,
                learner=learner,
                logger=logger,
                evaluator=None,
                checkpoint_dir=output_dir / "checkpoints",
                checkpoint_interval=settings.CHECKPOINT_INTERVAL,
                run_config=run_config,
                resume_from=settings.RESUME_FROM,
                policy_manifest=policy_manifest,
            )
            save_policy(
                output_dir / "policy",
                learner.inference_policy,
                spec=spec,
                model_id=model_id,
                model_config=model_config,
                provenance={**provenance, "training_step": cfg.total_timesteps},
            )

    print(f"Run saved to {output_dir}")


if __name__ == "__main__":
    main()
