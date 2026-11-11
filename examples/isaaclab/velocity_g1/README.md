# Isaac Lab G1 Rough Terrain

This example trains our PPO on Isaac Lab 3.0.0's
`Isaac-Velocity-Rough-G1` task. Follow the shared
[Isaac Lab setup](../README.md) on a Linux NVIDIA machine, then run from the
repository root:

```bash
uv run --locked --project examples/isaaclab python -m \
  examples.isaaclab.velocity_g1.run_g1_ppo
```

The default in [ppo_g1_config.py](ppo_g1_config.py) is a smoke run: 64
environments and 10 PPO updates. After checking its episode endings, task
metrics, memory use, and steps per second, set `SMOKE_TEST = False` for the
4096-environment, 5000-update rough-terrain recipe. Change `DEVICE` and
`RESUME_FROM` in the same file when needed. There are no command-line options.

The task generates transitions online; no dataset is needed. Its float32
`policy` observation contains base velocities, projected gravity, velocity
commands, joint positions and velocities, previous actions, and terrain
heights. The Gaussian policy samples joint-position-offset commands. The
example reads observation and action dimensions from the task rather than
hardcoding the robot's joint count, and Isaac Lab applies its own action
scaling.

Isaac Lab resets finished environments inside `step()`. Training enables
`compute_final_obs` so the collector stores the true final observation and
bootstraps time limits correctly. Initial episode lengths are randomized as
in the reference runner. Isaac Lab's reset-time task metrics are written to
TensorBoard and `metrics.jsonl` alongside learning metrics. Periodic policy
evaluation is off during training.

Each run writes `config.json`, training checkpoints, metrics, TensorBoard
events, and, after successful completion, an inference-only `policy/` folder
under `outputs/isaaclab/velocity_g1/ppo/{smoke,full}/`. The config and policy
record the installed Isaac Lab, Newton, and PyTorch versions, physics
backend, and solver. The smoke run is disposable; switching to 4096
environments starts a new full run. To continue a run with the same number
of environments, set `RESUME_FROM` to its `last.pt` checkpoint and increase
`NUM_UPDATES` as needed. The resumed run gets a new output directory.

To watch completed PPO updates, run from the repository root:

```bash
uv run --locked --project examples/isaaclab tensorboard \
  --logdir outputs/isaaclab/velocity_g1/ppo --port 6006
```

An event file alone does not indicate training progress; a run stopped before
its first update has no learning metrics to display.

To play an exported policy, set `PLAY_POLICY_DIR` in the config and run:

```bash
uv run --locked --project examples/isaaclab python -m \
  examples.isaaclab.velocity_g1.play_g1_ppo
```

Playback uses deterministic actions and saves episode returns and lengths in
`evaluation.json`. Set `PLAY_RECORD_VIDEO = True` to record one fixed-length
clip with the Newton visualizer; it is not one clip per complete episode.
Playback is separate from training so it does not slow down the learning run.
Isaac Lab's `play_mode()` changes task conditions, so playback returns are not
directly comparable to training-environment returns.

The recipe matches the reference's rollout size, minibatches, epochs,
network sizes, PPO clip, entropy weight, and adaptive learning-rate rule. It
is not numerically identical to RSL-RL: our actor and critic have separate
optimizers, our Gaussian learns log standard deviation, and time limits use
the true final state for bootstrapping. No Isaac Lab training or playback
result has been verified yet.
