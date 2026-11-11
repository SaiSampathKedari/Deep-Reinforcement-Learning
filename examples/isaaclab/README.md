# Isaac Lab Examples

These examples use the released Isaac Lab 3.0.0 package with its Newton physics
backend. They share one Python environment, separate from the repository's main
Gymnasium environment because Isaac Lab pins PyTorch 2.11.0. The environment is
created at `examples/isaaclab/.venv/` and is not committed.

The environment definition is [pyproject.toml](pyproject.toml), with exact
resolved packages in [uv.lock](uv.lock). It pins Python 3.12, Isaac Lab 3.0.0,
CUDA 13 PyTorch 2.11.0, and an editable install of this repository's `deeprl`
library. The `video` extra supports optional recorded playback. The lock targets
Linux x86-64 with an NVIDIA GPU; use a driver compatible with CUDA 13.

## Install

Install [uv](https://docs.astral.sh/uv/) and run these commands from the
repository root on the GPU machine:

```bash
uv sync --locked --project examples/isaaclab
uv run --locked --project examples/isaaclab python -c \
  'import torch, deeprl, isaaclab; print(torch.__version__, torch.cuda.is_available())'
```

The version should start with `2.11.0+cu130`, and CUDA availability should be
`True`. Do not run the root project's `uv sync` for these examples: that selects
the separate root lock and PyTorch stack. You do not need to clone Isaac Lab or
install Isaac Sim for these Newton-backed tasks. See the [Isaac Lab 3.0.0
installation guide](https://isaac-sim.github.io/IsaacLab/v3.0.0/source/isaaclab/installation.html#isaac-lab-python-package)
for platform prerequisites and backend options.

## G1 Rough Terrain

The [G1 task guide](velocity_g1/README.md) describes its observations, actions,
PPO recipe, and outputs. Its config starts in smoke-test mode: 64 environments
and 10 updates. Inspect that short run's shapes, episode endings, CUDA memory,
and throughput before changing `SMOKE_TEST` to `False` for the full recipe.

```bash
uv run --locked --project examples/isaaclab python -m \
  examples.isaaclab.velocity_g1.run_g1_ppo
```

Run the command from the repository root, so Python can import the checkout's
`examples` modules. The scripts import both installed packages; they never
install dependencies themselves. A completed run writes checkpoints, logs, and
an inference-only policy under `outputs/isaaclab/velocity_g1/ppo/`.

After a run completes, set the policy path and episode count in
`velocity_g1/ppo_g1_config.py`, then run:

```bash
uv run --locked --project examples/isaaclab python -m \
  examples.isaaclab.velocity_g1.play_g1_ppo
```

No end-to-end G1 GPU smoke run has completed successfully yet. The lock records
a dependency resolution, not a completed training result.
