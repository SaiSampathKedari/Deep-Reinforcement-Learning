# PPO on CartPole

Train PPO on CartPole-v1, evaluate saved policies, and record videos. Networks
and settings live in this folder; training, evaluation, and checkpoint handling
use the shared `deeprl` library.

Run all commands from the repository root with Python 3.12 or later and
[uv](https://docs.astral.sh/uv/) installed:

```bash
uv sync --extra tb --extra video
```

## Train

```bash
uv run python -m examples.gymnasium.cartpole.run_cartpole_ppo \
  --output-dir outputs/gymnasium/cartpole/ppo/seed_0/run_01
```

The script imports defaults from [ppo_cartpole_config.py](ppo_cartpole_config.py):
CartPole-v1, eight environments, 102,400 training transitions, seed 0, and CPU.
[cartpole_networks.py](cartpole_networks.py) defines the policy and value networks.
The CLI overrides selected defaults with `--seed`, `--device`, and
`--total-timesteps`; other settings are edited in the Python config.

Every run requires a fresh output directory and writes:

- `config.json`: resolved experiment and learner settings.
- `metrics.jsonl`: training and evaluation metrics.
- `events/`: TensorBoard events.
- `checkpoints/last.pt`: latest saved training state.
- `checkpoints/step_<transitions>.pt`: periodic checkpoints.

`CHECKPOINT_INTERVAL = 10` counts completed rollout updates, not transitions.
Each default rollout contains 1,024 transitions. Periodic saves occur every ten
updates, and `last.pt` is also saved at training completion. There is no
best-checkpoint selection.
Total-timestep budgets that are not multiples of the rollout size round down to
complete rollouts.

```bash
uv run tensorboard --logdir outputs
```

## Evaluate

```bash
uv run python -m examples.gymnasium.cartpole.evaluate_cartpole \
  outputs/gymnasium/cartpole/ppo/seed_0/run_01/checkpoints/last.pt \
  --episodes 10 --seed 20000 --mode deterministic
```

Use `--mode stochastic` to sample actions instead. Evaluation reconstructs the
policy from checkpoint settings and reports undiscounted episode returns, their
mean, and the population standard deviation across episodes. Episode seeds are
consecutive, beginning at `--seed`. Results are saved under the run's
`evaluations/` directory; `--output <path>` selects another JSON output path.

To record every evaluation episode, use a fresh video directory:

```bash
uv run python -m examples.gymnasium.cartpole.evaluate_cartpole \
  outputs/gymnasium/cartpole/ppo/seed_0/run_01/checkpoints/last.pt \
  --episodes 3 --seed 20000 --mode deterministic \
  --video-dir outputs/gymnasium/cartpole/ppo/seed_0/run_01/videos/eval_01
```

For an interactive window on a machine with a display:

```bash
uv run python -m examples.gymnasium.cartpole.evaluate_cartpole \
  outputs/gymnasium/cartpole/ppo/seed_0/run_01/checkpoints/last.pt \
  --episodes 3 --seed 20000 --mode deterministic --render
```

`--render` and `--video-dir` are mutually exclusive. Recording needs the video
extra installed above; choose a new folder for each recording to avoid replacing
earlier videos.

## Resume

```bash
uv run python -m examples.gymnasium.cartpole.run_cartpole_ppo \
  --resume-from outputs/gymnasium/cartpole/ppo/seed_0/run_01/checkpoints/last.pt \
  --total-timesteps 204800 \
  --output-dir outputs/gymnasium/cartpole/ppo/seed_0/run_02
```

The target is 204,800 total transitions, not 204,800 additional transitions.
Resume restores learner, optimizer, progress, and sampling RNG state, but starts
fresh environment episodes: it is not exact simulator continuation. Keep the
checkpoint's environment, network, and learner settings compatible; use a fresh
output directory so the earlier run remains intact.
