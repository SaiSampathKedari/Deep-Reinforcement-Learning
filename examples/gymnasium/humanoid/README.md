# PPO on Humanoid

PPO on Gymnasium **Humanoid-v5**, with 348 observations and 17 action components.
This example ports one reference recipe instead of mixing hyperparameters from
several libraries. It is **not yet a reproduced benchmark result**.

## Reference

[SLM-Lab's archived configuration](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/data/ppo_mujoco_arc_humanoid_2026_02_12_185439/ppo_mujoco_arc_humanoid_spec.json)
and [recorded metrics](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/data/ppo_mujoco_arc_humanoid_2026_02_12_185439/ppo_mujoco_arc_humanoid_t0_trial_metrics_scalar.json)
report a training-return moving average of **2,661.26 at 10M transitions**,
aggregated across four sessions. That is not a held-out deterministic evaluation.

| Setting | This run |
| --- | --- |
| Actor / critic | Separate 256 x 256 Tanh MLPs |
| Optimizers | Separate AdamW, constant LR `3e-4`, epsilon `1e-8`, weight decay `0.01` |
| Collection | 16 environments x 2,048 steps = 32,768 transitions |
| PPO optimization | Minibatch 64, 10 epochs, ratio clip `0.2` |
| Discount / GAE | `0.99` / `0.95` |
| Gradient clipping | Norm `0.5`, separately for actor and critic |
| Entropy / target KL | `0.0` / disabled; KL is still logged |
| Gaussian | State-independent log std, initially `0`, bounded to `[-5, 2]` |
| Actions | Gaussian coordinates rescaled from `[-1, 1]` to native `[-0.4, 0.4]` |
| Advantages | Standardized per minibatch using sample standard deviation |
| Observation normalization | Shared online moments, epsilon `1e-8`, clip `10` |
| Reward scaling | Per-step running discounted-return std, clip `10` |
| Value loss | Scale both predictions and targets by running target std; clip `10` |
| Run budget | 12,288,000 transitions = 375 complete rollouts |

The [inspected source](https://github.com/kengz/SLM-Lab/tree/e56a41477b814a51d1bc8831a699c01315f04c94)
uses orthogonal Tanh gain `5/3` for all linear layers, including output heads.
We match that effective initialization: the `.01`/`1` output gains in its config
are not actually applied by that code path. The PPO objective also bounds log
ratios to `[-20, 20]` before exponentiation; KL diagnostics remain unclamped.
Value statistics update on each minibatch, including repeated epochs. The critic
still predicts values in the reward-scaled units used by GAE; only its loss is
normalized, without subtracting a mean.

### Differences Kept Explicit

- The archive records an unknown source commit and no session seeds. We can
  verify its saved configuration and results, but cannot promise exact replication.
- Our GAE stops at both termination and truncation, bootstrapping truncations from
  true final observations. The inspected reference continues traces across truncation.
- Reward accumulators reset **after** including the final reward at either episode
  boundary. Our running moments start without Gymnasium's small pseudocount.
- Disabled autoreset records true final observations before partial resets. Only
  newly reset observations update moments again and are normalized again.
  Continuing states keep their stored normalized representation, so bootstrap
  and next-current value inputs remain consistent. This corrects the earlier
  full-batch re-normalization after partial resets; older checkpoints can load,
  but continuation now uses the corrected collection behavior.
- Evaluation uses separate fixed seeds and frozen observation statistics. The
  12.288M budget extends the reference's 10M budget; one seed is not its four-seed study.

There is no explicit action clipping or squashing in this recipe. Rescaling is
affine even outside `[-1, 1]`, as in the reference; MuJoCo's native actuator control
limits still apply. Gaussian samples and their log probabilities stay together in
the rollout. Native reward and termination definitions are not modified.

## Run

From the repository root:

```bash
uv sync --extra mujoco --extra tb --extra video

RUN="outputs/gymnasium/humanoid/ppo/seed_0/slm_reference_$(date +%Y%m%d_%H%M%S)"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python -m examples.gymnasium.humanoid.run_humanoid_ppo \
  --device cuda:0 --vectorization-mode async --seed 0 \
  --total-timesteps 12288000 --output-dir "$RUN"
```

CUDA trains the networks; MuJoCo simulation runs in 16 CPU workers. This is not
GPU physics simulation. For a one-rollout smoke test, use `--total-timesteps 32768`
with a different output directory. Do not resume an old experimental checkpoint.

Each fresh output directory contains `config.json`, `metrics.jsonl`, TensorBoard
`events/`, and `checkpoints/`. Every three completed rollouts (98,304 transitions),
the trainer saves `last.pt` and a numbered checkpoint, and evaluates five episodes
in each of deterministic and stochastic modes. It also saves at completion.
There is no best-checkpoint selection. Interrupting a run may lose work since the
last completed checkpoint; an interrupted partial PPO update is not saved.

```bash
.venv/bin/tensorboard --logdir outputs/gymnasium/humanoid --port 6006
```

Logs report raw, undiscounted episode returns. `diagnostics/final_approximate_kl`
measures the final policy on the entire stored rollout. `action_std_mean`,
`action_std_min`, `action_std_max`, and `action_mean_abs_mean` describe the Gaussian
in rescaled policy coordinates, not executed native controls. `effective_epochs`
and `optimizer_steps` record actual update work. All use the `diagnostics/` prefix.

## Evaluate And Record

```bash
.venv/bin/python -m examples.gymnasium.humanoid.evaluate_humanoid \
  "$RUN/checkpoints/last.pt" --episodes 10 --mode deterministic --render

.venv/bin/python -m examples.gymnasium.humanoid.evaluate_humanoid \
  "$RUN/checkpoints/last.pt" --episodes 10 --mode stochastic \
  --video-dir "$RUN/videos/stochastic_01"
```

Use a new video directory each time. `--render` and `--video-dir` are mutually
exclusive. Evaluation reconstructs network sizes, action rescaling, log-std bounds,
and frozen observation moments from the checkpoint, independent of current defaults.
Older embedded-normalizer checkpoints remain readable by the evaluator.

## Resume

```bash
.venv/bin/python -m examples.gymnasium.humanoid.run_humanoid_ppo \
  --device cuda:0 --vectorization-mode async \
  --resume-from "$RUN/checkpoints/last.pt" --total-timesteps 12288000 \
  --output-dir "${RUN}_resumed"
```

The budget includes already completed transitions. Resume restores both networks,
optimizers, observation/reward/value statistics, counters, and sampling RNG state.
Simulation starts fresh episodes, not the exact interrupted simulator state.
Use a larger total budget if the previous run already finished.
