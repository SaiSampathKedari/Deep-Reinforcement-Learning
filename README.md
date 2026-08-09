# Deep Reinforcement Learning

Deep Reinforcement Learning from mathematical foundations to PyTorch
implementations: rigorous proofs, step-by-step derivations of objectives and
gradient estimators, correctness-focused actor-critic code, and reproducible
experiment infrastructure for Gymnasium.

> [!IMPORTANT]
> **This repository is under active development.** New mathematical
> derivations, algorithms, tests, experiment configurations, and benchmark
> studies will be added throughout the coming weeks.

## Current State

| Topic | Mathematical development | PyTorch implementation |
|---|---|---|
| Policy-gradient foundations | Discounted and average-reward policy-gradient theorems; Bellman and trajectory proofs | Foundation for the policy-based algorithms |
| REINFORCE and baselines | Monte Carlo estimator, causality, control variates, and variance reduction | Planned |
| One-step actor-critic | From the exact policy gradient to TD advantage estimation | [`one_step_actor_critic.py`](src/deeprl/algorithms/one_step_actor_critic.py) |
| GAE and A2C | n-step estimators, TD(lambda), forward and backward views, eligibility traces | [`advantages.py`](src/deeprl/advantages.py), [`a2c.py`](src/deeprl/algorithms/a2c.py) |
| NPG, TRPO, PPO | Next derivation and implementation sequence | Planned |
| DQN, DDPG, TD3, SAC | Value-based and off-policy continuous-control sequence | Planned |

The code currently includes one-step actor-critic, vectorized GAE/A2C, rollout
storage, structured logging, scheduled policy evaluation, and cross-seed
aggregation. The runnable experiment targets CartPole through Gymnasium;
benchmark curves and comparison tables have not yet been published. Box2D,
MuJoCo, value-based learning, and continuous control belong to the roadmap.

## From Equations to Tensors

For a stored transition, let `termination_t` mark a true terminal state and let
`done_t = termination_t or truncation_t`. The implementation evaluates the true
next observation and computes

$$
\delta_t
= R_{t+1}
+ \gamma (1-\mathrm{termination}_t)V(S_{t+1})
- V(S_t),
$$

followed by the finite-rollout GAE recursion

$$
\hat A_t
= \delta_t
+ \gamma\lambda(1-\mathrm{done}_t)\hat A_{t+1},
\qquad
V_t^{\mathrm{target}} = V(S_t) + \hat A_t.
$$

That correspondence is explicit in the code:

1. [`collect_rollout()`](src/deeprl/rollouts.py) stores the true
   `next_observation` before resetting completed environments.
2. [`evaluate_next_values()`](src/deeprl/rollouts.py) evaluates
   `V(S_{t+1})` for ordinary and truncated transitions; terminal entries remain
   zero.
3. [`generalized_advantage_estimate()`](src/deeprl/advantages.py) computes the
   TD(lambda) errors and value targets over `(T, N)` tensors under `no_grad()`.
4. [`update()`](src/deeprl/algorithms/a2c.py) recomputes `log pi(A|S)` and
   `V(S)` with new autograd graphs. The advantage is a fixed policy-loss weight;
   the fixed value target gives the critic its semi-gradient update.

## Implementation Architecture

The on-policy runtime is divided into reusable components. Collection produces
time-major `(T, N, ...)` data, estimators construct fixed targets, and each
algorithm owns its gradient update over flat `(B, ...)` batches.

| Module | Responsibility |
|---|---|
| [`rollouts.py`](src/deeprl/rollouts.py) | Interact with vector environments and preserve true transition data |
| [`buffers.py`](src/deeprl/buffers.py) | Store time-major rollouts and produce aligned flat batches or minibatches |
| [`advantages.py`](src/deeprl/advantages.py) | Compute gradient-free advantage estimators and value targets |
| [`algorithms/`](src/deeprl/algorithms) | Define the collection-estimation-update loop for each learning rule |
| [`policies.py`](src/deeprl/policies.py) | Map observations to PyTorch action distributions |
| [`value.py`](src/deeprl/value.py) | Define scalar state-value approximators |
| [`logger.py`](src/deeprl/logger.py) | Track episodes and learner statistics; write JSONL and optional TensorBoard metrics |
| [`evaluate.py`](src/deeprl/evaluate.py) | Evaluate stochastic and deterministic policies and aggregate independent seeds |

The shared A2C path is also the base structure for NPG, TRPO, and PPO. They can
reuse environment collection, rollout storage, next-state evaluation, GAE, and
logging; their essential difference belongs in `update()`.

## Correctness Invariants

Several implementation choices are treated as invariants rather than optional
details:

1. **Termination and truncation remain separate.** A true termination sets the
   next-state value to zero. A time-limit truncation uses the value of the true
   final observation.
2. **Episode boundaries stop recursive estimators.** Neither termination nor
   truncation allows GAE from the next episode to enter the current one.
3. **Collection does not retain computation graphs.** Rollout data is collected
   under `torch.no_grad()`; log-probabilities and values are recomputed during
   optimization.
4. **Targets do not carry gradients.** Advantages remain fixed in the actor
   loss, and value targets remain fixed in the critic loss.
5. **Shapes remain explicit.** Rollouts use `(T, N, ...)`, vector steps use
   `(N, ...)`, and update batches use `(B, ...)` with `B = T * N`.
6. **Corresponding fields never lose alignment.** A single permutation indexes
   observations, actions, old log-probabilities, old values, advantages, and
   value targets.

## Mathematical Development

The reports are not summaries of finished code. They are the mathematical path
used to construct it.

| Report | Main development |
|---|---|
| [Policy Gradient Preliminaries](reports/13_Policy-Gradient-Preliminaries.pdf) | Objectives, stochastic approximation, universal estimator notation, and the structure shared by policy-gradient algorithms |
| [Policy Gradient Theorem](reports/10_Policy-Gradient-Theorem.pdf) | Infinite-horizon discounted objective, recursive differentiation, visitation measures, and the likelihood-ratio form |
| [Policy Gradient Theorem: Trajectory Route](reports/12_Policy-Gradient-Theorem_Episodic-Trajectory-Route.pdf) | A second proof over trajectories, including the causality argument that removes past rewards |
| [Average-Reward Policy Gradient Theorem](reports/11_Average-Reward-Policy-Gradient-Theorem.pdf) | Continuing-task objective, stationary distribution, differential values, and the Bellman proof |
| [REINFORCE](reports/14_REINFORCE.pdf) | Monte Carlo policy gradient, reward-to-go, stochastic ascent, baselines, and variance reduction |
| [Actor-Critic](reports/15_Actor-Critic.pdf) | Actor-critic derived from the exact policy gradient, finite-rollout and online estimators, critic construction, and bias-variance tradeoffs |
| [Actor-Critic with a Baseline](reports/16_Actor-Critic-with-a-Baseline.pdf) | Baseline identity, the advantage function, TD error as a one-sample advantage estimator, and one-step actor-critic |
| [Generalized Advantage Estimation](reports/17_GAE_Actor-Critic.pdf) | One-step and n-step advantages, geometric mixtures, TD(lambda), forward and backward views, and eligibility traces |

## Installation

The project targets Python 3.12 and uses [uv](https://docs.astral.sh/uv/) for
reproducible dependency management.

```bash
git clone https://github.com/SaiSampathKedari/Deep-Reinforcement-Learning.git
cd Deep-Reinforcement-Learning
uv sync
```

Optional environment and visualization dependencies are separated by use case:

```bash
uv sync --extra video --extra tb  # rendering, video, TensorBoard
uv sync --extra box2d             # LunarLander, BipedalWalker
uv sync --extra mujoco            # MuJoCo continuous control
```

## Running the Current Experiment

Train one-step actor-critic on `CartPole-v1`:

```bash
uv sync --extra video
uv run python -m deeprl.train
```

With the TensorBoard extra installed, monitor the run with:

```bash
uv run tensorboard --logdir runs
```

Each run receives its own directory containing the resolved configuration,
append-only metrics, model checkpoint, and optional TensorBoard events and
videos:

```text
runs/CartPole-v1__one_step_ac__<seed>__<timestamp>/
  config.json
  metrics.jsonl
  checkpoint.pt
  events.out.tfevents.*
  videos/
```

`metrics.jsonl` is the experiment record. TensorBoard is an optional live
viewer rather than the source used for later aggregation.

## Evaluation Infrastructure

The current harness implements the following protocol; benchmark results will
be published after the corresponding multi-seed studies are run:

- The x-axis is total environment interactions.
- Training and evaluation use separate environments.
- Stochastic and deterministic policies are evaluated independently.
- Evaluation occurs at fixed environment-step checkpoints and once at the end.
- Each random seed is stored as an independent run.
- Episodes are reduced within each seed before statistics are computed across
  seeds.
- Seed aggregation returns the mean and bootstrap 95% confidence interval
  rather than selecting a best run.

The implementation is in [`evaluate.py`](src/deeprl/evaluate.py), including
RNG-preserving policy evaluation, JSONL curve loading, aligned seed aggregation,
and bootstrap confidence intervals.

## Tests

Correctness is checked with tests rather than inferred from a rising reward
curve:

```bash
uv run pytest -q
```

The current suite covers asynchronous episode accounting, partial vector resets,
complete-update logging boundaries, weighted metric reduction, JSONL
persistence, evaluation RNG isolation, aligned seed aggregation, and end-to-end
smoke runs for one-step actor-critic and A2C.

## Roadmap

The implementation order follows the mathematical dependencies:

1. **On-policy trust-region methods:** Natural Policy Gradient, TRPO, and PPO.
2. **Continuous on-policy control:** diagonal Gaussian policies and Gymnasium
   continuous-control benchmarks.
3. **Value-based learning:** DQN with replay, target networks, Double DQN, and
   dueling architectures.
4. **Off-policy actor-critic:** DDPG, TD3, and SAC.
5. **MuJoCo studies:** multi-seed comparisons, ablations, and deterministic
   versus stochastic evaluation on standard continuous-control tasks.

Each algorithm is expected to arrive with its derivation, implementation,
correctness tests, experiment configuration, and benchmark results.

## References

- Richard S. Sutton and Andrew G. Barto,
  *[Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)*,
  2nd edition, 2018.
- Volodymyr Mnih et al.,
  *[Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)*,
  2016.
- John Schulman et al.,
  *[High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)*,
  2016.
- Fabio Pardo et al.,
  *[Time Limits in Reinforcement Learning](https://arxiv.org/abs/1712.00378)*,
  2018.
- Shengyi Huang et al.,
  *[The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)*,
  2022.

## Related Repositories

A sequence from mathematical foundations to deep reinforcement learning:

- **Foundations** — [Real Analysis](https://github.com/SaiSampathKedari/Real-Analysis) · [Probability & Distribution Theory](https://github.com/SaiSampathKedari/Probability-and-Distribution-Theory) · [Statistical Inference Theory](https://github.com/SaiSampathKedari/Statistical-Inference-Theory)
- **Sequential-decision theory** — [Sequential Decision Making](https://github.com/SaiSampathKedari/Sequential-Decision-Making)
- **Classical reinforcement learning** — [Reinforcement Learning](https://github.com/SaiSampathKedari/Reinforcement-Learning)
- **Deep reinforcement learning** — this repository

## Contact

- Email: [sampath@umich.edu](mailto:sampath@umich.edu)
- LinkedIn: [sai-sampath-kedari](https://www.linkedin.com/in/sai-sampath-kedari)
