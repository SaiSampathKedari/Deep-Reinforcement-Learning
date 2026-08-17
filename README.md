# Deep Reinforcement Learning

Deep Reinforcement Learning from mathematical foundations to PyTorch
implementations: rigorous proofs, step-by-step derivations of objectives and
gradient estimators, correctness-focused actor-critic code, and reproducible
experiment infrastructure for Gymnasium.

> ⚠️ **This repository is under active development.** New mathematical
> derivations, algorithms, tests, experiment configurations, and benchmark
> studies will be added throughout the coming weeks.

## Current Algorithms

| Algorithm | Core mechanism | Status |
|---|---|:---:|
| One-step actor-critic | TD(0) policy and value updates | Implemented |
| A2C with GAE | Synchronous rollouts and TD(lambda) targets | Implemented |
| Natural Policy Gradient | Matrix-free Fisher-vector products and conjugate gradient | Implemented |
| Trust Region Policy Optimization | KL-constrained natural-gradient steps with backtracking | Implemented |
| Proximal Policy Optimization | Clipped surrogate optimization with shuffled minibatches | Next |

The current code includes vectorized rollout storage, explicit termination and
truncation handling, a shared on-policy training engine, matrix-free natural
gradient updates, TRPO backtracking, structured logging, scheduled policy
evaluation, and cross-seed aggregation.

## From Mathematics to Code

Each algorithm is developed from its objective and gradient estimator before
being translated into PyTorch. GAE is implemented directly from

```math
\delta_t = R_{t+1}
+ \gamma(1-\mathrm{termination}_t)V(S_{t+1}) - V(S_t),
\qquad
\hat A_t = \delta_t
+ \gamma\lambda(1-\mathrm{done}_t)\hat A_{t+1},
\qquad
V_t^{\mathrm{target}} = V(S_t) + \hat A_t.
```

[`rollouts.py`](src/deeprl/rollouts.py) preserves the true next observation,
[`advantages.py`](src/deeprl/advantages.py) constructs fixed advantages and
value targets, and [`a2c.py`](src/deeprl/algorithms/a2c.py) performs the policy
and value-function updates.

NPG and TRPO use the damped natural-gradient system without materializing the
Fisher information matrix:

```math
(F + \eta I)x = g,
\qquad
\Delta\theta =
\sqrt{\frac{2\delta}{x^\top(F + \eta I)x}}\,x.
```

[`npg.py`](src/deeprl/algorithms/npg.py) computes Fisher-vector products and
solves this system with conjugate gradient. [`trpo.py`](src/deeprl/algorithms/trpo.py)
reuses the NPG step and adds a backtracking line search for surrogate improvement
and the sampled KL constraint.

## Mathematical Reports

Self-contained derivations and their corresponding PyTorch implementations.

| Mathematical development | Reports | Implementation |
|---|---|---|
| Policy-gradient foundations | [Preliminaries](reports/13_Policy-Gradient-Preliminaries.pdf) · [Policy Gradient Theorem](reports/10_Policy-Gradient-Theorem.pdf) · [Trajectory Proof](reports/12_Policy-Gradient-Theorem_Episodic-Trajectory-Route.pdf) · [Average-Reward Theorem](reports/11_Average-Reward-Policy-Gradient-Theorem.pdf) | Used throughout the policy-gradient algorithms |
| REINFORCE | [REINFORCE](reports/14_REINFORCE.pdf) | Report only |
| Actor-critic and baselines | [Actor-Critic](reports/15_Actor-Critic.pdf) · [Baselines and Advantages](reports/16_Actor-Critic-with-a-Baseline.pdf) | [`one_step_actor_critic.py`](src/deeprl/algorithms/one_step_actor_critic.py) |
| GAE and A2C | [Generalized Advantage Estimation](reports/17_GAE_Actor-Critic.pdf) | [`advantages.py`](src/deeprl/advantages.py) · [`a2c.py`](src/deeprl/algorithms/a2c.py) |
| Natural Policy Gradient | [Natural Policy Gradient](reports/18_Natural-Policy-Gradient.pdf) | [`npg.py`](src/deeprl/algorithms/npg.py) |
| Trust Region Policy Optimization | [TRPO](reports/19_Trust-Region-Policy-Optimization.pdf) | [`trpo.py`](src/deeprl/algorithms/trpo.py) |

## Code Map

- [`on_policy.py`](src/deeprl/algorithms/on_policy.py): shared on-policy training lifecycle.
- [`a2c.py`](src/deeprl/algorithms/a2c.py), [`npg.py`](src/deeprl/algorithms/npg.py), and [`trpo.py`](src/deeprl/algorithms/trpo.py): algorithm-specific update rules.
- [`rollouts.py`](src/deeprl/rollouts.py), [`buffers.py`](src/deeprl/buffers.py), and [`advantages.py`](src/deeprl/advantages.py): on-policy data collection and target construction.
- [`policies.py`](src/deeprl/policies.py) and [`value.py`](src/deeprl/value.py): policy distributions and value functions.
- [`logger.py`](src/deeprl/logger.py) and [`evaluate.py`](src/deeprl/evaluate.py): experiment tracking, evaluation, and seed aggregation.

## Setup

Requires Python 3.12 or later and uses [uv](https://docs.astral.sh/uv/) for
dependency management.

```bash
git clone https://github.com/SaiSampathKedari/Deep-Reinforcement-Learning.git
cd Deep-Reinforcement-Learning
uv sync
```

Optional dependencies are available for TensorBoard, video recording, Box2D,
and MuJoCo through the `tb`, `video`, `box2d`, and `mujoco` extras.

## Roadmap

The phases are ordered by implementation dependencies and relevance to
continuous control and robotics.

| Phase | Area | Implementation order |
|:---:|---|---|
| 1 | On-policy completion | PPO → continuous-action PPO |
| 2 | Off-policy foundations | Replay buffer and target networks → DQN → Double DQN |
| 3 | Continuous off-policy actor-critic | DDPG → TD3 → SAC |
| 4 | Offline RL | TD3+BC → AWAC → IQL → CQL |
| 5 | Model-based RL | Neural dynamics → random shooting → CEM-MPC → PETS → MBPO → Dreamer-style latent world models |
| 6 | Goal-conditioned and multi-task RL | Goal-conditioned policies → universal value functions → hindsight experience replay → goal-conditioned SAC → multi-task policies |
| 7 | Exploration | Count and pseudo-count bonuses → curiosity and intrinsic motivation → random network distillation → ensemble uncertainty |
| 8 | Hierarchical and skill-based RL | Options → Option-Critic → skill-conditioned policies → DIAYN → subgoal and manager-worker methods |
| 9 | Robustness, safety, and sim-to-real | Domain and dynamics randomization → constrained policy optimization → residual RL → robust policy optimization → sim-to-real adaptation |
| 10 | Advanced value-based RL | Dueling DQN → prioritized experience replay → n-step DQN → C51 → QR-DQN → Rainbow |
| 11 | Recurrent and distributed RL | Recurrent PPO → IMPALA and V-trace → distributed replay → D4PG → R2D2 |
| 12 | Advanced robotics research | Meta-RL → RL with sequence models → reward and preference learning → multi-agent RL → VLA policies |

Evaluation will progress from Gymnasium correctness environments to MuJoCo
continuous-control tasks and OGBench offline and robotic-control benchmarks.

New algorithms will be accompanied by their mathematical development,
implementation, correctness tests, and multi-seed experiments.

## Related Repositories

A sequence from mathematical foundations to deep reinforcement learning:

- **Foundations:** [Real Analysis](https://github.com/SaiSampathKedari/Real-Analysis) · [Probability & Distribution Theory](https://github.com/SaiSampathKedari/Probability-and-Distribution-Theory) · [Statistical Inference Theory](https://github.com/SaiSampathKedari/Statistical-Inference-Theory)
- **Sequential-decision theory:** [Sequential Decision Making](https://github.com/SaiSampathKedari/Sequential-Decision-Making)
- **Classical reinforcement learning:** [Reinforcement Learning](https://github.com/SaiSampathKedari/Reinforcement-Learning)
- **Deep reinforcement learning:** [Deep Reinforcement Learning](https://github.com/SaiSampathKedari/Deep-Reinforcement-Learning)

## Contact

- Email: [sampath@umich.edu](mailto:sampath@umich.edu)
- LinkedIn: [sai-sampath-kedari](https://www.linkedin.com/in/sai-sampath-kedari)
