# Deep Reinforcement Learning

Deep Reinforcement Learning from mathematical foundations to PyTorch
implementations: rigorous proofs, step-by-step derivations of objectives and
gradient estimators, correctness-focused actor-critic code, and reproducible
experiment infrastructure for Gymnasium.

> ⚠️ **This repository is under active development.** New mathematical
> derivations, algorithms, tests, experiment configurations, and benchmark
> studies will be added throughout the coming weeks.

## Current Scope

| Topic | Status |
|---|:---:|
| Policy-gradient foundations and mathematical reports | Available |
| One-step actor-critic | Implemented |
| Generalized Advantage Estimation and A2C | Implemented |
| Natural Policy Gradient and TRPO | Implemented |
| Proximal Policy Optimization | Planned |
| DQN, DDPG, TD3, and SAC | Planned |

The current code includes vectorized rollout storage, explicit termination and
truncation handling, a shared on-policy training engine, matrix-free natural
gradient updates, TRPO backtracking, structured logging, scheduled policy
evaluation, and cross-seed aggregation.

## From Mathematics to Code

Each algorithm is developed from its objective and gradient estimator before
being translated into PyTorch. For example, GAE is implemented directly from

```math
\delta_t = R_{t+1}
+ \gamma(1-\mathrm{termination}_t)V(S_{t+1}) - V(S_t),
```

```math
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

- **Policy-gradient foundations:** [preliminaries](reports/13_Policy-Gradient-Preliminaries.pdf), [discounted theorem](reports/10_Policy-Gradient-Theorem.pdf), [trajectory proof](reports/12_Policy-Gradient-Theorem_Episodic-Trajectory-Route.pdf), and [average-reward theorem](reports/11_Average-Reward-Policy-Gradient-Theorem.pdf).
- **Algorithms and estimators:** [REINFORCE](reports/14_REINFORCE.pdf), [actor-critic](reports/15_Actor-Critic.pdf), [baselines and advantages](reports/16_Actor-Critic-with-a-Baseline.pdf), and [Generalized Advantage Estimation](reports/17_GAE_Actor-Critic.pdf).
- **Second-order policy optimization:** [Natural Policy Gradient](reports/18_Natural-Policy-Gradient.pdf) and [Trust Region Policy Optimization](reports/19_Trust-Region-Policy-Optimization.pdf).

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

1. Proximal Policy Optimization and continuous-action policies.
2. Gymnasium control and MuJoCo benchmark studies.
3. DQN and its principal extensions.
4. DDPG, TD3, and SAC.

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
