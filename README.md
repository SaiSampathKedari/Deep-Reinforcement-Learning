# Deep Reinforcement Learning

**Mathematical derivations, proofs, and correctness-focused PyTorch implementations of deep reinforcement learning algorithms.**

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0081A5)](https://gymnasium.farama.org/)
[![Development](https://img.shields.io/badge/status-active_development-F4B942)](#roadmap)

This repository develops each algorithm from its mathematical objective to its
tensor implementation. The emphasis is on understanding every estimator,
preserving the semantics of the underlying Markov decision process, and building
components that can be reused across increasingly capable RL algorithms.

> **Active development:** the on-policy foundation is implemented. Continuous
> control, off-policy learning, offline RL, and robotics experiments are the next
> major stages.

## Implemented

- **[One-step actor-critic](src/deeprl/algorithms/one_step_actor_critic.py):** TD(0) policy and value updates.
- **[A2C](src/deeprl/algorithms/a2c.py):** synchronous vector rollouts with generalized advantage estimation.
- **[Natural Policy Gradient](src/deeprl/algorithms/npg.py):** matrix-free Fisher-vector products and conjugate gradient.
- **[TRPO](src/deeprl/algorithms/trpo.py):** natural-gradient proposals with a KL-constrained backtracking line search.
- **[PPO-Clip](src/deeprl/algorithms/ppo.py):** clipped surrogate optimization over shuffled minibatches and multiple epochs.

These algorithms share one [on-policy training engine](src/deeprl/algorithms/on_policy.py),
while each learner owns its optimization rule.

## From Mathematics To Code

The repository follows one path throughout:

```text
objective and derivation
        |
        v
rollout data and episode-boundary semantics
        |
        v
fixed advantages and value targets
        |
        v
algorithm-specific optimization
        |
        v
evaluation and reproducible metrics
```

For the current on-policy algorithms, that path is implemented by:

```text
collect_rollout
    -> evaluate_next_values
    -> generalized_advantage_estimate
    -> learner.update
```

The GAE implementation keeps termination and truncation distinct:

```text
delta_t        = reward_t + gamma * (1 - terminated_t) * V(next_state_t) - V(state_t)
advantage_t    = delta_t + gamma * lambda * (1 - done_t) * advantage_(t+1)
value_target_t = V(state_t) + advantage_t
```

See [rollout collection](src/deeprl/rollouts.py),
[advantage estimation](src/deeprl/advantages.py), and the
[rollout buffer](src/deeprl/buffers.py) for the corresponding tensor operations.
NPG and TRPO solve the damped natural-gradient system through
Fisher-vector products, so the full Fisher matrix is never materialized.

## Mathematical Reports

The reports are part of the implementation, not supplementary notes.

- **Policy-gradient foundations:** [Preliminaries](reports/13_Policy-Gradient-Preliminaries.pdf), [Policy Gradient Theorem](reports/10_Policy-Gradient-Theorem.pdf), [episodic trajectory derivation](reports/12_Policy-Gradient-Theorem_Episodic-Trajectory-Route.pdf), and [average-reward theorem](reports/11_Average-Reward-Policy-Gradient-Theorem.pdf).
- **REINFORCE:** [derivation](reports/14_REINFORCE.pdf).
- **Actor-critic:** [derivation](reports/15_Actor-Critic.pdf), [baselines and advantages](reports/16_Actor-Critic-with-a-Baseline.pdf), and [implementation](src/deeprl/algorithms/one_step_actor_critic.py).
- **GAE and A2C:** [derivation](reports/17_GAE_Actor-Critic.pdf) and [implementation](src/deeprl/algorithms/a2c.py).
- **Natural Policy Gradient:** [derivation](reports/18_Natural-Policy-Gradient.pdf) and [implementation](src/deeprl/algorithms/npg.py).
- **TRPO:** [derivation](reports/19_Trust-Region-Policy-Optimization.pdf) and [implementation](src/deeprl/algorithms/trpo.py).
- **PPO-Clip:** [implementation](src/deeprl/algorithms/ppo.py); mathematical report in progress.

## Correctness Decisions

- Terminated transitions do not bootstrap.
- Truncated transitions bootstrap from the true final observation.
- Both episode boundaries stop the recursive GAE trace.
- Rollouts and targets are collected without retaining neural-network graphs.
- Advantages and value targets remain fixed throughout an update phase.
- PPO compares current log probabilities with collection-time log probabilities.
- NPG and TRPO keep the rollout policy fixed while constructing the natural-gradient step.

## Repository Structure

```text
reports/                    mathematical derivations and proofs
src/deeprl/
  advantages.py             GAE and value-target construction
  buffers.py                on-policy rollout storage and batching
  rollouts.py               vectorized environment collection
  policies.py               stochastic policy modules
  value.py                  state-value modules
  logger.py                 training metrics and run artifacts
  evaluate.py               policy evaluation and seed aggregation
  algorithms/
    on_policy.py            shared on-policy training lifecycle
    a2c.py                  A2C learner
    npg.py                  Natural Policy Gradient learner
    trpo.py                 TRPO learner
    ppo.py                  PPO-Clip learner
```

## Installation

Python 3.12 or later is required. Dependencies are managed with
[uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/SaiSampathKedari/Deep-Reinforcement-Learning.git
cd Deep-Reinforcement-Learning
uv sync
```

Optional dependency groups are available for TensorBoard, video recording,
Box2D, and MuJoCo.

## Roadmap

Development proceeds along two connected tracks.

**Deep RL algorithms**

1. Continuous-action policies and Gymnasium/MuJoCo validation.
2. Replay buffers, target networks, DQN, and Double DQN.
3. DDPG, TD3, and SAC for continuous control.
4. TD3+BC, AWAC, IQL, and CQL for offline RL.
5. Model-based, goal-conditioned, and advanced value-based methods.

**Robotic manipulation**

1. Reproduce the official OpenArm MuJoCo and ACT pipelines.
2. Build behavioral-cloning baselines with visual and temporal observations.
3. Add ACT and diffusion-policy training for bimanual action sequences.
4. Add interactive imitation learning and offline-to-online improvement.
5. Validate deployment, latency, safety, and sim-to-real transfer on OpenArm.

The detailed dependency order and deployment criteria are documented in the
[OpenArm manipulation roadmap](docs/openarm_manipulation_roadmap.md).

## Related Repositories

- [Real Analysis](https://github.com/SaiSampathKedari/Real-Analysis)
- [Probability and Distribution Theory](https://github.com/SaiSampathKedari/Probability-and-Distribution-Theory)
- [Statistical Inference Theory](https://github.com/SaiSampathKedari/Statistical-Inference-Theory)
- [Sequential Decision Making](https://github.com/SaiSampathKedari/Sequential-Decision-Making)
- [Reinforcement Learning](https://github.com/SaiSampathKedari/Reinforcement-Learning)

## Contact

[Email](mailto:sampath@umich.edu) | [LinkedIn](https://www.linkedin.com/in/sai-sampath-kedari)
