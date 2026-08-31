# Deep Reinforcement Learning

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0081A5)](https://gymnasium.farama.org/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-00599C)](https://mujoco.org/)
[![Development](https://img.shields.io/badge/status-active_development-F4B942)](#roadmap)

This project builds a PyTorch reinforcement learning library from the
mathematics outward. Each algorithm begins with a derivation, and the same
objects then appear explicitly in code: sampled transitions, estimators,
objectives, constraints, and parameter updates.

The current implementation covers the on-policy path from one-step
actor-critic through PPO. Work is now moving into off-policy learning, followed
by offline and model-based RL, richer policy architectures, reproducible
Gymnasium and MuJoCo benchmarks, and robotics policy training and deployment.

> **Active development:** Algorithms, mathematical derivations, tests, and
> benchmark configurations are being added continuously. APIs may change as
> the shared on-policy and off-policy foundations are completed.

## Algorithms and Derivations

| Algorithm | Implementation | Mathematics |
|---|---|---|
| One-step actor-critic | [TD(0) actor and critic updates](src/deeprl/algorithms/one_step_actor_critic.py) | [Actor-Critic](mathematical_derivations/15_Actor-Critic.pdf), [Actor-Critic with a Baseline](mathematical_derivations/16_Actor-Critic-with-a-Baseline.pdf) |
| A2C | [Synchronous rollouts and GAE](src/deeprl/algorithms/a2c.py) | [GAE Actor-Critic](mathematical_derivations/17_GAE_Actor-Critic.pdf) |
| Natural Policy Gradient | [Matrix-free natural-gradient updates](src/deeprl/algorithms/npg.py) | [Natural Policy Gradient](mathematical_derivations/18_Natural-Policy-Gradient.pdf) |
| TRPO | [KL-constrained updates with backtracking](src/deeprl/algorithms/trpo.py) | [Trust Region Policy Optimization](mathematical_derivations/19_Trust-Region-Policy-Optimization.pdf) |
| PPO-Clip | [Multi-epoch shuffled-minibatch updates](src/deeprl/algorithms/ppo.py) | In progress |
| DQN | In progress | [Deep Q-Network](mathematical_derivations/21_Deep-Q-Network.pdf) |
| Deterministic Policy Gradient | Planned | [Deterministic Policy Gradient](mathematical_derivations/23_Deterministic-Policy-Gradient.pdf) |

Supporting derivations cover the
[Policy Gradient Theorem](mathematical_derivations/10_Policy-Gradient-Theorem.pdf),
its [average-reward](mathematical_derivations/11_Average-Reward-Policy-Gradient-Theorem.pdf)
and [episodic trajectory](mathematical_derivations/12_Policy-Gradient-Theorem_Episodic-Trajectory-Route.pdf)
forms, [policy-gradient preliminaries](mathematical_derivations/13_Policy-Gradient-Preliminaries.pdf),
and [REINFORCE](mathematical_derivations/14_REINFORCE.pdf).

The on-policy learners share one
[collection and estimation lifecycle](src/deeprl/algorithms/on_policy.py), while
each algorithm owns its optimization rule.

## From Mathematics to Code

The derivations are design documents for the implementation, not separate
explanations added afterward. They determine what the buffers preserve, where
gradients stop, which quantities remain fixed during an update, and which
operations belong to the shared training lifecycle.

For example, GAE uses different masks for bootstrapping and trace continuation:

```text
td_error_t = reward_t
           + gamma * (1 - terminated_t) * V(next_state_t)
           - V(state_t)

advantage_t = td_error_t
            + gamma * lambda * (1 - terminated_t) * (1 - truncated_t)
            * advantage_(t+1)

value_target_t = V(state_t) + advantage_t
```

A time-limit truncation therefore bootstraps from its true final state, while
neither a termination nor a truncation lets the recursive trace cross into the
next episode. The corresponding operations are implemented in
[advantages.py](src/deeprl/advantages.py), with final observations preserved by
[rollouts.py](src/deeprl/rollouts.py) and [buffers.py](src/deeprl/buffers.py).

The same approach is now shaping the off-policy foundation. The DQN derivation
separates generalized Q-learning into three independently scheduled processes:
data collection, target-network refresh, and Q-function fitting. This is the
architecture currently being translated into the replay and Q-learning code.

[![An agent performing generalized Q-learning through data collection, target-network refresh, and Q-function fitting](assets/generalized-q-learning.png)](mathematical_derivations/21_Deep-Q-Network.pdf)

Across the implemented algorithms:

- rollout collection and target construction do not retain computation graphs;
- advantages and value targets stay fixed throughout an on-policy update;
- PPO compares current action log probabilities with those stored at collection;
- NPG and TRPO use Fisher-vector products without materializing the Fisher matrix;
- TRPO checks finite candidate steps and restores the old policy if its line
  search fails.

## Quick Start

Python 3.12 or later is required. Dependencies are managed with
[uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/SaiSampathKedari/Deep-Reinforcement-Learning.git
cd Deep-Reinforcement-Learning
uv sync
```

Policies and value functions are ordinary PyTorch modules supplied by the
caller:

```python
import torch

from deeprl.algorithms.ppo import PPOConfig, ppo
from deeprl.env import make_vec_env
from deeprl.policies import CategoricalPolicy
from deeprl.utils import mlp
from deeprl.value import ValueFunction

cfg = PPOConfig(total_timesteps=50_000, seed=0)
torch.manual_seed(cfg.seed)

envs = make_vec_env("CartPole-v1", num_envs=8, device=cfg.device)
obs_dim = envs.single_observation_space.shape[0]
num_actions = int(envs.single_action_space.n)

policy = CategoricalPolicy(
    mlp([obs_dim, 64, 64, num_actions], output_gain=0.01)
)
value_fn = ValueFunction(
    mlp([obs_dim, 64, 64, 1], output_gain=1.0)
)

try:
    history = ppo(envs, policy, value_fn, cfg)
finally:
    envs.close()
```

## Roadmap

Work will proceed in this order:

1. Complete DQN and Double DQN on the shared replay-buffer foundation.
2. Add configurable optimizers and reusable policy, value, and Q-function
   interfaces for discrete and continuous control.
3. Implement DDPG, TD3, and SAC.
4. Establish reproducible multi-seed Gymnasium and MuJoCo benchmarks.
5. Add offline RL, beginning with TD3+BC, AWAC, IQL, and CQL.
6. Add model-based RL and recurrent, visual, transformer, ACT, and diffusion
   policy architectures.
7. Extend the training systems to real-robot data, deployment, and VLA
   post-training.

## Related Repositories

- **Mathematical foundations:** [Real Analysis](https://github.com/SaiSampathKedari/Real-Analysis) | [Probability and Distribution Theory](https://github.com/SaiSampathKedari/Probability-and-Distribution-Theory) | [Statistical Inference Theory](https://github.com/SaiSampathKedari/Statistical-Inference-Theory)
- **Sequential-decision theory:** [Sequential Decision Making](https://github.com/SaiSampathKedari/Sequential-Decision-Making)
- **Classical reinforcement learning:** [Reinforcement Learning](https://github.com/SaiSampathKedari/Reinforcement-Learning)

## Contact

- Email: [sampath@umich.edu](mailto:sampath@umich.edu)
- LinkedIn: [Sai Sampath Kedari](https://www.linkedin.com/in/sai-sampath-kedari)
