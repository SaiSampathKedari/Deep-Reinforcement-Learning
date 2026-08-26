# Deep Reinforcement Learning

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0081A5)](https://gymnasium.farama.org/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-00599C)](https://mujoco.org/)
[![Development](https://img.shields.io/badge/status-active_development-F4B942)](#direction)

This project builds a PyTorch reinforcement learning library from the
mathematics outward. Each algorithm begins with a derivation and carries the
same objects into code: rollout data, estimators, policy objectives, and
parameter updates.

## Direction

The on-policy foundation is implemented. Development now expands across five
connected tracks:

- **RL algorithms:** off-policy, offline, and model-based reinforcement
  learning.
- **Imitation and policy learning:** behavior cloning, ACT, and diffusion
  policies.
- **Model architectures:** recurrent, visual, and transformer-based policies.
- **Evaluation:** reproducible multi-seed benchmarks on Gymnasium and MuJoCo.
- **Robotics:** learning from real-robot data, policy deployment, and VLA
  post-training.

The immediate work is continuous-action policy support and reproducible
Gymnasium/MuJoCo evaluation, followed by DQN, Double DQN, DDPG, TD3, and SAC.
Later stages will build on those foundations rather than introducing separate,
disconnected training systems.

## Implemented

| Algorithm | What is implemented | Derivation |
|---|---|---|
| [One-step actor-critic](src/deeprl/algorithms/one_step_actor_critic.py) | TD(0) actor and critic updates | [Actor-Critic](reports/15_Actor-Critic.pdf), [Actor-Critic with a Baseline](reports/16_Actor-Critic-with-a-Baseline.pdf) |
| [A2C](src/deeprl/algorithms/a2c.py) | Synchronous vector rollouts and generalized advantage estimation | [GAE Actor-Critic](reports/17_GAE_Actor-Critic.pdf) |
| [Natural Policy Gradient](src/deeprl/algorithms/npg.py) | Matrix-free Fisher-vector products and conjugate gradient | [Natural Policy Gradient](reports/18_Natural-Policy-Gradient.pdf) |
| [TRPO](src/deeprl/algorithms/trpo.py) | KL-constrained natural-gradient step with backtracking | [Trust Region Policy Optimization](reports/19_Trust-Region-Policy-Optimization.pdf) |
| [PPO-Clip](src/deeprl/algorithms/ppo.py) | Clipped surrogate optimization over shuffled minibatches and multiple epochs | Report in progress |

The algorithms share a small [on-policy training lifecycle](src/deeprl/algorithms/on_policy.py),
while each learner owns its optimization rule.

## Mathematics In The Implementation

The estimator is not treated as a preprocessing detail. For GAE, the code
implements two different boundary decisions:

```text
td_error_t = reward_t
           + gamma * (1 - terminated_t) * V(next_state_t)
           - V(state_t)

advantage_t = td_error_t
            + gamma * lambda * (1 - terminated_t) * (1 - truncated_t)
            * advantage_(t+1)

value_target_t = V(state_t) + advantage_t
```

This means a time-limit truncation still bootstraps from the true final state,
but neither termination nor truncation allows the recursive trace to enter the
next episode. The corresponding tensor operations are in
[advantages.py](src/deeprl/advantages.py), with final observations preserved by
[rollouts.py](src/deeprl/rollouts.py) and [buffers.py](src/deeprl/buffers.py).

Other deliberate implementation choices include:

- rollout collection and target construction do not retain neural-network
  computation graphs;
- advantages and value targets remain fixed throughout an update;
- PPO evaluates new log probabilities against the actions and log probabilities
  stored by the collecting policy;
- NPG and TRPO compute Fisher-vector products without materializing a parameter
  by parameter Fisher matrix;
- TRPO evaluates the finite candidate step and restores the old parameters when
  every line-search candidate fails.

## Using PPO

Python 3.12 or later is required. Dependencies are managed with
[uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/SaiSampathKedari/Deep-Reinforcement-Learning.git
cd Deep-Reinforcement-Learning
uv sync
```

The policy and value networks are ordinary PyTorch modules supplied by the
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

## Code Structure

```text
src/deeprl/
  advantages.py          return, advantage, and value-target estimators
  buffers.py             time-major rollout storage and minibatching
  rollouts.py            vector-environment collection
  policies.py            stochastic policy modules
  value.py               state-value modules
  evaluate.py            isolated policy evaluation and seed aggregation
  logger.py              interval metrics and run artifacts
  algorithms/
    on_policy.py         shared collection-update lifecycle
    a2c.py               Advantage Actor-Critic
    npg.py               Natural Policy Gradient
    trpo.py              Trust Region Policy Optimization
    ppo.py               Proximal Policy Optimization
```

## Mathematical Notes

The supporting reports include
[policy-gradient preliminaries](reports/13_Policy-Gradient-Preliminaries.pdf),
the [Policy Gradient Theorem](reports/10_Policy-Gradient-Theorem.pdf), an
[episodic trajectory derivation](reports/12_Policy-Gradient-Theorem_Episodic-Trajectory-Route.pdf),
the [average-reward theorem](reports/11_Average-Reward-Policy-Gradient-Theorem.pdf),
and [REINFORCE](reports/14_REINFORCE.pdf). Algorithm-specific reports are linked
in the table above.

## Next

1. Add configurable learner-owned optimizers and continuous-action policy
   distributions.
2. Validate PPO, NPG, and TRPO on Gymnasium and MuJoCo continuous-control tasks.
3. Add replay storage, target networks, DQN, Double DQN, DDPG, TD3, and SAC.
4. Harden the library through public tests, reproducible benchmarks,
   checkpointing, and GPU-native environment support.

## Related Repositories

- **Mathematical foundations:** [Real Analysis](https://github.com/SaiSampathKedari/Real-Analysis) · [Probability and Distribution Theory](https://github.com/SaiSampathKedari/Probability-and-Distribution-Theory) · [Statistical Inference Theory](https://github.com/SaiSampathKedari/Statistical-Inference-Theory)
- **Sequential-decision theory:** [Sequential Decision Making](https://github.com/SaiSampathKedari/Sequential-Decision-Making)
- **Classical reinforcement learning:** [Reinforcement Learning](https://github.com/SaiSampathKedari/Reinforcement-Learning)

## Contact

- Email: [sampath@umich.edu](mailto:sampath@umich.edu)
- LinkedIn: [Sai Sampath Kedari](https://www.linkedin.com/in/sai-sampath-kedari)
