# Deep Reinforcement Learning

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0081A5)](https://gymnasium.farama.org/)
[![Development](https://img.shields.io/badge/status-active_development-F4B942)](#roadmap)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A PyTorch reinforcement learning library in which every algorithm starts as a
mathematical derivation and ends as readable, tested code you can train with.

- **Derivation first.** Each algorithm has a written derivation, and the code
  follows it: what is stored, where gradients stop, what stays fixed during an
  update.
- **Readable.** The core modules explain, step by step, why each step exists.
- **Practical.** Checkpoints and resume, saved policies that load without any
  training code, and observation normalization saved with the policy.

> **Active development:** APIs may change. The examples are runnable recipes,
> not reproduced benchmarks.

## Algorithms

| Algorithm | Code | Derivation | Status |
|---|---|---|---|
| One-step actor-critic | [one_step_actor_critic.py](src/deeprl/algorithms/one_step_actor_critic.py) | [Actor-Critic](mathematical_derivations/18_Actor-Critic.pdf), [with a baseline](mathematical_derivations/19_Actor-Critic-with-a-Baseline.pdf) | ✅ |
| A2C (GAE) | [a2c.py](src/deeprl/algorithms/a2c.py) | [GAE Actor-Critic](mathematical_derivations/20_GAE_Actor-Critic.pdf) | ✅ |
| Natural Policy Gradient | [npg.py](src/deeprl/algorithms/npg.py) | [NPG](mathematical_derivations/21_Natural-Policy-Gradient.pdf) | ✅ |
| TRPO | [trpo.py](src/deeprl/algorithms/trpo.py) | [TRPO](mathematical_derivations/22_Trust-Region-Policy-Optimization.pdf) | ✅ |
| PPO-Clip | [ppo.py](src/deeprl/algorithms/ppo.py) | [PPO](mathematical_derivations/23_Proximal-Policy-Optimization.pdf) | ✅ |
| DQN | [dqn.py](src/deeprl/algorithms/dqn.py) | [DQN](mathematical_derivations/24_Deep-Q-Network.pdf) | ✅ |
| Double DQN | [dqn.py](src/deeprl/algorithms/dqn.py) (`double_q=True`) | [Double DQN](mathematical_derivations/25_Double-DQN.pdf) | ✅ |
| DDPG | — | [DPG](mathematical_derivations/26_Deterministic-Policy-Gradient.pdf), [DDPG](mathematical_derivations/27_Deep-Deterministic-Policy-Gradient.pdf) | derivation done |

A2C, NPG, TRPO, and PPO share the [on-policy trainer](src/deeprl/training/train_on_policy.py);
DQN and Double DQN use the [off-policy trainer](src/deeprl/training/train_off_policy.py).
One-step actor-critic is a standalone teaching implementation.

## Quickstart

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/SaiSampathKedari/Deep-Reinforcement-Learning.git
cd Deep-Reinforcement-Learning
uv sync --extra tb --extra video
```

Train PPO on CartPole-v1 (CPU, 8 environments, 102,400 steps):

```bash
uv run python -m examples.gymnasium.cartpole.run_cartpole_ppo \
  --output-dir outputs/gymnasium/cartpole/ppo/seed_0/run_01
```

Evaluate the saved policy, and view the training curves:

```bash
uv run python -m examples.gymnasium.cartpole.evaluate_cartpole \
  outputs/gymnasium/cartpole/ppo/seed_0/run_01/policy --episodes 10

uv run tensorboard --logdir outputs/gymnasium/cartpole/ppo/seed_0/run_01/events
```

Each run writes one folder (use a new one per run):

```text
run_01/
├── config.json      every setting of the run
├── metrics.jsonl    episodes, training reports, evaluations
├── events/          TensorBoard logs
├── checkpoints/     last.pt and periodic step_<N>.pt, for resuming
└── policy/          the trained policy, loadable without any training code
```

The [CartPole guide](examples/gymnasium/cartpole/README.md) covers video
recording, rendering, and resuming from a checkpoint.

## Examples

| Task | What it shows |
|---|---|
| [CartPole-v1](examples/gymnasium/cartpole/README.md) | Discrete actions; the CPU starting point |
| [InvertedPendulum-v5](examples/gymnasium/inverted_pendulum/) | A Gaussian policy for continuous control |
| [Hopper-v5](examples/gymnasium/hopper/) | Continuous-action locomotion |
| [Humanoid-v5](examples/gymnasium/humanoid/README.md) | High-dimensional control with observation normalization |

Each example keeps its networks, settings, training script, and evaluator
together, and calls the shared library. The MuJoCo tasks need one more extra:

```bash
uv sync --extra mujoco --extra tb --extra video
```

## How the Library Is Organized

| Part | Files | Responsibility |
|---|---|---|
| Learners | [algorithms/](src/deeprl/algorithms/) | Configs, targets, losses, and parameter updates |
| Trainers | [train_on_policy.py](src/deeprl/training/train_on_policy.py), [train_off_policy.py](src/deeprl/training/train_off_policy.py) | When to collect, learn, evaluate, and save |
| Data | [collectors.py](src/deeprl/collectors.py), [buffers.py](src/deeprl/buffers.py), [advantages.py](src/deeprl/advantages.py) | Environment stepping, rollout/replay storage, GAE |
| Networks and processing | [policies.py](src/deeprl/policies.py), [value.py](src/deeprl/value.py), [processors.py](src/deeprl/processors.py), [inference.py](src/deeprl/inference.py) | Wrappers for your networks; observation and action processing |
| Evaluation and logging | [evaluate.py](src/deeprl/evaluate.py), [logger.py](src/deeprl/logger.py) | Episode returns, JSONL metrics, TensorBoard |
| Saving and loading | [checkpoint.py](src/deeprl/checkpoint.py), [policy_io.py](src/deeprl/policy_io.py) | Training checkpoints and exported policies |

Buffers store raw observations; normalization is applied when a network reads
them, with statistics the trainer updates on a fixed schedule. Network
architectures stay with each example. The
[CartPole training script](examples/gymnasium/cartpole/run_cartpole_ppo.py)
shows the full assembly: environments, networks, learner, logger, evaluator,
and the training call.

## Checkpoints and Saved Policies

- **Checkpoints** (`checkpoints/`) hold everything needed to continue training:
  networks, optimizers, observation statistics, progress, and random-number
  state. Pass `--resume-from .../checkpoints/last.pt` to continue a run. A
  resumed run starts fresh episodes, and off-policy runs refill an empty replay
  buffer before learning again.
- **Policies** (`policy/`) hold only what acting needs: a manifest, the model
  weights, and the observation statistics. They are written when training
  finishes, and load without a learner or optimizer.

To evaluate a checkpoint before training finishes, export it to a new folder:

```python
from deeprl.policy_io import export_policy_from_checkpoint

export_policy_from_checkpoint(
    "outputs/gymnasium/cartpole/ppo/seed_0/run_01/checkpoints/last.pt",
    "outputs/gymnasium/cartpole/ppo/seed_0/run_01/policy_snapshot",
)
```

The full rules are documented in [checkpoint.py](src/deeprl/checkpoint.py) and
[policy_io.py](src/deeprl/policy_io.py).

## From Derivations to Code

The derivations are working specifications. They decide what the buffers keep,
where gradients stop, and which quantities stay fixed during an update: for
example, termination-aware GAE, PPO's fixed old log-probabilities, and TRPO's
rollback. The DQN derivation splits Q-learning into three independently
scheduled steps (data collection, target-network refresh, and Q-function
fitting), which is exactly how the off-policy trainer and DQN learner are built.

[![Generalized Q-learning: data collection, target-network refresh, and Q-function fitting](assets/generalized-q-learning.png)](mathematical_derivations/24_Deep-Q-Network.pdf)

[Browse all derivations](mathematical_derivations/README.md). Their numbering
continues from the [Reinforcement Learning](https://github.com/SaiSampathKedari/Reinforcement-Learning/tree/master/mathematical_derivations)
repository: 13–20 cover policy gradients and actor-critic; 21–27 cover NPG,
TRPO, PPO, DQN, Double DQN, deterministic policy gradients, and DDPG.

## Roadmap

- **Done:** on-policy methods (actor-critic, A2C, NPG, TRPO, PPO), DQN, and Double DQN.
- **Next:** DDPG, TD3, SAC, and multi-seed Gymnasium and MuJoCo benchmarks.
- **Later:** offline RL, imitation learning, model-based RL and world models,
  robotics integration, and VLA post-training.

## Related Repositories

- **Mathematical foundations:** [Real Analysis](https://github.com/SaiSampathKedari/Real-Analysis) · [Probability and Distribution Theory](https://github.com/SaiSampathKedari/Probability-and-Distribution-Theory) · [Statistical Inference Theory](https://github.com/SaiSampathKedari/Statistical-Inference-Theory)
- **Sequential decision making:** [Sequential Decision Making](https://github.com/SaiSampathKedari/Sequential-Decision-Making)
- **Classical reinforcement learning:** [Reinforcement Learning](https://github.com/SaiSampathKedari/Reinforcement-Learning)

## Contact

- Email: [sampath@umich.edu](mailto:sampath@umich.edu)
- LinkedIn: [Sai Sampath Kedari](https://www.linkedin.com/in/sai-sampath-kedari)

## License

Released under the [MIT License](LICENSE).
