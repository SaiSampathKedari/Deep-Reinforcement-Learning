# Deep Reinforcement Learning

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0081A5)](https://gymnasium.farama.org/)
[![Development](https://img.shields.io/badge/status-active_development-F4B942)](#roadmap)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A PyTorch reinforcement learning library built from mathematical derivations
to explicit learning algorithms and runnable experiments.

The code keeps objectives, estimators, gradient updates, and environment
interaction visible. Task-specific networks and settings live with each example;
the learning algorithms, buffers, evaluation, and checkpoint handling live in
the library. The aim is to make the same implementation useful for studying an
algorithm and training a policy.

On-policy implementations and discrete/continuous-control PPO examples are
available. DQN and Double DQN are the current implementation focus.

> **Active development:** Public APIs may change. Runnable examples are not
> claims of reproduced benchmarks or optimal policies.

## Algorithms

### On-Policy Optimization (Implemented)

- [One-step actor-critic](src/deeprl/algorithms/one_step_actor_critic.py) ([Actor-Critic](mathematical_derivations/18_Actor-Critic.pdf), [baseline](mathematical_derivations/19_Actor-Critic-with-a-Baseline.pdf))
- [A2C](src/deeprl/algorithms/a2c.py) ([GAE Actor-Critic](mathematical_derivations/20_GAE_Actor-Critic.pdf))
- [Natural Policy Gradient](src/deeprl/algorithms/npg.py) ([derivation](mathematical_derivations/21_Natural-Policy-Gradient.pdf))
- [TRPO](src/deeprl/algorithms/trpo.py) ([derivation](mathematical_derivations/22_Trust-Region-Policy-Optimization.pdf))
- [PPO-Clip](src/deeprl/algorithms/ppo.py) ([derivation](mathematical_derivations/23_Proximal-Policy-Optimization.pdf))

A2C, NPG, TRPO, and PPO use the shared rollout trainer. One-step actor-critic is
a separate educational implementation.

### Off-Policy Learning (In Development)

- **DQN:** [derivation](mathematical_derivations/24_Deep-Q-Network.pdf); implementation in progress
- **Double DQN:** [derivation](mathematical_derivations/25_Double-DQN.pdf); implementation in progress
- **DDPG:** [Deterministic Policy Gradient](mathematical_derivations/26_Deterministic-Policy-Gradient.pdf) and [DDPG](mathematical_derivations/27_Deep-Deterministic-Policy-Gradient.pdf) derivations complete

## Quickstart

Python 3.12 or later is required. Dependencies are managed with
[uv](https://docs.astral.sh/uv/). Run these commands from the repository checkout.

```bash
git clone https://github.com/SaiSampathKedari/Deep-Reinforcement-Learning.git
cd Deep-Reinforcement-Learning
uv sync --extra tb --extra video

uv run python -m examples.gymnasium.cartpole.run_cartpole_ppo \
  --output-dir outputs/gymnasium/cartpole/ppo/seed_0/run_01
```

This trains PPO on **CartPole-v1** on CPU with eight environments and 102,400
transitions. Choose a new output directory for each run. The run saves its
configuration, JSONL metrics, TensorBoard events, training checkpoints, and a
final inference policy directory.

Evaluate the final saved policy:

```bash
uv run python -m examples.gymnasium.cartpole.evaluate_cartpole \
  outputs/gymnasium/cartpole/ppo/seed_0/run_01/policy \
  --episodes 10 --seed 20000
```

Evaluation uses the inference mode saved with the policy. Pass
`--mode deterministic` or `--mode stochastic` to override it explicitly.

View the training and evaluation curves:

```bash
uv run tensorboard --logdir outputs/gymnasium/cartpole/ppo/seed_0/run_01/events
```

See the [CartPole guide](examples/gymnasium/cartpole/README.md) for video
recording, interactive rendering, and continuation from a checkpoint. The
`tb` and `video` extras are optional; `uv sync` installs the base training setup.

## Environment Examples

Each task keeps its networks, configuration, training entry point, and evaluator
together. The examples call the shared library rather than reimplementing PPO.

- [CartPole-v1](examples/gymnasium/cartpole/README.md): discrete actions and a CPU starting point.
- [InvertedPendulum-v5](examples/gymnasium/inverted_pendulum/): Gaussian policy for continuous control.
- [Hopper-v5](examples/gymnasium/hopper/): continuous-action locomotion.
- [Humanoid-v5](examples/gymnasium/humanoid/README.md): higher-dimensional control, normalization, and a documented reference recipe.

For the MuJoCo examples, install the additional environment dependencies:

```bash
uv sync --extra mujoco --extra tb --extra video
```

CUDA can train the networks; these Gymnasium MuJoCo environments run their
physics on CPU. Each recipe specifies its settings and evaluation procedure.
Multi-seed comparisons, confidence intervals, and ablation results remain
benchmark work, separate from having a runnable example.

## Library Structure

- **Learning rules:** [algorithms/](src/deeprl/algorithms/) contains configs, target preparation, losses, and parameter updates. Learners use native PyTorch optimizers; NPG/TRPO perform their actor updates directly.
- **Training lifecycle:** [train_on_policy.py](src/deeprl/training/train_on_policy.py) coordinates rollouts, learner target preparation and updates, evaluation, and checkpoint timing. [train_off_policy.py](src/deeprl/training/train_off_policy.py) coordinates replay collection, minibatch updates, target-network refresh, evaluation, and checkpoints.
- **Data and targets:** [buffers.py](src/deeprl/buffers.py), [collectors.py](src/deeprl/collectors.py), and [advantages.py](src/deeprl/advantages.py) separate storage, interaction, and advantage estimation.
- **Networks and preprocessing:** [policies.py](src/deeprl/policies.py) and [value.py](src/deeprl/value.py) wrap caller-supplied networks; [processors.py](src/deeprl/processors.py) handles shared input normalization and action conversion, composed with the model by [inference.py](src/deeprl/inference.py). Buffers retain raw observations; the trainer explicitly schedules statistics updates. Architectures stay with the task examples.
- **Evaluation and logging:** [evaluate.py](src/deeprl/evaluate.py) and [logger.py](src/deeprl/logger.py) handle policy assessment, episode statistics, JSONL metrics, and optional TensorBoard output.
- **State and continuation:** Concrete learners explicitly save/load their native component state. [learner_state.py](src/deeprl/learner_state.py) validates state keys and resume configuration; [checkpoint.py](src/deeprl/checkpoint.py) and [lifecycle.py](src/deeprl/lifecycle.py) handle checkpoint I/O, configuration, Torch RNG state, and provenance.
- **Policy artifacts:** [policy_io.py](src/deeprl/policy_io.py) saves inference configuration, model state, and observation statistics separately from training checkpoints. The acting model is an actor or DQN's online Q-network. A caller-supplied factory rebuilds that architecture; loading needs no learner or optimizer.

Rollout buffers preserve the time/environment layout `(T, N)`. Replay buffers
use flat FIFO transition storage. Both support fixed-shape tensors and
TensorDict data; structured storage does not automatically make a policy support
every observation or action space.

Import the shared on-policy components through:

```python
from deeprl.algorithms.ppo import PPOConfig, PPOLearner
from deeprl.training.train_on_policy import train_on_policy
```

The [CartPole training script](examples/gymnasium/cartpole/run_cartpole_ppo.py)
shows the complete assembly: environments, networks, learner, logger, evaluator,
and training call. The learner contract is in `deeprl.algorithms.on_policy`.
These are the only on-policy import paths; there are no compatibility shims.

### Policies and Checkpoints

Examples export `policy/manifest.json`, `policy/model.pt`, and
`policy/processor.pt` only after training returns successfully. A running or
interrupted job is not guaranteed to have a current `policy/` export.
The manifest declares the input/action spec, model configuration, action
selector, processing settings, and inference defaults. The model file holds
the actor's state or DQN's online Q-network state. Target networks, separate
value networks, optimizers, and reward/value-target statistics are not part of
the inference artifact. Exports require a fresh directory and do not overwrite
an existing policy.

`load_policy()` receives an explicit trusted model factory and expected model
ID; it does not import code named in JSON. Task factories live in their network
files. Built-in action selectors are restored from their saved names; custom
selectors also require caller-supplied code and a matching ID. Evaluators use
this artifact, fixed observation statistics, and the policy's `act()` interface.
They report raw, undiscounted episode returns, with the resolved action mode
and episode seeds.

The shared trainers save `last.pt` and periodic `step_<transitions>.pt` files
after complete rollouts or off-policy collection/update cycles. Checkpoints
include learner state (networks, optimizers, and any learner-owned statistics),
shared observation-processing state, progress, supported Torch RNG state,
configuration, and informational dependency/source provenance. The PPO examples
also supply a policy manifest for the checkpoint, without duplicating model
state or observation statistics. Only the current checkpoint contract is
supported, without schema versions, migrations, or legacy fallbacks. There is
no automatic best-checkpoint selection.

To evaluate a saved snapshot before a run finishes, export it to a new policy
directory:

```python
from deeprl.policy_io import export_policy_from_checkpoint

export_policy_from_checkpoint(
    "outputs/gymnasium/cartpole/ppo/seed_0/run_01/checkpoints/last.pt",
    "outputs/gymnasium/cartpole/ppo/seed_0/run_01/policy_snapshot",
)
```

Then use the CartPole PPO evaluator:

```bash
uv run python -m examples.gymnasium.cartpole.evaluate_cartpole \
  outputs/gymnasium/cartpole/ppo/seed_0/run_01/policy_snapshot \
  --episodes 10 --seed 20000
```

Export uses the saved policy manifest, model state, observation statistics,
action processing, and inference defaults without constructing a model or
learner. It selects `learner["q_network"]` when the checkpoint's `algorithm` is
`"dqn"` (both DQN and Double DQN), and `learner["policy"]` otherwise. A missing
selected key is an error; the exporter never falls back to a target network.
The exported observations and statistics include only ungrouped features and
those in the `"actor"` group. This input-selection rule also applies to DQN's
online Q-network, even though it has no separate actor.

Provenance records the snapshot's `global_step` as `training_step`. Checkpoints
without the required manifest or state fail; architecture and preprocessing
are never inferred from current example defaults. Loading a DQN export requires
a matching Q-network factory, and its saved `"greedy"` decision is restored
automatically. The existing task evaluators use actor factories; there is not
yet a DQN evaluation entry point.

Resume restores learning state but starts **fresh environment episodes**;
it does not restore the simulator's exact trajectory. Off-policy checkpoints
also omit replay data, so resumed runs refill an empty buffer before learning.
In the PPO examples, use `checkpoints/last.pt` with `--resume-from`, not as the
evaluator's policy argument. Deterministic and stochastic actor evaluations
remain separate measurements; DQN's inference selector is always greedy.

The [one-step CartPole tutorial](examples/gymnasium/cartpole/run_cartpole_one_step_actor_critic.py)
is intentionally separate: it exports the same `policy/` artifact, without the shared
trainer's checkpoint/resume contract.

## From Derivations to Systems

The derivations are working specifications for the implementation. They determine
what the buffers preserve, where gradients stop, and which quantities remain
fixed during an update. This is visible in termination-aware GAE, fixed old-policy
log probabilities in PPO, matrix-free Fisher products, and TRPO's parameter
rollback.

The DQN derivation separates generalized Q-learning into data collection,
target-network refresh, and Q-function fitting. These independently scheduled
operations guide the shared off-policy trainer and DQN learner.

[![Generalized Q-learning: data collection, target-network refresh, and Q-function fitting](assets/generalized-q-learning.png)](mathematical_derivations/24_Deep-Q-Network.pdf)

[Browse the mathematical derivations](mathematical_derivations/README.md).
Their numbering follows a shared reading sequence with
[Reinforcement Learning](https://github.com/SaiSampathKedari/Reinforcement-Learning/tree/master/mathematical_derivations).
Reports 13-20 cover policy-gradient and actor-critic foundations; 21-27 cover
NPG, TRPO, PPO, DQN, Double DQN, deterministic policy gradients, and DDPG.

## Roadmap

**Now:** Implement and validate DQN and Double DQN with a separate off-policy training loop.

**Next:** DDPG, TD3, SAC, and multi-seed Gymnasium and MuJoCo benchmarks.

**Longer term:** Offline RL, imitation learning, model-based RL and world models,
richer policy architectures, robotics integration, and VLA post-training.

## Related Repositories

- **Mathematical foundations:** [Real Analysis](https://github.com/SaiSampathKedari/Real-Analysis) | [Probability and Distribution Theory](https://github.com/SaiSampathKedari/Probability-and-Distribution-Theory) | [Statistical Inference Theory](https://github.com/SaiSampathKedari/Statistical-Inference-Theory)
- **Sequential-decision theory:** [Sequential Decision Making](https://github.com/SaiSampathKedari/Sequential-Decision-Making)
- **Classical reinforcement learning:** [Reinforcement Learning](https://github.com/SaiSampathKedari/Reinforcement-Learning)

## Contact

- Email: [sampath@umich.edu](mailto:sampath@umich.edu)
- LinkedIn: [Sai Sampath Kedari](https://www.linkedin.com/in/sai-sampath-kedari)

## License

Released under the [MIT License](LICENSE).
