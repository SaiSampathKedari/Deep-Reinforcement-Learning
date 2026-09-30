# Mathematical derivations

The mathematical path from policy-gradient foundations to deep RL algorithms.
Numbers belong to one shared reading sequence with
[Reinforcement Learning](https://github.com/SaiSampathKedari/Reinforcement-Learning/blob/master/mathematical_derivations/README.md):
the same report has the same number and filename in both repositories. Keep
assigned numbers stable and use these indexes to guide readers to future
supplementary material.

## Prerequisites in Reinforcement Learning

Start with the [RL derivation index](https://github.com/SaiSampathKedari/Reinforcement-Learning/blob/master/mathematical_derivations/README.md)
for 01–03 (policy evaluation in Sutton notation, Bellman operators, policy
improvement), 04–10 (tabular MC and TD methods), and 11 (value-function
approximation). Slot 12 is reserved for on-policy control with approximation.

## Shared policy-gradient and actor-critic foundations

| # | Derivation | Topic | Previous # |
|---|---|---|---|
| 13 | [Policy Gradient Theorem](13_Policy-Gradient-Theorem.pdf) | discounted objective and exact gradient | 10 |
| 14 | [Average-Reward Policy Gradient Theorem](14_Average-Reward-Policy-Gradient-Theorem.pdf) | continuing tasks and stationary distributions | 11 |
| 15 | [Policy Gradient Theorem: Episodic Trajectory Route](15_Policy-Gradient-Theorem_Episodic-Trajectory-Route.pdf) | trajectory likelihoods and reward-to-go | 12 |
| 16 | [Policy Gradient Preliminaries](16_Policy-Gradient-Preliminaries.pdf) | objectives, stochastic gradient estimators, shared estimator decomposition | 13 |
| 17 | [REINFORCE](17_REINFORCE.pdf) | Monte Carlo policy gradient | 14 |
| 18 | [Actor-Critic](18_Actor-Critic.pdf) | from the exact gradient to an actor-critic update | 15 |
| 19 | [Actor-Critic with a Baseline](19_Actor-Critic-with-a-Baseline.pdf) | baselines, advantages, and TD errors | 16 |
| 20 | [GAE Actor-Critic](20_GAE_Actor-Critic.pdf) | n-step advantages and their λ-mixture | 17 |

Report 16 uses the results of 13–15 to introduce the estimator notation used by
the subsequent algorithms.

## Policy optimization and deep off-policy learning

| # | Derivation | Previous # |
|---|---|---|
| 21 | [Natural Policy Gradient](21_Natural-Policy-Gradient.pdf) | 18 |
| 22 | [Trust Region Policy Optimization](22_Trust-Region-Policy-Optimization.pdf) | 19 |
| 23 | **Proximal Policy Optimization — derivation in progress** | 20 (reserved) |
| 24 | [Deep Q-Network](24_Deep-Q-Network.pdf) | 21 |
| 25 | [Double DQN](25_Double-DQN.pdf) | 22 |
| 26 | [Deterministic Policy Gradient](26_Deterministic-Policy-Gradient.pdf) | 23 |
| 27 | [Deep Deterministic Policy Gradient](27_Deep-Deterministic-Policy-Gradient.pdf) | 24 |

## Earlier report numbers

The **Previous #** column maps the former RL/Deep RL filename numbers to the
current sequence. Existing numbers shifted by three when the Sutton-notation
foundations were added at the start. Compiled PDFs retain their original text,
including historical report-number references. For earlier tabular reports,
consult the Previous # column in the
[RL index](https://github.com/SaiSampathKedari/Reinforcement-Learning/blob/master/mathematical_derivations/README.md).
This mapping does not apply to other repositories' numbering or to pre-existing
inconsistent citations within a PDF. Reserved entries have no PDF yet.
