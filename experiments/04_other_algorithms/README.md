# Study 4 — Other algorithms

**Question.** Does a different learner, or a direct search over the rule's parameters, do better than PPO?

## Setup

- `dqn_branching`: dueling Q-network with one branch per input, double-DQN target averaged over the active branches, 1M steps, gamma = 1, lr 5e-4, epsilon 1 -> 0.05 over half the run.
- `cem_threshold`: cross-entropy search (population 64, 40 iterations, 20% elite) over the ten parameters of the threshold rule, scored on the training seasons with the training reward.
- Same seasons, seeds and reused rule tuning as study 2.

Every configuration is trained with 5 seeds and scored on the same held-out seasons (2017–2023, 30 pest-growth replicates each). Profit is INR/ha before environmental penalties; ± is the standard deviation over training seeds. A gap is agent minus reference (negative = below). The Shapley columns split the agent's gain over `threshold_tuned` between nitrogen, water, pesticide and sale.

## Results

| Configuration | Seeds | Profit | Gap to threshold_tuned | Gap to oracle | % of oracle | Shapley N / W / P / S | N kg · water mm · sprays |
|---|---|---|---|---|---|---|---|
| PPO v4 (200k steps, original settings) | 5 | 86,135 ± 179 | -1,096 ± 179 | -3,518 ± 179 | 96.1% | +17 / -1,188 / +0 / +75 | 199 · 232 · 0.00 |
| improved PPO (study 2 best) | 5 | 86,200 ± 542 | -1,032 ± 542 | -3,454 ± 542 | 96.1% | -144 / -1,005 / +0 / +117 | 213 · 235 · 0.00 |
| branching dueling DQN, 1M steps | 5 | 83,617 ± 1,741 | -3,615 ± 1,741 | -6,036 ± 1,741 | 93.3% | -343 / -2,602 / -733 / +63 | 212 · 221 · 0.83 |
| CEM over threshold-rule parameters | 5 | 84,178 ± 192 | -3,054 ± 192 | -5,476 ± 192 | 93.9% | -126 / +82 / -3,088 / +77 | 205 · 238 · 3.46 |

### Reference policies

| Run | threshold_tuned | recommended_tuned | recommended | random | oracle | threshold_tuned % of oracle |
|---|---|---|---|---|---|---|
| PPO v4 (200k steps, original settings) | 87,232 | 81,072 | 74,649 | 55,336 | 89,654 | 97.3% |
| improved PPO (study 2 best) | 87,232 | 81,072 | 74,649 | 55,336 | 89,654 | 97.3% |
| branching dueling DQN, 1M steps | 87,232 | 81,072 | 74,649 | 55,336 | 89,654 | 97.3% |
| CEM over threshold-rule parameters | 87,232 | 81,072 | 74,649 | 55,336 | 89,654 | 97.3% |

## Readings off the table

- Environment with oracle 89,654 INR/ha: best agent **improved PPO (study 2 best)** at 86,200 ± 542 INR/ha (96.1% of the oracle); `threshold_tuned` 87,232 (97.3%).
- Ahead of its own `threshold_tuned` on average: none.

## Analysis

Branching DQN reaches 83,617 ± 1,741 (seeds from 80,954 to 85,545), about 2,600 below the best PPO and far less stable. CEM over the threshold rule's ten parameters reaches 84,178 ± 192, 3,054 below the grid-tuned rule of the same family.

CEM's shortfall is almost entirely the pesticide decision (Shapley −3,088): it settles on about 3.5 sprays a season where the tuned rule sprays never. With 64 candidates x 40 iterations and 2 replicates per candidate, the pest cut-offs are hard to separate from noise, so the search did not converge to the 'never spray' corner that the block-wise grid search found. This says the grid-searched rule is a strong baseline, not that CEM is a poor method in general.

## Runs

- [`baseline_v4_oracle`](../01_oracle_bound/runs/baseline_v4_oracle/README.md)
- [`improved_ent0p003_ep10`](../02_improved_ppo/runs/improved_ent0p003_ep10/README.md)
- [`dqn_branching`](runs/dqn_branching/README.md)
- [`cem_threshold`](runs/cem_threshold/README.md)
