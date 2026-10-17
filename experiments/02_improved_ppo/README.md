# Study 2 — Improved PPO and a small sweep

**Question.** Do running observation/return normalisation, linear LR decay, orthogonal init, gamma = 1, a masked storage head and 1M steps close the gap, and how sensitive is it to ent_coef and epochs?

## Setup

- Bundle `train.variant: improved`: obs_norm, ret_norm, lr_decay, ortho_init, mask_storage, gamma = 1.0, total_timesteps = 1,000,000.
- Sweep: ent_coef in {0, 0.003, 0.01} x epochs in {4, 10}; everything else as v4.
- The tuned rule baselines are reused from study 1 (same environment).

Every configuration is trained with 5 seeds and scored on the same held-out seasons (2017–2023, 30 pest-growth replicates each). Profit is INR/ha before environmental penalties; ± is the standard deviation over training seeds. A gap is agent minus reference (negative = below). The Shapley columns split the agent's gain over `threshold_tuned` between nitrogen, water, pesticide and sale.

## Results

| Configuration | Seeds | Profit | Gap to threshold_tuned | Gap to oracle | % of oracle | Shapley N / W / P / S | N kg · water mm · sprays |
|---|---|---|---|---|---|---|---|
| PPO v4 (200k steps, original settings) | 5 | 86,135 ± 179 | -1,096 ± 179 | -3,518 ± 179 | 96.1% | +17 / -1,188 / +0 / +75 | 199 · 232 · 0.00 |
| improved PPO, ent_coef=0.01, epochs=10 | 5 | 85,864 ± 412 | -1,368 ± 412 | -3,789 ± 412 | 95.8% | -216 / -1,251 / -17 / +116 | 206 · 227 · 0.00 |
| improved PPO, ent_coef=0.003, epochs=10 | 5 | 86,200 ± 542 | -1,032 ± 542 | -3,454 ± 542 | 96.1% | -144 / -1,005 / +0 / +117 | 213 · 235 · 0.00 |
| improved PPO, ent_coef=0, epochs=10 | 5 | 86,104 ± 554 | -1,128 ± 554 | -3,550 ± 554 | 96.0% | -53 / -1,198 / -25 / +148 | 206 · 235 · 0.06 |
| improved PPO, ent_coef=0.01, epochs=4 | 5 | 85,820 ± 358 | -1,412 ± 358 | -3,834 ± 358 | 95.7% | -9 / -1,456 / +0 / +52 | 197 · 230 · 0.00 |
| improved PPO, ent_coef=0.003, epochs=4 | 5 | 85,975 ± 386 | -1,257 ± 386 | -3,679 ± 386 | 95.9% | +53 / -1,367 / +0 / +57 | 200 · 237 · 0.00 |
| improved PPO, ent_coef=0, epochs=4 | 5 | 86,144 ± 271 | -1,088 ± 271 | -3,509 ± 271 | 96.1% | +7 / -1,155 / +0 / +61 | 197 · 234 · 0.00 |

### Reference policies

| Run | threshold_tuned | recommended_tuned | recommended | random | oracle | threshold_tuned % of oracle |
|---|---|---|---|---|---|---|
| PPO v4 (200k steps, original settings) | 87,232 | 81,072 | 74,649 | 55,336 | 89,654 | 97.3% |
| improved PPO, ent_coef=0.01, epochs=10 | 87,232 | 81,072 | 74,649 | 55,336 | 89,654 | 97.3% |
| improved PPO, ent_coef=0.003, epochs=10 | 87,232 | 81,072 | 74,649 | 55,336 | 89,654 | 97.3% |
| improved PPO, ent_coef=0, epochs=10 | 87,232 | 81,072 | 74,649 | 55,336 | 89,654 | 97.3% |
| improved PPO, ent_coef=0.01, epochs=4 | 87,232 | 81,072 | 74,649 | 55,336 | 89,654 | 97.3% |
| improved PPO, ent_coef=0.003, epochs=4 | 87,232 | 81,072 | 74,649 | 55,336 | 89,654 | 97.3% |
| improved PPO, ent_coef=0, epochs=4 | 87,232 | 81,072 | 74,649 | 55,336 | 89,654 | 97.3% |

## Readings off the table

- Environment with oracle 89,654 INR/ha: best agent **improved PPO, ent_coef=0.003, epochs=10** at 86,200 ± 542 INR/ha (96.1% of the oracle); `threshold_tuned` 87,232 (97.3%).
- Ahead of its own `threshold_tuned` on average: none.

## Analysis

No configuration is distinguishable from the original agent: the six improved variants span 85,820–86,200 against 86,135 ± 179 for the original, and the best one (ent_coef 0.003, epochs 10) is +65 INR/ha, well inside the seed-to-seed std of 270–550. All six are behind `threshold_tuned` in every seed (0 of 5 seeds ahead), by 1,032–1,412.

The improvements (running observation and return normalisation, linear LR decay, orthogonal init, gamma = 1, the storage-head mask and five times more steps) are standard fixes for optimisation problems, so the remaining gap to the rule is unlikely to be an optimisation problem. The Shapley split points the same way for every configuration: the loss against the rule is in the water decisions (−1,005 to −1,456); nitrogen, pesticide and sale contribute almost nothing.

ent_coef and epochs move the mean by at most about 380 INR/ha, less than the seed spread of the worst-performing settings, so these runs do not support a claim about which value is better.

## Runs

- [`baseline_v4_oracle`](../01_oracle_bound/runs/baseline_v4_oracle/README.md)
- [`improved_ent0p01_ep10`](runs/improved_ent0p01_ep10/README.md)
- [`improved_ent0p003_ep10`](runs/improved_ent0p003_ep10/README.md)
- [`improved_ent0_ep10`](runs/improved_ent0_ep10/README.md)
- [`improved_ent0p01_ep4`](runs/improved_ent0p01_ep4/README.md)
- [`improved_ent0p003_ep4`](runs/improved_ent0p003_ep4/README.md)
- [`improved_ent0_ep4`](runs/improved_ent0_ep4/README.md)
