# Study 5 — Does PPO spray when pests cost more?

**Question.** With the old prior pest_loss = 0.35 (calibrated value 0.115), does the best PPO learn to spray?

## Setup

- Best study-2 configuration retrained with `pest_loss = 0.35`; `y_scale` recalibrated so the recommended calendar still averages the observed yield; rule baselines retuned and oracle recomputed in this environment.
- The pesticide head is kept; the last column of the table gives the mean sprays per season.

Every configuration is trained with 5 seeds and scored on the same held-out seasons (2017–2023, 30 pest-growth replicates each). Profit is INR/ha before environmental penalties; ± is the standard deviation over training seeds. A gap is agent minus reference (negative = below). The Shapley columns split the agent's gain over `threshold_tuned` between nitrogen, water, pesticide and sale.

## Results

| Configuration | Seeds | Profit | Gap to threshold_tuned | Gap to oracle | % of oracle | Shapley N / W / P / S | N kg · water mm · sprays |
|---|---|---|---|---|---|---|---|
| improved PPO, pest_loss = 0.115 (study 2 best) | 5 | 86,200 ± 542 | -1,032 ± 542 | -3,454 ± 542 | 96.1% | -144 / -1,005 / +0 / +117 | 213 · 235 · 0.00 |
| improved PPO, pest_loss = 0.35 | 5 | 86,410 ± 492 | +2,664 ± 492 | -4,151 ± 492 | 95.4% | -227 / -1,266 / +4,057 / +100 | 208 · 227 · 3.59 |

### Reference policies

| Run | threshold_tuned | recommended_tuned | recommended | random | oracle | threshold_tuned % of oracle |
|---|---|---|---|---|---|---|
| improved PPO, pest_loss = 0.115 (study 2 best) | 87,232 | 81,072 | 74,649 | 55,336 | 89,654 | 97.3% |
| improved PPO, pest_loss = 0.35 | 83,746 | 80,960 | 71,968 | 57,619 | 90,561 | 92.5% |

## Readings off the table

- Environment with oracle 89,654 INR/ha: best agent **improved PPO, pest_loss = 0.115 (study 2 best)** at 86,200 ± 542 INR/ha (96.1% of the oracle); `threshold_tuned` 87,232 (97.3%).
- Ahead of its own `threshold_tuned` on average: none.
- Environment with oracle 90,561 INR/ha: best agent **improved PPO, pest_loss = 0.35** at 86,410 ± 492 INR/ha (95.4% of the oracle); `threshold_tuned` 83,746 (92.5%).
- Ahead of its own `threshold_tuned` on average: improved PPO, pest_loss = 0.35 (5 of 5 seeds).

## Analysis

With pest_loss = 0.35, PPO learns to spray: 3.59 doses a season against 0 at the calibrated 0.115. It earns 86,410 ± 492 (95.4% of the oracle's 90,561) and is ahead of its own tuned threshold rule by +2,664 ± 492 in all 5 seeds; the rule earns only 83,746 here (92.5% of the oracle), and the whole advantage is the pest decision (Shapley +4,057).

So when pests matter, PPO adapts the spray decision where the rule's tuned spray cut-offs are poor. y_scale was recalibrated for the new loss; ky was not.

## Runs

- [`improved_ent0p003_ep10`](../02_improved_ppo/runs/improved_ent0p003_ep10/README.md)
- [`improved_best_pest035`](runs/improved_best_pest035/README.md)
