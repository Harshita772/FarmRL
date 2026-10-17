# Study 3 — Action structure

**Question.** Does a joint softmax over nitrogen x water, or a finer water grid, help?

## Setup

- Base: the best configuration of study 2 (same bundle, ent_coef and epochs).
- `joint_nw`: one softmax over the 9 nitrogen x water pairs; pesticide and storage keep their own heads.
- `fine_water`: water grid {0, 25, 50, 75} mm instead of {0, 37.5, 75}. The rule baselines are retuned on the new grid and map their doses onto it (their 37.5 mm becomes 50 mm); the oracle is recomputed on the grid.

Every configuration is trained with 5 seeds and scored on the same held-out seasons (2017–2023, 30 pest-growth replicates each). Profit is INR/ha before environmental penalties; ± is the standard deviation over training seeds. A gap is agent minus reference (negative = below). The Shapley columns split the agent's gain over `threshold_tuned` between nitrogen, water, pesticide and sale.

## Results

| Configuration | Seeds | Profit | Gap to threshold_tuned | Gap to oracle | % of oracle | Shapley N / W / P / S | N kg · water mm · sprays |
|---|---|---|---|---|---|---|---|
| improved PPO, factored heads (study 2 best) | 5 | 86,200 ± 542 | -1,032 ± 542 | -3,454 ± 542 | 96.1% | -144 / -1,005 / +0 / +117 | 213 · 235 · 0.00 |
| improved PPO, joint N x water head | 5 | 86,250 ± 194 | -982 ± 194 | -3,403 ± 194 | 96.2% | -151 / -985 / +0 / +155 | 198 · 233 · 0.00 |
| improved PPO, water grid 0/25/50/75 | 5 | 86,587 ± 286 | +196 ± 286 | -3,014 ± 286 | 96.6% | -371 / +428 / +0 / +139 | 198 · 227 · 0.00 |

### Reference policies

| Run | threshold_tuned | recommended_tuned | recommended | random | oracle | threshold_tuned % of oracle |
|---|---|---|---|---|---|---|
| improved PPO, factored heads (study 2 best) | 87,232 | 81,072 | 74,649 | 55,336 | 89,654 | 97.3% |
| improved PPO, joint N x water head | 87,232 | 81,072 | 74,649 | 55,336 | 89,654 | 97.3% |
| improved PPO, water grid 0/25/50/75 | 86,390 | 83,238 | 74,649 | 55,846 | 89,601 | 96.4% |

## Readings off the table

- Environment with oracle 89,654 INR/ha: best agent **improved PPO, joint N x water head** at 86,250 ± 194 INR/ha (96.2% of the oracle); `threshold_tuned` 87,232 (97.3%).
- Ahead of its own `threshold_tuned` on average: none.
- Environment with oracle 89,601 INR/ha: best agent **improved PPO, water grid 0/25/50/75** at 86,587 ± 286 INR/ha (96.6% of the oracle); `threshold_tuned` 86,390 (96.4%).
- Ahead of its own `threshold_tuned` on average: improved PPO, water grid 0/25/50/75 (4 of 5 seeds).

## Analysis

A joint softmax over nitrogen x water gives 86,250 ± 194, +50 over the best factored head (86,200 ± 542): no measurable change, so head independence is not what limits PPO here.

The finer water grid is the one configuration ahead of its own `threshold_tuned` (+196 ± 286, 4 of 5 seeds), but that comparison is not like for like: on the {0, 25, 50, 75} grid the rule's 37.5 mm 'half' irrigation becomes 50 mm and it cannot use 25 mm, so the rule falls from 87,232 to 86,390. In absolute terms PPO gains +387 over the best factored run (86,587 vs 86,200, about one seed std) and is still 645 below the standard-grid rule. The evidence that a finer grid helps PPO is weak.

## Runs

- [`improved_ent0p003_ep10`](../02_improved_ppo/runs/improved_ent0p003_ep10/README.md)
- [`joint_nw`](runs/joint_nw/README.md)
- [`fine_water`](runs/fine_water/README.md)
