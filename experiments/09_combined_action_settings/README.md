# Study 9 — Combining the best action settings

**Question.** Does the improved PPO with a finer water grid and the joint nitrogen x water head, together, beat the rule?

## Setup

- All runs use the improved bundle (ent_coef 0.003, epochs 10, 1M steps, 5 seeds).
- `fine4_joint`: water grid {0, 25, 50, 75} mm and the joint head. `fine6_joint`: grid {0, 15, 30, 45, 60, 75} mm and the joint head. `fine6_factored`: the same 6-level grid with separate heads, to isolate the head.
- On each grid the rule is retuned and maps its doses onto the grid (its 37.5 mm becomes the nearest level, ties upward), so it is handicapped relative to the standard grid; the last table below therefore also gives the gap to the **standard-grid** `threshold_tuned` of study 1.

Every configuration is trained with 5 seeds and scored on the same held-out seasons (2017–2023, 30 pest-growth replicates each). Profit is INR/ha before environmental penalties; ± is the standard deviation over training seeds. A gap is agent minus reference (negative = below). The Shapley columns split the agent's gain over `threshold_tuned` between nitrogen, water, pesticide and sale.

## Results

| Configuration | Seeds | Profit | Gap to threshold_tuned | Gap to oracle | % of oracle | Shapley N / W / P / S | N kg · water mm · sprays |
|---|---|---|---|---|---|---|---|
| PPO v4 (200k steps, original settings) | 5 | 86,135 ± 179 | -1,096 ± 179 | -3,518 ± 179 | 96.1% | +17 / -1,188 / +0 / +75 | 199 · 232 · 0.00 |
| improved PPO, no forecast (study 2 best) | 5 | 86,200 ± 542 | -1,032 ± 542 | -3,454 ± 542 | 96.1% | -144 / -1,005 / +0 / +117 | 213 · 235 · 0.00 |
| improved PPO, grid 0/25/50/75, factored (study 3) | 5 | 86,587 ± 286 | +196 ± 286 | -3,014 ± 286 | 96.6% | -371 / +428 / +0 / +139 | 198 · 227 · 0.00 |
| improved PPO, standard grid, joint head (study 3) | 5 | 86,250 ± 194 | -982 ± 194 | -3,403 ± 194 | 96.2% | -151 / -985 / +0 / +155 | 198 · 233 · 0.00 |
| improved PPO, grid 0/25/50/75, joint head | 5 | 86,821 ± 315 | +431 ± 315 | -2,780 ± 315 | 96.9% | -341 / +636 / -17 / +154 | 207 · 229 · 0.00 |
| improved PPO, grid 0/15/30/45/60/75, joint head | 5 | 86,548 ± 273 | -158 ± 273 | -3,155 ± 273 | 96.5% | -357 / +125 / +0 / +74 | 194 · 229 · 0.00 |
| improved PPO, grid 0/15/30/45/60/75, factored heads | 5 | 86,708 ± 69 | +1 ± 69 | -2,996 ± 69 | 96.7% | -157 / -15 / +0 / +174 | 201 · 231 · 0.00 |

### Reference policies

| Run | threshold_tuned | recommended_tuned | recommended | random | oracle | threshold_tuned % of oracle |
|---|---|---|---|---|---|---|
| PPO v4 (200k steps, original settings) | 87,232 | 81,072 | 74,649 | 55,336 | 89,654 | 97.3% |
| improved PPO, no forecast (study 2 best) | 87,232 | 81,072 | 74,649 | 55,336 | 89,654 | 97.3% |
| improved PPO, grid 0/25/50/75, factored (study 3) | 86,390 | 83,238 | 74,649 | 55,846 | 89,601 | 96.4% |
| improved PPO, standard grid, joint head (study 3) | 87,232 | 81,072 | 74,649 | 55,336 | 89,654 | 97.3% |
| improved PPO, grid 0/25/50/75, joint head | 86,390 | 83,238 | 74,649 | 55,846 | 89,601 | 96.4% |
| improved PPO, grid 0/15/30/45/60/75, joint head | 86,706 | 84,110 | 74,649 | 56,546 | 89,703 | 96.7% |
| improved PPO, grid 0/15/30/45/60/75, factored heads | 86,706 | 84,110 | 74,649 | 56,546 | 89,703 | 96.7% |

## Readings off the table

- Environment with oracle 89,654 INR/ha: best agent **improved PPO, standard grid, joint head (study 3)** at 86,250 ± 194 INR/ha (96.2% of the oracle); `threshold_tuned` 87,232 (97.3%).
- Ahead of its own `threshold_tuned` on average: none.
- Environment with oracle 89,601 INR/ha: best agent **improved PPO, grid 0/25/50/75, joint head** at 86,821 ± 315 INR/ha (96.9% of the oracle); `threshold_tuned` 86,390 (96.4%).
- Ahead of its own `threshold_tuned` on average: improved PPO, grid 0/25/50/75, factored (study 3) (4 of 5 seeds), improved PPO, grid 0/25/50/75, joint head (5 of 5 seeds).
- Environment with oracle 89,703 INR/ha: best agent **improved PPO, grid 0/15/30/45/60/75, factored heads** at 86,708 ± 69 INR/ha (96.7% of the oracle); `threshold_tuned` 86,706 (96.7%).
- Ahead of its own `threshold_tuned` on average: improved PPO, grid 0/15/30/45/60/75, factored heads (3 of 5 seeds).

## Against the standard-grid rule (87,232 INR/ha)

| Configuration | Profit | Gap to standard-grid `threshold_tuned` |
|---|---|---|
| PPO v4 (200k steps, original settings) | 86,135 ± 179 | -1,096 |
| improved PPO, no forecast (study 2 best) | 86,200 ± 542 | -1,032 |
| improved PPO, grid 0/25/50/75, factored (study 3) | 86,587 ± 286 | -645 |
| improved PPO, standard grid, joint head (study 3) | 86,250 ± 194 | -982 |
| improved PPO, grid 0/25/50/75, joint head | 86,821 ± 315 | -410 |
| improved PPO, grid 0/15/30/45/60/75, joint head | 86,548 ± 273 | -684 |
| improved PPO, grid 0/15/30/45/60/75, factored heads | 86,708 ± 69 | -524 |

## Runs

- [`baseline_v4_oracle`](../01_oracle_bound/runs/baseline_v4_oracle/README.md)
- [`improved_ent0p003_ep10`](../02_improved_ppo/runs/improved_ent0p003_ep10/README.md)
- [`fine_water`](../03_action_structure/runs/fine_water/README.md)
- [`joint_nw`](../03_action_structure/runs/joint_nw/README.md)
- [`fine4_joint`](runs/fine4_joint/README.md)
- [`fine6_joint`](runs/fine6_joint/README.md)
- [`fine6_factored`](runs/fine6_factored/README.md)
