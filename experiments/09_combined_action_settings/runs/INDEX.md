# Experiment runs

One folder per run. Each has a `README.md` run card (methods, parameters, results, provenance),
`params/` (every parameter file used), `code/` (the code that produced it) and the tables the
dashboard reads. Profits are held-out seasons, INR/ha, PPO's first training seed.

| Run | Created | Region | Yield model | pest_loss | Mechanisms | Train seeds | PPO profit | Best other policy | PPO − best |
|---|---|---|---|---|---|---|---|---|---|
| [`fine4_joint`](fine4_joint/README.md) | 2026-10-17 14:18 | Punjab | pooled panel | 0.12 | groundwater, waterlogging, pest_resistance, storage | 5 | 86,820 | threshold_tuned (86,390) | +430 |
| [`fine6_factored`](fine6_factored/README.md) | 2026-10-17 14:18 | Punjab | pooled panel | 0.12 | groundwater, waterlogging, pest_resistance, storage | 5 | 86,757 | threshold_tuned (86,706) | +51 |
| [`fine6_joint`](fine6_joint/README.md) | 2026-10-17 14:18 | Punjab | pooled panel | 0.12 | groundwater, waterlogging, pest_resistance, storage | 5 | 86,215 | threshold_tuned (86,706) | -491 |

_Generated 2026-10-17 15:56 by `python scripts/make_runcard.py --all`._
