# Experiment runs

One folder per run. Each has a `README.md` run card (methods, parameters, results, provenance),
`params/` (every parameter file used), `code/` (the code that produced it) and the tables the
dashboard reads. Profits are held-out seasons, INR/ha, PPO's first training seed.

| Run | Created | Region | Yield model | pest_loss | Mechanisms | Train seeds | PPO profit | Best other policy | PPO − best |
|---|---|---|---|---|---|---|---|---|---|
| [`joint_nw`](joint_nw/README.md) | 2026-10-17 05:57 | Punjab | pooled panel | 0.12 | groundwater, waterlogging, pest_resistance, storage | 5 | 86,476 | threshold_tuned (87,232) | -756 |
| [`fine_water`](fine_water/README.md) | 2026-10-17 06:42 | Punjab | pooled panel | 0.12 | groundwater, waterlogging, pest_resistance, storage | 5 | 86,876 | threshold_tuned (86,390) | +486 |

_Generated 2026-10-17 12:57 by `python scripts/make_runcard.py --all`._
