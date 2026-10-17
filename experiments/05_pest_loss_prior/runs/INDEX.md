# Experiment runs

One folder per run. Each has a `README.md` run card (methods, parameters, results, provenance),
`params/` (every parameter file used), `code/` (the code that produced it) and the tables the
dashboard reads. Profits are held-out seasons, INR/ha, PPO's first training seed.

| Run | Created | Region | Yield model | pest_loss | Mechanisms | Train seeds | PPO profit | Best other policy | PPO − best |
|---|---|---|---|---|---|---|---|---|---|
| [`improved_best_pest035`](improved_best_pest035/README.md) | 2026-10-17 08:45 | Punjab | pooled panel | 0.35 | groundwater, waterlogging, pest_resistance, storage | 5 | 86,483 | threshold_tuned (83,746) | +2,737 |

_Generated 2026-10-17 12:57 by `python scripts/make_runcard.py --all`._
