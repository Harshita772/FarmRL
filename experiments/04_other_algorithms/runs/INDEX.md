# Experiment runs

One folder per run. Each has a `README.md` run card (methods, parameters, results, provenance),
`params/` (every parameter file used), `code/` (the code that produced it) and the tables the
dashboard reads. Profits are held-out seasons, INR/ha, PPO's first training seed.

| Run | Created | Region | Yield model | pest_loss | Mechanisms | Train seeds | PPO profit | Best other policy | PPO − best |
|---|---|---|---|---|---|---|---|---|---|
| [`dqn_branching`](dqn_branching/README.md) | 2026-10-17 07:31 | Punjab | pooled panel | 0.12 | groundwater, waterlogging, pest_resistance, storage | 5 | 80,954 | threshold_tuned (87,232) | -6,278 |
| [`cem_threshold`](cem_threshold/README.md) | 2026-10-17 08:40 | Punjab | pooled panel | 0.12 | groundwater, waterlogging, pest_resistance, storage | 5 | 84,198 | threshold_tuned (87,232) | -3,034 |

_Generated 2026-10-17 12:57 by `python scripts/make_runcard.py --all`._
