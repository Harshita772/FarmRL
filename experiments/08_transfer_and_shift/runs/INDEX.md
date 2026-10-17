# Experiment runs

One folder per run. Each has a `README.md` run card (methods, parameters, results, provenance),
`params/` (every parameter file used), `code/` (the code that produced it) and the tables the
dashboard reads. Profits are held-out seasons, INR/ha, PPO's first training seed.

| Run | Created | Region | Yield model | pest_loss | Mechanisms | Train seeds | PPO profit | Best other policy | PPO − best |
|---|---|---|---|---|---|---|---|---|---|
| [`haryana_native`](haryana_native/README.md) | 2026-10-17 14:18 | Haryana | pooled panel | 0.12 | groundwater, waterlogging, pest_resistance, storage | 5 | 73,628 | threshold_tuned (73,831) | -203 |

_Generated 2026-10-17 15:56 by `python scripts/make_runcard.py --all`._
