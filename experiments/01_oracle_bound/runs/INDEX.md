# Experiment runs

One folder per run. Each has a `README.md` run card (methods, parameters, results, provenance),
`params/` (every parameter file used), `code/` (the code that produced it) and the tables the
dashboard reads. Profits are held-out seasons, INR/ha, PPO's first training seed.

| Run | Created | Region | Yield model | pest_loss | Mechanisms | Train seeds | PPO profit | Best other policy | PPO − best |
|---|---|---|---|---|---|---|---|---|---|
| [`baseline_v4_oracle`](baseline_v4_oracle/README.md) | 2026-10-17 01:16 | Punjab | pooled panel | 0.12 | groundwater, waterlogging, pest_resistance, storage | 5 | 86,053 | threshold_tuned (87,232) | -1,179 |

_Generated 2026-10-17 12:57 by `python scripts/make_runcard.py --all`._
