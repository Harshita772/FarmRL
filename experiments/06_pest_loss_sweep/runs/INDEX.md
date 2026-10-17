# Experiment runs

One folder per run. Each has a `README.md` run card (methods, parameters, results, provenance),
`params/` (every parameter file used), `code/` (the code that produced it) and the tables the
dashboard reads. Profits are held-out seasons, INR/ha, PPO's first training seed.

| Run | Created | Region | Yield model | pest_loss | Mechanisms | Train seeds | PPO profit | Best other policy | PPO − best |
|---|---|---|---|---|---|---|---|---|---|
| [`pest_0p115`](pest_0p115/README.md) | 2026-10-17 13:20 | Punjab | pooled panel | 0.12 | groundwater, waterlogging, pest_resistance, storage | 5 | 86,139 | threshold_tuned (87,232) | -1,093 |
| [`pest_0p2`](pest_0p2/README.md) | 2026-10-17 14:18 | Punjab | pooled panel | 0.20 | groundwater, waterlogging, pest_resistance, storage | 5 | 86,137 | threshold_tuned (85,904) | +233 |
| [`pest_0p3`](pest_0p3/README.md) | 2026-10-17 14:18 | Punjab | pooled panel | 0.30 | groundwater, waterlogging, pest_resistance, storage | 5 | 85,889 | threshold_tuned (83,406) | +2,483 |
| [`pest_0p35`](pest_0p35/README.md) | 2026-10-17 14:18 | Punjab | pooled panel | 0.35 | groundwater, waterlogging, pest_resistance, storage | 5 | 86,483 | threshold_tuned (83,746) | +2,737 |
| [`pest_0p5`](pest_0p5/README.md) | 2026-10-17 14:18 | Punjab | pooled panel | 0.50 | groundwater, waterlogging, pest_resistance, storage | 5 | 88,229 | threshold_tuned (84,762) | +3,467 |

_Generated 2026-10-17 15:56 by `python scripts/make_runcard.py --all`._
