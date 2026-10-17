# Experiment runs

One folder per run. Each has a `README.md` run card (methods, parameters, results, provenance),
`params/` (every parameter file used), `code/` (the code that produced it) and the tables the
dashboard reads. Profits are held-out seasons, INR/ha, PPO's first training seed.

| Run | Created | Region | Yield model | pest_loss | Mechanisms | Train seeds | PPO profit | Best other policy | PPO − best |
|---|---|---|---|---|---|---|---|---|---|
| [`improved_ent0p01_ep10`](improved_ent0p01_ep10/README.md) | 2026-10-17 01:33 | Punjab | pooled panel | 0.12 | groundwater, waterlogging, pest_resistance, storage | 5 | 86,438 | threshold_tuned (87,232) | -794 |
| [`improved_ent0p003_ep10`](improved_ent0p003_ep10/README.md) | 2026-10-17 02:20 | Punjab | pooled panel | 0.12 | groundwater, waterlogging, pest_resistance, storage | 5 | 86,139 | threshold_tuned (87,232) | -1,093 |
| [`improved_ent0_ep10`](improved_ent0_ep10/README.md) | 2026-10-17 03:08 | Punjab | pooled panel | 0.12 | groundwater, waterlogging, pest_resistance, storage | 5 | 86,648 | threshold_tuned (87,232) | -584 |
| [`improved_ent0p01_ep4`](improved_ent0p01_ep4/README.md) | 2026-10-17 03:56 | Punjab | pooled panel | 0.12 | groundwater, waterlogging, pest_resistance, storage | 5 | 85,949 | threshold_tuned (87,232) | -1,283 |
| [`improved_ent0p003_ep4`](improved_ent0p003_ep4/README.md) | 2026-10-17 04:37 | Punjab | pooled panel | 0.12 | groundwater, waterlogging, pest_resistance, storage | 5 | 85,528 | threshold_tuned (87,232) | -1,703 |
| [`improved_ent0_ep4`](improved_ent0_ep4/README.md) | 2026-10-17 05:17 | Punjab | pooled panel | 0.12 | groundwater, waterlogging, pest_resistance, storage | 5 | 85,910 | threshold_tuned (87,232) | -1,322 |

_Generated 2026-10-17 12:57 by `python scripts/make_runcard.py --all`._
