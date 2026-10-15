# Experiment runs

One folder per run. Each has a `README.md` run card (methods, parameters, results, provenance),
`params/` (every parameter file used), `code/` (the code that produced it) and the tables the
dashboard reads. Profits are held-out seasons, INR/ha, PPO's first training seed.

| Run | Created | Region | Yield model | pest_loss | Mechanisms | Train seeds | PPO profit | Best other policy | PPO − best |
|---|---|---|---|---|---|---|---|---|---|
| [`v1`](v1/README.md) | 2026-10-10 17:09 | Punjab | single region | 0.35 | – | 3 | 84,457 | threshold (74,400) | +10,057 |
| [`v2`](v2/README.md) | 2026-10-10 17:45 | Punjab | pooled panel | 0.35 | – | 3 | 86,630 | recommended (76,404) | +10,227 |
| [`v3_haryana`](v3_haryana/README.md) | 2026-10-11 13:26 | Haryana | pooled panel | 0.12 | groundwater, waterlogging, pest_resistance, storage | 5 | 74,754 | threshold_tuned (73,831) | +923 |
| [`v3`](v3/README.md) | 2026-10-11 13:55 | Punjab | pooled panel | 0.12 | groundwater, waterlogging, pest_resistance, storage | 5 | 86,736 | threshold_tuned (87,232) | -496 |
| [`v4`](v4/README.md) | 2026-10-15 16:40 | Punjab | pooled panel | 0.12 | groundwater, waterlogging, pest_resistance, storage | 5 | 86,599 | threshold_tuned (87,232) | -633 |
| [`v4_haryana`](v4_haryana/README.md) | 2026-10-15 22:04 | Haryana | pooled panel | 0.12 | groundwater, waterlogging, pest_resistance, storage | 5 | 74,119 | threshold_tuned (73,831) | +289 |

_Generated 2026-10-15 22:39 by `python scripts/make_runcard.py --all`._
