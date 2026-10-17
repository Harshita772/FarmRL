# Experiment runs

One folder per run. Each has a `README.md` run card (methods, parameters, results, provenance),
`params/` (every parameter file used), `code/` (the code that produced it) and the tables the
dashboard reads. Profits are held-out seasons, INR/ha, PPO's first training seed.

| Run | Created | Region | Yield model | pest_loss | Mechanisms | Train seeds | PPO profit | Best other policy | PPO − best |
|---|---|---|---|---|---|---|---|---|---|
| [`layout_fc25mm`](layout_fc25mm/README.md) | 2026-10-17 22:41 | Punjab | pooled panel | 0.12 | groundwater, waterlogging, pest_resistance, forecast, storage | 5 | 87,501 | threshold_tuned (86,545) | +956 |
| [`layout_fc0mm`](layout_fc0mm/README.md) | 2026-10-17 22:41 | Punjab | pooled panel | 0.12 | groundwater, waterlogging, pest_resistance, forecast, storage | 5 | 88,562 | threshold_tuned (86,545) | +2,018 |
| [`layout_fc10mm`](layout_fc10mm/README.md) | 2026-10-17 22:41 | Punjab | pooled panel | 0.12 | groundwater, waterlogging, pest_resistance, forecast, storage | 5 | 88,427 | threshold_tuned (86,545) | +1,883 |

_Generated 2026-10-17 23:35 by `python scripts/make_runcard.py --all`._
