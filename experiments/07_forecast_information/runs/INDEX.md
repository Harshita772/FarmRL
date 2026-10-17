# Experiment runs

One folder per run. Each has a `README.md` run card (methods, parameters, results, provenance),
`params/` (every parameter file used), `code/` (the code that produced it) and the tables the
dashboard reads. Profits are held-out seasons, INR/ha, PPO's first training seed.

| Run | Created | Region | Yield model | pest_loss | Mechanisms | Train seeds | PPO profit | Best other policy | PPO − best |
|---|---|---|---|---|---|---|---|---|---|
| [`forecast_sd0mm`](forecast_sd0mm/README.md) | 2026-10-17 14:18 | Punjab | pooled panel | 0.12 | groundwater, waterlogging, pest_resistance, forecast, storage | 5 | 88,796 | threshold_tuned (87,385) | +1,411 |
| [`forecast_sd10mm`](forecast_sd10mm/README.md) | 2026-10-17 14:18 | Punjab | pooled panel | 0.12 | groundwater, waterlogging, pest_resistance, forecast, storage | 5 | 88,105 | threshold_tuned (87,385) | +720 |
| [`forecast_sd25mm`](forecast_sd25mm/README.md) | 2026-10-17 14:18 | Punjab | pooled panel | 0.12 | groundwater, waterlogging, pest_resistance, forecast, storage | 5 | 86,764 | threshold_tuned (87,385) | -621 |

_Generated 2026-10-17 15:56 by `python scripts/make_runcard.py --all`._
