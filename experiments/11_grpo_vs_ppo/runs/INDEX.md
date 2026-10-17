# Experiment runs

One folder per run. Each has a `README.md` run card (methods, parameters, results, provenance),
`params/` (every parameter file used), `code/` (the code that produced it) and the tables the
dashboard reads. Profits are held-out seasons, INR/ha, PPO's first training seed.

| Run | Created | Region | Yield model | pest_loss | Mechanisms | Train seeds | PPO profit | Best other policy | PPO − best |
|---|---|---|---|---|---|---|---|---|---|
| [`grpo_g16`](grpo_g16/README.md) | 2026-10-17 14:18 | Punjab | pooled panel | 0.12 | groundwater, waterlogging, pest_resistance, storage | 5 | 85,340 | threshold_tuned (87,232) | -1,892 |
| [`grpo_g8`](grpo_g8/README.md) | 2026-10-17 14:18 | Punjab | pooled panel | 0.12 | groundwater, waterlogging, pest_resistance, storage | 5 | 86,442 | threshold_tuned (87,232) | -790 |
| [`grpo_g8_stepwise`](grpo_g8_stepwise/README.md) | 2026-10-17 14:18 | Punjab | pooled panel | 0.12 | groundwater, waterlogging, pest_resistance, storage | 5 | 85,493 | threshold_tuned (87,232) | -1,739 |

_Generated 2026-10-17 15:56 by `python scripts/make_runcard.py --all`._
