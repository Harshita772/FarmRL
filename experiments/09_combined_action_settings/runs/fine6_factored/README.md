# Run `fine6_factored`

**Punjab**, weather of Ludhiana; yield model: pooled panel with state fixed effects; simulator mechanisms: groundwater, waterlogging, pest_resistance, storage; created 2026-10-17 14:18:06.

## Results (held-out seasons)

| Policy | Profit (INR/ha) | Std | Yield (q/ha) | N (kg/ha) | Irrigation (mm) | Pesticide (doses) |
|---|---|---|---|---|---|---|
| Oracle | 89,703 | 5,500 | 51.3 | 200 | 199 | 0.0 |
| PPO | 86,757 | 4,721 | 50.4 | 218 | 232 | 0.0 |
| Threshold (tuned) | 86,706 | 5,021 | 50.4 | 200 | 231 | 0.0 |
| Calendar (tuned) | 84,110 | 3,359 | 49.5 | 200 | 270 | 0.0 |
| Recommended | 74,649 | 3,452 | 45.8 | 120 | 300 | 1.0 |
| Threshold rules | 63,246 | 5,439 | 46.2 | 140 | 246 | 11.8 |
| Random | 56,546 | 9,650 | 45.4 | 241 | 456 | 11.8 |
| No inputs | 12,234 | 8,253 | 17.2 | 0 | 0 | 0.0 |

The agent (`ppo` rows) across its 5 training seeds: mean 86,708, std 69, range 86,621–86,787 INR/ha. The table shows seed 42.

Gain over `threshold_tuned` split by decision (Shapley, mean over seeds): nitrogen -157, water -15, pests +0, sale +174 INR/ha.

## Methods

### Data

- Raw inputs present: not recorded (29 files listed with size and date in `metadata.json`).
- Seasons: 32 training (1985–2016), 7 held out (2017–2023); sowing years, 12 fortnights from 1 November.
- Prices: season price index from `params/prices.csv` around a base of 2425.0 INR/quintal.

### Yield model

- Form: pooled panel with state fixed effects; 1719 records, 13 states.
- In-sample R² 0.686, RMSE 6.02 q/ha.
- Held-out years ≥ 2015: R² 0.512 (state means alone 0.400).
- Identified by the data: n_coef, k_dry, heat_coef, y_max; reset to neutral/prior: r0, w_excess, r_opt.
- Pesticide loss: 0.115 (calibrated); weak: the estimate is 1.4 standard errors from zero, so it cannot be told apart from no effect; adding it raises R2 of log-yield only from 0.827 to 0.840.
- Parameters used: y_max = 49.82, n_coef = 0.02084, w_opt = 400, w_excess = 0, pest_loss = 0.1151, soil_n = 30, ky = 0.7065, y_scale = 1.123, heat_coef = 0.002337, k_dry = 0.3923, r0 = 1e+06 (`params/fitted_params.yaml`).

### Simulator

| Mechanism | Setting |
|---|---|
| groundwater | **on**: depth_m = 25.0, depth_year = 2024, decline_m_per_year = 0.5, min_depth_m = 3.0, specific_yield = 0.12, pump_efficiency = 0.35, energy_price_per_kwh = 6.0 |
| waterlogging | **on**: drain_mm_per_day = 20.0, top_layer_mm = 45.0, aeration_buffer_mm = 30.0, saturation_mm = 45.0 |
| pest_resistance | **on**: efficacy_decay = 0.25 |
| storage | **on**: max_months = 2, monthly_drift = 0.01, mean_reversion = 0.3, monthly_volatility = 0.03, cost_per_quintal_month = 10.0, loss_per_month = 0.003, interest_per_month = 0.0058 |

Costs (INR): nitrogen_per_kg = 12.9, irrigation_per_mm = 2.86, pesticide_per_level = 1183.0, fixed_per_ha = 29450.0; penalties: excess_nitrogen = 5.0, pesticide_use = 100.0, irrigation_per_mm = 0.0. Actions: N [0, 20, 40] kg/ha, water [0, 15, 30, 45, 60, 75] mm, pesticide [0, 1, 2], sale/storage heads [3, 6, 3, 3].

### Agent and training

- PPO (own implementation, `code/src/farmrl/agents/ppo.py`), 1,000,000 steps per seed, training seeds [42, 43, 44, 45, 46].
- Hyperparameters: total_timesteps = 1000000, lr = 0.0003, gamma = 1.0, gae_lambda = 0.95, clip_eps = 0.2, n_steps = 2048, batch_size = 64, epochs = 10, hidden = 64, ent_coef = 0.003, vf_coef = 0.5, variant = improved, obs_norm = True, ret_norm = True, lr_decay = True, ortho_init = True, mask_storage = True.
- Config overrides given on the command line: `train.variant=improved`, `train.ent_coef=0.003`, `train.epochs=10`, `baselines.fine_pests=true`, `env.irrigation_levels=[0,15,30,45,60,75]`.
- Compute: {'torch_threads': 1, 'parallel_seeds': 4}.
- Observation (13): stage, soil_moisture, soil_n, pest, cum_rain, cum_n, cum_water, price_index, last_rain, last_et0, et_ratio, pump_cost, spray_efficacy.

### Baselines and evaluation

- Policies: ppo, no_input, random, recommended, threshold, recommended_tuned, threshold_tuned, oracle; auxiliary (not ranked): ppo_untrained.
- `recommended_tuned`: grid search on the training seasons only (482 settings, objective reward): {'n_steps': [0, 1, 2, 3, 4], 'n_level': 2, 'w_steps': [0, 2, 4, 6, 8, 10], 'w_level': 1, 'p_steps': [], 'p_level': 0, 'store': 0}.
- `threshold_tuned`: grid search on the training seasons only (1470 settings, objective reward): {'n_below': 100, 'n_level': 2, 'n_until': 6, 'w_full_below': 0.1, 'w_half_below': 0.5, 'w_until': 12, 'p_high_above': 1.01, 'p_low_above': 1.01, 'store_below': 0.98, 'store_months': 2}.
- `oracle`: hindsight oracle, the best fixed 12-step plan per held-out season found by CEM with that season's weather and price known in advance (`oracle_plans.json`; settings default).
- Each policy replayed on every season with 30 pest-growth seeds shared across policies.
- Also computed: gain attribution, per-seed evaluation.

## Provenance

- Code: `code/` — code snapshot (details in `code/SNAPSHOT.md`).
- Commit recorded at run time: `dfb4354fee` (with uncommitted code changes).
- Software: {'python': '3.10.18', 'torch': '2.6.0+cu124', 'numpy': '2.2.5', 'pandas': '2.3.3'}.

## Files

- `models/` — one model per training seed (5 files)
- `params/` — parameter copies (4 files)
- `code/` — code snapshot (52 files)
- tables and model: `attribution.csv`, `baseline_tuning.json`, `config_snapshot.yaml`, `dry_wet.csv`, `episodes.csv`, `fitted_params.yaml`, `metadata.json`, `model.pt`, `oracle_plans.json`, `prices.csv`, `seed_episodes.csv`, `seed_summary.csv`, `summary.csv`, `train_history.csv`

_Card generated 2026-10-17 15:56 by `farmrl.runcard`; regenerate with `python scripts/make_runcard.py --run fine6_factored`._
