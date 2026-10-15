# Run `v2`

**Punjab**, weather of Ludhiana; yield model: pooled panel with state fixed effects; simulator mechanisms: none beyond the base water/N/pest model; created 2026-10-10 17:45:46.

## Results (held-out seasons)

| Policy | Profit (INR/ha) | Std | Yield (q/ha) | N (kg/ha) | Irrigation (mm) | Pesticide (doses) |
|---|---|---|---|---|---|---|
| PPO | 86,630 | 4,430 | 50.8 | 202 | 251 | 3.3 |
| Recommended | 76,404 | 4,992 | 45.1 | 120 | 300 | 1.0 |
| Threshold rules | 72,157 | 3,790 | 46.4 | 140 | 236 | 7.1 |
| Random | 70,908 | 8,199 | 48.9 | 244 | 437 | 11.8 |
| No inputs | 8,982 | 7,457 | 15.8 | 0 | 0 | 0.0 |

## Methods

### Data

- Raw inputs present: `cacp_rabi_price_policy_latest.pdf`, `icrisat_crops.csv`, `icrisat_fertiliser.csv`, `icrisat_irrigation.csv`, `icrisat_prices.csv`, `rbi_msp_foodgrains_2014.html`, `rbi_msp_foodgrains_latest.html`, `weather_ambala.csv`, `weather_gurgaon.csv`, `weather_hisar.csv`, `weather_jind.csv`, `weather_karnal.csv`, `weather_ludhiana.csv`, `weather_narnaul.csv`, `weather_rohtak.csv` (43 files listed with size and date in `metadata.json`).
- Seasons: 32 training (1985–2016), 7 held out (2017–2023); sowing years, 12 fortnights from 1 November.
- Prices: season price index from `params/prices.csv` around a base of 2425.0 INR/quintal.

### Yield model

- Form: pooled panel with state fixed effects; 1719 records, 13 states.
- In-sample R² 0.686, RMSE 6.02 q/ha.
- Held-out years ≥ 2015: R² 0.512 (state means alone 0.400).
- Identified by the data: n_coef, k_dry, heat_coef, y_max; reset to neutral/prior: r0, w_excess, r_opt, pest_loss.
- Pesticide loss: 0.350 (prior (no pesticide data)).
- Parameters used: y_max = 49.82, n_coef = 0.02084, w_opt = 400, w_excess = 0, pest_loss = 0.35, soil_n = 30, ky = 0.7065, y_scale = 1.115, heat_coef = 0.002337, k_dry = 0.3923, r0 = 1e+06 (`params/fitted_params.yaml`).

### Simulator

| Mechanism | Setting |
|---|---|
| groundwater | not in this version of the simulator |
| waterlogging | not in this version of the simulator |
| pest_resistance | not in this version of the simulator |
| storage | not in this version of the simulator |

Costs (INR): nitrogen_per_kg = 12.9, irrigation_per_mm = 2.86, pesticide_per_level = 1183.0, fixed_per_ha = 29450.0; penalties: excess_nitrogen = 5.0, pesticide_use = 100.0, irrigation_per_mm = 5.0. Actions: N [0, 20, 40] kg/ha, water [0, 37.5, 75] mm, pesticide [0, 1, 2].

### Agent and training

- PPO (own implementation, `code/src/farmrl/agents/ppo.py`), 200,000 steps per seed, training seeds [42, 43, 44].
- Hyperparameters: lr = 0.0003, gamma = 0.99, gae_lambda = 0.95, clip_eps = 0.2, n_steps = 2048, batch_size = 64, epochs = 10, hidden = 64, ent_coef = 0.01, vf_coef = 0.5.

### Baselines and evaluation

- Policies: ppo, no_input, random, recommended, threshold; auxiliary (not ranked): ppo_untrained.
- Each policy replayed on every season with 30 pest-growth seeds shared across policies.
- Also computed: per-district evaluation, sensitivity analysis.

## Provenance

- Code: `code/` — code snapshot (reconstructed) (details in `code/SNAPSHOT.md`).
- Commit recorded at run time: `7f8dadbfd0`.
- Software: {'python': '3.12.7', 'torch': '2.5.0+cpu', 'numpy': '1.26.4', 'pandas': '3.0.5'}.

## Files

- `figures/` — report figures (PNG 300 dpi + PDF) (20 files)
- `animations/` — season animations (1 files)
- `maps/` — district maps (3 files)
- `params/` — parameter copies (3 files)
- `code/` — code snapshot (37 files)
- tables and model: `config_snapshot.yaml`, `district_summary.csv`, `episodes.csv`, `fitted_params.yaml`, `metadata.json`, `model.pt`, `policy_grids.parquet`, `prices.csv`, `rollouts.parquet`, `sensitivity.csv`, `state_medians.csv`, `summary.csv`, `train_history.csv`

_Card generated 2026-10-15 22:39 by `farmrl.runcard`; regenerate with `python scripts/make_runcard.py --run v2`._
