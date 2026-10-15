# Run `v1`

**Punjab**, weather of Ludhiana; yield model: single-region fit (nitrogen + rain); simulator mechanisms: none beyond the base water/N/pest model; created 2026-10-10 17:09:40.

## Results (held-out seasons)

| Policy | Profit (INR/ha) | Std | Yield (q/ha) | N (kg/ha) | Irrigation (mm) | Pesticide (doses) |
|---|---|---|---|---|---|---|
| PPO | 84,457 | 3,521 | 50.0 | 173 | 225 | 3.7 |
| Threshold rules | 74,400 | 3,486 | 47.3 | 140 | 236 | 7.1 |
| Recommended | 69,518 | 8,373 | 42.3 | 120 | 300 | 1.0 |
| Random | 31,853 | 32,384 | 32.9 | 244 | 437 | 11.8 |
| No inputs | 3,208 | 11,915 | 13.4 | 0 | 0 | 0.0 |

## Methods

### Data

- Raw inputs present: `cacp_rabi_price_policy_latest.pdf`, `icrisat_crops.csv`, `icrisat_fertiliser.csv`, `rbi_msp_foodgrains_2014.html`, `rbi_msp_foodgrains_latest.html`, `weather_ambala.csv`, `weather_gurgaon.csv`, `weather_hisar.csv`, `weather_jind.csv`, `weather_karnal.csv`, `weather_ludhiana.csv`, `weather_narnaul.csv`, `weather_rohtak.csv` (35 files listed with size and date in `metadata.json`).
- Seasons: 32 training (1985–2016), 7 held out (2017–2023); sowing years, 12 fortnights from 1 November.
- Prices: season price index from `params/prices.csv` around a base of 2425.0 INR/quintal.

### Yield model

- Form: single-region fit (nitrogen + rain); 70 records.
- In-sample R² 0.237, RMSE 3.66 q/ha.
- Identified by the data: y_max, n_coef, w_opt, w_excess; reset to neutral/prior: pest_loss.
- Pesticide loss: 0.350 (prior (no pesticide data)).
- Parameters used: y_max = 46.76, n_coef = 0.02583, w_opt = 332.1, w_excess = 0.04253, pest_loss = 0.35, soil_n = 30, ky = 1.15, y_scale = 1.125 (`params/fitted_params.yaml`).

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
- `tensorboard/` — TensorBoard logs (3 files)
- tables and model: `config_snapshot.yaml`, `district_summary.csv`, `episodes.csv`, `fitted_params.yaml`, `metadata.json`, `model.pt`, `policy_grids.parquet`, `prices.csv`, `rollouts.parquet`, `sensitivity.csv`, `state_medians.csv`, `summary.csv`, `train_history.csv`

_Card generated 2026-10-15 22:39 by `farmrl.runcard`; regenerate with `python scripts/make_runcard.py --run v1`._
