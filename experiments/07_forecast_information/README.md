# Study 7 — Does a Rain Forecast Help PPO Close the Oracle Gap?

## Research Question

Can giving PPO a forecast of upcoming rainfall improve its decisions and help it get closer to the hindsight oracle? More specifically, how does the quality of the forecast affect the agent's profitability?

## Experimental Setup

In this study, we investigate whether information about future rainfall helps PPO make better irrigation decisions. The observation is extended with an additional value representing the rainfall expected during the upcoming fortnight. To simulate forecasts of different quality, we add Gaussian noise with standard deviations of 0, 10, and 25 mm. Negative forecast values are clipped to zero, and one noisy forecast is sampled per season.

A noise level of 0 mm represents a perfect forecast, while larger standard deviations represent increasingly uncertain predictions. We compare these configurations with the improved PPO agent from Study 2, which uses the same training setup but receives no rainfall forecast.

All forecast-enabled agents use the improved PPO configuration, with `ent_coef = 0.003`, 10 epochs, and 1 million training timesteps. Each configuration is trained using five random seeds.

The rule-based policies do not use forecast information, so we reuse the tuned reference policies from Study 1. The hindsight oracle is recomputed for the forecast experiments.

Every configuration is evaluated on the same held-out seasons from 2017 to 2023, with 30 pest-growth replicates per season. Profit is reported in INR per hectare before environmental penalties, and ± represents the standard deviation across training seeds.

The performance gap is calculated as the agent's profit minus the corresponding reference policy's profit. A negative gap indicates that the agent performs worse than the reference, while a positive gap indicates that it performs better. The Shapley values break down the agent's gain over `threshold_tuned` into contributions from nitrogen, water, pesticide, and sale decisions.

## Results

| Configuration | Seeds | Profit (INR/ha) | Gap to `threshold_tuned` | Gap to oracle | Oracle performance | N kg · water mm · sprays |
|---|---:|---:|---:|---:|---:|---|
| Improved PPO, no forecast | 5 | 86,200 ± 542 | −1,032 ± 542 | −3,454 ± 542 | 96.1% | 213 · 235 · 0.00 |
| Forecast noise 0 mm (perfect) | 5 | 88,689 ± 295 | +1,304 ± 295 | −787 ± 295 | 99.1% | 200 · 206 · 0.00 |
| Forecast noise 10 mm | 5 | 87,972 ± 412 | +586 ± 412 | −1,504 ± 412 | 98.3% | 207 · 217 · 0.00 |
| Forecast noise 25 mm | 5 | 86,979 ± 258 | −406 ± 258 | −2,497 ± 258 | 97.2% | 200 · 223 · 0.00 |

### Reference Policy Performance

| Configuration | `threshold_tuned` | `recommended_tuned` | `recommended` | `random` | `oracle` | Threshold as % of oracle |
|---|---:|---:|---:|---:|---:|---:|
| Improved PPO, no forecast | 87,232 | 81,072 | 74,649 | 55,336 | 89,654 | 97.3% |
| Forecast noise 0 mm | 87,385 | 81,026 | 74,594 | 56,845 | 89,476 | 97.7% |
| Forecast noise 10 mm | 87,385 | 81,026 | 74,594 | 56,845 | 89,476 | 97.7% |
| Forecast noise 25 mm | 87,385 | 81,026 | 74,594 | 56,845 | 89,476 | 97.7% |

## Key Findings

The results show that access to rainfall forecasts can substantially improve PPO's performance, with the largest gains occurring when the forecast is accurate.

Without a forecast, PPO earns 86,200 INR/ha, which is 3,454 INR/ha below the hindsight oracle and 1,032 INR/ha below the tuned threshold policy.

With a perfect forecast, average profit rises to 88,689 INR/ha. This is a gain of 2,489 INR/ha over the no-forecast agent, and the oracle gap shrinks from 3,454 to just 787 INR/ha. PPO also outperforms the tuned threshold policy by 1,304 INR/ha.

As forecast uncertainty increases, performance declines. With 10 mm of forecast noise, PPO earns 87,972 INR/ha and remains ahead of the tuned threshold policy by 586 INR/ha. At 25 mm of noise, profit falls to 86,979 INR/ha, leaving PPO 406 INR/ha behind the threshold baseline.

All forecast-enabled configurations avoid pesticide use, averaging zero sprays per season. Their main differences are therefore reflected in nitrogen and irrigation decisions rather than pesticide management.

## Analysis

### 1. Accurate rainfall forecasts help PPO make better decisions

The strongest result comes from the perfect-forecast configuration. Its average profit is 88,689 INR/ha, compared with 86,200 INR/ha for the agent without forecast information.

The additional observation appears to help PPO coordinate its decisions with upcoming rainfall. Better information about future rain can make it easier to decide how much irrigation is necessary and avoid applying water when rainfall is expected to provide it.

The resource-use figures are consistent with this interpretation. The perfect-forecast agent uses an average of 206 mm of water per season, compared with 235 mm for the no-forecast agent. It also uses slightly less nitrogen, at 200 kg compared with 213 kg.

These results suggest that the benefit of the forecast is not simply greater resource use. Instead, the additional information may help the agent allocate resources more effectively.

### 2. The benefit decreases as forecast uncertainty increases

The results show a clear performance trend across the three forecast-noise settings:

- **0 mm noise:** 88,689 INR/ha, or 99.1% of the oracle.
- **10 mm noise:** 87,972 INR/ha, or 98.3% of the oracle.
- **25 mm noise:** 86,979 INR/ha, or 97.2% of the oracle.

The agent's profit decreases as the forecast becomes less reliable. With 10 mm of noise, PPO still outperforms the tuned threshold policy by 586 INR/ha. At 25 mm of noise, this advantage disappears, and PPO falls 406 INR/ha below the rule.

This pattern suggests that forecast quality matters: noisy future-rainfall information is less useful than an accurate forecast. However, the experiments test only three noise levels, so they do not establish the precise point at which forecast information stops being beneficial.

### 3. The forecast helps PPO close most of the oracle gap

The hindsight oracle represents a stronger benchmark because it can use future information when determining its actions. Without a forecast, PPO achieves 96.1% of the oracle's profit. With a perfect forecast, this rises to 99.1%.

The absolute gap falls from 3,454 INR/ha to 787 INR/ha, meaning the perfect-forecast agent closes approximately 77% of the original gap.

This is an important result: a relatively small addition to the observation can substantially improve PPO's performance. It suggests that part of the remaining gap is related to uncertainty about future conditions, rather than only to the agent's ability to learn a good policy.

Nevertheless, the oracle still earns more. A forecast provides information about upcoming rainfall, but it does not necessarily provide the complete future information available to a hindsight-based policy.

### 4. PPO's advantage is mainly associated with water management

The Shapley values help explain where the improvement comes from.

| Configuration | Nitrogen | Water | Pesticide | Sale |
|---|---:|---:|---:|---:|
| No forecast | −144 | −1,005 | 0 | +117 |
| Perfect forecast | −31 | +1,390 | 0 | −55 |
| 10 mm noise | −105 | +802 | 0 | −110 |
| 25 mm noise | −70 | −305 | 0 | −32 |

<text color="secondary" size="sm">Shapley contributions in INR/ha.</text>

For the no-forecast agent, water contributes −1,005 INR/ha relative to the tuned threshold policy. With a perfect forecast, the water contribution becomes +1,390 INR/ha. At 10 mm of noise, it remains positive at +802 INR/ha, but at 25 mm it becomes negative again at −305 INR/ha.

This change aligns with the overall performance trend and supports the interpretation that rainfall information primarily helps PPO improve its water-management decisions. Nitrogen and sale contributions are comparatively small, while the pesticide contribution remains zero across all configurations.

The Shapley decomposition indicates which action components account for the difference in performance; it does not, by itself, prove the exact causal mechanism behind the improvement.

## Conclusion

This study shows that rainfall forecasts can help PPO improve profitability and close much of the gap to the hindsight oracle.

Without a forecast, PPO achieves 96.1% of oracle profit and remains behind the tuned threshold policy. With a perfect forecast, it reaches 99.1% of oracle profit and outperforms the threshold policy by 1,304 INR/ha. The advantage becomes smaller with 10 mm of noise and disappears at 25 mm.

The results suggest that accurate rainfall information is particularly valuable for irrigation management. The change in the water Shapley contribution, together with lower average water use, supports the idea that PPO uses the additional information to make more effective irrigation decisions.

Overall, the findings indicate that **forecast quality can determine whether adaptive PPO control outperforms a well-tuned rule-based policy**. The next step is to test additional noise levels, evaluate multiple independently sampled forecasts per season, and investigate whether the improvement persists when forecasts are generated by a more realistic rainfall-prediction model rather than by adding Gaussian noise.

## Experiment Runs

- [Improved PPO without a forecast](../02_improved_ppo/runs/improved_ent0p003_ep10/README.md)
- [Perfect forecast — 0 mm noise](runs/forecast_sd0mm/README.md)
- [Forecast with 10 mm noise](runs/forecast_sd10mm/README.md)
- [Forecast with 25 mm noise](runs/forecast_sd25mm/README.md)