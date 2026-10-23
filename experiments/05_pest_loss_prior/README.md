# Study 5 — Does PPO Learn to Spray When Pest Damage Increases?

## Research Question

When pest damage becomes more costly, does PPO learn to use pesticides more frequently, and can it outperform the tuned threshold-based policy?

## Experimental Setup

In the previous studies, the best-performing PPO agent rarely used pesticides under the calibrated pest-loss setting of `pest_loss = 0.115`. In this study, we increased the pest-loss parameter to `0.35` to investigate whether the agent would adapt its spraying decisions when pest damage became more expensive.

We retrained the best configuration from Study 2, using `ent_coef = 0.003` and 10 training epochs, with the higher pest-loss setting. We also recalibrated `y_scale` so that the recommended planting calendar continued to produce the observed average yield. The rule-based baselines were retuned, and the hindsight oracle was recomputed for the modified environment. The parameter `ky` was left unchanged.

The pesticide action head was retained, allowing PPO to decide when spraying would be worthwhile under the new conditions. We measured pesticide use as the average number of sprays per season.

Both configurations were trained with five random seeds and evaluated on the same held-out seasons from 2017 to 2023, using 30 pest-growth replicates per season. Profit is reported in INR per hectare before environmental penalties, and ± represents the standard deviation across training seeds.

The performance gap is calculated as the agent's profit minus the corresponding reference policy's profit. The Shapley values break down the agent's gain over `threshold_tuned` into nitrogen, water, pesticide, and sale decisions.

## Results

| Configuration | Seeds | Profit (INR/ha) | Gap to `threshold_tuned` | Gap to oracle | Oracle performance | N kg · water mm · sprays |
|---|---:|---:|---:|---:|---:|---|
| Improved PPO, `pest_loss = 0.115` | 5 | 86,200 ± 542 | −1,032 ± 542 | −3,454 ± 542 | 96.1% | 213 · 235 · 0.00 |
| Improved PPO, `pest_loss = 0.35` | 5 | 86,410 ± 492 | +2,664 ± 492 | −4,151 ± 492 | 95.4% | 208 · 227 · 3.59 |

### Reference Policy Performance

| Policy | `pest_loss = 0.115` | `pest_loss = 0.35` |
|---|---:|---:|
| `threshold_tuned` | 87,232 | 83,746 |
| `recommended_tuned` | 81,072 | 80,960 |
| `recommended` | 74,649 | 71,968 |
| `random` | 55,336 | 57,619 |
| `oracle` | 89,654 | 90,561 |
| Threshold policy as % of oracle | 97.3% | 92.5% |

All profit values are in INR/ha.

## Key Findings

The most important result is that PPO begins using pesticides when the pest-loss parameter increases. Under the calibrated setting of `pest_loss = 0.115`, the agent averages zero sprays per season. At `pest_loss = 0.35`, it averages 3.59 sprays per season.

The higher-loss configuration achieves an average profit of 86,410 INR/ha and outperforms its own tuned threshold baseline by 2,664 INR/ha. All five training seeds outperform that baseline.

This contrasts with the earlier studies, where PPO consistently fell short of the tuned threshold policy. The result suggests that PPO can adapt its pesticide decisions to a change in the cost of pest damage, particularly when the existing threshold rule does not respond as effectively.

## Analysis

### 1. PPO adapts its pesticide use when pest damage becomes more costly

Under the calibrated pest-loss setting, the best PPO agent does not use pesticides. When `pest_loss` increases to 0.35, the agent learns a different strategy, applying an average of 3.59 sprays per season.

This change is consistent with the economic trade-off between pesticide costs and the losses caused by pests. When pest damage becomes more expensive, spraying can become worthwhile even after accounting for the cost of pesticide application.

The profit breakdown supports this interpretation. The pesticide component contributes +4,057 INR/ha to PPO's advantage over `threshold_tuned` in the higher-loss environment. By comparison, the contributions from nitrogen, water, and sale decisions are −227, −1,266, and +100 INR/ha, respectively.

The results therefore point to pesticide management as the main source of PPO's advantage in this experiment.

### 2. PPO outperforms the tuned threshold policy in the higher-loss environment

With `pest_loss = 0.35`, PPO achieves 86,410 ± 492 INR/ha, while the tuned threshold policy earns 83,746 INR/ha. This gives PPO an average advantage of 2,664 INR/ha, with all five training seeds ahead of the baseline.

The difference is particularly interesting because the threshold policy was retuned for the modified environment. PPO's improvement therefore cannot be explained simply by comparing it with an unchanged rule from the previous experiment.

Instead, the result suggests that PPO learns a pesticide-use strategy that is more effective than the particular threshold rule found by the tuning procedure.

However, this does not establish that PPO is always better than threshold-based control when pest damage is high. The result applies to the tested environment, reward settings, and tuning procedure.

### 3. The oracle remains a useful benchmark

The hindsight oracle achieves 90,561 INR/ha in the higher-loss environment, compared with 89,654 INR/ha under the calibrated setting.

PPO reaches 95.4% of the oracle's profit at `pest_loss = 0.35`, while the tuned threshold policy reaches 92.5%.

Although PPO outperforms the rule-based baseline, it still falls short of the oracle by 4,151 INR/ha. There remains room for improvement, even when the agent learns to use pesticides more effectively.

The oracle has access to future seasonal weather and prices, giving it an information advantage over PPO. Its profit should therefore be interpreted as a hindsight benchmark rather than a directly achievable target for a policy that must make decisions as the season unfolds.

### 4. The reward recalibration matters when interpreting the comparison

The `y_scale` parameter was recalibrated for the higher pest-loss setting to preserve the recommended calendar's average observed yield, while `ky` remained unchanged.

This is important because the experiment changes the economic importance of pest damage while maintaining the intended yield calibration. Nevertheless, the results compare two separately configured environments, each with its own tuned baselines and oracle.

The increase in PPO's profit from 86,200 to 86,410 INR/ha is therefore not, by itself, evidence that higher pest damage increases profitability. The more informative result is that PPO changes its spraying behaviour and gains an advantage over the baseline evaluated under the same higher-loss conditions.

## Conclusion

This study provides evidence that PPO can adapt its pesticide-use decisions when pest damage becomes more costly. Under the calibrated pest-loss setting, the agent does not spray. When `pest_loss` increases to 0.35, it learns to apply approximately 3.59 sprays per season.

In the higher-loss environment, PPO earns 86,410 INR/ha, outperforming the retuned threshold policy by 2,664 INR/ha across all five training seeds. The Shapley analysis attributes most of this advantage to pesticide management.

These findings suggest that PPO can learn a more effective spraying strategy than the tested threshold-based policy when pest damage is sufficiently costly. However, it still achieves only 95.4% of the oracle's profit, leaving room for further improvement.

A useful follow-up would be to evaluate PPO across several intermediate pest-loss values. This would help determine whether pesticide use increases gradually as pest damage becomes more expensive, identify when spraying becomes economically worthwhile, and establish whether PPO's advantage over the tuned threshold policy persists across a wider range of conditions.

## Experiment Runs

- [Best improved PPO configuration from Study 2](../02_improved_ppo/runs/improved_ent0p003_ep10/README.md)
- [Improved PPO with `pest_loss = 0.35`](runs/improved_best_pest035/README.md)