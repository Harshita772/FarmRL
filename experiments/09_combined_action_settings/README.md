# Study 9 — Combining the Best Action Settings

## Research Question

Can combining a finer irrigation action grid with a joint nitrogen–water policy head help PPO outperform a tuned rule-based policy?

Previous experiments investigated these two changes separately. Study 3 found that a finer water grid improved PPO's performance, while using a joint nitrogen–water head produced little measurable improvement on the standard grid. This study combines the two ideas to determine whether they work better together and whether increasing the number of available irrigation levels provides any additional benefit.

## Experimental Setup

All experiments use the improved PPO training configuration from Study 2, with `ent_coef = 0.003`, 10 epochs, 1 million training timesteps, and five random seeds.

We evaluate three action configurations:

- **`fine4_joint`:** A four-level water grid of `{0, 25, 50, 75}` mm, with a joint nitrogen–water action head.
- **`fine6_joint`:** A six-level water grid of `{0, 15, 30, 45, 60, 75}` mm, with a joint nitrogen–water action head.
- **`fine6_factored`:** The same six-level water grid, but with separate nitrogen and water heads. This configuration helps isolate the effect of the joint head.

For comparison, we also include the original PPO v4, the improved PPO without a forecast, and the action configurations evaluated in Study 3.

The threshold-based policy is retuned separately for each water grid. Its irrigation doses are mapped to the nearest available level, with ties rounded upward. Consequently, the threshold policy's performance depends on the grid being used. To avoid drawing misleading conclusions from comparisons against a grid-specific baseline, we also compare every agent against the standard-grid `threshold_tuned` policy from Study 1, which earns 87,232 INR/ha.

Every configuration is evaluated on the same held-out seasons from 2017 to 2023, with 30 pest-growth replicates per season. Profit is reported in INR per hectare before environmental penalties, and ± represents the standard deviation across training seeds.

The reported performance gap is the agent's profit minus that of the corresponding reference policy. The Shapley values break down the agent's gain over `threshold_tuned` into contributions from nitrogen, water, pesticide, and sale decisions.

## Results

| Configuration | Seeds | Profit (INR/ha) | Gap to grid-specific rule | Gap to oracle | Oracle performance | N kg · water mm · sprays |
|---|---:|---:|---:|---:|---:|---|
| PPO v4, original settings | 5 | 86,135 ± 179 | −1,096 ± 179 | −3,518 ± 179 | 96.1% | 199 · 232 · 0.00 |
| Improved PPO, no forecast | 5 | 86,200 ± 542 | −1,032 ± 542 | −3,454 ± 542 | 96.1% | 213 · 235 · 0.00 |
| Four-level grid, factored heads (Study 3) | 5 | 86,587 ± 286 | +196 ± 286 | −3,014 ± 286 | 96.6% | 198 · 227 · 0.00 |
| Standard grid, joint head (Study 3) | 5 | 86,250 ± 194 | −982 ± 194 | −3,403 ± 194 | 96.2% | 198 · 233 · 0.00 |
| `fine4_joint` | 5 | 86,821 ± 315 | +431 ± 315 | −2,780 ± 315 | 96.9% | 207 · 229 · 0.00 |
| `fine6_joint` | 5 | 86,548 ± 273 | −158 ± 273 | −3,155 ± 273 | 96.5% | 194 · 229 · 0.00 |
| `fine6_factored` | 5 | 86,708 ± 69 | +1 ± 69 | −2,996 ± 69 | 96.7% | 201 · 231 · 0.00 |

### Reference Policy Performance

| Configuration | `threshold_tuned` | `recommended_tuned` | `recommended` | `random` | `oracle` | Threshold as % of oracle |
|---|---:|---:|---:|---:|---:|---:|
| PPO v4, original settings | 87,232 | 81,072 | 74,649 | 55,336 | 89,654 | 97.3% |
| Improved PPO, no forecast | 87,232 | 81,072 | 74,649 | 55,336 | 89,654 | 97.3% |
| Four-level grid, factored heads | 86,390 | 83,238 | 74,649 | 55,846 | 89,601 | 96.4% |
| Standard grid, joint head | 87,232 | 81,072 | 74,649 | 55,336 | 89,654 | 97.3% |
| `fine4_joint` | 86,390 | 83,238 | 74,649 | 55,846 | 89,601 | 96.4% |
| `fine6_joint` | 86,706 | 84,110 | 74,649 | 56,546 | 89,703 | 96.7% |
| `fine6_factored` | 86,706 | 84,110 | 74,649 | 56,546 | 89,703 | 96.7% |

All profit values are in INR/ha.

## Key Findings

The best result comes from `fine4_joint`, which combines the four-level water grid with a joint nitrogen–water head. It achieves an average profit of 86,821 INR/ha, or 96.9% of the oracle. It outperforms its grid-specific tuned threshold policy by 431 INR/ha, with all five seeds ahead of that baseline.

However, the agent still falls 410 INR/ha short of the standard-grid threshold policy, which earns 87,232 INR/ha. This distinction matters because the grid-specific rule is constrained by the same reduced set of irrigation levels as the agent.

The six-level grid does not improve performance further. `fine6_joint` earns 86,548 INR/ha, while `fine6_factored` earns 86,708 INR/ha. Both remain below the standard-grid threshold baseline.

Overall, the results suggest that combining a finer action grid with a joint nitrogen–water head can improve PPO relative to some grid-specific baselines, but the improvement is not sufficient to establish that PPO outperforms the standard rule.

## Analysis

### 1. Combining the four-level grid and joint head gives the best average result

The `fine4_joint` configuration achieves 86,821 INR/ha, the highest average profit among the PPO configurations in this study.

Compared with the improved PPO without a forecast, its average profit is higher by 621 INR/ha. It also outperforms the four-level factored-head agent from Study 3, which earns 86,587 INR/ha, by 234 INR/ha.

These results are consistent with the possibility that combining the two action-design choices is useful. However, the difference between the joint and factored configurations is modest relative to their training variability, and a controlled paired-seed analysis would be needed to establish whether the gain is statistically reliable.

### 2. A finer grid does not automatically lead to better performance

Increasing the water grid from four levels to six does not improve the joint-head agent's results. The six-level joint configuration earns 86,548 INR/ha, which is 273 INR/ha lower than the four-level joint configuration.

The six-level factored configuration performs somewhat better, at 86,708 INR/ha, but still does not exceed `fine4_joint`.

This suggests that adding more available actions does not necessarily make policy learning easier. A larger action space may provide greater flexibility, but it can also make the policy harder to learn. These results alone do not identify which mechanism explains the observed differences.

### 3. The effect of the joint head is not consistent across grids

The joint nitrogen–water head performs differently depending on the water grid.

On the four-level grid, the joint-head configuration earns 86,821 INR/ha, compared with 86,587 INR/ha for the factored-head configuration. The joint-head agent is ahead by 234 INR/ha.

On the six-level grid, the pattern reverses. The factored-head configuration earns 86,708 INR/ha, compared with 86,548 INR/ha for the joint-head configuration, a difference of 160 INR/ha in favour of the factored version.

The joint head therefore does not provide a consistent advantage across both grids. Its usefulness may depend on how the action space is structured, but further experiments would be needed to confirm that interpretation.

### 4. Grid-specific baselines can make the results look more favourable

The threshold policy earns 86,390 INR/ha on the four-level grid and 86,706 INR/ha on the six-level grid. Both are lower than the 87,232 INR/ha achieved by the standard-grid threshold policy.

Against the grid-specific rule, `fine4_joint` has a positive gap of 431 INR/ha. Against the standard-grid rule, its gap is negative at −410 INR/ha.

Similarly, `fine6_factored` is almost level with its grid-specific rule, earning 86,708 versus 86,706 INR/ha. Yet it remains 524 INR/ha below the standard-grid baseline.

This comparison shows why the choice of reference policy matters. Restricting the threshold policy to a coarser action grid changes its achievable performance, so beating that version does not necessarily mean the agent has beaten the stronger standard-grid rule.

### 5. The agents remain below the hindsight oracle

The best configuration, `fine4_joint`, achieves 96.9% of the oracle's profit, leaving a gap of 2,780 INR/ha.

The other configurations achieve between 96.1% and 96.7% of their respective oracle values, with the exception of `fine4_joint` at 96.9%.

Thus, the action-space changes do not close the oracle gap substantially. Although the four-level joint configuration improves the average result, there is still room to improve the agent's resource-allocation decisions.

All configurations also average zero pesticide sprays per season. Their differences in profit therefore arise primarily from nitrogen, irrigation, and sale decisions rather than pesticide use.

## Conclusion

This study combines a finer irrigation action grid with a joint nitrogen–water policy head to test whether the two changes improve PPO performance together.

The four-level joint configuration, `fine4_joint`, achieves the highest average profit at 86,821 INR/ha and outperforms its grid-specific threshold baseline by 431 INR/ha. However, it remains 410 INR/ha below the stronger standard-grid threshold policy.

The six-level grid does not provide an additional benefit, and the joint head does not consistently outperform factored heads across different grids. These findings suggest that action-space design can influence PPO performance, but simply increasing the number of available actions or combining design changes does not guarantee better results.

The main conclusion is that **the four-level joint configuration is the strongest PPO variant tested here, but the evidence does not show that it beats the standard-grid rule**. A useful next step would be to compare these configurations using paired seeds and examine the learned nitrogen–water action combinations to understand why the four-level joint configuration performs best.

## Experiment Runs

- [PPO v4 oracle baseline](../01_oracle_bound/runs/baseline_v4_oracle/README.md)
- [Improved PPO](../02_improved_ppo/runs/improved_ent0p003_ep10/README.md)
- [Four-level water grid, factored heads](../03_action_structure/runs/fine_water/README.md)
- [Standard grid, joint nitrogen–water head](../03_action_structure/runs/joint_nw/README.md)
- [`fine4_joint`](runs/fine4_joint/README.md)
- [`fine6_joint`](runs/fine6_joint/README.md)
- [`fine6_factored`](runs/fine6_factored/README.md)