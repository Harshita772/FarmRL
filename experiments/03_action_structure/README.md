# Study 3 — Does Action Structure Affect PPO Performance?

## Research Question

Does changing how PPO represents its actions improve performance? Specifically, we investigate whether combining nitrogen and irrigation decisions into a single policy head, or using a finer set of irrigation options, helps the agent make better farming decisions.

## Experimental Setup

We used the best-performing configuration from Study 2 as the starting point: improved PPO with `ent_coef = 0.003` and 10 training epochs. All other settings were kept unchanged unless required by the action-space modification.

We tested two changes:

- **Joint nitrogen–water head (`joint_nw`):** Instead of predicting nitrogen and irrigation actions independently, the agent uses a single softmax distribution over all nine nitrogen–water combinations. Pesticide and storage decisions retain their separate policy heads.
- **Finer irrigation grid (`fine_water`):** We expanded the available irrigation levels from `{0, 37.5, 75}` mm to `{0, 25, 50, 75}` mm, giving the agent more flexibility in choosing water quantities.

For the finer-grid experiment, the rule-based baselines were retuned to use the new irrigation options. Their irrigation doses were mapped to the new grid, with the previous 37.5 mm option replaced by 50 mm. The oracle was also recomputed using the finer grid.

Each configuration was trained with five random seeds and evaluated on the same held-out seasons from 2017 to 2023, using 30 pest-growth replicates per season. Profit is reported in INR per hectare before environmental penalties, and the ± values represent the standard deviation across training seeds.

The performance gap is calculated as the agent's profit minus the corresponding reference policy's profit. Negative values indicate that the agent earns less than the reference. The Shapley values break down the agent's gain over `threshold_tuned` into nitrogen, water, pesticide, and sale decisions.

## Results

| Configuration | Seeds | Profit (INR/ha) | Gap to `threshold_tuned` | Gap to oracle | Oracle performance |
|---|---:|---:|---:|---:|---:|
| Improved PPO, factored heads | 5 | 86,200 ± 542 | −1,032 ± 542 | −3,454 ± 542 | 96.1% |
| Improved PPO, joint nitrogen–water head | 5 | 86,250 ± 194 | −982 ± 194 | −3,403 ± 194 | 96.2% |
| Improved PPO, finer water grid (0/25/50/75 mm) | 5 | 86,587 ± 286 | +196 ± 286 | −3,014 ± 286 | 96.6% |

### Reference Policy Performance

The reference results differ between the standard and finer irrigation grids because the available irrigation actions change.

| Policy | Standard grid | Finer water grid |
|---|---:|---:|
| `threshold_tuned` | 87,232 | 86,390 |
| `recommended_tuned` | 81,072 | 83,238 |
| `recommended` | 74,649 | 74,649 |
| `random` | 55,336 | 55,846 |
| `oracle` | 89,654 | 89,601 |

All values are in INR/ha. The standard-grid oracle reaches 89,654 INR/ha, while the oracle on the finer grid reaches 89,601 INR/ha.

## Key Findings

The joint nitrogen–water head produced an average profit of 86,250 INR/ha, only 50 INR/ha higher than the factored-head PPO agent from Study 2. This small difference does not provide convincing evidence that combining the two action heads improves performance.

The finer irrigation grid achieved the highest average profit of the three configurations, at 86,587 INR/ha. It also exceeded its corresponding tuned threshold baseline by 196 INR/ha, with four of the five training seeds outperforming that baseline.

However, the finer-grid result needs to be interpreted carefully. The threshold baseline itself earns less under the new grid, and the comparison with the standard-grid results is therefore not directly equivalent.

## Analysis

### 1. Combining nitrogen and irrigation decisions makes little difference

The joint-head agent achieved 86,250 ± 194 INR/ha, compared with 86,200 ± 542 INR/ha for the original factored-head agent.

The improvement is just 50 INR/ha, which is small relative to the variation across training seeds. Although the joint-head configuration also showed a lower standard deviation, five seeds are not enough to establish that it is consistently more stable.

These results suggest that predicting nitrogen and irrigation actions independently is not the main factor limiting PPO's performance. Combining them into a single action distribution did not produce a meaningful improvement in average profit.

### 2. A finer irrigation grid shows a promising but inconclusive improvement

The finer-grid agent achieved 86,587 ± 286 INR/ha, an increase of 387 INR/ha over the best factored-head configuration from Study 2.

It also outperformed its own tuned threshold baseline by 196 INR/ha, with four out of five training seeds ahead of the rule.

However, the comparison is complicated by the change in the action space. Under the original grid, the threshold policy could choose 37.5 mm of irrigation. Under the finer grid, this option was replaced by 50 mm, and the rule could no longer select 25 mm as an intermediate dose below that level.

As a result, the tuned threshold policy's profit fell from 87,232 to 86,390 INR/ha. The apparent PPO advantage on the finer grid is therefore partly influenced by a change in the baseline's available actions.

In absolute terms, the finer-grid agent improves on the original factored-head agent by 387 INR/ha, but it still earns 645 INR/ha less than the standard-grid threshold policy.

The finer grid may give PPO more flexibility to adjust irrigation, but the current results do not establish that this change alone is responsible for the improvement.

### 3. The remaining gap to the oracle is still substantial

Even with the joint action head or finer irrigation grid, PPO achieves only 96.2% and 96.6% of the corresponding oracle profits, respectively.

The finer-grid configuration comes closest to the oracle in absolute terms, falling short by 3,014 INR/ha. This suggests that modifying the action representation can help to some extent, but it does not resolve the broader performance gap observed in Studies 1 and 2.

## Conclusion

This study investigated whether changing PPO's action structure could improve farming decisions. Combining nitrogen and irrigation into a joint policy head produced almost no change in average profit, suggesting that independent action heads are not the main bottleneck.

Using a finer irrigation grid produced a more promising result, with the highest average profit among the three configurations and better performance than its corresponding tuned threshold baseline in four out of five seeds. However, the change in irrigation options also reduced the baseline's profit, making the comparison with the original action space less straightforward.

Overall, the results provide limited evidence that a finer irrigation grid helps PPO, but they do not establish a clear causal improvement. A stronger follow-up experiment would compare both irrigation grids under carefully matched evaluation conditions, retune the baselines consistently, and use more training seeds to determine whether the observed gain is robust.

## Experiment Runs

- [Best improved PPO configuration from Study 2](../02_improved_ppo/runs/improved_ent0p003_ep10/README.md)
- [Joint nitrogen–water action head](runs/joint_nw/README.md)
- [Finer irrigation grid](runs/fine_water/README.md)