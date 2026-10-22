# Study 2 — Improving PPO and Testing Hyperparameters

## Research Question

Can we improve PPO's performance by using observation and return normalisation, linear learning-rate decay, orthogonal weight initialisation, a discount factor of 1.0, a masked storage head, and 1 million training steps? We also investigate how sensitive the results are to the entropy coefficient (`ent_coef`) and the number of training epochs.

## Experimental Setup

We introduced an improved PPO configuration that includes observation normalisation, return normalisation, linear learning-rate decay, orthogonal initialisation, a masked storage head, and `gamma = 1.0`. The total training budget was increased to 1 million timesteps.

We then conducted a small hyperparameter sweep, testing three entropy coefficients (`0`, `0.003`, and `0.01`) and two epoch settings (`4` and `10`). All other settings were kept the same as in PPO v4.

Each configuration was trained using five different random seeds and evaluated on the same held-out seasons from 2017 to 2023, with 30 pest-growth replicates per season. The tuned rule-based baselines from Study 1 were reused to ensure a fair comparison.

Profit is reported in INR per hectare before environmental penalties. The reported ± values represent the standard deviation across training seeds. The performance gap is calculated as the agent's profit minus the reference policy's profit, so a negative value indicates that the agent performs worse than the reference.

The Shapley values break down the agent's performance difference relative to `threshold_tuned` into four components: nitrogen, water, pesticide, and sale decisions.

## Results

| Configuration | Seeds | Profit (INR/ha) | Gap to `threshold_tuned` | Gap to oracle | Oracle performance |
|---|---:|---:|---:|---:|---:|
| PPO v4 (200k steps) | 5 | 86,135 ± 179 | −1,096 ± 179 | −3,518 ± 179 | 96.1% |
| Improved PPO, `ent_coef=0.01`, epochs=10 | 5 | 85,864 ± 412 | −1,368 ± 412 | −3,789 ± 412 | 95.8% |
| Improved PPO, `ent_coef=0.003`, epochs=10 | 5 | 86,200 ± 542 | −1,032 ± 542 | −3,454 ± 542 | 96.1% |
| Improved PPO, `ent_coef=0`, epochs=10 | 5 | 86,104 ± 554 | −1,128 ± 554 | −3,550 ± 554 | 96.0% |
| Improved PPO, `ent_coef=0.01`, epochs=4 | 5 | 85,820 ± 358 | −1,412 ± 358 | −3,834 ± 358 | 95.7% |
| Improved PPO, `ent_coef=0.003`, epochs=4 | 5 | 85,975 ± 386 | −1,257 ± 386 | −3,679 ± 386 | 95.9% |
| Improved PPO, `ent_coef=0`, epochs=4 | 5 | 86,144 ± 271 | −1,088 ± 271 | −3,509 ± 271 | 96.1% |

### Reference Policies

The reference policies produced the same results across all configurations:

| Policy | Profit (INR/ha) |
|---|---:|
| `threshold_tuned` | 87,232 |
| `recommended_tuned` | 81,072 |
| `recommended` | 74,649 |
| `random` | 55,336 |
| `oracle` | 89,654 |

The tuned threshold policy achieves 97.3% of the oracle's profit, making it a strong benchmark for evaluating the learned agents.

## Key Findings

The best-performing configuration was improved PPO with `ent_coef = 0.003` and 10 epochs. It achieved an average profit of 86,200 INR/ha, corresponding to 96.1% of the oracle's profit.

However, it still fell short of the `threshold_tuned` baseline, which achieved 87,232 INR/ha. None of the tested PPO configurations outperformed this baseline on average.

The original PPO v4 agent achieved 86,135 INR/ha after 200,000 training steps. Despite increasing the training budget to 1 million steps and introducing several standard PPO improvements, the new configurations did not produce a clear performance gain.

## Analysis

### 1. More training and improved PPO settings did not close the gap

The six improved configurations achieved average profits between 85,820 and 86,200 INR/ha, compared with 86,135 INR/ha for the original PPO agent.

The best configuration improved the mean profit by just 65 INR/ha. This difference is small compared with the variation across training seeds, so the results do not provide convincing evidence that the improved setup performs better than the original agent.

More importantly, all six improved configurations remained below `threshold_tuned`, with average gaps ranging from 1,032 to 1,412 INR/ha. According to the reported seed-level results, none of the five seeds for any configuration outperformed the tuned threshold policy.

These findings suggest that simply increasing the training budget or adding standard PPO optimisation techniques is not enough to close the performance gap.

### 2. Water management appears to be the main source of the gap

The Shapley analysis helps identify which decisions contribute most to the difference between PPO and the tuned threshold policy.

Across all configurations, water management has the largest negative contribution, ranging from −1,005 to −1,456 INR/ha. In comparison, the contributions from nitrogen, pesticide, and sale decisions are much smaller.

This pattern suggests that the agent's main weakness lies in its irrigation decisions rather than in the other three components of the policy.

A useful next step would be to investigate the agent's water-use behaviour in more detail, including irrigation timing, total water consumption, and how its decisions respond to changing weather and crop conditions.

### 3. The hyperparameter sweep shows no clear winner

Changing the entropy coefficient and the number of epochs led to relatively small differences in average profit.

Among the tested configurations, `ent_coef = 0.003` with 10 epochs achieved the highest mean profit, while `ent_coef = 0.01` with 4 epochs achieved the lowest. However, the differences are small relative to the observed variation across seeds.

Consequently, these experiments do not establish that any particular entropy coefficient or epoch setting is consistently better. More seeds or a more targeted hyperparameter search would be needed to draw a stronger conclusion.

## Conclusion

The second study tested whether a more carefully configured PPO agent, trained for five times longer than the original agent, could outperform the existing baseline. The results show that these changes did not lead to a meaningful improvement in average profit.

The best configuration achieved 86,200 INR/ha, or 96.1% of the oracle's profit, but still remained 1,032 INR/ha below the tuned threshold policy. The Shapley analysis consistently identifies water management as the largest contributor to this gap.

Overall, the results suggest that the remaining performance gap is not easily resolved through the PPO settings tested here. Rather than continuing to make broad optimisation changes, future experiments should focus on understanding and improving irrigation decisions. The current results point towards this direction, although they do not rule out other sources of error or limitations in the learning setup.

The hyperparameter sweep also provides no strong evidence that a particular entropy coefficient or number of epochs is superior.

## Experiment Runs

- [Baseline PPO v4 and oracle comparison](../01_oracle_bound/runs/baseline_v4_oracle/README.md)
- [Improved PPO — `ent_coef=0.01`, epochs=10](runs/improved_ent0p01_ep10/README.md)
- [Improved PPO — `ent_coef=0.003`, epochs=10](runs/improved_ent0p003_ep10/README.md)
- [Improved PPO — `ent_coef=0`, epochs=10](runs/improved_ent0_ep10/README.md)
- [Improved PPO — `ent_coef=0.01`, epochs=4](runs/improved_ent0p01_ep4/README.md)
- [Improved PPO — `ent_coef=0.003`, epochs=4](runs/improved_ent0p003_ep4/README.md)
- [Improved PPO — `ent_coef=0`, epochs=4](runs/improved_ent0_ep4/README.md)