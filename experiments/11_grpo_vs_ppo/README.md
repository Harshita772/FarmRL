# Study 11 — GRPO vs PPO: Does the Critic Matter?

## Research Question

Does PPO need a learned value function to achieve strong performance, or can a critic-free method such as Group Relative Policy Optimization (GRPO) perform just as well by comparing returns across multiple episodes under similar conditions?

PPO uses a critic to estimate the expected future return and calculate advantages for policy updates. GRPO removes this value function and instead estimates relative advantages by comparing episodes within a group. This study investigates whether that simpler approach can match PPO in the agricultural control environment.

## Experimental Setup

We implemented GRPO from scratch in `agents/grpo.py`. Its policy network and action heads follow the improved PPO configuration, including observation normalisation, orthogonal initialisation, the storage-head mask, and factored action heads. The main difference is that GRPO has no value head.

For each update, GRPO samples a batch of training seasons. Within each season, it collects multiple episodes using the same weather and prices but different pest-growth random seeds. This allows the agent to compare outcomes under similar environmental conditions while accounting for variation in pest growth.

We tested three GRPO configurations:

- **Group 8, episode advantage:** Eight episodes are collected per season, with 21 seasons per update.
- **Group 16, episode advantage:** Sixteen episodes are collected per season, with 10 seasons per update.
- **Group 8, per-step advantage:** Uses the same grouping as the first configuration but calculates advantages separately at each step.

In episode-level mode, the advantage is calculated by subtracting the group's mean episode return from each episode's return and dividing by the group standard deviation plus `0.01` in reward units, approximately 100 INR/ha. The resulting advantage is applied to every step in that episode.

In per-step mode, the return-to-go from each step is normalised within the group at that step. This provides more local information about the relative quality of actions, rather than assigning the same advantage to every step of an episode.

Both approaches use PPO's clipped policy-ratio objective with an entropy bonus. The entropy coefficient is `0.003`, training uses 10 epochs and a batch size of 64, and the optional KL penalty against the policy that collected the data is disabled (`kl_coef` off).

Each configuration receives a budget of 1 million environment steps per seed, matching PPO. The combined group size is approximately 168 episodes, or around 2,000 environment steps, per update.

The rule-based policies and hindsight oracle are taken from Study 1. The improved PPO agent from Study 2, trained with `ent_coef = 0.003` and 10 epochs, serves as the main reference and is not retrained for this experiment.

All configurations are trained using five random seeds and evaluated on the same held-out seasons from 2017 to 2023, with 30 pest-growth replicates per season. Profit is reported in INR per hectare before environmental penalties, and ± represents the standard deviation across training seeds.

The learning curves show mean training reward, which includes environmental penalties, against environment steps. PPO's curve is taken from its saved training history, which records reward rather than profit. Consequently, the learning curves and the final profit results measure related but different quantities.

## Results

| Configuration | Seeds | Profit (INR/ha) | Gap to `threshold_tuned` | Gap to oracle | Oracle performance | N kg · water mm · sprays |
|---|---:|---:|---:|---:|---:|---|
| Improved PPO with critic | 5 | 86,200 ± 542 | −1,032 ± 542 | −3,454 ± 542 | 96.1% | 213 · 235 · 0.00 |
| GRPO, group 8, episode advantage | 5 | 86,117 ± 189 | −1,115 ± 189 | −3,537 ± 189 | 96.1% | 208 · 232 · 0.14 |
| GRPO, group 16, episode advantage | 5 | 86,223 ± 680 | −1,009 ± 680 | −3,431 ± 680 | 96.2% | 205 · 232 · 0.01 |
| GRPO, group 8, per-step advantage | 5 | 85,482 ± 449 | −1,750 ± 449 | −4,171 ± 449 | 95.3% | 223 · 236 · 2.17 |

### Reference Policy Performance

| Configuration | `threshold_tuned` | `recommended_tuned` | `recommended` | `random` | `oracle` | Threshold as % of oracle |
|---|---:|---:|---:|---:|---:|---:|
| Improved PPO with critic | 87,232 | 81,072 | 74,649 | 55,336 | 89,654 | 97.3% |
| GRPO, group 8, episode advantage | 87,232 | 81,072 | 74,649 | 55,336 | 89,654 | 97.3% |
| GRPO, group 16, episode advantage | 87,232 | 81,072 | 74,649 | 55,336 | 89,654 | 97.3% |
| GRPO, group 8, per-step advantage | 87,232 | 81,072 | 74,649 | 55,336 | 89,654 | 97.3% |

## Key Findings

The main result is that **GRPO with episode-level advantages achieves approximately the same profit as PPO, despite having no learned critic**.

The best GRPO configuration, using groups of 16 episodes, earns 86,223 INR/ha. This is just 23 INR/ha higher than the improved PPO mean of 86,200 INR/ha. Both agents achieve approximately 96% of the oracle's profit, and neither outperforms the tuned threshold policy on average.

The group-8 episode-level configuration also performs similarly, earning 86,117 INR/ha. Its standard deviation across seeds is lower than that of PPO, although five seeds are not enough to establish that GRPO is inherently more stable.

By contrast, the per-step advantage configuration earns 85,482 INR/ha, around 718 INR/ha below the improved PPO mean. It also uses more water and applies more pesticides on average.

These results suggest that the way GRPO constructs its advantages matters. Comparing complete episodes under similar conditions appears to work better in this environment than normalising returns separately at each step.

## Analysis

### 1. Removing the critic does not significantly change average performance in these experiments

PPO's critic estimates expected future returns and provides a baseline for calculating policy advantages. GRPO replaces this learned value estimate with a comparison between episodes in the same group.

The two episode-level GRPO configurations achieve profits close to PPO:

- Improved PPO: 86,200 INR/ha.
- GRPO, group 8: 86,117 INR/ha.
- GRPO, group 16: 86,223 INR/ha.

The difference between the best GRPO mean and PPO is only 23 INR/ha, which is small compared with the variation across training seeds.

This indicates that a learned critic is not essential for achieving PPO-like average performance in this particular environment, provided the agent can compare episodes that share weather and price conditions.

However, this should not be interpreted as proof that the critic is unnecessary in general. The experiment covers one environment, one training budget, and a small number of seeds. A paired statistical comparison and additional environments would be needed to establish whether the methods are genuinely equivalent.

### 2. Group size has little effect on average profit, but seed variability differs

Increasing the group size from 8 to 16 episodes changes the average profit from 86,117 to 86,223 INR/ha, a difference of just 106 INR/ha.

The larger-group configuration has a substantially higher standard deviation across seeds: 680 INR/ha compared with 189 INR/ha for group 8. Its observed profit range is also wider.

| Configuration | Mean profit | Standard deviation | Minimum | Maximum | Range |
|---|---:|---:|---:|---:|---:|
| Improved PPO with critic | 86,200 | 542 | 85,585 | 86,781 | 1,196 |
| GRPO, group 8, episode advantage | 86,117 | 189 | 85,955 | 86,442 | 488 |
| GRPO, group 16, episode advantage | 86,223 | 680 | 85,340 | 87,166 | 1,826 |
| GRPO, group 8, per-step advantage | 85,482 | 449 | 85,124 | 86,229 | 1,105 |

*Profit values and variability measures are in INR/ha.*

A larger group could provide a more informative estimate of relative performance within each update, but it also changes the number of seasons contributing to an update. The results do not show a clear average-profit benefit from doubling the group size, and the higher observed seed variability warrants further investigation.

### 3. Episode-level advantages outperform the per-step variant

The group-8 per-step configuration earns 85,482 INR/ha, compared with 86,117 INR/ha for episode-level advantages using the same group size. The difference is 635 INR/ha.

The per-step configuration also uses more resources on average:

- Nitrogen application rises to 223 kg, compared with 208 kg for episode-level GRPO.
- Water use rises to 236 mm, compared with 232 mm.
- Pesticide use rises to 2.17 sprays per season, compared with 0.14.

The episode-level approach gives every action in a successful episode the same positive or negative signal relative to the group. This may be useful when the episode's overall outcome is a meaningful indicator of the quality of the policy's decisions.

The per-step approach instead normalises return-to-go values at each step. In this environment, that alternative produces lower final profit and a different resource-use pattern.

These results show that the tested per-step formulation is less effective here. They do not establish that per-step advantage estimation is generally inferior, since its performance may depend on reward structure, normalisation, and other implementation choices.

### 4. Neither method beats the tuned threshold policy

The tuned threshold policy earns 87,232 INR/ha, ahead of all four learning configurations.

The best GRPO result, 86,223 INR/ha, remains 1,009 INR/ha below the rule and 3,431 INR/ha below the hindsight oracle. It achieves 96.2% of oracle profit.

This is consistent with the results of earlier studies: the learned policies capture much of the achievable profit, but the threshold policy remains a strong baseline in the calibrated environment.

The Shapley results also show that water management remains a weakness. For episode-level GRPO, water contributes −1,325 INR/ha relative to the threshold rule in the group-8 configuration and −1,221 INR/ha in the group-16 configuration. The per-step version has an even larger negative water contribution of −1,608 INR/ha.

The critic-free approach therefore appears capable of matching PPO, but removing the critic does not by itself solve the remaining resource-allocation gap.

## Learning Curves

The learning curves compare mean training reward against environment steps for PPO and the three GRPO configurations.

Because the plotted quantity is reward after environmental penalties rather than the held-out profit reported in the results table, the curves should be interpreted as evidence about training dynamics rather than as a direct comparison of final agricultural profit.

The saved PPO training history is used for its curve, while the GRPO curves come from the corresponding training runs.

![Learning curves](learning_curves.png)

## Wall-Clock Training Time

Training time was estimated using a probe that trained each agent for 100,000 environment steps with four processes running concurrently and one PyTorch thread per process. The probe time was scaled to estimate the cost of one million steps. Actual GRPO run times are also reported below; evaluation time is excluded.

| Agent | Estimated minutes per 1M steps | Measured mean run time |
|---|---:|---:|
| Improved PPO with critic | 22.2 | Not logged |
| GRPO, group 8, episode advantage | 20.9 | 35.5 minutes |
| GRPO, group 16, episode advantage | 21.0 | 36.3 minutes |
| GRPO, group 8, per-step advantage | 19.3 | 35.3 minutes |

The probe-based estimates are lower than the measured GRPO run times. The measured times should therefore be treated as the more relevant indication of the actual training cost for these runs.

A direct wall-clock comparison with PPO is not available because the corresponding measured PPO run time was not logged. The current results are therefore insufficient to conclude that GRPO is faster than PPO, even though its probe-based estimate is similar.

## Conclusion

This study compares improved PPO with a learned critic against GRPO, which removes the value head and estimates advantages by comparing episodes under similar environmental conditions.

The two episode-level GRPO configurations match PPO's average profit closely. Group-16 GRPO achieves 86,223 INR/ha, compared with 86,200 INR/ha for PPO. The difference is negligible relative to the observed seed variability, suggesting that a learned critic may not be necessary to achieve comparable performance in this particular agricultural control task.

The advantage-estimation method matters, however. Episode-level GRPO performs better than the tested per-step variant, which achieves lower profit and uses more water and pesticides. Increasing the group size from 8 to 16 provides little improvement in mean profit and is accompanied by higher variability across seeds.

Despite matching PPO, GRPO does not outperform the tuned threshold policy, and both methods remain below the hindsight oracle. The main conclusion is that **critic-free, episode-level GRPO is a viable alternative to PPO in this environment, but it does not yet provide a clear performance advantage**.

Future work should use paired seeds and formal statistical comparisons, test more group sizes and advantage-normalisation methods, and record PPO and GRPO wall-clock times under identical hardware and execution settings. These experiments would help determine when removing the critic is worthwhile and whether GRPO offers practical advantages beyond comparable average profit.

## Experiment Runs

- [Improved PPO with critic](../02_improved_ppo/runs/improved_ent0p003_ep10/README.md)
- [GRPO, group 8, episode advantage](runs/grpo_g8/README.md)
- [GRPO, group 16, episode advantage](runs/grpo_g16/README.md)
- [GRPO, group 8, per-step advantage](runs/grpo_g8_stepwise/README.md)