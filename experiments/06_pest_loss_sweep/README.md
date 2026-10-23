# Study 6 — Pest-Loss Sweep: When Does PPO Outperform the Rule?

## Research Question

As pest damage becomes more costly, at what point does PPO's ability to adapt its pesticide use give it an advantage over a tuned threshold-based policy?

## Experimental Setup

In Study 5, PPO began using pesticides when the pest-loss parameter increased from the calibrated value of `0.115` to `0.35`. This study extends that experiment to investigate how the agent's behaviour and profitability change across a wider range of pest-loss settings.

We tested five values of `pest_loss`: `0.115`, `0.2`, `0.3`, `0.35`, and `0.5`.

At each value, we reran the complete experimental pipeline. This included recalibrating `y_scale`, retuning the threshold-based policy on the training seasons, retraining PPO, and recomputing the hindsight oracle. The PPO agent used the best configuration from Study 2, with the improved training setup, `ent_coef = 0.003`, 10 epochs, and 1 million training timesteps.

To ensure that the rule-based baseline was not disadvantaged by a coarse search, its pesticide thresholds were optimised over a fine grid, testing every pair of cut-offs at increments of `0.02`.

Each configuration was trained with five random seeds and evaluated on the same held-out seasons from 2017 to 2023, using 30 pest-growth replicates per season. Profit is reported in INR per hectare before environmental penalties, and ± represents the standard deviation across training seeds.

The performance gap is calculated as the agent's profit minus the profit of `threshold_tuned`. A positive gap means PPO earns more than the tuned rule, while a negative gap means it earns less. The Shapley values break down PPO's gain over the rule into nitrogen, water, pesticide, and sale decisions.

The final column reports the average nitrogen application, water use, and number of pesticide sprays per season.

## Results

| Pest-loss setting | Seeds | PPO profit (INR/ha) | Gap to `threshold_tuned` | Gap to oracle | Oracle performance | N kg · water mm · sprays |
|---|---:|---:|---:|---:|---:|---|
| 0.115 (calibrated) | 5 | 86,200 ± 542 | −1,032 ± 542 | −3,454 ± 542 | 96.1% | 213 · 235 · 0.00 |
| 0.2 | 5 | 85,745 ± 311 | −159 ± 311 | −3,547 ± 311 | 96.0% | 201 · 233 · 2.03 |
| 0.3 | 5 | 86,007 ± 395 | +2,601 ± 395 | −3,996 ± 395 | 95.6% | 197 · 225 · 2.29 |
| 0.35 | 5 | 86,410 ± 492 | +2,664 ± 492 | −4,151 ± 492 | 95.4% | 208 · 227 · 3.59 |
| 0.5 | 5 | 88,589 ± 628 | +3,827 ± 628 | −3,932 ± 628 | 95.8% | 197 · 234 · 4.42 |

### Reference Policy Performance

| Pest-loss setting | `threshold_tuned` | `recommended_tuned` | `recommended` | `random` | `oracle` | Threshold as % of oracle |
|---|---:|---:|---:|---:|---:|---:|
| 0.115 | 87,232 | 81,072 | 74,649 | 55,336 | 89,654 | 97.3% |
| 0.2 | 85,904 | 80,528 | 73,706 | 56,139 | 89,292 | 96.2% |
| 0.3 | 83,406 | 80,917 | 72,558 | 57,117 | 90,002 | 92.7% |
| 0.35 | 83,746 | 80,960 | 71,968 | 57,619 | 90,561 | 92.5% |
| 0.5 | 84,762 | 81,094 | 70,132 | 59,183 | 92,521 | 91.6% |

All profit values are in INR/ha.

## Key Findings

The results show a clear change in PPO's performance as pest damage becomes more costly.

At the calibrated pest-loss value of `0.115`, PPO earns 86,200 INR/ha, falling 1,032 INR/ha short of the tuned threshold policy. At `0.2`, the gap narrows to just 159 INR/ha.

At `0.3`, PPO begins to outperform the tuned rule by 2,601 INR/ha. This advantage persists at `0.35` and increases to 3,827 INR/ha at `0.5`.

Pesticide use also increases across the sweep, from zero sprays per season at `0.115` to 4.42 sprays at `0.5`. This pattern suggests that PPO adapts its spraying decisions as pest damage becomes more expensive.

## Analysis

### 1. PPO's advantage emerges between pest-loss values of 0.2 and 0.3

At `pest_loss = 0.115`, PPO remains behind the tuned threshold policy. At `0.2`, the two approaches perform similarly, with PPO trailing by only 159 INR/ha.

The picture changes at `0.3`, where PPO achieves an average profit of 86,007 INR/ha, compared with 83,406 INR/ha for the tuned rule. All five PPO seeds outperform the baseline at this setting. The same is true at `0.35` and `0.5`.

This places the observed crossover somewhere between `0.2` and `0.3` for the tested configurations. However, the exact point at which PPO starts to outperform the rule has not been established. Additional experiments at intermediate values would be needed to locate it more precisely.

Importantly, this crossover is observed even though the rule's pesticide thresholds are retuned using a fine grid at every pest-loss value. PPO's advantage is therefore not simply the result of comparing it with a rule that uses poorly chosen pesticide cut-offs.

### 2. PPO increases pesticide use as pest damage becomes more costly

The average number of sprays per season rises across the sweep:

- At `pest_loss = 0.115`, PPO does not spray.
- At `pest_loss = 0.2`, it averages 2.03 sprays.
- At `pest_loss = 0.3`, it averages 2.29 sprays.
- At `pest_loss = 0.35`, it averages 3.59 sprays.
- At `pest_loss = 0.5`, it averages 4.42 sprays.

The overall trend is consistent with the expected economic trade-off: when pest damage becomes more expensive, preventing that damage through pesticide use becomes more worthwhile.

The relationship is not perfectly linear. For example, spray frequency increases only slightly between `0.2` and `0.3`, before rising more sharply at higher values. This may reflect changes in the learned policy's response to pest conditions, but the current results do not establish the exact mechanism behind these differences.

### 3. Pesticide decisions account for most of PPO's advantage

The Shapley analysis provides further evidence that pesticide management is central to the results.

At `pest_loss = 0.115`, the pesticide contribution to PPO's advantage is zero, while the water component contributes −1,005 INR/ha. The agent remains behind the tuned rule.

At higher pest-loss values, the pesticide contribution becomes increasingly positive:

| Pest-loss setting | Nitrogen | Water | Pesticide | Sale |
|---|---:|---:|---:|---:|
| 0.115 | −144 | −1,005 | 0 | +117 |
| 0.2 | −151 | −1,372 | +1,179 | +185 |
| 0.3 | −68 | −1,217 | +3,782 | +103 |
| 0.35 | −227 | −1,266 | +4,057 | +100 |
| 0.5 | −196 | −1,217 | +5,110 | +130 |

<text color="secondary" size="sm">Shapley contributions in INR/ha; values may not sum exactly to the reported profit gap because of rounding.</text>

By `pest_loss = 0.5`, the pesticide contribution reaches +5,110 INR/ha. This more than offsets the negative contributions from nitrogen and water, producing an overall advantage of 3,827 INR/ha.

The pattern suggests that PPO's ability to adapt pesticide use is the main reason it outperforms the threshold rule in the higher-loss environments. Its water decisions remain a consistent weakness relative to the baseline.

### 4. Higher pest losses change both the attainable profit and the baseline comparison

PPO's average profit rises from 86,200 INR/ha at `pest_loss = 0.115` to 88,589 INR/ha at `0.5`. Meanwhile, the oracle's profit increases from 89,654 to 92,521 INR/ha.

These figures should be interpreted within each environment rather than as a direct measure of the benefit of increasing pest loss. Each setting has its own recalibrated `y_scale`, retuned reference policies, and recomputed oracle.

The more meaningful comparison is how PPO performs against the corresponding threshold rule at each value. By that measure, the gap moves from negative at the lower settings to strongly positive at the higher settings.

Even at `0.5`, PPO achieves 95.8% of the oracle's profit, leaving a gap of 3,932 INR/ha. Adaptive pesticide use improves PPO's relative performance, but it does not eliminate the remaining difference from the hindsight benchmark.

## Conclusion

This study shows that PPO's advantage over the tuned threshold policy depends strongly on the cost of pest damage.

At low pest-loss values, the threshold rule remains competitive, and PPO either falls short or performs at a similar level. Between `0.2` and `0.3`, the observed results show a crossover: PPO begins to outperform the retuned threshold policy, and its advantage grows as pest damage becomes more costly.

The increase in pesticide use, together with the positive pesticide Shapley contributions, suggests that adaptive spraying is the main driver of this improvement. By contrast, water management continues to contribute negatively to PPO's performance relative to the rule.

The results support the conclusion that PPO can learn useful adaptive pesticide strategies in environments where pest damage is sufficiently expensive. However, they do not establish an exact crossover threshold or prove that PPO will outperform every possible threshold-based policy.

A natural next step is to run additional experiments between `0.2` and `0.3` to locate the crossover more precisely, while examining pesticide timing and irrigation decisions to identify further opportunities for improvement.

## Experiment Runs

- [Pest loss = 0.115](runs/pest_0p115/README.md)
- [Pest loss = 0.2](runs/pest_0p2/README.md)
- [Pest loss = 0.3](runs/pest_0p3/README.md)
- [Pest loss = 0.35](runs/pest_0p35/README.md)
- [Pest loss = 0.5](runs/pest_0p5/README.md)