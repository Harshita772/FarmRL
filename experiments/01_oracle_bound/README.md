# Study 1 — Hindsight Oracle: How Much Room Is There for Improvement?

## Research Question

Is PPO genuinely underperforming, or is the tuned threshold-based policy already close to the best performance achievable in this environment?

## Experimental Setup

We used the default environment configuration (`configs/default.yaml`), which models farming conditions in Punjab using Ludhiana weather data, with all four mechanisms enabled.

The baseline agent was the original PPO model, trained for 200,000 timesteps using the default hyperparameters. We used five training seeds to reproduce the results from run v4.

To estimate how much performance could improve with perfect foresight, we also constructed a hindsight oracle. For each held-out season, the oracle uses Cross-Entropy Method (CEM) search to optimise a 12-step action plan. The plan specifies nitrogen application, irrigation, and pesticide use for each fortnight, along with the months in which produce should be stored for later sale.

Unlike PPO, the oracle has access to the season's weather and prices in advance. It optimises the training reward by evaluating candidate plans across six pest-growth seeds, then replays the selected plan across the 30 evaluation seeds. The search uses a population of 240 candidates over 40 iterations, followed by coordinate-ascent refinement.

All policies were evaluated on the same held-out seasons from 2017 to 2023, using 30 pest-growth replicates per season. Profit is reported in INR per hectare before environmental penalties, and the ± values represent the standard deviation across the five training seeds.

We compare PPO against the tuned threshold policy and the hindsight oracle. The performance gap is calculated as the agent's profit minus the reference policy's profit, so negative values indicate lower profit. The Shapley values break down PPO's gain over `threshold_tuned` into nitrogen, water, pesticide, and sale decisions.

## Results

| Configuration | Seeds | Profit (INR/ha) | Gap to `threshold_tuned` | Gap to oracle | Oracle performance | Shapley N / W / P / S | N kg · water mm · sprays |
|---|---:|---:|---:|---:|---:|---|---|
| PPO v4 (200k steps, original settings) | 5 | 86,135 ± 179 | −1,096 ± 179 | −3,518 ± 179 | 96.1% | +17 / −1,188 / +0 / +75 | 199 · 232 · 0.00 |

### Reference Policy Performance

| Policy | Profit (INR/ha) |
|---|---:|
| `threshold_tuned` | 87,232 |
| `recommended_tuned` | 81,072 |
| `recommended` | 74,649 |
| `random` | 55,336 |
| `oracle` | 89,654 |

The tuned threshold policy achieves 97.3% of the oracle's profit, while PPO reaches 96.1%.

## Key Findings

The reproduced PPO v4 agent achieves an average profit of 86,135 INR/ha, compared with 87,232 INR/ha for the tuned threshold policy and 89,654 INR/ha for the hindsight oracle.

The threshold policy therefore earns 2,422 INR/ha less than the oracle, while PPO falls short by 3,518 INR/ha. PPO does not outperform the tuned threshold policy on average, and none of the five training seeds exceeds it.

These results suggest that the threshold policy is already performing close to the upper limit estimated by the oracle. However, PPO still has a measurable performance gap to close.

## Analysis

### 1. Reproducing PPO v4 gives slightly different results

The reproduced PPO v4 agent achieves 86,135 ± 179 INR/ha, compared with the 86,584 ± 153 INR/ha recorded in `results/runs/v4`.

We investigated this difference by comparing the agent implementations and training behaviour. The agent code was identical to the original, and five PPO iterations produced matching returns to four decimal places for the same seed on the current server. The tuned baselines also matched the original results exactly, with profits of 87,232 INR/ha and 81,072 INR/ha.

The most likely explanation is a difference in the computing environment: the original v4 run was trained on Windows with PyTorch 2.5, whereas the reproduced runs used Linux with PyTorch 2.6.

For this reason, results within the current experiment folder should be compared with one another rather than directly against the original v4 result.

### 2. The tuned threshold policy is already close to the oracle

The hindsight oracle achieves 89,654 INR/ha, setting a strong benchmark for the environment.

The tuned threshold policy reaches 87,232 INR/ha, or 97.3% of the oracle's profit. Its remaining gap is only 2,422 INR/ha, equivalent to approximately 2.7% of the oracle's profit.

PPO achieves 86,135 INR/ha, or 96.1% of the oracle's profit. Its gap to the tuned threshold policy is 1,096 INR/ha, which is about 45% of the threshold policy's remaining gap to the oracle.

This indicates that PPO is not dramatically underperforming. The rule-based policy is already strong, leaving relatively limited room for any learned policy to improve upon it.

### 3. Perfect foresight gives the oracle an advantage

The oracle knows the season's weather and prices before selecting its actions. PPO, in contrast, must make decisions as the season unfolds, without knowing future conditions.

This difference is important because the oracle benefits from information that a realistic, reactive policy cannot access. Its results should therefore be interpreted as a benchmark for the value of foresight, rather than as a directly achievable target for PPO.

The oracle also has two limitations: its action plan is fixed in advance rather than adapted to future pest outcomes, and its search relies on a heuristic combination of CEM and coordinate ascent. It averages over six pest-growth seeds when selecting a plan, rather than optimising directly over every possible future outcome.

Consequently, the reported oracle profit of 89,654 INR/ha is a lower bound on the true optimum under the same foresight assumptions, not a guarantee that the global optimum has been found.

## Conclusion

This study suggests that both observations are true: the tuned threshold policy is already close to the performance estimated by the hindsight oracle, and PPO still performs somewhat worse than the tuned rule.

The threshold policy achieves 97.3% of the oracle's profit, while PPO achieves 96.1%. The remaining gap between PPO and the rule is 1,096 INR/ha, and none of the five PPO training seeds outperforms the rule.

The results suggest that future improvements may require more than simply increasing training time or adjusting PPO's hyperparameters. It may be more productive to investigate the decisions where PPO differs from the tuned rule and identify whether those differences reflect limitations in the learned policy, the available observations, or the training objective.

At the same time, the oracle should be treated as an approximate benchmark because its search is heuristic and its action plan does not adapt to pest-growth outcomes. The estimated gap to the oracle therefore provides a useful reference point, rather than a definitive measure of the maximum possible improvement.

## Experiment Run

- [Baseline PPO v4 and hindsight oracle](runs/baseline_v4_oracle/README.md)