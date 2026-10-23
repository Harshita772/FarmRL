# Study 4 — Comparing Alternative Algorithms

## Research Question

Can an alternative reinforcement learning algorithm or direct optimisation of the threshold rule's parameters outperform PPO?

## Experimental Setup

We compared two alternatives against the original PPO agent and the best-performing improved PPO configuration from Study 2.

The first approach, `dqn_branching`, uses a branching dueling Deep Q-Network (DQN). Instead of predicting values for every possible combination of actions, the network uses a separate output branch for each action component. It uses a Double-DQN target averaged across the active branches and is trained for 1 million timesteps with a discount factor of 1.0 and a learning rate of \(5 \times 10^{-4}\). Exploration is controlled by an epsilon-greedy schedule, with epsilon decreasing from 1.0 to 0.05 over the first half of training.

The second approach, `cem_threshold`, applies the Cross-Entropy Method (CEM) directly to the ten parameters of the threshold-based farming rule. It uses a population of 64 candidates over 40 iterations, retaining the top 20% of candidates at each iteration. Candidate rules are evaluated on the training seasons using the training reward.

Both approaches were evaluated using the same held-out seasons, random seeds, and reference policies as in Study 2.

Each configuration was trained with five seeds and evaluated on the 2017–2023 seasons, using 30 pest-growth replicates per season. Profit is reported in INR per hectare before environmental penalties, with ± representing the standard deviation across training seeds.

The performance gap is calculated as the agent's profit minus the profit of the reference policy. The Shapley values break down the agent's gain over `threshold_tuned` into nitrogen, water, pesticide, and sale decisions.

## Results

| Configuration | Seeds | Profit (INR/ha) | Gap to `threshold_tuned` | Gap to oracle | Oracle performance |
|---|---:|---:|---:|---:|---:|
| PPO v4 (200k steps) | 5 | 86,135 ± 179 | −1,096 ± 179 | −3,518 ± 179 | 96.1% |
| Improved PPO (Study 2 best) | 5 | 86,200 ± 542 | −1,032 ± 542 | −3,454 ± 542 | 96.1% |
| Branching dueling DQN (1M steps) | 5 | 83,617 ± 1,741 | −3,615 ± 1,741 | −6,036 ± 1,741 | 93.3% |
| CEM over threshold-rule parameters | 5 | 84,178 ± 192 | −3,054 ± 192 | −5,476 ± 192 | 93.9% |

### Reference Policy Performance

| Policy | Profit (INR/ha) |
|---|---:|
| `threshold_tuned` | 87,232 |
| `recommended_tuned` | 81,072 |
| `recommended` | 74,649 |
| `random` | 55,336 |
| `oracle` | 89,654 |

The tuned threshold policy achieves 97.3% of the oracle's profit, making it the strongest reference for comparing the learned agents and the direct parameter search.

## Key Findings

The improved PPO configuration from Study 2 remains the strongest agent, achieving an average profit of 86,200 INR/ha. Branching DQN reaches 83,617 INR/ha, while CEM over the threshold-rule parameters reaches 84,178 INR/ha.

Neither alternative outperforms the tuned threshold baseline, which earns 87,232 INR/ha. Branching DQN has the largest variation across training seeds, whereas the CEM results are more consistent but still fall well short of the tuned rule.

These findings suggest that changing the learning algorithm or searching directly over the rule's parameters does not automatically improve performance. The reasons for the two methods falling short, however, appear to be different.

## Analysis

### 1. Branching DQN performs worse than PPO and is less stable

Branching DQN achieves an average profit of 83,617 ± 1,741 INR/ha, approximately 2,583 INR/ha below the best improved PPO configuration.

Its results also vary considerably across seeds, ranging from 80,954 to 85,545 INR/ha. This is substantially more variation than observed for either PPO configuration.

The agent uses a branching architecture to handle multiple action components, but this design does not translate into better performance in the current environment. The results suggest that branching DQN struggles to learn a policy that consistently matches PPO's performance under the tested training settings.

However, this experiment compares specific implementations and hyperparameters rather than the full capabilities of the two algorithms. It therefore does not establish that DQN is inherently inferior to PPO for this task.

### 2. CEM does not recover the performance of the tuned threshold rule

The CEM-based search achieves 84,178 ± 192 INR/ha. Although its results are relatively consistent across seeds, its average profit remains 3,054 INR/ha below `threshold_tuned`.

This is an important result because CEM searches within the same family of threshold-based rules as the reference policy. In principle, a successful search should be able to identify parameter settings that produce strong farming decisions.

The Shapley analysis shows that the main source of the performance gap is pesticide use, which contributes −3,088 INR/ha. The CEM-optimised rule applies approximately 3.46 sprays per season, whereas the tuned threshold policy applies none.

This difference accounts for most of the gap between the two approaches. The nitrogen, water, and sale contributions are comparatively small.

### 3. Search noise may explain why CEM falls short

The CEM search uses 64 candidates over 40 iterations, with two evaluation replicates per candidate. Under these settings, differences between candidate rules can be difficult to distinguish when pest-growth randomness introduces substantial variation in the reward.

In particular, the search may struggle to identify the benefit of avoiding pesticide applications if the reward differences between nearby pest thresholds are obscured by evaluation noise.

The block-wise grid search used to tune the reference rule successfully identified a configuration that never sprays. CEM did not converge to the same solution.

This suggests that the current CEM search budget and evaluation procedure may be insufficient for reliably optimising these threshold parameters. It does not mean that CEM is generally unsuitable for the task; rather, the particular search setup did not recover the stronger rule found by grid search.

## Conclusion

This study compared improved PPO with branching DQN and direct CEM optimisation of the threshold rule's parameters. Neither alternative outperformed the tuned threshold policy, and improved PPO remained the strongest learned agent.

Branching DQN performed worse than PPO and showed substantial variation across training seeds. CEM produced more consistent results but failed to recover the performance of the grid-tuned rule, primarily because its optimised policy used pesticides when the reference policy did not.

The results highlight two different challenges: learning a reliable policy with a branching value-based architecture, and optimising threshold parameters when candidate evaluations are noisy.

A useful next step for DQN would be to investigate its training stability and hyperparameters. For CEM, increasing the number of evaluation replicates, improving the search budget, or using a more targeted search around the pesticide thresholds could help determine whether the gap is caused by the optimisation method or by the current evaluation setup.

Overall, the experiments reinforce the finding from the earlier studies that the tuned threshold policy is a strong baseline and that improving upon it requires more than simply changing the learning algorithm.

## Experiment Runs

- [Baseline PPO v4 and oracle comparison](../01_oracle_bound/runs/baseline_v4_oracle/README.md)
- [Improved PPO — best configuration from Study 2](../02_improved_ppo/runs/improved_ent0p003_ep10/README.md)
- [Branching dueling DQN](runs/dqn_branching/README.md)
- [CEM over threshold-rule parameters](runs/cem_threshold/README.md)