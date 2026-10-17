| Configuration | Seeds | Profit | Gap to threshold_tuned | Gap to oracle | % of oracle | Shapley N / W / P / S | N kg · water mm · sprays |
|---|---|---|---|---|---|---|---|
| PPO v4 (200k steps, original settings) | 5 | 86,135 ± 179 | -1,096 ± 179 | -3,518 ± 179 | 96.1% | +17 / -1,188 / +0 / +75 | 199 · 232 · 0.00 |
| improved PPO (study 2 best) | 5 | 86,200 ± 542 | -1,032 ± 542 | -3,454 ± 542 | 96.1% | -144 / -1,005 / +0 / +117 | 213 · 235 · 0.00 |
| branching dueling DQN, 1M steps | 5 | 83,617 ± 1,741 | -3,615 ± 1,741 | -6,036 ± 1,741 | 93.3% | -343 / -2,602 / -733 / +63 | 212 · 221 · 0.83 |
| CEM over threshold-rule parameters | 5 | 84,178 ± 192 | -3,054 ± 192 | -5,476 ± 192 | 93.9% | -126 / +82 / -3,088 / +77 | 205 · 238 · 3.46 |
