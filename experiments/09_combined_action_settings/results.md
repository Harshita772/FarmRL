| Configuration | Seeds | Profit | Gap to threshold_tuned | Gap to oracle | % of oracle | Shapley N / W / P / S | N kg · water mm · sprays |
|---|---|---|---|---|---|---|---|
| PPO v4 (200k steps, original settings) | 5 | 86,135 ± 179 | -1,096 ± 179 | -3,518 ± 179 | 96.1% | +17 / -1,188 / +0 / +75 | 199 · 232 · 0.00 |
| improved PPO, no forecast (study 2 best) | 5 | 86,200 ± 542 | -1,032 ± 542 | -3,454 ± 542 | 96.1% | -144 / -1,005 / +0 / +117 | 213 · 235 · 0.00 |
| improved PPO, grid 0/25/50/75, factored (study 3) | 5 | 86,587 ± 286 | +196 ± 286 | -3,014 ± 286 | 96.6% | -371 / +428 / +0 / +139 | 198 · 227 · 0.00 |
| improved PPO, standard grid, joint head (study 3) | 5 | 86,250 ± 194 | -982 ± 194 | -3,403 ± 194 | 96.2% | -151 / -985 / +0 / +155 | 198 · 233 · 0.00 |
| improved PPO, grid 0/25/50/75, joint head | 5 | 86,821 ± 315 | +431 ± 315 | -2,780 ± 315 | 96.9% | -341 / +636 / -17 / +154 | 207 · 229 · 0.00 |
| improved PPO, grid 0/15/30/45/60/75, joint head | 5 | 86,548 ± 273 | -158 ± 273 | -3,155 ± 273 | 96.5% | -357 / +125 / +0 / +74 | 194 · 229 · 0.00 |
| improved PPO, grid 0/15/30/45/60/75, factored heads | 5 | 86,708 ± 69 | +1 ± 69 | -2,996 ± 69 | 96.7% | -157 / -15 / +0 / +174 | 201 · 231 · 0.00 |
