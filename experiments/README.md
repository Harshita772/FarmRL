# Comparative study: is PPO underperforming, or is the rule already near the best possible?

Held-out profit (INR/ha, seasons 2017–2023, mean ± std over 5 training seeds). Every study card lists its setup, run folders and the full numbers; every run folder has its own run card, config, logs and the exact code that produced it.

![Summary](summary.png)

Dot: the agent (± std over seeds). Diamond: `threshold_tuned` in the same environment. Open square: hindsight oracle in the same environment.

| Configuration | Seeds | Profit | Gap to threshold_tuned | Gap to oracle | % of oracle | Shapley N / W / P / S | N kg · water mm · sprays |
|---|---|---|---|---|---|---|---|
| PPO v4 (200k steps, original settings) | 5 | 86,135 ± 179 | -1,096 ± 179 | -3,518 ± 179 | 96.1% | +17 / -1,188 / +0 / +75 | 199 · 232 · 0.00 |
| improved PPO, ent_coef=0.01, epochs=10 | 5 | 85,864 ± 412 | -1,368 ± 412 | -3,789 ± 412 | 95.8% | -216 / -1,251 / -17 / +116 | 206 · 227 · 0.00 |
| improved PPO, ent_coef=0.003, epochs=10 | 5 | 86,200 ± 542 | -1,032 ± 542 | -3,454 ± 542 | 96.1% | -144 / -1,005 / +0 / +117 | 213 · 235 · 0.00 |
| improved PPO, ent_coef=0, epochs=10 | 5 | 86,104 ± 554 | -1,128 ± 554 | -3,550 ± 554 | 96.0% | -53 / -1,198 / -25 / +148 | 206 · 235 · 0.06 |
| improved PPO, ent_coef=0.01, epochs=4 | 5 | 85,820 ± 358 | -1,412 ± 358 | -3,834 ± 358 | 95.7% | -9 / -1,456 / +0 / +52 | 197 · 230 · 0.00 |
| improved PPO, ent_coef=0.003, epochs=4 | 5 | 85,975 ± 386 | -1,257 ± 386 | -3,679 ± 386 | 95.9% | +53 / -1,367 / +0 / +57 | 200 · 237 · 0.00 |
| improved PPO, ent_coef=0, epochs=4 | 5 | 86,144 ± 271 | -1,088 ± 271 | -3,509 ± 271 | 96.1% | +7 / -1,155 / +0 / +61 | 197 · 234 · 0.00 |
| improved PPO, joint N x water head | 5 | 86,250 ± 194 | -982 ± 194 | -3,403 ± 194 | 96.2% | -151 / -985 / +0 / +155 | 198 · 233 · 0.00 |
| improved PPO, water grid 0/25/50/75 | 5 | 86,587 ± 286 | +196 ± 286 | -3,014 ± 286 | 96.6% | -371 / +428 / +0 / +139 | 198 · 227 · 0.00 |
| branching dueling DQN, 1M steps | 5 | 83,617 ± 1,741 | -3,615 ± 1,741 | -6,036 ± 1,741 | 93.3% | -343 / -2,602 / -733 / +63 | 212 · 221 · 0.83 |
| CEM over threshold-rule parameters | 5 | 84,178 ± 192 | -3,054 ± 192 | -5,476 ± 192 | 93.9% | -126 / +82 / -3,088 / +77 | 205 · 238 · 3.46 |
| improved PPO, pest_loss = 0.35 | 5 | 86,410 ± 492 | +2,664 ± 492 | -4,151 ± 492 | 95.4% | -227 / -1,266 / +4,057 / +100 | 208 · 227 · 3.59 |

## Studies

- [Study 1 — Hindsight oracle: how much is there to gain?](01_oracle_bound/README.md)
- [Study 2 — Improved PPO and a small sweep](02_improved_ppo/README.md)
- [Study 3 — Action structure](03_action_structure/README.md)
- [Study 4 — Other algorithms](04_other_algorithms/README.md)
- [Study 5 — Does PPO spray when pests cost more?](05_pest_loss_prior/README.md)

## Conclusion

- **The rule is already near the best possible.** A hindsight oracle that knows each held-out season's weather and price in advance earns 89,654 INR/ha; the grid-tuned threshold rule earns 97.3% of that, and no policy that reacts to the season as it unfolds can reach the oracle.
- **PPO is slightly below the rule, and this is not an optimisation problem.** Seven PPO variants (original, six improved/sweep configurations, a joint head) sit at 85.8–86.3k, 1.0–1.4k below the rule in every seed; normalisation, LR decay, orthogonal init, gamma = 1, a masked storage head and five times more steps changed nothing measurable. The shortfall is in the water decisions.
- **A finer water grid is the only change with a hint of benefit** (+387 over the best factored run, about one seed std; ahead of its own, handicapped, rule in 4 of 5 seeds).
- **DQN and CEM over the rule's parameters are worse** (83.6k and 84.2k).
- **When pests cost more (pest_loss 0.35), PPO does learn to spray** and beats the tuned rule by 2.7k.
- Caveats: 7 held-out seasons; the v4 reproduction is 450 INR/ha below the original v4 run because of the platform (identical code, Windows/torch 2.5 vs Linux/torch 2.6); the oracle is a heuristic search, hence a lower bound on the foresight optimum.

## Layout

- `code/` — self-contained copy of the code the experiments use (`src/farmrl`, `scripts`, `configs`, `data`, `tests`); the repository's own `src/` is untouched.
- `<study>/runs/<run>/` — one folder per run (run card, tables, models, config and code snapshot).
- `<study>/logs/` — the stdout/stderr of every run.
- `run_all_studies.sh` — the sequential driver that produced everything.
