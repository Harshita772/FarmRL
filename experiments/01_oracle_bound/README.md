# Study 1 — Hindsight oracle: how much is there to gain?

**Question.** Is PPO underperforming, or is the tuned threshold rule already close to the best possible?

## Setup

- Environment: `configs/default.yaml` (Punjab, Ludhiana weather, all four mechanisms on).
- Agent: the original PPO (200k steps, default hyperparameters), 5 training seeds, reproducing run v4.
- Oracle: for each held-out season, a CEM search over the 12-step action plan (N, water, pesticide per fortnight and the months to store) with that season's weather and price known in advance, maximising the training reward and averaged over 6 pest-growth seeds; the plan is then replayed on the 30 evaluation seeds. Search: population 240, 40 iterations, then coordinate-ascent polishing.

Every configuration is trained with 5 seeds and scored on the same held-out seasons (2017–2023, 30 pest-growth replicates each). Profit is INR/ha before environmental penalties; ± is the standard deviation over training seeds. A gap is agent minus reference (negative = below). The Shapley columns split the agent's gain over `threshold_tuned` between nitrogen, water, pesticide and sale.

## Results

| Configuration | Seeds | Profit | Gap to threshold_tuned | Gap to oracle | % of oracle | Shapley N / W / P / S | N kg · water mm · sprays |
|---|---|---|---|---|---|---|---|
| PPO v4 (200k steps, original settings) | 5 | 86,135 ± 179 | -1,096 ± 179 | -3,518 ± 179 | 96.1% | +17 / -1,188 / +0 / +75 | 199 · 232 · 0.00 |

### Reference policies

| Run | threshold_tuned | recommended_tuned | recommended | random | oracle | threshold_tuned % of oracle |
|---|---|---|---|---|---|---|
| PPO v4 (200k steps, original settings) | 87,232 | 81,072 | 74,649 | 55,336 | 89,654 | 97.3% |

## Readings off the table

- Environment with oracle 89,654 INR/ha: best agent **PPO v4 (200k steps, original settings)** at 86,135 ± 179 INR/ha (96.1% of the oracle); `threshold_tuned` 87,232 (97.3%).
- Ahead of its own `threshold_tuned` on average: none.

## Analysis

The reproduction of v4 gives 86,135 ± 179, not the 86,584 ± 153 recorded in `results/runs/v4`. The agent code is bit-identical to the original for the same seed (5 PPO iterations compared on this server, returns equal to four decimals) and the tuned baselines match v4 exactly (87,232 and 81,072), so the difference comes from the platform (v4 was trained on Windows with torch 2.5, these runs on Linux with torch 2.6), not from the code. Numbers in this folder should be compared with each other, not with the v4 card.

The hindsight oracle reaches 89,654 INR/ha. `threshold_tuned` earns 97.3% of it (2,422 below) and PPO 96.1% (3,518 below). PPO's shortfall to the rule (1,096, 0 of 5 seeds ahead) is about 45% of the rule's own distance to the oracle.

The oracle knows the whole season's weather and price in advance, so no policy that reacts to the season as it unfolds can reach it. The tuned rule is therefore within 2.7% of a ceiling it cannot attain, and the room above it that any learned policy could use is smaller than 2,422 INR/ha. Both readings hold: the rule is already near the best possible, and PPO is still a little below it.

Caveat: the oracle plan is open-loop in the pest noise (averaged over 6 seeds) and its search is heuristic (CEM plus coordinate ascent), so 89,654 is a lower bound on the true foresight optimum.

## Runs

- [`baseline_v4_oracle`](runs/baseline_v4_oracle/README.md)
