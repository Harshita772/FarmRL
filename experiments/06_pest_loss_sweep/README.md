# Study 6 — Pest-loss sweep: when does PPO beat the rule?

**Question.** As pests become more costly, where does adaptive control start to pay? Pest loss is swept from the calibrated 0.115 to 0.5.

## Setup

- At every value the whole pipeline is rerun: `y_scale` recalibrated, the threshold rule retuned on the training seasons, PPO retrained (study-2 best: improved bundle, ent_coef 0.003, epochs 10, 1M steps, 5 seeds) and the hindsight oracle recomputed.
- The rule's pest cut-offs are searched on a fine grid (every pair of cut-offs on a 0.02 step) at every value, so the rule is not handicapped by a coarse spray grid (an earlier concern about study 5).
- The last table column gives sprays per season.

Every configuration is trained with 5 seeds and scored on the same held-out seasons (2017–2023, 30 pest-growth replicates each). Profit is INR/ha before environmental penalties; ± is the standard deviation over training seeds. A gap is agent minus reference (negative = below). The Shapley columns split the agent's gain over `threshold_tuned` between nitrogen, water, pesticide and sale.

## Results

| Configuration | Seeds | Profit | Gap to threshold_tuned | Gap to oracle | % of oracle | Shapley N / W / P / S | N kg · water mm · sprays |
|---|---|---|---|---|---|---|---|
| pest_loss = 0.115 (calibrated) | 5 | 86,200 ± 542 | -1,032 ± 542 | -3,454 ± 542 | 96.1% | -144 / -1,005 / +0 / +117 | 213 · 235 · 0.00 |
| pest_loss = 0.2 | 5 | 85,745 ± 311 | -159 ± 311 | -3,547 ± 311 | 96.0% | -151 / -1,372 / +1,179 / +185 | 201 · 233 · 2.03 |
| pest_loss = 0.3 | 5 | 86,007 ± 395 | +2,601 ± 395 | -3,996 ± 395 | 95.6% | -68 / -1,217 / +3,782 / +103 | 197 · 225 · 2.29 |
| pest_loss = 0.35 | 5 | 86,410 ± 492 | +2,664 ± 492 | -4,151 ± 492 | 95.4% | -227 / -1,266 / +4,057 / +100 | 208 · 227 · 3.59 |
| pest_loss = 0.5 | 5 | 88,589 ± 628 | +3,827 ± 628 | -3,932 ± 628 | 95.8% | -196 / -1,217 / +5,110 / +130 | 197 · 234 · 4.42 |

### Reference policies

| Run | threshold_tuned | recommended_tuned | recommended | random | oracle | threshold_tuned % of oracle |
|---|---|---|---|---|---|---|
| pest_loss = 0.115 (calibrated) | 87,232 | 81,072 | 74,649 | 55,336 | 89,654 | 97.3% |
| pest_loss = 0.2 | 85,904 | 80,528 | 73,706 | 56,139 | 89,292 | 96.2% |
| pest_loss = 0.3 | 83,406 | 80,917 | 72,558 | 57,117 | 90,002 | 92.7% |
| pest_loss = 0.35 | 83,746 | 80,960 | 71,968 | 57,619 | 90,561 | 92.5% |
| pest_loss = 0.5 | 84,762 | 81,094 | 70,132 | 59,183 | 92,521 | 91.6% |

## Readings off the table

- Environment with oracle 89,654 INR/ha: best agent **pest_loss = 0.115 (calibrated)** at 86,200 ± 542 INR/ha (96.1% of the oracle); `threshold_tuned` 87,232 (97.3%).
- Ahead of its own `threshold_tuned` on average: none.
- Environment with oracle 89,292 INR/ha: best agent **pest_loss = 0.2** at 85,745 ± 311 INR/ha (96.0% of the oracle); `threshold_tuned` 85,904 (96.2%).
- Ahead of its own `threshold_tuned` on average: none.
- Environment with oracle 90,002 INR/ha: best agent **pest_loss = 0.3** at 86,007 ± 395 INR/ha (95.6% of the oracle); `threshold_tuned` 83,406 (92.7%).
- Ahead of its own `threshold_tuned` on average: pest_loss = 0.3 (5 of 5 seeds).
- Environment with oracle 90,561 INR/ha: best agent **pest_loss = 0.35** at 86,410 ± 492 INR/ha (95.4% of the oracle); `threshold_tuned` 83,746 (92.5%).
- Ahead of its own `threshold_tuned` on average: pest_loss = 0.35 (5 of 5 seeds).
- Environment with oracle 92,521 INR/ha: best agent **pest_loss = 0.5** at 88,589 ± 628 INR/ha (95.8% of the oracle); `threshold_tuned` 84,762 (91.6%).
- Ahead of its own `threshold_tuned` on average: pest_loss = 0.5 (5 of 5 seeds).

## Figure

![Held-out profit against pest loss](pest_sweep.png)

## Runs

- [`pest_0p115`](runs/pest_0p115/README.md)
- [`pest_0p2`](runs/pest_0p2/README.md)
- [`pest_0p3`](runs/pest_0p3/README.md)
- [`pest_0p35`](runs/pest_0p35/README.md)
- [`pest_0p5`](runs/pest_0p5/README.md)
