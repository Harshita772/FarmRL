# Study 10 — What does the oracle need to know?

**Question.** Does the remaining headroom above the tuned rule come from weather uncertainty or from price uncertainty?

Four plan-searching oracles, each choosing one fixed 12-step plan per held-out season and then replayed on the true season: **full** knows the season's weather and price; **weather** knows the weather but only the average price of the training seasons; **price** knows the price but only the weather of ten training seasons spread over the record; **neither** is the best single plan for the average price and those ten seasons.

Held-out seasons 2017–2023, 30 pest replicates; PPO is the study-2 best configuration, mean ± std over 5 seeds. The oracles are heuristic searches, so each is a lower bound on what its information allows.

![What the oracle needs to know](oracle_breakdown.png)

| Policy | Profit (INR/ha) | % of full oracle |
|---|---|---|
| oracle: weather and price known | 89,637 | 100.0% |
| oracle: weather known, average price | 89,151 | 99.5% |
| oracle: price known, typical weather | 83,972 | 93.7% |
| oracle: neither (best fixed plan) | 83,784 | 93.5% |
| threshold_tuned | 87,232 | 97.3% |
| recommended_tuned | 81,072 | 90.4% |
| PPO | 86,200 ± 542 | 96.2% |

Value of information over the best fixed plan (83,784): knowing the weather adds +5,367, knowing the price adds +188, knowing both adds +5,853 INR/ha.
