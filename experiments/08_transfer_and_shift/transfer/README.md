# Study 8 — Transfer across regions and stress tests

**Question.** Does one learned policy adapt to a new region or shifted conditions better than a rule that has to be retuned for each setting?

Nothing is retrained here. `PPO trained elsewhere` and `rule tuned elsewhere` are the Punjab-tuned agent and rule (or the Haryana ones for the reverse direction) applied unchanged. `Rule tuned here` is the threshold rule retuned on the training seasons of the scenario (the rule's best case); `PPO trained here` is the agent trained on that region. The oracle knows each held-out season in advance. Held-out seasons 2017–2023, 30 pest replicates, 5 PPO training seeds (± is std over seeds).

Shifted conditions are assumptions, not data: Each function returns a modified copy of the weather / price table (or env config) that is handed to ``FarmEnv`` in place of the processed one, so the same trained policy can be replayed under conditions it was not trained on. The physical assumptions are stated here because they are assumptions: ET0 rises 4% per degree of warming (the order of the sensitivity of temperature-based ET0 formulas), and the daily maximum temperature within a fortnight is taken to be normal with a standard deviation of 3.5 degrees around the fortnight's mean maximum, which turns a shift of the mean into extra heat-stress days (days above 30 C between 15 February and 31 March, as in data_loader).

![Transfer and stress tests](transfer_summary.png)

## Punjab → Haryana

| Policy | Profit (INR/ha) | % of oracle | Gap to rule tuned here |
|---|---|---|---|
| rule tuned elsewhere, unchanged | 73,831 | 95.6% | +0 |
| rule tuned here | 73,831 | 95.6% | – |
| hindsight oracle | 77,245 | 100.0% | – |
| PPO trained elsewhere, unchanged | 74,303 ± 463 | 96.2% | +472 |
| PPO trained here | 73,778 ± 847 | 95.5% | -53 |

## Haryana → Punjab

| Policy | Profit (INR/ha) | % of oracle | Gap to rule tuned here |
|---|---|---|---|
| rule tuned elsewhere, unchanged | 87,232 | 97.3% | +0 |
| rule tuned here | 87,232 | 97.3% | – |
| hindsight oracle | 89,654 | 100.0% | – |
| PPO trained elsewhere, unchanged | 85,908 ± 680 | 95.8% | -1,324 |
| PPO trained here | 86,200 ± 542 | 96.1% | -1,032 |

## Punjab, as trained

| Policy | Profit (INR/ha) | % of oracle | Gap to rule tuned here |
|---|---|---|---|
| rule tuned elsewhere, unchanged | 87,232 | 97.3% | +0 |
| rule tuned here | 87,232 | 97.3% | – |
| hindsight oracle | 89,654 | 100.0% | – |
| PPO trained elsewhere, unchanged | 86,200 ± 542 | 96.1% | -1,032 |

## +1 °C

| Policy | Profit (INR/ha) | % of oracle | Gap to rule tuned here |
|---|---|---|---|
| rule tuned elsewhere, unchanged | 85,387 | 96.7% | +361 |
| rule tuned here | 85,027 | 96.3% | – |
| hindsight oracle | 88,332 | 100.0% | – |
| PPO trained elsewhere, unchanged | 85,418 ± 164 | 96.7% | +391 |

## +2 °C

| Policy | Profit (INR/ha) | % of oracle | Gap to rule tuned here |
|---|---|---|---|
| rule tuned elsewhere, unchanged | 83,870 | 96.2% | +0 |
| rule tuned here | 83,870 | 96.2% | – |
| hindsight oracle | 87,196 | 100.0% | – |
| PPO trained elsewhere, unchanged | 83,773 ± 422 | 96.1% | -97 |

## 20% less rain

| Policy | Profit (INR/ha) | % of oracle | Gap to rule tuned here |
|---|---|---|---|
| rule tuned elsewhere, unchanged | 86,856 | 97.3% | +151 |
| rule tuned here | 86,705 | 97.1% | – |
| hindsight oracle | 89,250 | 100.0% | – |
| PPO trained elsewhere, unchanged | 86,867 ± 306 | 97.3% | +162 |

## 2x price volatility

| Policy | Profit (INR/ha) | % of oracle | Gap to rule tuned here |
|---|---|---|---|
| rule tuned elsewhere, unchanged | 87,773 | 98.0% | +0 |
| rule tuned here | 87,773 | 98.0% | – |
| hindsight oracle | 89,600 | 100.0% | – |
| PPO trained elsewhere, unchanged | 86,579 ± 417 | 96.6% | -1,194 |

## +2 °C and 20% less rain

| Policy | Profit (INR/ha) | % of oracle | Gap to rule tuned here |
|---|---|---|---|
| rule tuned elsewhere, unchanged | 84,641 | 97.5% | -60 |
| rule tuned here | 84,701 | 97.6% | – |
| hindsight oracle | 86,809 | 100.0% | – |
| PPO trained elsewhere, unchanged | 84,187 ± 521 | 97.0% | -514 |
