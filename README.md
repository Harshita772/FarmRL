# FarmRL — Reinforcement Learning for Crop Input Management

**Team:** Harshita Vachhani (B23EE1026), Aurindum Banerjee (B23CS1006)
**Course:** AI Assignment 2 — Real-Life Use Cases of RL

## Problem
Indian farmers must decide fertiliser, irrigation and pesticide doses under uncertain
weather and prices. We train an RL agent that observes crop, soil and market conditions
and chooses fortnightly nitrogen, irrigation and pesticide doses for one wheat season
(Ludhiana, Punjab), rewarding net profit per hectare and penalising wasteful or harmful
input use.

## RL Formulation
| Component | Description |
|---|---|
| State | stage, soil moisture, soil N, pest pressure, cumulative rain / N / water, price index, last fortnight's rain and ET0, share of water demand met |
| Action | MultiDiscrete: nitrogen {0, 20, 40 kg/ha} × irrigation {0, 37.5, 75 mm} × pesticide {0, 1, 2} |
| Reward | yield × price − fixed cost − input costs − environmental penalties |
| Step | 1 fortnight; 12 steps per season (1 Nov – end of April) |
| Episode | one real historical season: NASA POWER weather + that year's price index |
| Split | seasons sown ≤ 2016 for training, 2017+ held out |
| Algorithm | PPO, own implementation (baselines: recommended schedule, threshold rules, random, no-input) |

## Repository Structure
```
FarmRL/
├── configs/default.yaml      # environment, costs (CACP, cited), training hyperparameters
├── configs/fitted_params.yaml# yield-model parameters written by the fit
├── data/raw/                 # downloaded datasets (not committed)
├── data/processed/           # cleaned tables
├── data/reference/           # tables transcribed from published reports
├── src/farmrl/
│   ├── env.py                # Gymnasium environment (crop season MDP, train/test split)
│   ├── yield_model.py        # yield response function + fit()
│   ├── data_loader.py        # loading / cleaning weather, ICRISAT, prices, costs, FAO
│   ├── baselines.py          # rule-based comparison policies
│   ├── evaluation.py         # rollouts and result tables
│   ├── utils.py
│   └── agents/ppo.py         # PPO agent
├── scripts/                  # download / prepare / train / evaluate / plot
├── notebooks/FarmRL_main.ipynb  # full pipeline with explanations and figures
├── results/                  # figures (300 dpi), trained models, logs
├── report/                   # 2-page LaTeX report
└── tests/
```

## Setup
Everything runs in the existing conda environment `TogAI`:
```bash
conda run -n TogAI python -m pip install -r requirements.txt
conda run -n TogAI python -m ipykernel install --user --name TogAI
```

## Usage
```bash
conda run -n TogAI python scripts/download_data.py      # fetch raw data; prints manual steps for what it cannot fetch
conda run -n TogAI python scripts/prepare_data.py       # build data/processed, fit the yield model -> configs/fitted_params.yaml
conda run -n TogAI python scripts/train.py              # PPO on the training seasons (~6 min on CPU)
conda run -n TogAI python scripts/evaluate.py           # all policies on the held-out seasons
conda run -n TogAI python scripts/plot_results.py       # training curve + comparison figure
conda run -n TogAI python -m pytest                     # tests (env, yield model, data loaders)

# the whole pipeline as a notebook, executed in place:
conda run -n TogAI python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 --ExecutePreprocessor.kernel_name=togai notebooks/FarmRL_main.ipynb
```
`download_data.py` accepts step names (`weather kaggle icrisat msp costs prices`) to run
only some of them. Optional environment variables: `DATA_GOV_IN_API_KEY`,
`DATA_GOV_IN_COC_RESOURCE`, `DATA_GOV_IN_AGMARKNET_RESOURCE`.

## Data Sources
See `data/README.md` for sources, licences, dates and the datasets that still need a
manual download.
