import sys, argparse, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from farmrl.env import FarmEnv
from farmrl.agents.ppo import PPO
from farmrl.baselines import BASELINES
from farmrl.evaluation import evaluate, summarise
from farmrl.utils import load_config

p = argparse.ArgumentParser()
p.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
p.add_argument("--model", default=str(ROOT / "results/models/ppo_latest.pt"))
p.add_argument("--split", default="test", choices=["train", "test", "all"])
p.add_argument("--seeds", type=int, default=None, help="replicates per season")
a = p.parse_args()
cfg = load_config(a.config)
env = FarmEnv(cfg["env"], split=a.split)       # default: held-out seasons (> train_end)

policies = dict(BASELINES)
if Path(a.model).exists():
    agent = PPO(env, cfg["train"]); agent.load(a.model)
    policies["ppo"] = lambda o, e: agent.predict(o)
episodes = evaluate(env, policies, seeds_per_season=a.seeds or cfg["eval"]["seeds_per_season"])
table = summarise(episodes)
print(f"{a.split} seasons: {env.seasons}")
print(f"{'policy':<14}{'profit INR/ha':>15}{'std':>9}{'yield q/ha':>12}{'N kg':>8}{'irr mm':>8}{'pest':>6}")
for k, r in table.iterrows():
    print(f"{k:<14}{r['profit']:>15.0f}{r['profit_std']:>9.0f}{r['yield']:>12.2f}"
          f"{r['n_total']:>8.0f}{r['irrigation_total']:>8.0f}{r['pesticide_total']:>6.1f}")
(ROOT / "results").mkdir(exist_ok=True)
episodes.to_csv(ROOT / "results/eval_episodes.csv", index=False)
json.dump(table.round(3).to_dict(orient="index"), open(ROOT / "results/eval.json", "w"), indent=2)
