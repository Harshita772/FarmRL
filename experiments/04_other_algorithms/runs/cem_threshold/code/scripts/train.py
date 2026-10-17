import sys, argparse, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from farmrl.env import FarmEnv
from farmrl.agents.ppo import PPO
from farmrl.utils import load_config, set_seed

p = argparse.ArgumentParser()
p.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
p.add_argument("--timesteps", type=int, default=None)
a = p.parse_args()

cfg = load_config(a.config)
set_seed(cfg["env"]["seed"])
env = FarmEnv(cfg["env"], split="train")      # seasons <= train_end only
env.reset(seed=cfg["env"]["seed"])
agent = PPO(env, cfg["train"])
k = env.reward_scale / 1000.0                 # env reward units -> kINR
hist = agent.learn(a.timesteps or cfg["train"]["total_timesteps"],
                   log_fn=lambda i, r: print(f"iter {i:4d}  mean ep return {r * k:8.2f} kINR"))
(ROOT / "results/models").mkdir(parents=True, exist_ok=True)
agent.save(ROOT / "results/models/ppo_latest.pt")
json.dump([h * k for h in hist], open(ROOT / "results/train_history.json", "w"))
print("saved model + history")
