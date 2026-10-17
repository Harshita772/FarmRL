import sys, argparse, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from farmrl.utils import save_fig

h = json.load(open(ROOT / "results/train_history.json"))
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(h); ax.set_xlabel("PPO iteration"); ax.set_ylabel("mean episode return (kINR/ha)")
ax.set_title("Training curve")
save_fig(fig, "training_curve"); plt.close(fig)

ev = json.load(open(ROOT / "results/eval.json"))
names = list(ev); means = [ev[k]["profit"] / 1000 for k in names]; stds = [ev[k]["profit_std"] / 1000 for k in names]
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.bar(names, means, yerr=stds, capsize=4); ax.set_ylabel("net profit (kINR/ha)")
ax.set_title("Policy comparison, held-out seasons")
save_fig(fig, "policy_comparison"); plt.close(fig)
print("figures saved to results/figures")
