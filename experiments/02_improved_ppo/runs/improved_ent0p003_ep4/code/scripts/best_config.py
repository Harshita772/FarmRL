"""Print the `--set` arguments of the best study-2 configuration (highest mean held-out
profit over training seeds), for the driver script to reuse in studies 3-5.

    python scripts/best_config.py ../02_improved_ppo/runs
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from farmrl.study import run_row
from farmrl.runs import Run

rows = []
for meta in sorted(Path(sys.argv[1]).glob("improved_*/metadata.json")):
    rows.append((run_row(meta.parent)["profit_mean"], Run(meta.parent)))
if not rows:
    sys.exit("no finished study-2 runs found")
profit, run = max(rows, key=lambda r: r[0])
train = run.config["train"]
print(f"best study-2 run: {run.name} ({profit:,.0f} INR/ha)", file=sys.stderr)
print(f"--set train.ent_coef={train['ent_coef']} --set train.epochs={train['epochs']}")
