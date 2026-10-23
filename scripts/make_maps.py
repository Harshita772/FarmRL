"""District choropleth of a run: static PNG/PDF and interactive HTML in results/runs/<name>/maps/.

    python scripts/make_maps.py --run v1
    python scripts/make_maps.py --run v1 --metric ppo_water_saved_mm --policy threshold

Needs district_summary.csv in the run (run_experiments.py --districts) and the boundary
file from `python scripts/download_data.py geo`.
"""
import sys, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import matplotlib
matplotlib.use("Agg")
from farmrl import runs
from farmrl.viz import maps

p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
p.add_argument("--run", default=None, help="run name (default: newest run)")
p.add_argument("--metric", default="ppo_profit_gain", help=f"column of district_summary.csv, e.g. {list(maps.METRICS)}")
p.add_argument("--policy", default=None, help="baseline for gain metrics, else the policy shown")
a = p.parse_args()

try:
    for path in maps.make_maps(runs.get(a.run), a.metric, a.policy):
        print("saved", path)
except maps.MapUnavailable as why:
    print(f"map skipped: {why}")
