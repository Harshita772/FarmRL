"""Save every report figure of a run to results/runs/<name>/figures/ as PNG (300 dpi) and PDF.

    python scripts/make_figures.py --run v1
    python scripts/make_figures.py --run v1 --only season_timeline --season 2019
"""
import sys, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import matplotlib
matplotlib.use("Agg")
from farmrl import runs
from farmrl.viz import figures

p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
p.add_argument("--run", default=None, help="run name (default: newest run)")
p.add_argument("--only", nargs="*", choices=list(figures.FIGURES), help="draw just these figures")
p.add_argument("--season", type=int, default=None, help="season for the timeline figure")
p.add_argument("--baseline", default=None, help="baseline for the dry-vs-wet figure")
a = p.parse_args()

run = runs.get(a.run)
report = figures.make_all(run, only=a.only, season=a.season, baseline=a.baseline)
print(f"run '{run.name}':")
for name, result in report.items():
    print(f"  {name:<22} {result}")
