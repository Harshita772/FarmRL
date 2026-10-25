"""District choropleth of a run: static PNG/PDF and interactive HTML in results/runs/<name>/maps/.

    python scripts/make_maps.py --run v1
    python scripts/make_maps.py --run v1 --metric ppo_water_saved_mm --policy threshold

Needs district_summary.csv in the run (run_experiments.py --districts) and the boundary
file from `python scripts/download_data.py geo`.
"""
import sys, argparse
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import matplotlib
matplotlib.use("Agg")
from farmrl import runs
from farmrl.viz import maps


def merge_district_rollouts(target_run="v4_haryana"):
    """Scan additional runs in results/runs/ and experiments/ for any newly evaluated district rows
    and merge them into results/runs/<target_run>/district_summary.csv without duplicating existing (policy, district) tuples."""
    target_dir = ROOT / "results" / "runs" / target_run
    target_csv = target_dir / "district_summary.csv"
    if not target_dir.exists():
        return
    target_df = pd.read_csv(target_csv) if target_csv.exists() else pd.DataFrame()
    initial_len = len(target_df)

    search_dirs = [ROOT / "results" / "runs", ROOT / "experiments"]
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for p in search_dir.rglob("district_summary.csv"):
            if p.resolve() == target_csv.resolve():
                continue
            try:
                cand = pd.read_csv(p)
                if "policy" not in cand.columns or "district" not in cand.columns:
                    continue
                if not target_df.empty:
                    existing = set(zip(target_df["policy"], target_df["district"]))
                    cand_keys = list(zip(cand["policy"], cand["district"]))
                    new_mask = [k not in existing for k in cand_keys]
                    new_rows = cand[new_mask]
                    if not new_rows.empty:
                        target_df = pd.concat([target_df, new_rows], ignore_index=True)
                else:
                    target_df = cand
            except Exception:
                pass

    if not target_df.empty:
        target_df = target_df.drop_duplicates(subset=["policy", "district"], keep="first")
        if len(target_df) != initial_len or not target_csv.exists():
            target_df.to_csv(target_csv, index=False)
            print(f"merged district rollouts into {target_csv.relative_to(ROOT)} ({len(target_df)} rows)")


p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
p.add_argument("--run", default=None, help="run name (default: newest run)")
p.add_argument("--metric", default="ppo_profit_gain", help=f"column of district_summary.csv, e.g. {list(maps.METRICS)}")
p.add_argument("--policy", default=None, help="baseline for gain metrics, else the policy shown")
a = p.parse_args()

target_run = a.run or "v4_haryana"
merge_district_rollouts(target_run)

try:
    for path in maps.make_maps(runs.get(a.run), a.metric, a.policy):
        print("saved", path)
except maps.MapUnavailable as why:
    print(f"map skipped: {why}")
