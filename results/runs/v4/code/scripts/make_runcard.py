"""Write the run card (README.md), params/ copies and results/runs/INDEX.md.

    conda run -n TogAI python scripts/make_runcard.py --run v4          # one run
    conda run -n TogAI python scripts/make_runcard.py --all             # every run + the index
    conda run -n TogAI python scripts/make_runcard.py --run v1 --code-from 5458344 --reason "..."

New runs save their code snapshot themselves; --code-from rebuilds code/ from a git commit
for runs made before that, and labels it as reconstructed.
"""
import sys, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from farmrl import runcard, runs

p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
g = p.add_mutually_exclusive_group(required=True)
g.add_argument("--run", help="run name")
g.add_argument("--all", action="store_true", help="every run in results/runs, then INDEX.md")
p.add_argument("--code-from", default=None, help="git commit to rebuild code/ from (old runs only)")
p.add_argument("--reason", default="nearest commit containing the code the run was made with")
a = p.parse_args()

names = runs.list_runs() if a.all else [a.run]
for name in names:
    run = runs.get(name)
    if a.code_from:
        runcard.snapshot_from_git(run.path, a.code_from, a.reason)
    print("card:", runcard.write_card(run).relative_to(ROOT))
print("index:", runcard.write_index().relative_to(ROOT))
