"""Animate one season of a run as a GIF (and MP4 if ffmpeg is installed).

    python scripts/make_animation.py --run v1 --season 2019 --compare recommended
    python scripts/make_animation.py --run v1 --compare ppo_untrained --demo

Without --season the held-out season where the agent gains most over --compare is used.
Files go to results/runs/<name>/animations/; --demo also copies the GIF to docs/demo.gif.
"""
import sys, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import matplotlib
matplotlib.use("Agg")
from farmrl import runs
from farmrl.viz import animate

p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
p.add_argument("--run", default=None, help="run name (default: newest run)")
p.add_argument("--season", type=int, default=None)
p.add_argument("--policy", default="ppo")
p.add_argument("--compare", default=None, help="second policy shown side by side, e.g. recommended or ppo_untrained")
p.add_argument("--fps", type=int, default=6)
p.add_argument("--demo", action="store_true", help="copy the GIF to docs/demo.gif")
a = p.parse_args()

run = runs.get(a.run)
known = run.policies(include_aux=True)
for name in filter(None, [a.policy, a.compare]):
    if name not in known:
        sys.exit(f"policy '{name}' is not in run '{run.name}'; choose from {known}")
for path in animate.save_animation(run, a.season, a.policy, a.compare, fps=a.fps,
                                   demo_path=ROOT / "docs/demo.gif" if a.demo else None):
    print("saved", path)
