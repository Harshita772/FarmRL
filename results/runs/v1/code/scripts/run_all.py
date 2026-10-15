"""Rebuild everything with one command.

    conda run -n TogAI python scripts/run_all.py --run-name v2
    conda run -n TogAI python scripts/run_all.py --run-name smoke --quick
    conda run -n TogAI python scripts/run_all.py --run-name v2 --skip data experiments   # redraw only
    conda run -n TogAI python scripts/run_all.py --run-name haryana --config configs/haryana.yaml

Stages, in order (each is a script that can also be run on its own):
    data         scripts/prepare_data.py     raw -> processed tables, fit the yield model
    experiments  scripts/run_experiments.py  train PPO, roll out all policies -> results/runs/<name>/
    figures      scripts/make_figures.py     report figures (PNG + PDF)
    animation    scripts/make_animation.py   season GIF, copied to docs/demo.gif
    maps         scripts/make_maps.py        district choropleth (PNG + HTML)
A failing stage stops the pipeline; later stages only read what earlier ones saved, so
the pipeline can be resumed with --skip.
"""
import sys, argparse, subprocess, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
STAGES = ["data", "experiments", "figures", "animation", "maps"]

p = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                            epilog=__doc__.split("\n\n", 1)[1])
p.add_argument("--run-name", required=True, help="folder under results/runs/")
p.add_argument("--config", default=None, help="config file (default configs/default.yaml), e.g. configs/haryana.yaml")
p.add_argument("--quick", action="store_true", help="few timesteps and seeds (smoke test)")
p.add_argument("--skip", nargs="*", default=[], choices=STAGES, help="stages to skip")
p.add_argument("--only", nargs="*", default=None, choices=STAGES, help="run just these stages")
p.add_argument("--seeds", type=int, default=None, help="rollout replicates per season")
p.add_argument("--multi-seed-train", type=int, default=None, metavar="K", help="PPO training seeds (default 3, quick 2)")
p.add_argument("--compare", default="recommended", help="policy shown next to PPO in the animation")
p.add_argument("--no-demo", action="store_true", help="do not overwrite docs/demo.gif")
a = p.parse_args()

k = a.multi_seed_train or (2 if a.quick else 3)
experiments = ["run_experiments.py", "--run-name", a.run_name, "--multi-seed-train", str(k), "--districts", "--sensitivity"]
experiments += ["--quick"] if a.quick else []
experiments += ["--seeds", str(a.seeds)] if a.seeds else []
config = ["--config", a.config] if a.config else []
experiments += config
commands = {
    "data": ["prepare_data.py", *config],
    "experiments": experiments,
    "figures": ["make_figures.py", "--run", a.run_name],
    "animation": ["make_animation.py", "--run", a.run_name, "--compare", a.compare] + ([] if a.no_demo else ["--demo"]),
    "maps": ["make_maps.py", "--run", a.run_name],
}
todo = [s for s in STAGES if s not in a.skip and (a.only is None or s in a.only)]
for stage in todo:
    print(f"\n=== {stage} " + "=" * (60 - len(stage)), flush=True)
    t0 = time.time()
    code = subprocess.call([sys.executable, str(ROOT / "scripts" / commands[stage][0]), *commands[stage][1:]], cwd=ROOT)
    if code != 0:
        sys.exit(f"stage '{stage}' failed (exit code {code}); fix it and resume with --skip "
                 + " ".join(STAGES[:STAGES.index(stage)]))
    print(f"--- {stage} done in {time.time() - t0:.0f} s", flush=True)
print(f"\nrun '{a.run_name}' is in results/runs/{a.run_name}/ -- view it with: streamlit run app/streamlit_app.py")
