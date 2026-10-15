"""Train or load PPO, roll out every policy on all seasons and save a run directory.

    conda run -n TogAI python scripts/run_experiments.py --run-name v1 --multi-seed-train 3 --districts --sensitivity
    conda run -n TogAI python scripts/run_experiments.py --run-name smoke --quick

Training seeds run in a multiprocessing pool sized by the config's ``compute`` block
(``parallel_seeds`` processes, ``torch_threads`` threads each). See farmrl/experiment.py
for the files written to results/runs/<run-name>/.
"""
import sys, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from farmrl.experiment import run_experiment
from farmrl.utils import load_config

def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    p.add_argument("--run-name", default=None, help="folder under results/runs (default: timestamp)")
    p.add_argument("--seeds", type=int, default=None, help="rollout replicates per season")
    p.add_argument("--multi-seed-train", type=int, default=1, metavar="K",
                   help="train K PPO seeds; all are evaluated (mean and std), the first one is shown in detail")
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--load-model", default=None, help="skip training and use these weights")
    p.add_argument("--districts", action="store_true", help="also evaluate under each district's weather")
    p.add_argument("--sensitivity", action="store_true",
                   help="also evaluate with prices, costs, yield-model and mechanism parameters scaled by +-20%%")
    p.add_argument("--no-tune", action="store_true", help="skip the grid search of the rule baselines")
    p.add_argument("--tensorboard", action="store_true", help="write TensorBoard logs into the run")
    p.add_argument("--quick", action="store_true", help="few timesteps and seeds, for a smoke test")
    p.add_argument("--parallel-seeds", type=int, default=None,
                   help="training seeds run at once (default: compute.parallel_seeds in the config)")
    p.add_argument("--torch-threads", type=int, default=None,
                   help="torch threads per process (default: compute.torch_threads in the config)")
    a = p.parse_args()

    run_experiment(load_config(a.config), run_name=a.run_name, seeds=a.seeds,
                   train_seeds=a.multi_seed_train, timesteps=a.timesteps, load_model=a.load_model,
                   districts=a.districts, with_sensitivity=a.sensitivity, quick=a.quick,
                   tensorboard=a.tensorboard, parallel_seeds=a.parallel_seeds, torch_threads=a.torch_threads,
                   tune_baselines=False if a.no_tune else None)


# The guard matters: the training pool starts fresh processes that import this file.
if __name__ == "__main__":
    main()
