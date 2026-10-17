"""Wall-clock probe: training seconds per 100k environment steps of PPO and the GRPO variants.

    python scripts/time_probe.py --out ../11_grpo_vs_ppo/timing --steps 100000

Each agent is trained for ``--steps`` steps by 4 processes at once (1 torch thread each, as in
the experiment runs, so the figure includes the contention of a pool of 4), and the mean
seconds are scaled to 1M steps. The probe does not evaluate anything.
"""
import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy  # noqa: F401  numpy before torch: otherwise MKL and libgomp clash on this server and the workers die
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

AGENTS = {
    "improved PPO": ["train.variant=improved", "train.ent_coef=0.003", "train.epochs=10"],
    "GRPO, group 8": ["train.variant=improved", "train.algo=grpo", "train.group_size=8", "train.ent_coef=0.003", "train.epochs=10"],
    "GRPO, group 16": ["train.variant=improved", "train.algo=grpo", "train.group_size=16", "train.ent_coef=0.003", "train.epochs=10"],
    "GRPO, group 8, stepwise": ["train.variant=improved", "train.algo=grpo", "train.group_size=8", "train.adv_mode=stepwise",
                                "train.ent_coef=0.003", "train.epochs=10"],
}


def _job(args):
    name, sets, steps, seed = args
    import numpy  # noqa: F401
    import torch
    from farmrl.agents import build_agent
    from farmrl.env import FarmEnv
    from farmrl.utils import apply_overrides, load_config, set_seed
    torch.set_num_threads(1)
    cfg = load_config(ROOT / "configs/default.yaml")
    apply_overrides(cfg, sets)
    set_seed(seed)
    env = FarmEnv(cfg["env"], split="train")
    env.reset(seed=seed)
    agent = build_agent(env, cfg["train"])
    t0 = time.time()
    agent.learn(steps)
    return name, time.time() - t0


def main():
    import pandas as pd
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--procs", type=int, default=4)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, sets in AGENTS.items():
        with mp.get_context("spawn").Pool(a.procs) as pool:
            times = [t for _, t in pool.map(_job, [(name, sets, a.steps, 100 + k) for k in range(a.procs)])]
        rows.append({"agent": name, "steps": a.steps, "processes": a.procs, "seconds_mean": sum(times) / len(times),
                     "seconds_per_1M": sum(times) / len(times) * 1_000_000 / a.steps,
                     "minutes_per_1M": sum(times) / len(times) * 1_000_000 / a.steps / 60})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(out / "time_probe.csv", index=False)


if __name__ == "__main__":
    main()
