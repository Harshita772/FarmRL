"""Train (or load) the agent, roll every policy out and save a self-contained run.

``run_experiment`` writes results/runs/<run_name>/ with
    model.pt                trained PPO weights (first training seed)
    train_history.csv       per training seed and iteration: return, losses, entropy, KL
    rollouts.parquet        one row per step for every policy x season x seed (rollout.py)
    episodes.csv            one row per episode
    summary.csv             one row per policy and split
    policy_grids.parquet    greedy / expected action over (soil moisture x stage) and (soil N x stage)
    state_medians.csv       per-stage median observation used to fill the other grid dimensions
    district_summary.csv    optional: test-season results under each district's weather
    sensitivity.csv         optional: test-season results under scaled prices and costs
    prices.csv              the season price index the environment used
    config_snapshot.yaml, fitted_params.yaml, metadata.json
"""
import json
import multiprocessing as mp
import platform
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from . import data_loader as dl
from . import rollout
from .agents.ppo import PPO
from .baselines import BASELINES
from .env import FarmEnv, OBS_NAMES, scale_config
from .runs import RUNS_DIR
from .utils import set_seed
from .yield_model import YieldModel

ROOT = Path(__file__).resolve().parents[2]
AUX_POLICIES = ["ppo_untrained"]          # rolled out for comparison, left out of rankings
QUICK = {"timesteps": 4 * 2048, "seeds": 2}
SENSITIVITY = {"price": [0.8, 1.2], "nitrogen_cost": [0.5, 2.0], "irrigation_cost": [0.5, 5.0],
               "pesticide_cost": [0.5, 2.0], "penalty": [0.0, 3.0]}


def env_maker(yield_model, weather=None, prices=None):
    """``make(cfg_env, split) -> FarmEnv``. ``weather`` / ``prices`` replace the processed
    tables (used by tests and for experiments on other data)."""
    return lambda cfg_env, split: FarmEnv(cfg_env, weather=weather, prices=prices, split=split,
                                          yield_model=yield_model)


def agent_policy(agent):
    return lambda obs, env: agent.predict(obs)


def train_agent(cfg, seed, timesteps, make, tensorboard_dir=None, log_fn=None):
    """One PPO agent trained on the train seasons. Returns (agent, per-iteration table)."""
    set_seed(seed)
    env = make(cfg["env"], "train")
    env.reset(seed=seed)
    agent = PPO(env, cfg["train"])
    agent.learn(timesteps, log_fn=log_fn, tensorboard_dir=tensorboard_dir)
    log = pd.DataFrame(agent.log)
    log["mean_return"] *= env.reward_scale            # INR per hectare
    log.insert(0, "train_seed", seed)
    return agent, log


def _train_worker(job):
    """Pool worker: train one seed and send back its weights and history. Runs in a fresh
    process (spawn on Windows), so everything it needs arrives in ``job``."""
    cfg, seed, timesteps, yield_params, weather, prices, tb_dir, threads = job
    torch.set_num_threads(threads)
    make = env_maker(YieldModel(**yield_params), weather, prices)
    agent, log = train_agent(cfg, seed, timesteps, make, tensorboard_dir=tb_dir)
    return seed, agent.net.state_dict(), log


def train_in_pool(cfg, seeds, timesteps, yield_model, weather=None, prices=None, tb_root=None,
                parallel=1, threads=1, log=print):
    """Train one agent per seed, up to ``parallel`` at a time in a multiprocessing pool.
    Returns ({seed: state_dict}, history table). Each seed's result is the same whether
    it ran alone or in the pool: every worker seeds itself and uses ``threads`` threads."""
    jobs = [(cfg, s, timesteps, yield_model.params(), weather, prices,
             None if tb_root is None else tb_root / f"seed_{s}", threads) for s in seeds]
    workers = max(1, min(int(parallel), len(jobs)))
    log(f"training PPO seeds {list(seeds)} for {timesteps} steps each "
        f"({workers} process{'es' * (workers > 1)}, {threads} torch thread{'s' * (threads > 1)} each)")
    if workers == 1:
        results = [_train_worker(job) for job in jobs]
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            results = pool.map(_train_worker, jobs)
    weights = {seed: state for seed, state, _ in results}
    return weights, pd.concat([table for _, _, table in results], ignore_index=True)


def all_policies(agent, cfg, make):
    """PPO, an untrained copy of it, and every baseline registered in BASELINES."""
    set_seed(cfg["env"]["seed"] + 999)
    blank = PPO(make(cfg["env"], "train"), cfg["train"])
    return {"ppo": agent_policy(agent), **BASELINES, "ppo_untrained": agent_policy(blank)}


def collect(cfg_env, policies, make, seeds, splits=("train", "test")):
    parts = [rollout.run_many(make(cfg_env, s), policies, seeds=seeds) for s in splits]
    return pd.concat(parts, ignore_index=True)


def policy_grids(agent, env, reference, points=21):
    """Agent's action over a grid of one state variable x season stage.

    The remaining observation entries are held at their per-stage medians over
    ``reference`` (step rows of the trained agent), because cumulative quantities such as
    rain or nitrogen so far only make sense relative to the stage. Returns (grid rows,
    per-stage medians)."""
    obs_cols = rollout.OBS_COLS
    medians = reference.groupby("step")[obs_cols].median()
    axes = {"soil_moisture": (np.linspace(0, 1, points), 1.0, "fraction of capacity"),
            "soil_n": (np.linspace(0, 120, points), 100.0, "kg N/ha")}
    levels = [env.n_levels, env.w_levels, env.p_levels]
    names = ["nitrogen_kg", "irrigation_mm", "pesticide_level"]
    rows = []
    for var, (values, scale, unit) in axes.items():
        j = OBS_NAMES.index(var)
        for step in medians.index:
            for v in values:
                obs = medians.loc[step].to_numpy(np.float32).copy()
                obs[j] = v / scale
                probs = agent.action_probs(obs)
                row = {"grid": var, "unit": unit, "step": int(step), "value": float(v)}
                for name, lv, p in zip(names, levels, probs):
                    row[name] = float(lv[int(np.argmax(p))])          # greedy action
                    row[f"expected_{name}"] = float((lv * p).sum())   # mean under the policy
                rows.append(row)
    return pd.DataFrame(rows), medians.reset_index()


def district_summary(cfg_env, policies, make, seeds):
    """Test-season results when the same policies face each district's own weather."""
    rows = []
    for site, meta in dl.SITES.items():
        if not (dl.PROCESSED / f"weather_fortnightly_{site}.csv").exists():
            continue
        env = make(scale_config(cfg_env, weather_site=site), "test")
        eps = rollout.episodes(rollout.run_many(env, policies, seeds=seeds))
        table = rollout.summary(eps).assign(site=site, district=meta["district"], state=meta["state"],
                                            lat=meta["lat"], lon=meta["lon"])
        ppo = table.loc[table["policy"] == "ppo"].iloc[0]
        table["ppo_profit_gain"] = ppo["profit"] - table["profit"]
        table["ppo_water_saved_mm"] = table["irrigation_mm"] - ppo["irrigation_mm"]
        rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else None


def sensitivity(cfg_env, policies, make, seeds):
    """Test-season results with one price / cost / penalty multiplier changed at a time."""
    cases = [("baseline", 1.0, {})] + [(k, v, {k: v}) for k, vs in SENSITIVITY.items() for v in vs]
    rows = []
    for factor, value, kw in cases:
        env = make(scale_config(cfg_env, **kw), "test")
        eps = rollout.episodes(rollout.run_many(env, policies, seeds=seeds))
        rows.append(rollout.summary(eps).assign(factor=factor, multiplier=value))
    return pd.concat(rows, ignore_index=True)


def _git_hash():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def _data_files():
    out = []
    for folder in (dl.RAW, dl.PROCESSED, dl.REFERENCE):
        for p in sorted(folder.glob("*")):
            if p.is_file() and p.name != ".gitkeep":
                out.append({"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size,
                            "modified": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")})
    return out


def run_experiment(cfg, run_name=None, seeds=None, train_seeds=1, timesteps=None, load_model=None,
                   districts=False, with_sensitivity=False, quick=False, tensorboard=False,
                   runs_dir=None, log=print, weather=None, prices=None, yield_model=None,
                   parallel_seeds=None, torch_threads=None):
    """Run the whole experiment and return the run directory. ``weather``, ``prices`` and
    ``yield_model`` override what the config points to (per-district evaluation is skipped
    with a weather override, since it reads each district's processed weather).
    ``parallel_seeds`` and ``torch_threads`` default to the config's ``compute`` block."""
    created = datetime.now()
    run_name = run_name or created.strftime("run_%Y%m%d_%H%M%S")
    out = Path(runs_dir or RUNS_DIR) / run_name
    out.mkdir(parents=True, exist_ok=True)
    seeds = seeds or (QUICK["seeds"] if quick else cfg["eval"]["seeds_per_season"])
    timesteps = timesteps or (QUICK["timesteps"] if quick else cfg["train"]["total_timesteps"])
    ym = yield_model or YieldModel.from_config(ROOT / cfg["env"]["fitted_params"])
    make = env_maker(ym, weather, prices)
    base_seed = cfg["env"]["seed"]
    compute = cfg.get("compute", {})
    threads = int(torch_threads or compute.get("torch_threads", 1))
    parallel = int(parallel_seeds or compute.get("parallel_seeds", 1))
    torch.set_num_threads(threads)

    # --- agent(s) ---
    if load_model:
        agent = PPO(make(cfg["env"], "train"), cfg["train"])
        agent.load(load_model)
        history = None
        log(f"loaded model {load_model}")
    else:
        seeds_to_train = [base_seed + k for k in range(max(1, train_seeds))]
        weights, history = train_in_pool(cfg, seeds_to_train, timesteps, ym, weather, prices,
                                       out / "tensorboard" if tensorboard else None, parallel, threads, log)
        agent = PPO(make(cfg["env"], "train"), cfg["train"])
        agent.net.load_state_dict(weights[base_seed])        # the first seed is the one evaluated
        history.to_csv(out / "train_history.csv", index=False)
    agent.save(out / "model.pt")

    # --- rollouts and tables ---
    policies = all_policies(agent, cfg, make)
    log(f"rolling out {list(policies)} with {seeds} seeds per season")
    rollouts = collect(cfg["env"], policies, make, seeds)
    rollouts.to_parquet(out / "rollouts.parquet", index=False)
    eps = rollout.episodes(rollouts)
    eps.to_csv(out / "episodes.csv", index=False)
    rollout.summary(eps).to_csv(out / "summary.csv", index=False)

    env = make(cfg["env"], "all")
    reference = rollouts[rollouts["policy"] == "ppo"]
    grids, medians = policy_grids(agent, env, reference)
    grids.to_parquet(out / "policy_grids.parquet", index=False)
    medians.to_csv(out / "state_medians.csv", index=False)

    ranked = {k: v for k, v in policies.items() if k not in AUX_POLICIES}
    extra_seeds = max(2, seeds // 3)               # the what-if tables need less precision
    if districts and weather is None:
        log("evaluating per district")
        table = district_summary(cfg["env"], ranked, make, extra_seeds)
        if table is not None:
            table.to_csv(out / "district_summary.csv", index=False)
    if with_sensitivity:
        log("running sensitivity cases")
        sensitivity(cfg["env"], ranked, make, extra_seeds).to_csv(out / "sensitivity.csv", index=False)

    # --- provenance ---
    pd.DataFrame({"season": list(env.price_index), "price_index": list(env.price_index.values())}).to_csv(
        out / "prices.csv", index=False)
    (out / "config_snapshot.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    ym.save(out / "fitted_params.yaml", source=f"copied from {cfg['env']['fitted_params']}")
    fitted = ROOT / cfg["env"]["fitted_params"]
    meta = {
        "run_name": run_name, "created": created.isoformat(timespec="seconds"),
        "git_hash": _git_hash(), "quick": bool(quick), "timesteps": int(timesteps),
        "train_seeds": [] if history is None else sorted(int(s) for s in history["train_seed"].unique()),
        "loaded_model": str(load_model) if load_model else None,
        "rollout_seeds": int(seeds), "policies": list(policies), "aux_policies": AUX_POLICIES,
        "compute": {"torch_threads": threads, "parallel_seeds": parallel},
        "weather_site": cfg["env"]["weather_site"],
        "train_seasons": env.train_seasons, "test_seasons": env.test_seasons,
        "yield_model": ym.params(),
        "yield_fit": (yaml.safe_load(fitted.read_text()).get("fit") if fitted.exists() else None),
        "data_files": _data_files(),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    log(f"run saved to {out}")
    return out
