"""Train (or load) the agent, roll every policy out and save a self-contained run.

``run_experiment`` writes results/runs/<run_name>/ with
    model.pt                trained PPO weights (first training seed)
    models/seed_<s>.pt      weights of every training seed
    train_history.csv       per training seed and iteration: return, losses, entropy, KL
    baseline_tuning.json    rule baselines tuned on the training seasons (farmrl.tuning)
    rollouts.parquet        one row per step for every policy x season x seed (rollout.py)
    episodes.csv            one row per episode
    summary.csv             one row per policy and split
    seed_episodes.csv       one row per episode of every PPO training seed
    seed_summary.csv        one row per PPO training seed and split
    attribution.csv         PPO's gain over each rule baseline split by decision (Shapley)
    dry_wet.csv             profit in seasons drier / wetter than the median, per policy and split
    policy_grids.parquet    greedy / expected action over (soil moisture x stage) and (soil N x stage)
    state_medians.csv       per-stage median observation used to fill the other grid dimensions
    district_summary.csv    optional: test-season results under each district's weather
    sensitivity.csv         optional: test-season results with one input scaled by +-20%
    prices.csv              the season price index the environment used
    config_snapshot.yaml, fitted_params.yaml, metadata.json
    code/                   snapshot of src/farmrl, scripts, configs and requirements.txt (runcard.py)
"""
import copy
import itertools
import json
import math
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
from . import oracle as oracle_mod
from . import rollout, runcard, tuning
from .agents import build_agent
from .baselines import BASELINES, FAMILIES, TUNED_SUFFIX, recommended_schedule, tuned_policies
from .env import FarmEnv, OBS_NAMES, calibrate_yield_scale, scale_config
from .runs import RUNS_DIR
from .utils import set_seed
from .yield_model import YieldModel

ROOT = Path(__file__).resolve().parents[2]
AUX_POLICIES = ["ppo_untrained"]          # rolled out for comparison, left out of rankings
QUICK = {"timesteps": 4 * 2048, "seeds": 2}
# One factor at a time is multiplied by each of LEVELS and every policy re-evaluated on the
# test seasons (nothing is retrained or retuned). group -> {factor: how it is applied}:
# a keyword of env.scale_config, a YieldModel attribute, or a "block.key" of the env config.
LEVELS = (0.8, 1.2)
SENSITIVITY = {
    "price": {"price": "config"},
    "costs": {"nitrogen_cost": "config", "irrigation_cost": "config", "pesticide_cost": "config",
              "fixed_cost": "config", "penalty": "config"},
    "yield model": {"y_max": "yield", "n_coef": "yield", "ky": "yield", "heat_coef": "yield", "pest_loss": "yield"},
    "mechanisms": {"groundwater.specific_yield": "path", "waterlogging.drain_mm_per_day": "path",
                   "pest_resistance.efficacy_decay": "path", "storage.monthly_drift": "path",
                   "storage.monthly_volatility": "path"},
}
DECISIONS = ["nitrogen", "water", "pests", "sale"]     # the action heads, in order


def env_maker(yield_model, weather=None, prices=None):
    """``make(cfg_env, split, yield_model=None) -> FarmEnv``. ``weather`` / ``prices`` replace
    the processed tables (used by tests and for experiments on other data)."""
    return lambda cfg_env, split, ym=None: FarmEnv(cfg_env, weather=weather, prices=prices, split=split,
                                                   yield_model=ym or yield_model)


def agent_policy(agent):
    return lambda obs, env: agent.predict(obs, env)


def train_agent(cfg, seed, timesteps, make, tensorboard_dir=None, log_fn=None):
    """One agent (``train.algo``, PPO by default) trained on the train seasons.
    Returns (agent, per-iteration table)."""
    set_seed(seed)
    env = make(cfg["env"], "train")
    env.reset(seed=seed)
    agent = build_agent(env, cfg["train"])
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
    return seed, agent.state(), log


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


def all_policies(agent, cfg, make, tuned=None):
    """PPO, an untrained copy of it, every baseline registered in BASELINES and the tuned
    rule baselines (``tuned``: a farmrl.tuning record)."""
    set_seed(cfg["env"]["seed"] + 999)
    blank = build_agent(make(cfg["env"], "train"), cfg["train"])
    return {"ppo": agent_policy(agent), **BASELINES, **tuned_policies(tuned),
            "ppo_untrained": agent_policy(blank)}


def collect(cfg_env, policies, make, seeds, splits=("train", "test")):
    parts = [rollout.run_many(make(cfg_env, s), policies, seeds=seeds) for s in splits]
    return pd.concat(parts, ignore_index=True)


def policy_grids(agent, env, reference, points=21):
    """Agent's action over a grid of one state variable x season stage.

    The remaining observation entries are held at their per-stage medians over
    ``reference`` (step rows of the trained agent), because cumulative quantities such as
    rain or nitrogen so far only make sense relative to the stage. Returns (grid rows,
    per-stage medians)."""
    obs_cols = [f"obs_{k}" for k in env.obs_names]
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


def sensitivity_cases(cfg_env, yield_model, levels=LEVELS):
    """[(group, factor, multiplier, env config, yield model)], the unchanged case first.
    Mechanism factors are left out when their mechanism is switched off."""
    cases = [("baseline", "baseline", 1.0, cfg_env, yield_model)]
    for group, factors in SENSITIVITY.items():
        for factor, kind in factors.items():
            if kind == "path" and not (cfg_env.get(factor.split(".")[0]) or {}).get("enabled"):
                continue
            for k in levels:
                cfg, ym = cfg_env, yield_model
                if kind == "config":
                    cfg = scale_config(cfg_env, **{factor: k})
                elif kind == "path":
                    cfg = scale_config(cfg_env, scale={factor: k})
                else:
                    ym = copy.copy(yield_model)
                    setattr(ym, factor, getattr(yield_model, factor) * k)
                cases.append((group, factor, k, cfg, ym))
    return cases


def sensitivity(cfg_env, policies, make, seeds, yield_model):
    """Test-season results with one price, cost, penalty, yield-model parameter or mechanism
    parameter multiplied by each of LEVELS at a time. Policies are evaluated as they are."""
    rows = []
    for group, factor, value, cfg, ym in sensitivity_cases(cfg_env, yield_model):
        eps = rollout.episodes(rollout.run_many(make(cfg, "test", ym), policies, seeds=seeds))
        rows.append(rollout.summary(eps).assign(group=group, factor=factor, multiplier=value))
    return pd.concat(rows, ignore_index=True)


# ------------------------------------------------------------------ where the gain comes from
def hybrid_policy(agent, base, use_agent):
    """Policy that takes the decisions flagged in ``use_agent`` (one bool per action head)
    from ``agent`` and the others from ``base``, both asked in the state actually reached."""
    mask = np.asarray(use_agent, dtype=bool)

    def pad(action):
        a = np.asarray(action).astype(int)[:len(mask)]
        return np.concatenate([a, np.zeros(len(mask) - len(a), dtype=int)])

    return lambda obs, env: np.where(mask, pad(agent(obs, env)), pad(base(obs, env)))


def attribute_gain(env, agent, base, seeds, objective="profit"):
    """Split the agent's mean gain over ``base`` on ``env.seasons`` between the decisions.

    Every mix of "agent decides / baseline decides" over the action heads is played (2^k
    hybrid policies) and each decision gets its Shapley value: its average marginal effect
    over all orders in which the baseline's decisions can be handed to the agent. The
    values add up to the total gain exactly. Returns one row per decision with
    ``contribution``, ``alone`` (only this decision handed over) and ``last`` (this one
    handed over after all the others)."""
    k = len(env.action_space.nvec)
    value = {}
    for mask in itertools.product([False, True], repeat=k):
        value[mask] = tuning.score(env, hybrid_policy(agent, base, mask), env.seasons, seeds,
                                   objective, seed0=10_000)
    none, full = (False,) * k, (True,) * k
    rows = []
    for i, name in enumerate(DECISIONS[:k]):
        phi = 0.0
        for mask in value:
            if mask[i]:
                continue
            size = sum(mask)
            with_i = mask[:i] + (True,) + mask[i + 1:]
            phi += math.factorial(size) * math.factorial(k - size - 1) / math.factorial(k) * (value[with_i] - value[mask])
        only = none[:i] + (True,) + none[i + 1:]
        without = full[:i] + (False,) + full[i + 1:]
        rows.append({"decision": name, "contribution": phi, "alone": value[only] - value[none],
                     "last": value[full] - value[without], "gain": value[full] - value[none],
                     "agent": value[full], "baseline": value[none]})
    return pd.DataFrame(rows)


def dry_wet_table(eps, seed_eps=None):
    """Mean profit per policy and split in seasons drier / wetter than the median season
    rain of the run. PPO rows also carry the std over training seeds when ``seed_eps``
    (episodes of every seed) is given."""
    rain = eps.groupby("season")["rain_mm"].first()
    median = float(rain.median())
    kind = lambda d: np.where(d["rain_mm"] > median, "wet", "dry")
    table = (eps.assign(season_type=kind).groupby(["policy", "split", "season_type"], sort=False)
             .agg(profit=("profit", "mean"), irrigation_mm=("irrigation_mm", "mean"),
                  yield_q_ha=("yield_q_ha", "mean"), seasons=("season", "nunique")).reset_index())
    table["median_rain_mm"] = median
    table["profit_seed_mean"] = table["profit_seed_std"] = np.nan
    if seed_eps is not None and len(seed_eps):
        per_seed = (seed_eps.assign(season_type=kind)
                    .groupby(["split", "season_type", "train_seed"])["profit"].mean()
                    .groupby(["split", "season_type"]).agg(["mean", "std"]))
        for (split, season_type), r in per_seed.iterrows():
            row = (table["policy"] == "ppo") & (table["split"] == split) & (table["season_type"] == season_type)
            table.loc[row, ["profit_seed_mean", "profit_seed_std"]] = [r["mean"], r["std"]]
    return table


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


def apply_yield_overrides(cfg, ym, overrides):
    """Set yield-model parameters (``{"pest_loss": 0.35}``) and recalibrate ``y_scale`` so the
    recommended calendar still averages the observed yield on the training seasons."""
    for key, value in (overrides or {}).items():
        setattr(ym, key, float(value))
    if overrides:
        fitted = yaml.safe_load((ROOT / cfg["env"]["fitted_params"]).read_text())
        target = (fitted.get("fit") or {}).get("mean_observed_yield")
        if target:
            calibrate_yield_scale(cfg["env"], ym, float(target), recommended_schedule)
    return ym


def run_experiment(cfg, run_name=None, seeds=None, train_seeds=1, timesteps=None, load_model=None,
                   districts=False, with_sensitivity=False, quick=False, tensorboard=False,
                   runs_dir=None, log=print, weather=None, prices=None, yield_model=None,
                   parallel_seeds=None, torch_threads=None, tune_baselines=None, oracle=False, lean=False,
                   tuning_from=None, yield_overrides=None, overrides=None):
    """Run the whole experiment and return the run directory. ``weather``, ``prices`` and
    ``yield_model`` override what the config points to (per-district evaluation is skipped
    with a weather override, since it reads each district's processed weather).
    ``parallel_seeds`` and ``torch_threads`` default to the config's ``compute`` block,
    ``tune_baselines`` to the config's ``baselines.tune``. ``oracle`` adds the hindsight
    oracle (farmrl.oracle) on the held-out seasons; ``lean`` skips the step-level rollout
    table and policy grids; ``tuning_from`` reuses a saved baseline_tuning.json instead of
    searching again (same environment only); ``yield_overrides`` sets yield-model parameters."""
    created = datetime.now()
    run_name = run_name or created.strftime("run_%Y%m%d_%H%M%S")
    out = Path(runs_dir or RUNS_DIR) / run_name
    out.mkdir(parents=True, exist_ok=True)
    runcard.snapshot_code(out)                # the exact code of this run, before anything can change it
    seeds = seeds or (QUICK["seeds"] if quick else cfg["eval"]["seeds_per_season"])
    timesteps = timesteps or (QUICK["timesteps"] if quick else cfg["train"]["total_timesteps"])
    ym = yield_model or YieldModel.from_config(ROOT / cfg["env"]["fitted_params"])
    apply_yield_overrides(cfg, ym, yield_overrides)
    make = env_maker(ym, weather, prices)
    base_seed = cfg["env"]["seed"]
    compute = cfg.get("compute", {})
    threads = int(torch_threads or compute.get("torch_threads", 1))
    parallel = int(parallel_seeds or compute.get("parallel_seeds", 1))
    torch.set_num_threads(threads)

    # --- rule baselines tuned on the training seasons ---
    tune_cfg = cfg.get("baselines", {})
    tuned = None
    if tuning_from:
        tuned = json.loads(Path(tuning_from).read_text())
        (out / "baseline_tuning.json").write_text(json.dumps(tuned, indent=2))
        log(f"reusing the tuned rule baselines of {tuning_from}")
    elif tune_cfg.get("tune", False) if tune_baselines is None else tune_baselines:
        log(f"tuning the rule baselines {list(FAMILIES)} on the training seasons")
        tuned = tuning.tune_all(make(cfg["env"], "train"), seeds=int(tune_cfg.get("seeds", 3)),
                                objective=tune_cfg.get("objective", "reward"),
                                max_rounds=int(tune_cfg.get("max_rounds", 4)), log=log)
        (out / "baseline_tuning.json").write_text(json.dumps(tuned, indent=2))

    # --- agent(s) ---
    agents = {}
    if load_model:
        agent = build_agent(make(cfg["env"], "train"), cfg["train"])
        agent.load(load_model)
        history = None
        log(f"loaded model {load_model}")
    else:
        seeds_to_train = [base_seed + k for k in range(max(1, train_seeds))]
        weights, history = train_in_pool(cfg, seeds_to_train, timesteps, ym, weather, prices,
                                       out / "tensorboard" if tensorboard else None, parallel, threads, log)
        (out / "models").mkdir(exist_ok=True)
        for s, state in weights.items():
            agents[s] = build_agent(make(cfg["env"], "train"), cfg["train"])
            agents[s].set_state(state)
            agents[s].save(out / "models" / f"seed_{s}.pt")
        agent = agents[base_seed]                            # the first seed is the one shown in detail
        history.to_csv(out / "train_history.csv", index=False)
    agent.save(out / "model.pt")
    agents = agents or {base_seed: agent}

    # --- rollouts and tables ---
    policies = all_policies(agent, cfg, make, tuned)
    log(f"rolling out {list(policies)} with {seeds} seeds per season")
    rollouts = collect(cfg["env"], policies, make, seeds)
    if oracle:
        plans = oracle_mod.solve_all(cfg["env"], make(cfg["env"], "test").seasons, ym, cfg.get("oracle"),
                                     weather, prices, parallel, threads, log)
        (out / "oracle_plans.json").write_text(json.dumps({str(k): v for k, v in plans.items()}))
        policies["oracle"] = oracle_mod.plan_policy(plans)
        rollouts = pd.concat([rollouts, rollout.run_many(make(cfg["env"], "test"), {"oracle": policies["oracle"]},
                                                         seeds=seeds)], ignore_index=True)
    if not lean:
        rollouts.to_parquet(out / "rollouts.parquet", index=False)
    eps = rollout.episodes(rollouts)
    eps.to_csv(out / "episodes.csv", index=False)
    summary = rollout.summary(eps)
    summary.to_csv(out / "summary.csv", index=False)

    # the rule baseline PPO is measured against: the tuned one that did best in training
    rules = [n + TUNED_SUFFIX for n in (tuned or {})] or [n for n in FAMILIES if n in policies]
    train_reward = summary[summary["split"] == "train"].set_index("policy")["reward"]
    reference_policy = max(rules, key=lambda p: train_reward[p])

    # --- every training seed: episodes, summary, and where its gain comes from ---
    log(f"evaluating {len(agents)} PPO training seed{'s' * (len(agents) > 1)} and attributing the gain")
    seed_parts, attribution = [], []
    test_env = make(cfg["env"], "test")
    for s, a in agents.items():
        if a is agent:
            part = eps[eps["policy"] == "ppo"]
        else:
            part = rollout.episodes(collect(cfg["env"], {"ppo": agent_policy(a)}, make, seeds))
        seed_parts.append(part.assign(train_seed=s))
        for rule in rules + [n for n in FAMILIES if n in policies and n not in rules]:   # tuned, then as given
            table = attribute_gain(test_env, agent_policy(a), policies[rule], seeds)
            attribution.append(table.assign(train_seed=s, baseline_policy=rule, split="test"))
    seed_eps = pd.concat(seed_parts, ignore_index=True)
    seed_eps.to_csv(out / "seed_episodes.csv", index=False)
    seed_summary = rollout.summary(seed_eps.drop(columns="policy").rename(columns={"train_seed": "policy"}))
    seed_summary = seed_summary.rename(columns={"policy": "train_seed"})
    seed_summary.to_csv(out / "seed_summary.csv", index=False)
    pd.concat(attribution, ignore_index=True).to_csv(out / "attribution.csv", index=False)
    dry_wet_table(eps, seed_eps).to_csv(out / "dry_wet.csv", index=False)

    env = make(cfg["env"], "all")
    if not lean:
        reference = rollouts[rollouts["policy"] == "ppo"]
        grids, medians = policy_grids(agent, env, reference)
        grids.to_parquet(out / "policy_grids.parquet", index=False)
        medians.to_csv(out / "state_medians.csv", index=False)

    ranked = {k: v for k, v in policies.items() if k not in AUX_POLICIES and k != "oracle"}
    extra_seeds = max(2, seeds // 3)               # the what-if tables need less precision
    if districts and weather is None:
        log("evaluating per district")
        table = district_summary(cfg["env"], ranked, make, extra_seeds)
        if table is not None:
            table.to_csv(out / "district_summary.csv", index=False)
    if with_sensitivity:
        log("running sensitivity cases")
        sensitivity(cfg["env"], ranked, make, extra_seeds, ym).to_csv(out / "sensitivity.csv", index=False)

    # --- provenance ---
    pd.DataFrame({"season": list(env.price_index), "price_index": list(env.price_index.values())}).to_csv(
        out / "prices.csv", index=False)
    (out / "config_snapshot.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    ym.save(out / "fitted_params.yaml", source=f"copied from {cfg['env']['fitted_params']}")
    fitted = ROOT / cfg["env"]["fitted_params"]
    meta = {
        "run_name": run_name, "created": created.isoformat(timespec="seconds"),
        "algo": cfg.get("train", {}).get("algo", "ppo"), "variant": cfg.get("train", {}).get("variant"),
        "overrides": list(overrides or []), "yield_overrides": dict(yield_overrides or {}),
        "oracle": bool(oracle), "lean": bool(lean),
        "git_hash": _git_hash(), "git_dirty": runcard.git_state()["dirty"], "quick": bool(quick), "timesteps": int(timesteps),
        "train_seeds": [] if history is None else sorted(int(s) for s in history["train_seed"].unique()),
        "loaded_model": str(load_model) if load_model else None,
        "rollout_seeds": int(seeds), "policies": list(policies), "aux_policies": AUX_POLICIES,
        "tuned_policies": [n + TUNED_SUFFIX for n in (tuned or {})], "reference_policy": reference_policy,
        "mechanisms": dict(env.on), "obs_names": list(env.obs_names),
        "action_heads": [int(n) for n in env.action_space.nvec],
        "compute": {"torch_threads": threads, "parallel_seeds": parallel},
        "region": cfg["env"].get("region"), "weather_site": cfg["env"]["weather_site"],
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
