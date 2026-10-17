"""Hindsight oracle: the best fixed plan for a season when its weather and price are known.

A policy only sees the season as it unfolds. The oracle is handed the whole season in
advance: for each held-out season it searches the 12-step action plan (nitrogen, water,
pesticide per fortnight, plus the months to store the grain at the end) with the
cross-entropy method, then polishes it by coordinate ascent. It maximises the same reward
PPO is trained on, averaged over a few pest-growth seeds (the one thing it cannot know),
and the plan is then replayed on the evaluation seeds like any other policy.

It is a plan, not a feedback rule, so it does not react to the pest noise; with pests at
their calibrated strength nothing sprays, which makes that a small caveat. Treat it as the
best a policy with perfect foresight of weather and price could do, up to search error.
"""
import multiprocessing as mp
import zlib

import numpy as np
import torch

ORACLE_SEED0 = 900_000          # tuning uses 500_000 and evaluation 10_000
DEFAULTS = {"population": 240, "iterations": 40, "elite_frac": 0.1, "smoothing": 0.7,
            "search_seeds": 6, "polish_seeds": 20, "polish_rounds": 3}


def plan_reward(env, season, plan, seeds, seed0=ORACLE_SEED0):
    """Mean reward (INR/ha) of the plan over ``seeds`` pest-growth replicates."""
    total = 0.0
    for k in range(seeds):
        env.reset(seed=seed0 + 1000 * season + k, options={"season": season})
        for t in range(env.T):
            _, r, done, _, _ = env.step(plan[t])
            total += r
    return total * env.reward_scale / seeds


def scenario_reward(env, scenarios, plan, seeds, seed0=ORACLE_SEED0):
    """Mean reward of ``plan`` over (season, price index or None) scenarios x ``seeds`` replicates.
    A price index of None keeps the season's own price."""
    total = 0.0
    for season, price in scenarios:
        for k in range(seeds):
            env.reset(seed=seed0 + 1000 * season + k, options={"season": season, "price_index": price})
            for t in range(env.T):
                _, r, done, _, _ = env.step(plan[t])
                total += r
    return total * env.reward_scale / (seeds * len(scenarios))


def _variables(env):
    heads = [int(n) for n in env.action_space.nvec]
    return [(t, h) for t in range(env.T) for h in range(len(heads)) if h < 3 or t == env.T - 1], heads


def solve_season(env, season, cfg=None, rng=None):
    """Best plan found for ``season``: (plan [T x heads] ints, reward on the polish seeds)."""
    return solve_scenarios(env, [(season, None)], cfg, rng or np.random.default_rng(season))


def solve_scenarios(env, scenarios, cfg=None, rng=None, seeds_per_scenario=None):
    """Best single plan for a set of (season, price index) scenarios, i.e. the plan of an
    agent that knows only what those scenarios have in common. With one scenario this is the
    hindsight oracle of ``solve_season``. ``seeds_per_scenario`` replaces the pest seeds
    (default: the config's, divided over the scenarios, at least 1)."""
    c = {**DEFAULTS, **(cfg or {})}
    rng = rng or np.random.default_rng(scenarios[0][0])
    n_sc = len(scenarios)
    search_seeds = seeds_per_scenario or max(1, int(c["search_seeds"]) // n_sc)
    polish_seeds = seeds_per_scenario or max(1, int(c["polish_seeds"]) // n_sc)
    plan_reward = lambda env_, season_, plan_, k_: scenario_reward(env_, scenarios, plan_, k_)
    variables, heads = _variables(env)
    probs = {v: np.full(heads[v[1]], 1.0 / heads[v[1]]) for v in variables}
    n_elite = max(2, int(c["population"] * c["elite_frac"]))

    def to_plan(choice):
        plan = np.zeros((env.T, len(heads)), dtype=int)
        for v, a in choice.items():
            plan[v] = a
        return plan

    best_plan, best_val = None, -np.inf
    for _ in range(int(c["iterations"])):
        draws = {v: rng.choice(heads[v[1]], size=int(c["population"]), p=probs[v]) for v in variables}
        plans = [to_plan({v: draws[v][i] for v in variables}) for i in range(int(c["population"]))]
        scores = np.array([plan_reward(env, None, p, search_seeds) for p in plans])
        order = np.argsort(scores)[-n_elite:]
        if scores[order[-1]] > best_val:
            best_val, best_plan = float(scores[order[-1]]), plans[int(order[-1])]
        for v in variables:
            freq = np.bincount(draws[v][order], minlength=heads[v[1]]) / n_elite
            probs[v] = c["smoothing"] * freq + (1 - c["smoothing"]) * probs[v]
            probs[v] = np.maximum(probs[v], 1e-3); probs[v] /= probs[v].sum()
    mode = to_plan({v: int(np.argmax(probs[v])) for v in variables})
    cands = [best_plan, mode]
    plan = max(cands, key=lambda p: plan_reward(env, None, p, polish_seeds))
    value = plan_reward(env, None, plan, polish_seeds)
    for _ in range(int(c["polish_rounds"])):                    # coordinate ascent on fresh seeds
        improved = False
        for v in variables:
            for a in range(heads[v[1]]):
                if a == plan[v]:
                    continue
                trial = plan.copy(); trial[v] = a
                val = plan_reward(env, None, trial, polish_seeds)
                if val > value + 1e-6:
                    plan, value, improved = trial, val, True
        if not improved:
            break
    return plan, value


def _worker(job):
    cfg_env, season, oracle_cfg, yield_params, weather, prices, threads = job
    from .experiment import env_maker
    from .yield_model import YieldModel
    torch.set_num_threads(threads)
    env = env_maker(YieldModel(**yield_params), weather, prices)(cfg_env, "test")
    plan, value = solve_season(env, season, oracle_cfg)
    return season, plan.tolist(), value


def solve_all(cfg_env, seasons, yield_model, oracle_cfg=None, weather=None, prices=None,
              parallel=1, threads=1, log=print):
    """{season: {"plan": [[...]], "reward": float}} for every season, in a process pool."""
    jobs = [(cfg_env, int(s), oracle_cfg, yield_model.params(), weather, prices, threads) for s in seasons]
    workers = max(1, min(int(parallel), len(jobs)))
    log(f"oracle: searching plans for seasons {list(seasons)} ({workers} process{'es' * (workers > 1)})")
    if workers == 1:
        results = [_worker(j) for j in jobs]
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            results = pool.map(_worker, jobs)
    return {s: {"plan": plan, "reward": value} for s, plan, value in results}


def plan_policy(plans):
    """Policy replaying the stored plan of the current season (plans: output of solve_all)."""
    arrays = {int(s): np.asarray(v["plan"], dtype=int) for s, v in plans.items()}

    def oracle(obs, env):
        return arrays[env.season][env.t]
    return oracle


# ------------------------------------------------------------- what the oracle needs to know
KINDS = ("full", "weather", "price", "none")


def _info_worker(job):
    """One plan of an information-limited oracle. ``full`` knows the season's weather and price;
    ``weather`` its weather but only the average price; ``price`` its price but only the
    weather of a spread of other (training) seasons; ``none`` neither."""
    kind, season, cfg_env, oracle_cfg, yield_params, weather, prices, threads, sample = job
    from .experiment import env_maker
    from .yield_model import YieldModel
    torch.set_num_threads(threads)
    env = env_maker(YieldModel(**yield_params), weather, prices)(cfg_env, "all")
    train = env.train_seasons
    mean_price = float(np.mean([env.price_index.get(s, 1.0) for s in train]))
    spread = [train[int(i)] for i in np.linspace(0, len(train) - 1, int(sample))]
    own = env.price_index.get(season, 1.0) if season is not None else mean_price
    scenarios = {"full": [(season, None)], "weather": [(season, mean_price)],
                 "price": [(w, own) for w in spread], "none": [(w, mean_price) for w in spread]}[kind]
    seeds = None if kind in ("full", "weather") else 1
    plan, value = solve_scenarios(env, scenarios, oracle_cfg, np.random.default_rng(zlib.crc32(f"{kind}-{season}".encode())),
                                  seeds_per_scenario=seeds)
    return kind, season, plan.tolist(), value


def solve_info(cfg_env, seasons, yield_model, kinds=KINDS, oracle_cfg=None, weather=None, prices=None,
               parallel=1, threads=1, sample=10, log=print):
    """{kind: {season: {"plan", "reward"}}} for the information-limited oracles. ``none`` does not
    depend on the season, so it is searched once and shared."""
    jobs = []
    for kind in kinds:
        for s in ([None] if kind == "none" else seasons):
            jobs.append((kind, None if s is None else int(s), cfg_env, oracle_cfg, yield_model.params(),
                         weather, prices, threads, sample))
    workers = max(1, min(int(parallel), len(jobs)))
    log(f"oracle breakdown: {len(jobs)} plan searches ({workers} process{'es' * (workers > 1)})")
    if workers == 1:
        results = [_info_worker(j) for j in jobs]
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            results = pool.map(_info_worker, jobs)
    out = {k: {} for k in kinds}
    for kind, season, plan, value in results:
        for s in (seasons if kind == "none" else [season]):
            out[kind][int(s)] = {"plan": plan, "reward": value}
    return out
