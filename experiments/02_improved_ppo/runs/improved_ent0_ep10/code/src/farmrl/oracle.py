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


def _variables(env):
    heads = [int(n) for n in env.action_space.nvec]
    return [(t, h) for t in range(env.T) for h in range(len(heads)) if h < 3 or t == env.T - 1], heads


def solve_season(env, season, cfg=None, rng=None):
    """Best plan found for ``season``: (plan [T x heads] ints, reward on the polish seeds)."""
    c = {**DEFAULTS, **(cfg or {})}
    rng = rng or np.random.default_rng(season)
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
        scores = np.array([plan_reward(env, season, p, int(c["search_seeds"])) for p in plans])
        order = np.argsort(scores)[-n_elite:]
        if scores[order[-1]] > best_val:
            best_val, best_plan = float(scores[order[-1]]), plans[int(order[-1])]
        for v in variables:
            freq = np.bincount(draws[v][order], minlength=heads[v[1]]) / n_elite
            probs[v] = c["smoothing"] * freq + (1 - c["smoothing"]) * probs[v]
            probs[v] = np.maximum(probs[v], 1e-3); probs[v] /= probs[v].sum()
    mode = to_plan({v: int(np.argmax(probs[v])) for v in variables})
    cands = [best_plan, mode]
    plan = max(cands, key=lambda p: plan_reward(env, season, p, int(c["polish_seeds"])))
    value = plan_reward(env, season, plan, int(c["polish_seeds"]))
    for _ in range(int(c["polish_rounds"])):                    # coordinate ascent on fresh seeds
        improved = False
        for v in variables:
            for a in range(heads[v[1]]):
                if a == plan[v]:
                    continue
                trial = plan.copy(); trial[v] = a
                val = plan_reward(env, season, trial, int(c["polish_seeds"]))
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
