"""Grid search of the rule baselines on the training seasons.

The calendar's timings and the threshold rule's cut-offs are searched one decision at a
time (nitrogen, water, pests, sale): every combination inside a block is tried with the
other blocks held at their current best, and the blocks are cycled until a full round
brings no improvement. Only seasons of the "train" split are ever played, with seeds that
the evaluation does not use, and the score is the objective the agent itself is trained
on (reward = profit minus penalties) unless ``objective`` says "profit".
"""
import numpy as np

from .baselines import FAMILIES

TUNE_SEED0 = 500_000          # evaluation rollouts start at 10_000


def _steps(first, gap, count, T):
    return [s for s in (first + gap * i for i in range(count)) if s < T]


def calendar_grid(env):
    """Blocks of candidate settings for ``calendar_policy``: {block: [partial params]}."""
    T = env.T
    months = [int(m) for m in env.store_levels] if env.on["storage"] else [0]
    nitrogen = [{"n_steps": _steps(f, g, k, T), "n_level": lv}
                for lv in (1, 2) for f in (0, 1) for g in (1, 2) for k in range(1, 7)]
    water = [{"w_steps": [], "w_level": 0}] + [
        {"w_steps": _steps(f, g, k, T), "w_level": lv}
        for lv in (1, 2) for f in range(0, 6) for g in (1, 2, 3) for k in range(1, 7)]
    singles = [[s] for s in range(2, T - 1)]
    pairs = [[s, s + g] for s in range(2, T - 1) for g in (2, 3) if s + g < T - 1]
    pests = [{"p_steps": [], "p_level": 0}] + [{"p_steps": steps, "p_level": lv}
                                                for lv in (1, 2) for steps in singles + pairs]
    return {"nitrogen": nitrogen, "water": water, "pests": pests, "sale": [{"store": m} for m in months]}


def threshold_grid(env):
    """Blocks of candidate settings for ``threshold_policy``."""
    T = env.T
    nitrogen = [{"n_below": b, "n_level": lv, "n_until": u}
                for b in (20, 30, 40, 50, 60, 80, 100) for lv in (1, 2) for u in (4, 5, 6, 7, 8, 9, T)]
    water = [{"w_full_below": f, "w_half_below": h, "w_until": u}
             for h in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7) for f in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5) if f <= h
             for u in (T - 3, T - 2, T - 1, T)]
    never = 1.01              # pest pressure never exceeds 1
    pests = [{"p_low_above": lo, "p_high_above": hi}
             for lo in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, never)
             for hi in (0.2, 0.3, 0.4, 0.6, 0.8, never) if hi >= lo]
    sale = [{"store_below": 0.0, "store_months": 0}]
    if env.on["storage"]:
        sale += [{"store_below": b, "store_months": int(m)}
                 for m in env.store_levels[1:] for b in (0.96, 0.97, 0.98, 0.99, 1.0, 1.01, 1.02, 1.04, 9.0)]
    return {"nitrogen": nitrogen, "water": water, "pests": pests, "sale": sale}


GRIDS = {"recommended": calendar_grid, "threshold": threshold_grid}


def score(env, policy, seasons, seeds=3, objective="reward", seed0=TUNE_SEED0):
    """Mean episode reward (or profit), in INR/ha, of ``policy`` over ``seasons`` x ``seeds``."""
    total = []
    for season in seasons:
        for k in range(seeds):
            obs, _ = env.reset(seed=seed0 + 1000 * season + k, options={"season": season})
            done, ret = False, 0.0
            while not done:
                obs, r, done, _, info = env.step(policy(obs, env))
                ret += r
            total.append(ret * env.reward_scale if objective == "reward" else info["profit"])
    return float(np.mean(total))


def _unique(candidates):
    seen, out = set(), []
    for c in candidates:
        key = repr(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def tune(env, family, seeds=3, objective="reward", max_rounds=4, log=None):
    """Tune one rule family on ``env.train_seasons``. Returns a JSON-ready record:
    params, default_params, the train score of both, candidates per block, evaluations."""
    factory, default = FAMILIES[family]
    grid = {block: _unique(c) for block, c in GRIDS[family](env).items()}
    seasons = env.train_seasons
    evaluate = lambda params: score(env, factory(params), seasons, seeds, objective)
    best = dict(default)
    default_score = best_score = evaluate(best)
    evaluations = 1
    for round_ in range(max_rounds):
        improved = False
        for block, candidates in grid.items():
            for cand in candidates:
                trial = {**best, **cand}
                if trial == best:
                    continue
                value = evaluate(trial)
                evaluations += 1
                if value > best_score + 1e-9:
                    best, best_score, improved = trial, value, True
        if log:
            log(f"  {family}: round {round_ + 1}, train {objective} {best_score:,.0f} "
                f"(default {default_score:,.0f})")
        if not improved:
            break
    return {"params": best, "default_params": dict(default), "objective": objective,
            "train_score": best_score, "default_train_score": default_score,
            "train_seasons": [int(s) for s in seasons], "seeds": int(seeds),
            "candidates": {b: len(c) for b, c in grid.items()},
            "grid_size": int(np.prod([len(c) for c in grid.values()])), "evaluations": evaluations}


def tune_all(env, families=None, **kw):
    """{family: tuning record} for every tunable rule family."""
    return {name: tune(env, name, **kw) for name in (families or FAMILIES)}
