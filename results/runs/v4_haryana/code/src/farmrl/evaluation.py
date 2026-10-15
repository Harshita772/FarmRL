"""Roll policies out on chosen seasons and tabulate the results."""
import numpy as np
import pandas as pd

METRICS = ["profit", "reward", "yield", "n_total", "irrigation_total", "pesticide_total"]


def run_episode(env, policy, season=None, seed=None, trace=False):
    """Play one season with ``policy(obs, env) -> action``.

    Returns the terminal ``info`` dict plus ``reward`` (sum of env rewards, in INR). With
    ``trace=True`` it also carries ``steps``: one row per fortnight with the action taken
    and the weather / soil state the agent faced."""
    obs, _ = env.reset(seed=seed, options=None if season is None else {"season": season})
    done, total, rows = False, 0.0, []
    while not done:
        action = np.asarray(policy(obs, env))
        t, moist, pest = env.t, env.soil_moist, env.pest
        rain = env._wx[t, 0]
        obs, r, term, trunc, info = env.step(action)
        total += r
        done = term or trunc
        if trace:
            rows.append({"step": t, "nitrogen": env.n_levels[action[0]],
                         "irrigation": env.w_levels[action[1]],
                         "pesticide": env.p_levels[action[2]], "rain": rain,
                         "soil_moisture": moist, "pest": pest})
    out = dict(info, reward=total * env.reward_scale)
    if trace:
        out["steps"] = pd.DataFrame(rows)
    return out


def evaluate(env, policies, seasons=None, seeds_per_season=30, seed0=10_000):
    """One row per (policy, season, replicate). Every policy sees the same seeds, so the
    random pest growth is identical across policies for a given replicate."""
    seasons = list(env.seasons if seasons is None else seasons)
    rows = []
    for name, policy in policies.items():
        for season in seasons:
            for k in range(seeds_per_season):
                ep = run_episode(env, policy, season=season, seed=seed0 + 1000 * season + k)
                rows.append({"policy": name, "replicate": k, **ep})
    return pd.DataFrame(rows)


def summarise(episodes):
    """Mean of each metric per policy, plus the std of profit across episodes."""
    g = episodes.groupby("policy", sort=False)
    table = g[METRICS].mean()
    table.insert(1, "profit_std", g["profit"].std())
    return table
