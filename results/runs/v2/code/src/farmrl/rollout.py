"""Step-by-step simulation records: the single source every plot, animation and
dashboard page reads from.

``run_episode`` returns one row per fortnight; ``run_many`` stacks episodes over
policies, seasons and seeds; ``episodes`` collapses step rows to one row per episode and
``summary`` to one row per policy and split.
"""
import numpy as np
import pandas as pd

from .env import OBS_NAMES

EPISODE_KEYS = ["policy", "split", "season", "seed"]
OBS_COLS = [f"obs_{k}" for k in OBS_NAMES]


def run_episode(env, policy, season=None, seed=None, policy_name=None):
    """Play one season with ``policy(obs, env) -> action`` and record every step.

    Columns: season, step, date, policy, seed; the observation the agent saw (``obs_*``,
    scaled as in the env) and the same state in real units (soil_moisture as a fraction of
    capacity, soil_n in kg/ha, pest in [0, 1], price in INR/quintal); the weather of the
    fortnight (rain_mm, et0_mm, rh); the action in real units (nitrogen_kg, irrigation_mm,
    pesticide_level); input_cost, penalty and reward in INR; the state after the step
    (crop_growth, pest_damage, et_ratio, soil_moisture_end); cum_profit (input costs so
    far, plus revenue minus fixed cost at harvest) and the episode totals final_yield
    (q/ha), final_revenue and final_profit repeated on every row.
    """
    obs, _ = env.reset(seed=seed, options=None if season is None else {"season": season})
    dates = env.step_dates()
    season_etc = float((np.asarray(env.dyn["kc"]) * env._wx[:, 1]).sum())
    fixed = env.costs.get("fixed_per_ha", 0.0)
    name = policy_name or getattr(policy, "__name__", "policy")
    rows, done, cum_profit = [], False, 0.0
    while not done:
        t = env.t
        action = np.asarray(policy(obs, env)).astype(int)
        rain, et0, rh = env._wx[t]
        row = {"season": env.season, "step": t, "date": dates[t], "policy": name, "seed": seed,
               **dict(zip(OBS_COLS, obs.astype(float))),
               "soil_moisture": env.soil_moist, "soil_n": env.soil_n, "pest": env.pest,
               "price": env.price, "rain_mm": rain, "et0_mm": et0, "rh": rh,
               "nitrogen_kg": env.n_levels[action[0]], "irrigation_mm": env.w_levels[action[1]],
               "pesticide_level": env.p_levels[action[2]]}
        cost0, pen0 = env.cum_cost, env.cum_penalty
        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        cost = env.cum_cost - cost0
        cum_profit -= cost
        if done:
            cum_profit += info["revenue"] - fixed
        row.update({"input_cost": cost, "penalty": env.cum_penalty - pen0,
                    "reward": reward * env.reward_scale, "cum_profit": cum_profit,
                    "crop_growth": env.cum_eta / season_etc, "pest_damage": env.pest_damage,
                    "et_ratio": env.cum_eta / env.cum_etc, "soil_moisture_end": env.soil_moist})
        rows.append(row)
    df = pd.DataFrame(rows)
    df["final_yield"] = info["yield"]
    df["final_revenue"] = info["revenue"]
    df["final_profit"] = info["profit"]
    return df


def run_many(env, policies, seasons=None, seeds=10, seed0=10_000, split=None):
    """Every policy on every season with ``seeds`` replicates. All policies share the
    seeds, so the random pest growth is identical across them for a given replicate."""
    seasons = list(env.seasons if seasons is None else seasons)
    parts = []
    for name, policy in policies.items():
        for season in seasons:
            for k in range(seeds):
                ep = run_episode(env, policy, season, seed0 + 1000 * season + k, policy_name=name)
                ep["seed"] = k
                parts.append(ep)
    out = pd.concat(parts, ignore_index=True)
    out.insert(0, "split", split or env.split)
    return out


def episodes(rollouts):
    """One row per episode: profit, yield, reward and total inputs used."""
    keys = [k for k in EPISODE_KEYS if k in rollouts.columns]
    g = rollouts.groupby(keys, sort=False)
    return g.agg(profit=("final_profit", "first"), yield_q_ha=("final_yield", "first"),
                 revenue=("final_revenue", "first"), reward=("reward", "sum"),
                 nitrogen_kg=("nitrogen_kg", "sum"), irrigation_mm=("irrigation_mm", "sum"),
                 pesticide_level=("pesticide_level", "sum"), rain_mm=("rain_mm", "sum"),
                 input_cost=("input_cost", "sum"), penalty=("penalty", "sum"),
                 price=("price", "first")).reset_index()


def summary(episode_table):
    """One row per (policy, split): mean of every metric, std of profit, episode count."""
    keys = [k for k in ("policy", "split") if k in episode_table.columns]
    metrics = ["profit", "yield_q_ha", "reward", "nitrogen_kg", "irrigation_mm",
               "pesticide_level", "input_cost", "penalty"]
    g = episode_table.groupby(keys, sort=False)
    out = g[metrics].mean()
    out.insert(1, "profit_std", g["profit"].std())
    out["episodes"] = g.size()
    return out.reset_index()
