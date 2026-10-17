"""Tests for the follow-up studies: forecast observation, price override, finer pest grid,
climate / price shifts and the information-limited oracles."""
import copy
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import pytest

from farmrl import oracle, shifts, tuning
from farmrl.env import FarmEnv, OBS_NAMES
from farmrl.yield_model import YieldModel
from tests.test_env import CFG, toy_env, toy_weather


def with_forecast(noise):
    cfg = copy.deepcopy(CFG)
    cfg["forecast"] = {"enabled": True, "noise_mm": noise}
    return toy_env(cfg=cfg)


def test_forecast_adds_one_observation_equal_to_next_rain_when_noise_free():
    env = with_forecast(0.0)
    assert env.obs_names[-1] == "rain_forecast" and len(env.obs_names) == len(toy_env().obs_names) + 1
    obs, _ = env.reset(seed=1, options={"season": 2015})
    for t in range(env.T):
        assert obs[-1] == pytest.approx(env._wx[t, 0] / 50, abs=1e-6)      # the rain of the step about to be played
        obs, *_ = env.step(np.array([0, 0, 0, 0]))
    assert env.observation_space.contains(obs)


def test_noisy_forecast_is_fixed_within_a_season_and_differs_from_the_truth():
    env = with_forecast(15.0)
    obs, _ = env.reset(seed=3, options={"season": 2016})
    first = [env._forecast[t] for t in range(env.T)]
    assert not np.allclose(first, env._wx[:, 0]) and min(first) >= 0.0
    env.reset(seed=3, options={"season": 2016})
    assert np.allclose(first, env._forecast)                                 # same seed, same forecast
    obs, *_ = env.step(np.array([0, 0, 0, 0]))
    assert obs[-1] == pytest.approx(env._forecast[1] / 50, abs=1e-6)


def test_forecast_off_leaves_the_environment_unchanged():
    a, b = toy_env(), toy_env()
    oa, _ = a.reset(seed=5, options={"season": 2015})
    assert "rain_forecast" not in a.obs_names and len(oa) == len(OBS_NAMES) + 2      # pump cost + spray efficacy


def test_price_index_option_overrides_the_season_price():
    env = toy_env("all")
    env.reset(seed=0, options={"season": 2015, "price_index": 1.25})
    assert env.price == pytest.approx(CFG["base_price"] * 1.25)
    env.reset(seed=0, options={"season": 2015})
    assert env.price == pytest.approx(CFG["base_price"])


def test_fine_pest_grid_is_larger_and_includes_every_pair_on_the_step():
    env = toy_env()
    coarse, fine = tuning.threshold_grid(env)["pests"], tuning.threshold_grid(env, fine_pests=True)["pests"]
    assert len(fine) > 4 * len(coarse)
    assert {"p_low_above": 0.02, "p_high_above": 0.02} in fine and {"p_low_above": 1.01, "p_high_above": 1.01} in fine
    assert all(c["p_high_above"] >= c["p_low_above"] for c in fine)
    assert tuning.calendar_grid(env, fine_pests=True)["pests"]              # accepted and ignored


def climate_table():
    w = toy_weather(seasons=(2015, 2016))
    w["start_date"] = [str(pd.Timestamp(s, 11, 1) + pd.Timedelta(days=15 * t)) for s, t in zip(w["season"], w["step"])]
    w["t_max"], w["t_mean"], w["t_min"] = 24.0 + 0.8 * w["step"], 18.0 + 0.7 * w["step"], 11.0
    w["heat_days"] = 0
    return w


def test_warming_raises_temperature_et0_and_heat_days_only_where_the_window_is():
    w = climate_table()
    hot = shifts.warm(w, 2.0)
    assert np.allclose(hot["t_max"], w["t_max"] + 2) and np.allclose(hot["et0_mm"], w["et0_mm"] * 1.08)
    assert (hot["heat_days"] >= w["heat_days"]).all() and hot["heat_days"].sum() > 0
    early = w["step"] < 6                                                    # November-January: outside the window
    assert np.allclose(hot.loc[early, "heat_days"], 0.0)
    assert shifts.warm(w, 1.0)["heat_days"].sum() < hot["heat_days"].sum()
    assert shifts.warm(w, 0.0).equals(w)


def test_drier_and_volatile_prices_and_apply():
    w = climate_table()
    assert shifts.drier(w, 0.8)["rain_mm"].sum() == pytest.approx(0.8 * w["rain_mm"].sum())
    prices = pd.DataFrame({"season": [2015, 2016, 2017], "price_index": [0.9, 1.0, 1.2]})
    vol = shifts.volatile_prices(prices, 2.0)
    assert list(vol["price_index"].round(3)) == [0.8, 1.0, 1.4]
    cfg, weather, p = shifts.apply(CFG, {"rain": 0.8, "price_vol": 2.0, "storage_vol": 2.0}, w, prices)
    assert cfg["storage"]["monthly_volatility"] == pytest.approx(2 * CFG["storage"]["monthly_volatility"])
    assert weather["rain_mm"].sum() < w["rain_mm"].sum() and p["price_index"].max() == pytest.approx(1.4)
    assert set(shifts.SCENARIOS) >= {"+1 °C", "+2 °C", "20% less rain", "2x price volatility"}


def test_multi_scenario_plan_search_and_information_limited_oracles():
    env = toy_env("all")
    cfg = {"population": 30, "iterations": 4, "search_seeds": 2, "polish_seeds": 4, "polish_rounds": 1}
    plan, value = oracle.solve_scenarios(env, [(2014, None), (2015, 0.9)], cfg, np.random.default_rng(0), seeds_per_scenario=1)
    assert plan.shape == (env.T, 4) and np.isfinite(value)
    prices = pd.DataFrame({"season": [2014, 2018], "price_index": [0.9, 1.1]})
    plans = oracle.solve_info(CFG, [2017, 2018], YieldModel(), oracle.KINDS, cfg, weather=toy_weather(), prices=prices,
                              parallel=1, sample=2, log=lambda m: None)
    assert set(plans) == set(oracle.KINDS)
    assert all(set(d) == {2017, 2018} for d in plans.values())
    assert plans["none"][2017]["plan"] == plans["none"][2018]["plan"]       # does not depend on the season
    test = toy_env("test")
    full = oracle.plan_reward(test, 2017, np.asarray(plans["full"][2017]["plan"]), 4)
    blind = oracle.plan_reward(test, 2017, np.asarray(plans["none"][2017]["plan"]), 4)
    assert np.isfinite(full) and np.isfinite(blind)
