import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import pytest
from farmrl.env import FarmEnv
from farmrl.baselines import BASELINES
from farmrl.evaluation import evaluate, run_episode, summarise
from farmrl.utils import load_config
from farmrl.yield_model import YieldModel

CFG = load_config(ROOT / "configs/default.yaml")["env"]


def toy_weather(seasons=(2014, 2015, 2016, 2017, 2018), steps=12, rain=10.0):
    return pd.DataFrame([{"season": s, "step": t, "rain_mm": rain + (s - 2014), "et0_mm": 40.0, "rh": 50.0}
                         for s in seasons for t in range(steps)])


def toy_env(split="train", **kw):
    prices = pd.DataFrame({"season": [2014, 2018], "price_index": [0.9, 1.1]})
    return FarmEnv(CFG, weather=toy_weather(**kw), prices=prices, split=split, yield_model=YieldModel())


def test_episode_runs():
    env = FarmEnv(load_config(ROOT / "configs/default.yaml")["env"])
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    done, steps = False, 0
    while not done:
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        done = term or trunc; steps += 1
    assert steps == env.T and "yield" in info


def test_split_flag_separates_train_and_test_seasons():
    train, test, both = toy_env("train"), toy_env("test"), toy_env("all")
    assert train.seasons == [2014, 2015, 2016] and test.seasons == [2017, 2018]
    assert both.seasons == train.seasons + test.seasons
    seen = {test.reset(seed=s)[1]["season"] for s in range(40)}
    assert seen == {2017, 2018}
    real = FarmEnv(CFG, split="test")
    assert min(real.seasons) > CFG["train_end"] >= max(FarmEnv(CFG, split="train").seasons)


def test_season_option_replays_real_weather_and_price():
    env = toy_env("all")
    _, info = env.reset(seed=1, options={"season": 2018})
    assert info["season"] == 2018 and env.price == pytest.approx(CFG["base_price"] * 1.1)
    env.reset(seed=1, options={"season": 2016})
    assert env.price == CFG["base_price"]                 # no index for that year -> base price
    out = run_episode(env, BASELINES["no_input"], season=2018, seed=1)
    assert out["rain_total"] == pytest.approx(12 * 14.0)


def test_same_seed_same_episode():
    env = toy_env()
    a = run_episode(env, BASELINES["random"], season=2015, seed=7)
    b = run_episode(env, BASELINES["random"], season=2015, seed=7)
    assert a["profit"] == b["profit"] and a["yield"] == b["yield"]


def test_profit_accounting():
    env = toy_env()
    out = run_episode(env, BASELINES["recommended"], season=2015, seed=3, trace=True)
    c = CFG["costs"]
    inputs = (out["n_total"] * c["nitrogen_per_kg"] + out["irrigation_total"] * c["irrigation_per_mm"]
              + out["pesticide_total"] * c["pesticide_per_level"])
    assert out["input_cost"] == pytest.approx(inputs)
    assert out["profit"] == pytest.approx(out["yield"] * out["price"] - c["fixed_per_ha"] - inputs)
    assert out["reward"] == pytest.approx(out["profit"] - out["penalty"])
    assert len(out["steps"]) == env.T and out["steps"]["nitrogen"].sum() == out["n_total"]


def test_inputs_matter():
    env = toy_env(rain=2.0)
    table = summarise(evaluate(env, BASELINES, seeds_per_season=3))
    assert table.loc["recommended", "yield"] > table.loc["no_input", "yield"] + 5
    assert table.loc["recommended", "profit"] > table.loc["no_input", "profit"]


def test_yield_model_shape():
    m = YieldModel(y_max=50, n_coef=0.02, w_opt=400, w_excess=0.05)
    assert m.potential(0, 400) < m.potential(120, 400) < m.potential(240, 400) < 50
    assert m.potential(120, 300) == m.potential(120, 400) > m.potential(120, 600)
    assert m.final_yield(120, 400, 0, et_ratio=0.8) < m.final_yield(120, 400, 0)
    assert m.final_yield(120, 400, 0.5) == pytest.approx(m.final_yield(120, 400, 0) * (1 - 0.35 * 0.5))


def test_fit_recovers_parameters(tmp_path):
    rng = np.random.default_rng(0)
    truth = YieldModel(y_max=48.0, n_coef=0.02, w_opt=380.0, w_excess=0.06)
    df = pd.DataFrame({"n_kg_ha": rng.uniform(20, 250, 300), "rain_mm": rng.uniform(20, 320, 300)})
    df["yield_q_ha"] = truth.potential(df["n_kg_ha"], df["rain_mm"] + 300) + rng.normal(0, 0.5, 300)
    m = YieldModel()
    rep = m.fit(df, assumed_irrigation=300)
    assert rep["r2"] > 0.95 and all(rep["identified"].values())
    assert m.y_max == pytest.approx(48.0, rel=0.03) and m.n_coef == pytest.approx(0.02, rel=0.1)
    assert m.w_opt == pytest.approx(380.0, abs=15) and m.w_excess == pytest.approx(0.06, rel=0.2)
    m.save(tmp_path / "p.yaml", source="synthetic")
    assert YieldModel.load(tmp_path / "p.yaml").params() == m.params()


def test_fit_falls_back_to_priors_when_unidentified():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"n_kg_ha": rng.uniform(150, 250, 60), "rain_mm": rng.uniform(20, 60, 60)})
    df["yield_q_ha"] = 45 + rng.normal(0, 2, 60)          # flat: no information on any slope
    m = YieldModel()
    rep = m.fit(df, assumed_irrigation=300)
    assert not rep["identified"]["w_excess"] and m.w_excess == 0.0 and m.w_opt == 400.0
    assert m.y_max == pytest.approx(45, abs=2)
