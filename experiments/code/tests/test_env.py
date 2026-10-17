import copy
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import pytest
from farmrl import tuning
from farmrl.env import FarmEnv, MECHANISMS, OBS_NAMES, scale_config
from farmrl.baselines import BASELINES, calendar_policy, threshold_policy
from farmrl.experiment import attribute_gain, sensitivity_cases
from farmrl.evaluation import evaluate, run_episode, summarise
from farmrl.utils import load_config
from farmrl.yield_model import YieldModel

CFG = load_config(ROOT / "configs/default.yaml")["env"]


def toy_weather(seasons=(2014, 2015, 2016, 2017, 2018), steps=12, rain=10.0):
    return pd.DataFrame([{"season": s, "step": t, "rain_mm": rain + (s - 2014), "et0_mm": 40.0, "rh": 50.0}
                         for s in seasons for t in range(steps)])


def toy_env(split="train", cfg=CFG, **kw):
    prices = pd.DataFrame({"season": [2014, 2018], "price_index": [0.9, 1.1]})
    return FarmEnv(cfg, weather=toy_weather(**kw), prices=prices, split=split, yield_model=YieldModel())


def with_only(*mechanisms):
    """The default config with just these mechanisms switched on."""
    cfg = copy.deepcopy(CFG)
    for name in MECHANISMS:
        cfg[name]["enabled"] = name in mechanisms
    return cfg


def play(env, actions, season=2015, seed=0):
    """Play one action repeated for the whole season and return the final info."""
    env.reset(seed=seed, options={"season": season})
    for _ in range(env.T):
        _, _, _, _, info = env.step(np.asarray(actions))
    return info


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
              + out["pesticide_total"] * c["pesticide_per_level"] + out["pumping_cost"])
    assert out["input_cost"] == pytest.approx(inputs) and out["pumping_cost"] > 0
    assert out["store_months"] == 0 and out["revenue"] == pytest.approx(out["yield"] * out["price"])
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


# ------------------------------------------------------------- mechanisms added for v3
def test_mechanisms_off_give_the_earlier_simulator():
    old = {k: v for k, v in CFG.items() if k not in MECHANISMS}        # a config written before v3
    env, new = toy_env(cfg=old), toy_env()
    assert env.obs_names == OBS_NAMES and list(env.action_space.nvec) == [3, 3, 3]
    assert len(new.obs_names) == len(OBS_NAMES) + 2 and list(new.action_space.nvec) == [3, 3, 3, 3]
    out = run_episode(env, BASELINES["recommended"], season=2015, seed=3)
    c = CFG["costs"]
    inputs = (out["n_total"] * c["nitrogen_per_kg"] + out["irrigation_total"] * c["irrigation_per_mm"]
              + out["pesticide_total"] * c["pesticide_per_level"])
    assert out["input_cost"] == pytest.approx(inputs) and out["pumping_cost"] == 0 and out["waterlog_loss"] == 0
    assert out["profit"] == pytest.approx(out["yield"] * out["price"] - c["fixed_per_ha"] - inputs)
    assert new.observation_space.contains(new.reset(seed=0)[0])


def test_pumping_cost_rises_within_the_season_and_over_the_years():
    env = toy_env("all", cfg=with_only("groundwater"))
    env.reset(seed=0, options={"season": 2015})
    gw = env.mech["groundwater"]
    assert env.depth == pytest.approx(gw["depth_m"] - gw["decline_m_per_year"] * (gw["depth_year"] - 2016))
    costs = []
    for _ in range(4):                                   # four identical full irrigations
        before = env.cum_pump_cost
        env.step(np.array([0, 2, 0]))
        costs.append(env.cum_pump_cost - before)
    assert costs[0] < costs[1] < costs[2] < costs[3]
    lift = env.depth + 37.5 / (1000 * gw["specific_yield"])
    assert costs[0] == pytest.approx(0.02725 * gw["energy_price_per_kwh"] / gw["pump_efficiency"] * lift * 75)
    later = play(env, [0, 2, 0], season=2018)["pumping_cost"]
    assert later > play(env, [0, 2, 0], season=2014)["pumping_cost"]      # deeper water table


def test_waterlogging_costs_yield_only_when_the_topsoil_is_overfilled():
    on, off = toy_env(cfg=with_only("waterlogging")), toy_env(cfg=with_only())
    assert on._aeration_stress(30.0) == 0 and on._aeration_stress(0.0) == 0
    assert 0 < on._aeration_stress(60.0) < on._aeration_stress(120.0) <= 1
    full, half = play(on, [0, 2, 0]), play(on, [0, 1, 0])
    assert full["waterlog_loss"] > half["waterlog_loss"] >= 0 and full["waterlog_loss"] > 0.05
    assert play(on, [0, 0, 0])["waterlog_loss"] == 0
    same = play(off, [0, 2, 0])
    assert same["waterlog_loss"] == 0 and full["yield"] == pytest.approx(same["yield"] * (1 - full["waterlog_loss"]))


def test_repeated_sprays_work_less_well():
    env = toy_env(cfg=with_only("pest_resistance"))
    env.reset(seed=0, options={"season": 2015})
    decay, seen = env.mech["pest_resistance"]["efficacy_decay"], []
    for _ in range(3):
        seen.append(env.spray_eff)
        env.step(np.array([0, 0, 1]))
    assert seen == pytest.approx([1.0, 1 - decay, (1 - decay) ** 2])
    env.step(np.array([0, 0, 0]))
    assert env.spray_eff == pytest.approx((1 - decay) ** 3)             # not spraying does not restore it
    always = play(env, [0, 0, 2])["pest_damage"]
    assert always > play(toy_env(cfg=with_only()), [0, 0, 2])["pest_damage"]


def test_storage_action_trades_price_risk_against_storage_costs():
    env = toy_env("all", cfg=with_only("storage"))
    st = env.mech["storage"]
    now, kept = play(env, [0, 0, 0, 0], season=2018, seed=5), play(env, [0, 0, 0, 2], season=2018, seed=5)
    assert now["store_months"] == 0 and now["sale_price"] == now["price"] and now["storage_cost"] == 0
    assert now["revenue"] == pytest.approx(now["yield"] * now["price"])
    assert kept["store_months"] == 2 and kept["yield"] == now["yield"] and kept["sale_price"] != kept["price"]
    assert kept["storage_cost"] == pytest.approx(
        2 * kept["yield"] * (st["cost_per_quintal_month"] + st["interest_per_month"] * kept["price"]))
    assert kept["revenue"] == pytest.approx(
        kept["yield"] * (1 - st["loss_per_month"]) ** 2 * kept["sale_price"] - kept["storage_cost"])
    assert play(env, [0, 0, 0, 2], season=2018, seed=6)["sale_price"] != kept["sale_price"]      # stochastic
    assert play(env, [0, 0, 0], season=2018, seed=5)["revenue"] == pytest.approx(now["revenue"])  # default: sell
    # a price below trend is expected to recover, one above it to fall back
    gain = lambda season: np.mean([play(env, [0, 0, 0, 2], season, k)["sale_price"] for k in range(300)]) / env.price
    assert gain(2014) > gain(2018)


def test_rule_families_reproduce_the_untuned_baselines():
    env = toy_env()
    for name, rule in (("recommended", calendar_policy()), ("threshold", threshold_policy())):
        a = run_episode(env, BASELINES[name], season=2015, seed=2)
        b = run_episode(env, rule, season=2015, seed=2)
        assert a["profit"] == b["profit"] and a["n_total"] == b["n_total"]
    late = calendar_policy({"w_steps": [5], "w_level": 1, "store": 1})
    out = run_episode(env, late, season=2015, seed=2, trace=True)
    assert out["irrigation_total"] == 37.5 and out["steps"]["irrigation"].iloc[5] == 37.5 and out["store_months"] == 1


def test_tuning_uses_training_seasons_only_and_never_gets_worse():
    env = toy_env("train", rain=2.0)
    seen, reset = set(), env.reset

    def spy(seed=None, options=None):
        seen.add(options["season"])
        return reset(seed=seed, options=options)

    env.reset = spy
    rec = tuning.tune(env, "threshold", seeds=1, max_rounds=1)
    assert seen == set(env.train_seasons) and not seen & set(env.test_seasons)
    assert rec["train_score"] >= rec["default_train_score"] and rec["train_seasons"] == env.train_seasons
    assert set(rec["params"]) == set(rec["default_params"]) and rec["evaluations"] > 100
    again = tuning.score(env, threshold_policy(rec["params"]), env.train_seasons, 1)
    assert again == pytest.approx(rec["train_score"])


def test_gain_attribution_adds_up():
    env = toy_env("test", rain=2.0)
    agent = calendar_policy({"n_steps": [0, 1, 2, 3], "w_steps": [2, 4, 6, 8], "w_level": 1, "p_steps": [3], "store": 1})
    table = attribute_gain(env, agent, BASELINES["no_input"], seeds=2).set_index("decision")
    assert list(table.index) == ["nitrogen", "water", "pests", "sale"]
    assert table["contribution"].sum() == pytest.approx(table["gain"].iloc[0])
    assert table["gain"].iloc[0] == pytest.approx(table["agent"].iloc[0] - table["baseline"].iloc[0])
    assert table.loc["nitrogen", "contribution"] > 0 and table.loc["water", "contribution"] > 0
    same = attribute_gain(env, agent, agent, seeds=1)
    assert same["contribution"].abs().max() == pytest.approx(0, abs=1e-6)


def test_sensitivity_cases_scale_one_thing_by_twenty_percent():
    ym = YieldModel()
    cases = sensitivity_cases(CFG, ym)
    assert cases[0][:3] == ("baseline", "baseline", 1.0)
    assert {c[2] for c in cases[1:]} == {0.8, 1.2}
    by = {(c[1], c[2]): c for c in cases}
    assert by[("price", 1.2)][3]["base_price"] == pytest.approx(1.2 * CFG["base_price"])
    assert by[("n_coef", 0.8)][4].n_coef == pytest.approx(0.8 * ym.n_coef) and ym.n_coef == YieldModel().n_coef
    assert by[("storage.monthly_drift", 1.2)][3]["storage"]["monthly_drift"] == pytest.approx(
        1.2 * CFG["storage"]["monthly_drift"])
    dearer = scale_config(CFG, irrigation_cost=1.2)
    assert dearer["groundwater"]["energy_price_per_kwh"] == pytest.approx(1.2 * CFG["groundwater"]["energy_price_per_kwh"])
    assert {"y_max", "n_coef", "ky", "heat_coef"} <= {c[1] for c in cases}
    assert not any("." in c[1] for c in sensitivity_cases(with_only(), ym))    # mechanisms off -> no such factors
