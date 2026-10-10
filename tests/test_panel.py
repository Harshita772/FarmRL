"""All-India panel: ICRISAT irrigation / price files, district centroids and weather,
region harvest prices, the pooled yield fit and the heat / deficit terms in the env.
Everything runs on small synthetic files in tmp_path."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import pytest

from farmrl import data_loader as dl, districts
from farmrl.baselines import BASELINES, recommended_schedule
from farmrl.env import FarmEnv, calibrate_ky
from farmrl.evaluation import run_episode
from farmrl.utils import load_config
from farmrl.yield_model import NO_RAIN_EFFECT, YieldModel
from .test_data_loader import KEYS, rbi_page, write_power

CFG = load_config(ROOT / "configs/default.yaml")["env"]


def write_icrisat_panel(raw):
    rows = [(1, "4", "Haryana", "Hissar"), (2, "9", "Punjab", "Ludhiana"), (3, "9", "Punjab", "Tiny")]
    crops, fert, irr, price = [], [], [], []
    for code, sc, state, dist in rows:
        for year in (2010, 2011):
            area = 5.0 if dist == "Tiny" else 100.0
            crops.append(f'"{code}","{year}","{sc}","{state}","{dist}","{area}","450.0","4500.0"')
            fert.append(f'"{code}","{year}","{sc}","{state}","{dist}","{-1.0 if (code, year) == (1, 2011) else 150.0}"')
            irr.append(f'"{code}","{year}","{sc}","{state}","{dist}","{120.0 if code == 2 else 50.0}"')
            # price rows are labelled by marketing year: year + 1 holds the price of season year
            price.append(f'"{code}","{year + 1}","{sc}","{state}","{dist}","{1000 + 100 * (year - 2010) + 10 * code}"')
    price.append('"1","2010","4","Haryana","Hissar","-1"')
    (raw / "icrisat_crops.csv").write_text(
        f'{KEYS},"WHEAT AREA (1000 ha)","WHEAT PRODUCTION (1000 tons)","WHEAT YIELD (Kg per ha)"\n' + "\n".join(crops))
    (raw / "icrisat_fertiliser.csv").write_text(f'{KEYS},"NITROGEN PER HA OF GCA (Kg per ha)"\n' + "\n".join(fert))
    (raw / "icrisat_irrigation.csv").write_text(f'{KEYS},"WHEAT IRRIGATED AREA (1000 ha)"\n' + "\n".join(irr))
    (raw / "icrisat_prices.csv").write_text(f'{KEYS},"WHEAT HARVEST PRICE (Rs per Quintal)"\n' + "\n".join(price))


def test_icrisat_panel_columns_and_cleaning(tmp_path):
    write_icrisat_panel(tmp_path)
    d = dl.load_icrisat("wheat", min_area_kha=10, raw=tmp_path, save=False)
    assert sorted(d["district"].unique()) == ["Hissar", "Ludhiana"]          # area filter drops "Tiny"
    row = d.set_index(["district", "year"])
    assert row.loc[("Hissar", 2010), "irrigated_share"] == 0.5
    assert row.loc[("Ludhiana", 2010), "irrigated_share"] == 1.0             # 120/100 is clipped
    assert np.isnan(row.loc[("Hissar", 2011), "n_kg_ha"])                    # -1 means not reported
    assert row.loc[("Ludhiana", 2010), "harvest_price"] == 1020              # marketing year 2011 -> season 2010
    assert row.loc[("Ludhiana", 2011), "harvest_price"] == 1120
    assert len(d) == 4 and d["dist_code"].tolist() == [1, 1, 2, 2]           # the price-only row adds nothing


def test_region_harvest_prices_replace_msp_with_their_own_index(tmp_path):
    write_icrisat_panel(tmp_path)
    rows = [(f"{y}-{(y + 1) % 100:02d}", 200, 500 + 50 * (y - 2000)) for y in range(2000, 2016)]
    (tmp_path / "rbi_msp_foodgrains_latest.html").write_text(rbi_page(rows), encoding="utf-8")
    national = dl.load_prices(raw=tmp_path, processed=tmp_path).set_index("season")
    punjab = dl.load_prices(region="punjab", raw=tmp_path, processed=tmp_path).set_index("season")
    assert (tmp_path / "wheat_prices_punjab.csv").exists() and (tmp_path / "wheat_prices.csv").exists()
    # mean over the region's districts (Ludhiana 1020, Tiny 1030)
    assert punjab.loc[2010, "price"] == 1025 and punjab.loc[2010, "source"].startswith("ICRISAT")
    assert punjab.loc[2009, "source"].startswith("MSP") and punjab.loc[2009, "price"] == national.loc[2009, "price"]
    assert punjab.loc[2010, "msp"] == 1000                                   # MSP kept alongside
    # index is relative to the harvest series itself, not to the MSP level around it
    assert punjab.loc[[2010, 2011], "price_index"].between(0.9, 1.1).all()
    assert punjab.loc[2012, "price_index"] == national.loc[2012, "price_index"]


def test_heat_days_and_district_weather(tmp_path):
    folder = tmp_path / dl.DISTRICT_WEATHER
    folder.mkdir()
    write_power(folder / "7.csv", "2009-01-01", "2012-12-31")
    daily = dl.read_power(folder / "7.csv")
    assert daily["heat_day"].sum() == 0                                      # t_max is 22 in the synthetic file
    text = (folder / "7.csv").read_text().replace(",15.0,22.0,8.0,", ",25.0,35.0,18.0,")
    (folder / "7.csv").write_text(text)
    wx = dl.load_district_weather(raw=tmp_path, processed=tmp_path)
    assert set(wx["dist_code"]) == {7} and wx["year"].tolist() == [2009, 2010, 2011]
    assert (wx["heat_days"] == [45, 45, 46]).all()                           # 15 Feb - 31 Mar, leap year 2012
    seasons = dl.build_seasons(dl.read_power(folder / "7.csv"))
    assert seasons.groupby("season")["heat_days"].sum().tolist() == [45, 45, 46]


def test_fit_table_uses_district_weather_when_cached(tmp_path):
    write_icrisat_panel(tmp_path)
    folder = tmp_path / dl.DISTRICT_WEATHER
    folder.mkdir()
    write_power(folder / "2.csv", "2009-01-01", "2012-12-31")
    table = dl.build_fit_table(raw=tmp_path, processed=tmp_path / "out")
    assert set(table["district"]) == {"Ludhiana"} and {"heat_days", "irrigated_share", "harvest_price"} <= set(table)


def square(name, lon, lat):
    ring = [[lon, lat], [lon + 1, lat], [lon + 1, lat + 1], [lon, lat + 1], [lon, lat]]
    return {"type": "Feature", "properties": {"shapeName": name}, "geometry": {"type": "Polygon", "coordinates": [ring]}}


def test_district_sites_disambiguate_by_state():
    geo = {"type": "FeatureCollection", "features": [square("Hamirpur", 76, 31), square("Hamirpur", 80, 25),
                                                     square("Bathinda", 74.5, 30), square("Hisar", 75, 29)]}
    panel = pd.DataFrame({"dist_code": [1, 2, 3, 4, 5], "state": ["Himachal Pradesh", "Uttar Pradesh", "Punjab", "Haryana", "Punjab"],
                          "district": ["Hamirpur", "Hamirpur", "Bhatinda", "Hissar", "Nowhere"]})
    s = districts.district_sites(panel, geo).set_index("dist_code")
    assert s.loc[1, "lat"] == pytest.approx(31.4) and s.loc[2, "lat"] == pytest.approx(25.4)   # same name, two states
    assert s.loc[3, "geo_name"] == "Bathinda" and s.loc[4, "geo_name"] == "Hisar"              # aliases
    assert not s.loc[5, "matched"] and np.isnan(s.loc[5, "lat"])                               # never guessed


# ------------------------------------------------------------------------ pooled fit
def synthetic_panel(n=900, seed=0):
    rng = np.random.default_rng(seed)
    truth = {"A": {"North": 50.0, "Centre": 35.0, "South": 20.0}, "n_coef": 0.02, "k_dry": 0.5, "heat": 0.004}
    d = pd.DataFrame({"state": rng.choice(list(truth["A"]), n), "year": rng.integers(2008, 2018, n),
                      "dist_code": rng.integers(1, 60, n), "n_kg_ha": rng.uniform(10, 250, n),
                      "irrigated_share": rng.uniform(0, 1, n), "rain_mm": rng.uniform(10, 150, n),
                      "heat_days": rng.integers(0, 45, n).astype(float)})
    d["yield_q_ha"] = (d["state"].map(truth["A"]) * (1 - np.exp(-truth["n_coef"] * (d["n_kg_ha"] + 30)))
                       * (1 - truth["k_dry"] * (1 - d["irrigated_share"])) * (1 - truth["heat"] * d["heat_days"])
                       + rng.normal(0, 0.6, n))
    return d, truth


def test_fit_panel_recovers_parameters_and_scores_holdout(tmp_path):
    d, truth = synthetic_panel()
    m = YieldModel()
    rep = m.fit_panel(d, region="north", holdout_from=2015)
    assert m.state_effects["North"] == pytest.approx(50, rel=0.04) and m.y_max == m.state_effects["North"]
    assert m.n_coef == pytest.approx(truth["n_coef"], rel=0.12) and m.k_dry == pytest.approx(truth["k_dry"], abs=0.05)
    assert m.heat_coef == pytest.approx(truth["heat"], rel=0.15)
    assert not rep["identified"]["w_excess"] and m.w_excess == 0.0           # no excess-rain effect in the data
    assert not rep["identified"]["r0"] and m.r0 == NO_RAIN_EFFECT            # nor a rain dependence of the dry loss
    h = rep["holdout"]
    assert h["from_year"] == 2015 and h["n_train"] + h["n_test"] == len(d)
    assert h["r2_test"] > 0.95 > h["r2_test_state_means_only"] and h["within_state_skill"] > 0.9
    m.save(tmp_path / "p.yaml", source="synthetic")
    again = YieldModel.load(tmp_path / "p.yaml")
    assert again.params() == m.params() and again.heat_coef == m.heat_coef
    other = YieldModel()
    other.fit_panel(d, region="nowhere")
    assert other.report["region_in_panel"] is False and 20 < other.y_max < 50


def test_heat_term_lowers_yield_in_the_env():
    weather = pd.DataFrame([{"season": s, "step": t, "rain_mm": 8.0, "et0_mm": 40.0, "rh": 50.0,
                             "heat_days": (5.0 if t >= 8 else 0.0) * (s - 2015)} for s in (2015, 2016) for t in range(12)])
    env = FarmEnv(CFG, weather=weather, prices=None, split="all", yield_model=YieldModel(heat_coef=0.01))
    cool = run_episode(env, BASELINES["recommended"], season=2015, seed=1)
    hot = run_episode(env, BASELINES["recommended"], season=2016, seed=1)
    assert cool["heat_days"] == 0 and hot["heat_days"] == 20
    assert hot["yield"] == pytest.approx(cool["yield"] * 0.8)


def test_calibrate_ky_matches_fitted_rainfed_loss():
    weather = pd.DataFrame([{"season": s, "step": t, "rain_mm": 6.0, "et0_mm": 40.0, "rh": 50.0}
                            for s in (2014, 2015) for t in range(12)])
    cfg = dict(CFG)
    model = YieldModel(k_dry=0.4, r0=NO_RAIN_EFFECT)
    import farmrl.env as env_module
    original = env_module.FarmEnv
    env_module.FarmEnv = lambda c, split, yield_model: original(c, weather=weather, prices=None, split=split, yield_model=yield_model)
    try:
        ky = calibrate_ky(cfg, model, recommended_schedule)
        env = original(cfg, weather=weather, prices=None, split="train", yield_model=model)
        dry = run_episode(env, lambda o, e: recommended_schedule(o, e) * [1, 0, 1], season=2014, seed=0)
    finally:
        env_module.FarmEnv = original
    target = float(model.rainfed_loss(dry["rain_total"]))                    # what the panel says a rainfed crop loses
    assert target == pytest.approx(0.4, rel=1e-3) and ky == pytest.approx(target / (1 - dry["et_ratio"]), rel=1e-6)
    assert calibrate_ky(cfg, YieldModel(), recommended_schedule) == 1.15     # nothing fitted -> literature value


# ------------------------------------------------------------- pesticide calibration
def pesticide_series(loss, n=24, noise=0.004, seed=0, rain_varies=False):
    rng = np.random.default_rng(seed)
    year = np.arange(1990, 1990 + n)
    x = np.clip(1.6 - 0.05 * (year - 1990) + rng.normal(0, 0.25, n), 0.2, None)      # use falls while yield rises
    temp = 26 + 0.03 * (year - 1990) + rng.normal(0, 0.3, n)
    y = 21 * np.exp(0.013 * (year - 1990) - 0.01 * (temp - 26)) * (1 - loss * np.exp(-x / x.mean()))
    return pd.DataFrame({"year": year, "pesticide_x": x, "avg_temp": temp, "normalisation": "index (test)",
                         "rain_mm_year": 1083.0 + (rng.normal(0, 40, n) if rain_varies else 0.0),
                         "yield_q_ha": y * np.exp(rng.normal(0, noise, n))})


def test_pesticide_calibration_recovers_loss_despite_opposite_trends():
    d = pesticide_series(0.30)
    assert np.corrcoef(d["pesticide_x"], d["yield_q_ha"])[0, 1] < 0          # the raw relation has the wrong sign
    m = YieldModel()
    rep = m.calibrate_pesticide(d)
    assert rep["status"] == "calibrated" and m.pest_loss == pytest.approx(0.30, abs=0.06)
    assert rep["previous_value"] == 0.35 and rep["n_obs"] == 24 and rep["pesticide_measure"] == "index (test)"
    assert rep["controls"] == ["year", "temperature"] and rep["controls_dropped_no_variation"] == ["rainfall"]
    assert rep["r2_with_pesticide"] > rep["r2_controls_only"]
    assert YieldModel().calibrate_pesticide(pesticide_series(0.30, rain_varies=True))["controls"][-1] == "rainfall"


def test_pesticide_calibration_keeps_previous_value_without_signal(tmp_path):
    d = pesticide_series(0.0, noise=0.02, seed=3)
    d["yield_q_ha"] *= np.exp(-0.05 * (d["pesticide_x"] - 1))                 # more use, slightly lower yield
    m = YieldModel(pest_loss=0.35)
    rep = m.calibrate_pesticide(d)
    assert rep["status"].startswith("not calibrated") and m.pest_loss == 0.35 and "kept" in rep["quality_note"]
    weak = YieldModel()
    report = weak.calibrate_pesticide(pesticide_series(0.05, noise=0.03, seed=5))
    if report["status"] == "calibrated":
        assert "weak" in report["quality_note"] or abs(report["t_ratio"]) >= 2
    weak.report = {"identified": {}, "pesticide": report}
    weak.save(tmp_path / "p.yaml", source="test")
    import yaml
    saved = yaml.safe_load((tmp_path / "p.yaml").read_text())
    assert saved["fit"]["pest_loss_status"] == report["status"] and "quality_note" in saved["fit"]["pesticide"]
    with pytest.raises(ValueError):
        YieldModel().calibrate_pesticide(d.head(5))
