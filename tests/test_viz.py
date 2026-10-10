"""Rollout schema, experiment outputs, figures, animation, maps and the dashboard, all on
a tiny synthetic run (toy weather, one train and one test season, 2 seeds, a few PPO steps)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from farmrl import live, rollout, runs
from farmrl.baselines import BASELINES
from farmrl.env import FarmEnv, OBS_NAMES
from farmrl.experiment import run_experiment
from farmrl.utils import load_config
from farmrl.viz import animate, figures, interactive, maps, style
from farmrl.yield_model import YieldModel

APP = ROOT / "app"
PAGES = [APP / "streamlit_app.py", *sorted((APP / "pages").glob("*.py"))]


def toy_weather(seasons=(2016, 2017), steps=12):
    rows = []
    for s in seasons:
        for t in range(steps):
            rows.append({"season": s, "step": t, "start_date": pd.Timestamp(s, 11, 1) + pd.Timedelta(days=15 * t),
                         "rain_mm": 4.0 + 3 * t * (s - 2015), "et0_mm": 40.0, "rh": 50.0})
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def tiny_run(tmp_path_factory):
    cfg = load_config(ROOT / "configs/default.yaml")
    cfg["train"].update(n_steps=96, batch_size=32, epochs=2)
    cfg["compute"] = {"torch_threads": 1, "parallel_seeds": 1}      # in-process: keeps the fixture fast
    out = run_experiment(cfg, run_name="tiny", seeds=2, train_seeds=2, timesteps=192, with_sensitivity=True,
                         runs_dir=tmp_path_factory.mktemp("runs"), weather=toy_weather(),
                         yield_model=YieldModel(), log=lambda *_: None)
    return runs.get(out)


# ------------------------------------------------------------------------------ rollout
def test_rollout_schema():
    cfg = load_config(ROOT / "configs/default.yaml")["env"]
    env = FarmEnv(cfg, weather=toy_weather(), prices=None, split="all", yield_model=YieldModel())
    df = rollout.run_episode(env, BASELINES["recommended"], season=2017, seed=1, policy_name="recommended")
    need = {"season", "step", "date", "policy", "seed", "soil_moisture", "soil_n", "pest", "price", "rain_mm",
            "nitrogen_kg", "irrigation_mm", "pesticide_level", "input_cost", "penalty", "reward", "cum_profit",
            "crop_growth", "final_yield", "final_revenue", "final_profit", *[f"obs_{k}" for k in OBS_NAMES]}
    assert need <= set(df.columns) and len(df) == env.T
    assert df["step"].tolist() == list(range(env.T)) and df["date"].is_monotonic_increasing
    assert set(df["nitrogen_kg"]) <= set(env.n_levels) and set(df["irrigation_mm"]) <= set(env.w_levels)
    assert df["cum_profit"].iloc[-1] == pytest.approx(df["final_profit"].iloc[0])
    assert df["reward"].sum() == pytest.approx(df["final_profit"].iloc[0] - df["penalty"].sum())
    assert df["crop_growth"].is_monotonic_increasing and 0 < df["crop_growth"].iloc[-1] <= 1
    assert not df.isna().any().any()


def test_run_many_and_tables():
    cfg = load_config(ROOT / "configs/default.yaml")["env"]
    env = FarmEnv(cfg, weather=toy_weather(), prices=None, split="test", yield_model=YieldModel())
    r = rollout.run_many(env, {k: BASELINES[k] for k in ("no_input", "recommended")}, seeds=3)
    eps = rollout.episodes(r)
    assert len(r) == 2 * 1 * 3 * env.T and len(eps) == 6 and set(eps["split"]) == {"test"}
    s = rollout.summary(eps).set_index("policy")
    assert s.loc["recommended", "nitrogen_kg"] == 120 and s.loc["no_input", "irrigation_mm"] == 0
    assert s.loc["recommended", "profit"] > s.loc["no_input", "profit"]


# --------------------------------------------------------------------------- experiment
def test_run_directory_is_complete(tiny_run):
    for name in ["rollouts.parquet", "episodes.csv", "summary.csv", "train_history.csv", "model.pt",
                 "config_snapshot.yaml", "metadata.json", "policy_grids.parquet", "state_medians.csv",
                 "fitted_params.yaml", "sensitivity.csv"]:
        assert (tiny_run.path / name).exists(), name
    meta = tiny_run.meta
    assert meta["train_seasons"] == [2016] and meta["test_seasons"] == [2017] and len(meta["train_seeds"]) == 2
    assert set(BASELINES) | {"ppo", "ppo_untrained"} == set(meta["policies"])      # baselines are discovered
    assert "ppo_untrained" not in tiny_run.policies() and "ppo_untrained" in tiny_run.policies(include_aux=True)
    assert {"train_seed", "iteration", "mean_return", "policy_loss", "value_loss", "entropy", "kl"} <= set(tiny_run.history)
    assert set(tiny_run.rollouts["split"]) == {"train", "test"}
    assert set(tiny_run.grids["grid"]) == {"soil_moisture", "soil_n"}
    assert runs.list_runs(tiny_run.path.parent) == ["tiny"] and runs.get(None, tiny_run.path.parent).name == "tiny"


def test_parallel_training_matches_serial(tiny_run):
    from farmrl.experiment import train_in_pool
    cfg = tiny_run.config
    args = (cfg, [42, 43], 192, YieldModel(), toy_weather(), None)
    serial_w, serial_h = train_in_pool(*args, parallel=1, threads=1, log=lambda *_: None)
    pool_w, pool_h = train_in_pool(*args, parallel=2, threads=1, log=lambda *_: None)
    pd.testing.assert_frame_equal(serial_h, pool_h)                 # same seeds -> same curves
    assert all((serial_w[s][k] == pool_w[s][k]).all() for s in (42, 43) for k in serial_w[s])
    saved = tiny_run.history[["train_seed", "mean_return"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(saved, serial_h[["train_seed", "mean_return"]], check_exact=False)


def test_no_runs_gives_instructions(tmp_path):
    assert runs.list_runs(tmp_path) == []
    with pytest.raises(FileNotFoundError, match="run_all.py"):
        runs.get(None, tmp_path)


# ------------------------------------------------------------------------------ figures
@pytest.mark.parametrize("name", list(figures.FIGURES))
def test_figure_functions(tiny_run, name):
    style.apply()
    try:
        fig = figures.FIGURES[name](tiny_run)
    except figures.SkipFigure:
        return                                   # allowed: the data for it is not there
    assert isinstance(fig, plt.Figure) and fig.axes
    plt.close(fig)


def test_make_all_saves_png_and_pdf_and_skips_cleanly(tiny_run, tmp_path):
    with pytest.warns(UserWarning, match="skipped"):
        report = figures.make_all(tiny_run, out_dir=tmp_path, only=["learning_curve", "pesticide_response"],
                                  table=tmp_path / "missing.csv")
    assert (tmp_path / "learning_curve.png").exists() and (tmp_path / "learning_curve.pdf").exists()
    assert report["pesticide_response"].startswith("skipped")


def test_policy_colours_are_fixed_per_name():
    a = style.policy_colors(["ppo", "threshold", "new_rule"])
    b = style.policy_colors(["new_rule", "ppo"])
    assert a["ppo"] == b["ppo"] and a["new_rule"] == b["new_rule"] and len(set(a.values())) == 3


# ---------------------------------------------------------------------------- animation
def test_animation_gif(tiny_run, tmp_path):
    paths = animate.save_animation(tiny_run, policy="ppo", compare="recommended", out_dir=tmp_path,
                                   sub=1, hold=1, demo_path=tmp_path / "docs" / "demo.gif")
    assert paths[0].suffix == ".gif" and paths[0].stat().st_size > 1000 and (tmp_path / "docs/demo.gif").exists()


# --------------------------------------------------------------------------------- maps
def square(name, x, y):
    ring = [[x, y], [x + 1, y], [x + 1, y + 1], [x, y + 1], [x, y]]
    return {"type": "Feature", "properties": {"shapeName": name}, "geometry": {"type": "Polygon", "coordinates": [ring]}}


GEOJSON = {"type": "FeatureCollection", "features": [square("Hisar", 75, 29), square("Karnal", 76.5, 29),
                                                     square("Lakhisarai", 86, 25), square("Hamirpur", 76, 31),
                                                     square("Hamirpur", 80, 25)]}


def test_district_matching_uses_aliases_fuzziness_and_coordinates():
    m = maps.match_districts(["Hissar", "Karnaal", "Hamirpur", "Atlantis"], GEOJSON, coords={"Hamirpur": (25.5, 80.5)})
    assert m["Hissar"]["properties"]["shapeName"] == "Hisar"            # alias
    assert m["Karnaal"]["properties"]["shapeName"] == "Karnal"          # fuzzy
    assert m["Hamirpur"]["geometry"]["coordinates"][0][0] == [80, 25]   # the one containing the point
    assert "Atlantis" not in m


def test_maps_from_district_table(tiny_run, tmp_path, monkeypatch):
    table = pd.DataFrame({"policy": ["ppo", "recommended"] * 2, "district": ["Hissar", "Hissar", "Karnal", "Karnal"],
                          "lat": [29.5, 29.5, 29.5, 29.5], "lon": [75.5, 75.5, 77.0, 77.0],
                          "profit": [80e3, 70e3, 78e3, 79e3], "ppo_profit_gain": [0, 10e3, 0, -1e3]})
    monkeypatch.setitem(tiny_run.__dict__, "districts", table)
    fig = maps.static_map(tiny_run, "ppo_profit_gain", geojson=GEOJSON)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    assert isinstance(maps.interactive_map(tiny_run, "profit", "ppo", geojson=GEOJSON), go.Figure)
    monkeypatch.setitem(tiny_run.__dict__, "districts", None)
    with pytest.raises(maps.MapUnavailable):
        maps.static_map(tiny_run, geojson=GEOJSON)


# ------------------------------------------------------------------------ plotly + live
def test_interactive_charts(tiny_run):
    pol = tiny_run.policies()
    charts = [interactive.profit_box(tiny_run, pol), interactive.inputs_bar(tiny_run, pol),
              interactive.policy_heatmap(tiny_run, "soil_moisture", "expected_irrigation_mm"),
              interactive.learning_curve(tiny_run), interactive.diagnostics(tiny_run),
              interactive.profit_by_season(tiny_run, pol), interactive.dry_wet(tiny_run, pol),
              interactive.rain_response(tiny_run, pol), interactive.sensitivity(tiny_run, pol)]
    assert all(isinstance(c, go.Figure) and c.data for c in charts)
    k = interactive.headline(tiny_run)
    assert k["baseline"] == "recommended" and np.isfinite(k["profit_gain"])
    assert len(interactive.summary_table(tiny_run)) == len(pol)


def test_live_simulation_matches_saved_schema(tiny_run):
    agent = live.load_agent(tiny_run)
    steps = live.simulate(tiny_run, agent, 2017, ["ppo", "recommended"], price=1.2, irrigation_cost=3.0)
    assert set(steps) == {"ppo", "recommended"} and set(tiny_run.rollouts.columns) - {"split"} <= set(steps["ppo"].columns)
    assert steps["recommended"]["price"].iloc[0] == pytest.approx(1.2 * live.make_env(tiny_run).price_index.get(2017, 1.0)
                                                                  * tiny_run.config["env"]["base_price"])
    fig = interactive.season_player(steps)
    assert len(fig.frames) == len(steps["ppo"])
    dist = live.action_distribution(tiny_run, agent, {k: 0.5 for k in OBS_NAMES})
    assert all(p.sum() == pytest.approx(1.0, abs=1e-5) for _, p in dist.values())


# ---------------------------------------------------------------------------- dashboard
@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
def test_dashboard_pages_run_without_exceptions(tiny_run, page, monkeypatch):
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    monkeypatch.setattr(runs, "RUNS_DIR", tiny_run.path.parent)
    monkeypatch.syspath_prepend(str(APP))
    at = AppTest.from_file(str(page), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]


def test_dashboard_without_runs_shows_instructions(tmp_path, monkeypatch):
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path)
    monkeypatch.syspath_prepend(str(APP))
    at = AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=60).run()
    assert not at.exception and any("No experiment run" in w.value for w in at.warning)
    assert any("run_all.py" in c.value for c in at.code)
