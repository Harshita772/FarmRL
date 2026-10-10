"""EDA of whichever processed datasets are present."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402
from farmrl import data_loader as dl  # noqa: E402
from farmrl.viz import interactive, style  # noqa: E402
from farmrl.yield_model import YieldModel  # noqa: E402

run, policies = common.setup("Data")
st.title("Data")
st.markdown("Charts are drawn for every processed table found in `data/processed/`; a dataset that has "
            "not been downloaded yet is listed as missing. Sources and licences are in `data/README.md`.")

P = dl.PROCESSED
found = {p.name for p in P.glob("*.csv")}
train_end = run.config["env"]["train_end"]


def section(title, pattern, source):
    files = sorted(f for f in found if pd.Series([f]).str.fullmatch(pattern)[0])
    st.header(title)
    st.caption(source)
    if not files:
        st.info("Not available yet (no matching file in data/processed/).")
    return files


# ------------------------------------------------------------------------------ weather
files = section("Weather", r"weather_fortnightly_.+\.csv", "NASA POWER daily data, aggregated to fortnights from sowing.")
if files:
    sites = [f.removeprefix("weather_fortnightly_").removesuffix(".csv") for f in files]
    default = sites.index(run.meta["weather_site"]) if run.meta["weather_site"] in sites else 0
    site = st.selectbox("Site", sites, index=default, format_func=str.title)
    w = pd.read_csv(P / f"weather_fortnightly_{site}.csv")
    s = dl.season_summary(w)
    a, b = st.columns(2)
    with a:
        colors = [style.PALETTE[0] if y <= train_end else style.PALETTE[1] for y in s["season"]]
        common.chart(interactive.bars(s["season"], s["rain_mm"], "Season rainfall (blue = train, orange = held out)", "mm", colors))
    with b:
        common.chart(interactive.line(s, "season", "late_t_max", "Late-season mean daily maximum temperature", "°C", style.PALETTE[1]))
    clim = w.groupby("step")[["rain_mm", "et0_mm", "t_mean"]].mean().reset_index()
    clim["fortnight"] = [pd.Timestamp(d).strftime("%d %b") for d in w[w["season"] == w["season"].min()]["start_date"]]
    a, b = st.columns(2)
    with a:
        common.chart(interactive.bars(clim["fortnight"], clim["rain_mm"], "Mean rain per fortnight", "mm"))
    with b:
        common.chart(interactive.line(clim, "fortnight", "et0_mm", "Mean reference evapotranspiration per fortnight", "mm", style.PALETTE[1]))

# ------------------------------------------------------------------------------- yields
files = section("Crop yields and fertiliser", r"yield_fit_table\.csv", "ICRISAT district-level data joined with season weather.")
if files:
    t = pd.read_csv(P / files[0])
    st.caption(f"{len(t)} district-years · states: {', '.join(sorted(t['state'].unique()))} · "
               f"{t['district'].nunique()} districts · {t['year'].min()}–{t['year'].max()}")
    by = "district" if t["district"].nunique() <= 12 else "state"      # too many districts to tell apart
    trend = t if by == "district" else t.groupby(["state", "year"], as_index=False)["yield_q_ha"].mean()
    common.chart(interactive.line(trend, "year", "yield_q_ha", f"Yield by {by}" + (" (mean of districts)" if by == "state" else ""), "q/ha", group=by))
    pairs = [("n_kg_ha", "Yield vs nitrogen", "kg N per ha of cropped area"), ("rain_mm", "Yield vs season rain", "mm"),
             ("irrigated_share", "Yield vs irrigated share of the crop", "share of wheat area irrigated"),
             ("heat_days", "Yield vs heat-stress days", "days above the heat threshold during grain filling")]
    pairs = [x for x in pairs if x[0] in t.columns and t[x[0]].notna().any()]
    for row in range(0, len(pairs), 2):
        for column, (col, title, xlabel) in zip(st.columns(2), pairs[row:row + 2]):
            with column:
                common.chart(interactive.scatter(t.dropna(subset=[col]), col, "yield_q_ha", title, xlabel, "q/ha", group=by))
    fit = run.meta.get("yield_fit") or {}
    if fit:
        flags = ", ".join(f"{k}: {'fitted' if v else 'prior'}" for k, v in fit.get("identified", {}).items())
        st.caption(f"Yield model fit used by this run: R² = {fit.get('r2', float('nan')):.2f}, "
                   f"RMSE = {fit.get('rmse', float('nan')):.2f} q/ha, n = {fit.get('n_obs')}. {flags}")

# ------------------------------------------------------------------------------- prices
files = section("Prices", r"wheat_prices(_.+)?\.csv", "Season price series: farm harvest prices of the region where available, otherwise the MSP.")
if files:
    regional = f"wheat_prices_{str(run.config['env'].get('region', '')).lower()}.csv"
    p = pd.read_csv(P / (regional if regional in files else files[0]))
    st.caption("Sources in the table: " + ", ".join(sorted(p["source"].unique())))
    a, b = st.columns(2)
    with a:
        common.chart(interactive.line(p, "season", "price", "Price (nominal)", "INR per quintal"))
    with b:
        common.chart(interactive.bars(p["season"], (p["price_index"] - 1) * 100, "Price index used by the environment", "% deviation from 5-year trend"))

# -------------------------------------------------------------------------------- costs
files = section("Costs", r"cost_of_cultivation_.+\.csv", "CACP cost of cultivation, transcribed from the price policy report.")
if files:
    c = pd.read_csv(P / files[0])
    latest = c[(c["year"] == c["year"].max()) & (c["group"] == "operational")].sort_values("value")
    common.chart(interactive.bars(latest["item"].str.replace("_", " "), latest["value"],
                                  f"Operational cost items, {c['year'].max()}", "INR per hectare", horizontal=True))

# -------------------------------------------------------------------- optional datasets
files = section("Pesticide use (FAO)", r"fao_.+\.csv",
                "Kaggle crop yield prediction dataset (FAOSTAT yield and pesticide use, World Bank rainfall and temperature); "
                "national series, pesticide use is the total for all crops.")
if files:
    f = pd.read_csv(P / files[0])
    st.caption(f"{len(f)} years, {f['year'].min()}–{f['year'].max()} · pesticide measure: {f['normalisation'].iloc[0]} · "
               f"{int((f['source'] != 'yield_df.csv').sum())} year(s) restored from the source files")
    a, b = st.columns(2)
    with a:
        common.chart(interactive.line(f, "year", "pesticides_t", "National pesticide use", "tonnes of active ingredient"))
    with b:
        common.chart(interactive.line(f, "year", "yield_q_ha", "National wheat yield", "q/ha", style.PALETTE[1]))
    cal = YieldModel().calibrate_pesticide(f)
    chart_data = pd.DataFrame({"pesticide use (series mean = 1)": cal["x"], "yield": cal["observed"],
                               "adjusted": cal["partial_yield"], "year": cal["year"]})
    a, b = st.columns(2)
    with a:
        common.chart(interactive.scatter(chart_data, "pesticide use (series mean = 1)", "yield",
                                         "Yield vs pesticide use (raw)", "pesticide use (series mean = 1)", "q/ha"))
    with b:
        common.chart(interactive.scatter(chart_data, "pesticide use (series mean = 1)", "adjusted",
                                         "Yield adjusted to the mean year and weather", "pesticide use (series mean = 1)", "q/ha"))
    lo, hi = cal["interval_90"]
    used = run.meta["yield_model"].get("pest_loss")
    st.markdown(f"**Calibrated pesticide loss:** L = {cal['estimate']:.2f} ± {cal['std_err']:.2f} "
                f"(90% interval {lo:.2f}–{hi:.2f}), n = {cal['n_obs']} national years. {cal['quality_note'].capitalize()}. "
                f"Controls: {', '.join(cal['controls'])}"
                + (f"; no variation in the series for: {', '.join(cal['controls_dropped_no_variation'])}" if cal["controls_dropped_no_variation"] else "")
                + f". **This run used L = {used:.2f}.**")
files = section("Mandi prices (Agmarknet)", r"wheat_mandi_monthly\.csv", "Agmarknet / data.gov.in monthly mean modal price.")
if files:
    m = pd.read_csv(P / files[0])
    common.chart(interactive.line(m, "date", "modal_price", "Monthly modal price", "INR per quintal"))

st.header("Files this run was built from")
st.dataframe(pd.DataFrame(run.meta["data_files"]), hide_index=True, use_container_width=True)
