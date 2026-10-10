"""Load and clean the raw datasets into tidy tables in data/processed/.

Each ``load_*`` function reads from ``raw`` (default data/raw), returns a DataFrame and,
when ``save=True``, writes the matching CSV to ``processed``. A missing raw file raises
``FileNotFoundError`` naming the file and the script that fetches it, so callers can
decide whether the dataset is optional.

A *season* is labelled by its sowing year: season 2016 runs from 1 Nov 2016 for
``steps * step_days`` days (wheat harvested in April 2017). ICRISAT's "Year" and the
RBI "crop year" 2016-17 follow the same convention.
"""
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REFERENCE = ROOT / "data" / "reference"

# NASA POWER grid points. "ludhiana" drives the environment; the others are the head-
# quarters of the ICRISAT districts in the files we have and are only used to attach
# season weather to the district yield records when fitting the yield model.
SITES = {
    "ludhiana": {"lat": 30.90, "lon": 75.85, "state": "Punjab", "district": "Ludhiana"},
    "hisar": {"lat": 29.15, "lon": 75.72, "state": "Haryana", "district": "Hissar"},
    "gurgaon": {"lat": 28.46, "lon": 77.03, "state": "Haryana", "district": "Gurgaon"},
    "jind": {"lat": 29.32, "lon": 76.31, "state": "Haryana", "district": "Jind"},
    "narnaul": {"lat": 28.04, "lon": 76.11, "state": "Haryana",
                "district": "Mahendragarh / Narnaul"},
    "ambala": {"lat": 30.38, "lon": 76.78, "state": "Haryana", "district": "Ambala"},
    "karnal": {"lat": 29.69, "lon": 76.99, "state": "Haryana", "district": "Karnal"},
    "rohtak": {"lat": 28.90, "lon": 76.61, "state": "Haryana", "district": "Rohtak"},
}

SEASON_START = (11, 1)   # month, day
TRAIN_END = 2016         # seasons <= this are train, later ones are held out

_WEATHER_COLS = {"PRECTOTCORR": "rain_mm", "T2M": "t_mean", "T2M_MAX": "t_max",
                 "T2M_MIN": "t_min", "RH2M": "rh", "ALLSKY_SFC_SW_DWN": "srad"}
_ICRISAT_KEYS = ["Dist Code", "Year", "State Code", "State Name", "Dist Name"]


def _need(path, how):
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found -- {how}")
    return Path(path)


def _write(df, processed, name, save):
    if save:
        Path(processed).mkdir(parents=True, exist_ok=True)
        df.to_csv(Path(processed) / name, index=False)
    return df


# ------------------------------------------------------------------------------ weather
def load_weather(site="ludhiana", raw=RAW, processed=PROCESSED, save=True):
    """NASA POWER daily CSV -> tidy daily table (date, rain_mm, t_mean, t_max, t_min, rh,
    srad [MJ/m2/day], et0_mm). POWER's -999 fill value becomes NaN; short gaps (<=5 days)
    are interpolated, rain gaps are left as NaN."""
    path = _need(Path(raw) / f"weather_{site}.csv", "run scripts/download_data.py weather")
    lines = path.read_text().splitlines()
    start = next((i for i, l in enumerate(lines) if "-END HEADER-" in l), -1) + 1
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])))
    df["date"] = pd.to_datetime(df["YEAR"].astype(str) + df["DOY"].astype(str).str.zfill(3),
                                format="%Y%j")
    df = df.rename(columns=_WEATHER_COLS)[["date", *_WEATHER_COLS.values()]]
    df = df.replace(-999.0, np.nan).sort_values("date").drop_duplicates("date")
    smooth = ["t_mean", "t_max", "t_min", "rh", "srad"]
    df[smooth] = df[smooth].interpolate(limit=5, limit_direction="both")
    df["rain_mm"] = df["rain_mm"].clip(lower=0)
    # Reference evapotranspiration, Hargreaves radiation form (Hargreaves & Samani 1982):
    # ET0 = 0.0135 (T + 17.8) Rs / lambda, with Rs in MJ/m2/day and lambda = 2.45 MJ/kg.
    df["et0_mm"] = (0.0135 * (df["t_mean"] + 17.8) * df["srad"] / 2.45).clip(lower=0)
    return _write(df.reset_index(drop=True), processed, f"weather_daily_{site}.csv", save)


def build_seasons(daily, steps=12, step_days=15, start=SEASON_START):
    """Aggregate a daily table into (season, step) rows. Seasons with any missing day or
    missing rainfall are dropped rather than filled."""
    d = daily.set_index("date")
    rows = []
    for year in range(d.index.year.min(), d.index.year.max() + 1):
        t0 = pd.Timestamp(year, *start)
        season = d.loc[t0: t0 + pd.Timedelta(days=steps * step_days - 1)]
        if len(season) < steps * step_days or season[["rain_mm", "t_mean", "et0_mm"]].isna().any().any():
            continue
        step = np.arange(len(season)) // step_days
        g = season.groupby(step)
        out = pd.DataFrame({"season": year, "step": np.arange(steps),
                            "start_date": g.apply(lambda x: x.index[0]).dt.date.values,
                            "rain_mm": g["rain_mm"].sum().values,
                            "et0_mm": g["et0_mm"].sum().values,
                            "t_mean": g["t_mean"].mean().values,
                            "t_max": g["t_max"].mean().values,
                            "t_min": g["t_min"].mean().values,
                            "rh": g["rh"].mean().values,
                            "srad": g["srad"].mean().values})
        rows.append(out)
    if not rows:
        raise ValueError("no complete season in the daily weather table")
    return pd.concat(rows, ignore_index=True).round(3)


def load_seasons(site="ludhiana", steps=12, step_days=15, raw=RAW, processed=PROCESSED, save=True):
    """Fortnightly season table for one site (also written to data/processed)."""
    seasons = build_seasons(load_weather(site, raw, processed, save), steps, step_days)
    return _write(seasons, processed, f"weather_fortnightly_{site}.csv", save)


def season_summary(seasons):
    """One row per season: total rain / ET0, mean temperature, late-season heat."""
    late = seasons[seasons["step"] >= seasons["step"].max() - 4]   # roughly mid-Feb onwards
    out = seasons.groupby("season").agg(rain_mm=("rain_mm", "sum"), et0_mm=("et0_mm", "sum"),
                                        t_mean=("t_mean", "mean")).reset_index()
    out = out.merge(late.groupby("season").agg(late_t_max=("t_max", "mean"),
                                               late_rain_mm=("rain_mm", "sum")).reset_index())
    return out.round(3)


def split_seasons(seasons, train_end=TRAIN_END):
    """Sorted lists of (train, test) season labels: train <= train_end < test."""
    years = sorted(int(y) for y in pd.unique(seasons["season"]))
    return [y for y in years if y <= train_end], [y for y in years if y > train_end]


# ------------------------------------------------------------------------------ ICRISAT
def load_icrisat(crop="wheat", state=None, raw=RAW, processed=PROCESSED, save=True):
    """ICRISAT District Level Data -> one row per district-year for ``crop``.

    Every ``icrisat*.csv`` in ``raw`` is merged on the district/year keys, so the crops,
    fertiliser, irrigation, ... exports can be dropped in side by side. ``state=None``
    keeps every state present. Area/yield of 0 (ICRISAT's "not reported") becomes NaN.
    """
    files = sorted(Path(raw).glob("icrisat*.csv"))
    if not files:
        raise FileNotFoundError(f"no icrisat*.csv in {raw} -- see scripts/download_data.py icrisat")
    merged = None
    for f in files:
        part = pd.read_csv(f)
        part.columns = [c.strip() for c in part.columns]
        if not set(_ICRISAT_KEYS) <= set(part.columns):
            raise ValueError(f"{f.name}: expected ICRISAT key columns {_ICRISAT_KEYS}")
        new = [c for c in part.columns if merged is None or c not in merged.columns]
        merged = part if merged is None else merged.merge(part[_ICRISAT_KEYS + new],
                                                          on=_ICRISAT_KEYS, how="outer")
    if state is not None:
        merged = merged[merged["State Name"].str.lower() == state.lower()]
    C = crop.upper()
    need = [f"{C} AREA (1000 ha)", f"{C} PRODUCTION (1000 tons)", f"{C} YIELD (Kg per ha)"]
    if not set(need) <= set(merged.columns):
        raise ValueError(f"ICRISAT files have no {crop} area/production/yield columns")
    out = pd.DataFrame({"state": merged["State Name"], "district": merged["Dist Name"],
                        "year": merged["Year"].astype(int),
                        "area_kha": merged[need[0]], "production_kt": merged[need[1]],
                        "yield_kg_ha": merged[need[2]]})
    optional = {"NITROGEN PER HA OF GCA (Kg per ha)": "n_kg_ha",
                "PHOSPHATE PER HA OF GCA (Kg per ha)": "p_kg_ha",
                "POTASH PER HA OF GCA (Kg per ha)": "k_kg_ha",
                "NITROGEN PER HA OF NCA (Kg per ha)": "n_kg_ha_nca",
                "TOTAL PER HA OF GCA (Kg per ha)": "npk_kg_ha"}
    for src, dst in optional.items():
        if src in merged.columns:
            out[dst] = merged[src].values
    extra = [c for c in merged.columns if re.search(rf"{C}.*IRRIGATED AREA|RAINFALL|{C}.*PRICE", c)]
    for c in extra:   # only present once the Punjab irrigation / rainfall / price files are added
        out[re.sub(r"[^a-z0-9]+", "_", c.lower()).strip("_")] = merged[c].values
    num = out.columns.difference(["state", "district", "year"])
    out[num] = out[num].apply(pd.to_numeric, errors="coerce").where(lambda x: x >= 0)
    out.loc[out["area_kha"] <= 0, ["area_kha", "production_kt", "yield_kg_ha"]] = np.nan
    out.loc[out["yield_kg_ha"] <= 0, "yield_kg_ha"] = np.nan
    out["yield_q_ha"] = out["yield_kg_ha"] / 100.0
    out = out.dropna(subset=["yield_kg_ha"]).sort_values(["state", "district", "year"])
    return _write(out.reset_index(drop=True), processed, f"icrisat_{crop}.csv", save)


def build_fit_table(crop="wheat", raw=RAW, processed=PROCESSED, save=True):
    """District-year yield records joined with that district's season weather.

    Columns used by ``YieldModel.fit``: yield_q_ha, n_kg_ha, rain_mm (plus et0_mm,
    late_t_max for diagnostics). Districts with no weather file are dropped."""
    ic = load_icrisat(crop, raw=raw, processed=processed, save=save)
    parts = []
    for site, meta in SITES.items():
        rows = ic[(ic["district"] == meta["district"]) & (ic["state"] == meta["state"])]
        if rows.empty or not (Path(raw) / f"weather_{site}.csv").exists():
            continue
        wx = season_summary(load_seasons(site, raw=raw, processed=processed, save=save))
        parts.append(rows.merge(wx, left_on="year", right_on="season").drop(columns="season"))
    if not parts:
        raise FileNotFoundError("no ICRISAT district has a matching weather_<site>.csv in "
                                f"{raw} -- run scripts/download_data.py weather")
    table = pd.concat(parts, ignore_index=True)
    return _write(table, processed, "yield_fit_table.csv", save)


# -------------------------------------------------------------------- FAO yield/pesticides
def load_fao_pesticides(area="India", item="Wheat", raw=RAW, processed=PROCESSED, save=True):
    """Kaggle 'crop yield prediction dataset' (FAO + World Bank) -> country-year rows.

    Pesticide use is national tonnage, so it is converted to an index relative to the
    series mean; yield is converted from hg/ha to q/ha."""
    path = _need(Path(raw) / "yield_df.csv", "run scripts/download_data.py kaggle "
                                              "(needs ~/.kaggle/kaggle.json)")
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df = df[(df["Area"] == area) & (df["Item"] == item)]
    if df.empty:
        raise ValueError(f"yield_df.csv has no rows for {area} / {item}")
    out = (df.groupby("Year", as_index=False)
             .agg(yield_hg_ha=("hg/ha_yield", "mean"), pesticides_t=("pesticides_tonnes", "mean"),
                  rain_mm_year=("average_rain_fall_mm_per_year", "mean"),
                  avg_temp=("avg_temp", "mean"))
             .rename(columns={"Year": "year"}))
    out["yield_q_ha"] = out["yield_hg_ha"] / 1000.0
    out["pesticide_index"] = out["pesticides_t"] / out["pesticides_t"].mean()
    return _write(out, processed, f"fao_{item.lower()}_{area.lower()}.csv", save)


# -------------------------------------------------------------------- cost of cultivation
def load_cost_of_cultivation(crop="wheat", state="Punjab", reference=REFERENCE,
                             processed=PROCESSED, save=True):
    """CACP cost-of-cultivation break-up (Rs/ha), hand-transcribed from the price policy
    report into data/reference/. Returns long rows: year, item, group, value."""
    path = _need(Path(reference) / f"cacp_{crop}_cost_{state.lower()}.csv",
                 "transcribe Annex Table 5.6 of the CACP Rabi price policy report")
    wide = pd.read_csv(path, comment="#")
    years = [c for c in wide.columns if re.fullmatch(r"\d{4}-\d{2}", c)]
    long = wide.melt(id_vars=["item", "group"], value_vars=years, var_name="year",
                     value_name="value")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["value"]).sort_values(["year", "group", "item"])
    return _write(long.reset_index(drop=True), processed,
                  f"cost_of_cultivation_{crop}_{state.lower()}.csv", save)


def latest_costs(costs):
    """{item: Rs/ha} for the most recent year of a ``load_cost_of_cultivation`` table."""
    last = costs[costs["year"] == costs["year"].max()]
    return dict(zip(last["item"], last["value"]))


# -------------------------------------------------------------------------------- prices
def _rbi_msp_table(path, commodity="Wheat"):
    """Pull the (crop year, MSP) rows out of a saved RBI Handbook table page."""
    best = None
    for t in pd.read_html(io.StringIO(Path(path).read_text(encoding="utf-8", errors="replace"))):
        years = t.iloc[:, 0].astype(str).str.fullmatch(r"\d{4}-\d{2}")
        if years.sum() > 10 and (best is None or t.size < best.size):
            best = t                      # innermost table that still has the year rows
    if best is None:
        raise ValueError(f"{path}: no MSP table found")
    header = best[best.apply(lambda r: r.astype(str).str.contains(commodity).any(), axis=1)].iloc[0]
    col = header.astype(str).str.contains(commodity).idxmax()
    rows = best[best.iloc[:, 0].astype(str).str.fullmatch(r"\d{4}-\d{2}")]
    out = pd.DataFrame({"crop_year": rows.iloc[:, 0].astype(str),
                        "msp": pd.to_numeric(rows[col], errors="coerce")}).dropna()
    out["season"] = out["crop_year"].str[:4].astype(int)
    return out


def load_msp(raw=RAW, processed=PROCESSED, save=True):
    """Wheat MSP (Rs/quintal) by season from the RBI Handbook tables. The latest edition
    wins where the two overlap; the 2014 edition supplies the years before 1993-94."""
    files = [Path(raw) / "rbi_msp_foodgrains_2014.html", Path(raw) / "rbi_msp_foodgrains_latest.html"]
    have = [f for f in files if f.exists()]
    if not have:
        raise FileNotFoundError(f"no rbi_msp_foodgrains_*.html in {raw} -- run "
                                "scripts/download_data.py msp")
    msp = pd.concat([_rbi_msp_table(f) for f in have]).drop_duplicates("season", keep="last")
    msp = msp.sort_values("season")[["season", "crop_year", "msp"]].reset_index(drop=True)
    return _write(msp, processed, "wheat_msp.csv", save)


def _find_col(df, *needles):
    for c in df.columns:
        if any(n in c.lower().replace("_", " ") for n in needles):
            return c
    raise ValueError(f"no column matching {needles} in {list(df.columns)}")


def load_agmarknet(raw=RAW, processed=PROCESSED, save=True):
    """Agmarknet / data.gov.in daily mandi prices -> monthly mean modal price (Rs/quintal)."""
    path = _need(Path(raw) / "agmarknet_wheat_punjab.csv", "see scripts/download_data.py prices")
    df = pd.read_csv(path)
    date = pd.to_datetime(df[_find_col(df, "arrival date", "price date", "date")],
                          dayfirst=True, errors="coerce")
    modal = pd.to_numeric(df[_find_col(df, "modal")], errors="coerce")
    tidy = pd.DataFrame({"date": date, "modal_price": modal}).dropna()
    tidy = tidy[tidy["modal_price"] > 0]
    monthly = (tidy.set_index("date")["modal_price"].resample("MS").agg(["mean", "count"])
               .rename(columns={"mean": "modal_price", "count": "n_quotes"}).dropna().reset_index())
    return _write(monthly, processed, "wheat_mandi_monthly.csv", save)


def price_index(series, window=5):
    """Price relative to its own centred ``window``-year geometric-mean trend.

    Nominal prices grow ~8% a year, so the environment cannot use old rupee values next
    to today's costs; this keeps the real year-to-year deviations and drops the trend."""
    log = np.log(series.astype(float))
    trend = log.rolling(window, center=True, min_periods=window // 2 + 1).mean()
    return np.exp(log - trend)


def load_prices(crop="wheat", raw=RAW, processed=PROCESSED, save=True):
    """Season price table: season, price (Rs/quintal), source, price_index.

    The price of season y is what the farmer gets for the crop harvested in April y+1:
    the mean Apr-Jun mandi modal price where Agmarknet data exist, otherwise the MSP."""
    if crop != "wheat":
        raise ValueError("only wheat prices are wired up")
    table = load_msp(raw, processed, save)
    table["price"] = table["msp"].astype(float)
    table["source"] = "MSP (RBI Handbook)"
    try:
        mandi = load_agmarknet(raw, processed, save)
        harvest = mandi[mandi["date"].dt.month.isin([4, 5, 6])]
        h = harvest.groupby(harvest["date"].dt.year - 1)["modal_price"].mean()
        table = table.merge(h.rename("mandi").rename_axis("season").reset_index(),
                            on="season", how="outer").sort_values("season")
        use = table["mandi"].notna()
        table.loc[use, "price"] = table.loc[use, "mandi"]
        table.loc[use, "source"] = "Agmarknet modal (Apr-Jun)"
    except FileNotFoundError:
        pass
    table = table.dropna(subset=["price"]).reset_index(drop=True)
    table["price_index"] = price_index(table["price"]).round(4)
    return _write(table[["season", "price", "source", "price_index", "msp"]], processed,
                  "wheat_prices.csv", save)
