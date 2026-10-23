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
    # Punjab
    "ludhiana": {"lat": 30.90, "lon": 75.85, "state": "Punjab", "district": "Ludhiana"},
    "amritsar": {"lat": 31.63, "lon": 74.87, "state": "Punjab", "district": "Amritsar"},
    "bathinda": {"lat": 30.21, "lon": 74.95, "state": "Punjab", "district": "Bathinda"},
    "jalandhar": {"lat": 31.33, "lon": 75.58, "state": "Punjab", "district": "Jalandhar"},
    "patiala": {"lat": 30.34, "lon": 76.38, "state": "Punjab", "district": "Patiala"},
    "firozpur": {"lat": 30.92, "lon": 74.61, "state": "Punjab", "district": "Firozpur"},
    "sangrur": {"lat": 30.25, "lon": 75.84, "state": "Punjab", "district": "Sangrur"},
    # Haryana
    "hisar": {"lat": 29.15, "lon": 75.72, "state": "Haryana", "district": "Hissar"},
    "gurgaon": {"lat": 28.46, "lon": 77.03, "state": "Haryana", "district": "Gurgaon"},
    "jind": {"lat": 29.32, "lon": 76.31, "state": "Haryana", "district": "Jind"},
    "narnaul": {"lat": 28.04, "lon": 76.11, "state": "Haryana",
                "district": "Mahendragarh / Narnaul"},
    "ambala": {"lat": 30.38, "lon": 76.78, "state": "Haryana", "district": "Ambala"},
    "karnal": {"lat": 29.69, "lon": 76.99, "state": "Haryana", "district": "Karnal"},
    "rohtak": {"lat": 28.90, "lon": 76.61, "state": "Haryana", "district": "Rohtak"},
    "sirsa": {"lat": 29.53, "lon": 75.03, "state": "Haryana", "district": "Sirsa"},
    "kurukshetra": {"lat": 29.97, "lon": 76.88, "state": "Haryana", "district": "Kurukshetra"},
    "panipat": {"lat": 29.39, "lon": 76.97, "state": "Haryana", "district": "Panipat"},
    "sonipat": {"lat": 28.99, "lon": 77.02, "state": "Haryana", "district": "Sonipat"},
    # Western Uttar Pradesh
    "meerut": {"lat": 28.98, "lon": 77.71, "state": "Uttar Pradesh", "district": "Meerut"},
    "aligarh": {"lat": 27.89, "lon": 78.08, "state": "Uttar Pradesh", "district": "Aligarh"},
    "agra": {"lat": 27.18, "lon": 78.01, "state": "Uttar Pradesh", "district": "Agra"},
    "muzaffarnagar": {"lat": 29.47, "lon": 77.70, "state": "Uttar Pradesh", "district": "Muzaffarnagar"},
    "mathura": {"lat": 27.49, "lon": 77.67, "state": "Uttar Pradesh", "district": "Mathura"},
    # Rajasthan
    "ganganagar": {"lat": 29.90, "lon": 73.88, "state": "Rajasthan", "district": "Ganganagar"},
    "hanumangarh": {"lat": 29.58, "lon": 74.32, "state": "Rajasthan", "district": "Hanumangarh"},
    "alwar": {"lat": 27.55, "lon": 76.63, "state": "Rajasthan", "district": "Alwar"},
    "bharatpur": {"lat": 27.22, "lon": 77.49, "state": "Rajasthan", "district": "Bharatpur"},
}

SEASON_START = (11, 1)   # month, day
TRAIN_END = 2016         # seasons <= this are train, later ones are held out
# Terminal heat stress: days hotter than HEAT_TMAX (deg C, daily maximum) during grain
# filling, taken as HEAT_WINDOW (month, day) of the harvest year.
HEAT_TMAX = 30.0
HEAT_WINDOW = ((2, 15), (3, 31))
# ICRISAT labels crop years by their first calendar year (2008 = sown Nov 2008) but farm
# harvest prices by the marketing year: for Punjab the mean price of "year" Y sits on the
# MSP of crop year Y-1 in every year of the file. So a price row of year Y is season Y-1.
PRICE_YEAR_OFFSET = 1
DISTRICT_WEATHER = "weather_districts"   # sub-folder of data/raw, one POWER file per district

_WEATHER_COLS = {"PRECTOTCORR": "rain_mm", "T2M": "t_mean", "T2M_MAX": "t_max",
                 "T2M_MIN": "t_min", "RH2M": "rh", "ALLSKY_SFC_SW_DWN": "srad"}
_ICRISAT_KEYS = ["Dist Code", "Year", "State Code", "State Name", "Dist Name"]
_ICRISAT_ON = ["Dist Code", "Year"]


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
def read_power(path):
    """One NASA POWER daily CSV -> tidy daily table (date, rain_mm, t_mean, t_max, t_min,
    rh, srad [MJ/m2/day], et0_mm, heat_day). POWER's -999 fill value becomes NaN; short
    gaps (<=5 days) are interpolated, rain gaps are left as NaN."""
    lines = Path(path).read_text().splitlines()
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
    md = df["date"].dt.month * 100 + df["date"].dt.day
    (m0, d0), (m1, d1) = HEAT_WINDOW
    hot = (md >= m0 * 100 + d0) & (md <= m1 * 100 + d1) & (df["t_max"] > HEAT_TMAX)
    df["heat_day"] = hot.astype(int)
    return df.reset_index(drop=True)


def load_weather(site="ludhiana", raw=RAW, processed=PROCESSED, save=True):
    """Daily weather of one named site (data/raw/weather_<site>.csv), see ``read_power``."""
    path = _need(Path(raw) / f"weather_{site}.csv", "run scripts/download_data.py weather")
    return _write(read_power(path), processed, f"weather_daily_{site}.csv", save)


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
        if "heat_day" in season:
            out["heat_days"] = g["heat_day"].sum().values
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
    if "heat_days" in seasons:
        out = out.merge(seasons.groupby("season")["heat_days"].sum().reset_index())
    return out.round(3)


def split_seasons(seasons, train_end=TRAIN_END):
    """Sorted lists of (train, test) season labels: train <= train_end < test."""
    years = sorted(int(y) for y in pd.unique(seasons["season"]))
    return [y for y in years if y <= train_end], [y for y in years if y > train_end]


# ------------------------------------------------------------------------------ ICRISAT
def load_icrisat(crop="wheat", state=None, min_area_kha=0.0, raw=RAW, processed=PROCESSED, save=True):
    """ICRISAT District Level Data -> one row per district-year for ``crop``.

    Every ``icrisat*.csv`` in ``raw`` is merged on district code and year, so the crops,
    fertiliser, irrigation and price exports sit side by side. ``state=None`` keeps every
    state present; ``min_area_kha`` drops district-years with less crop area than that.
    The -1 / 0 "not reported" codes become NaN. Harvest prices are shifted to the season
    they belong to (see PRICE_YEAR_OFFSET) and the irrigated share is clipped to [0, 1].
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
        if any("HARVEST PRICE" in c for c in part.columns):
            part["Year"] = part["Year"] - PRICE_YEAR_OFFSET
        if merged is None:
            merged = part
            continue
        new = [c for c in part.columns if c not in merged.columns]
        names = part.drop_duplicates(_ICRISAT_ON).set_index(_ICRISAT_ON)[["State Name", "Dist Name"]]
        merged = merged.merge(part[_ICRISAT_ON + new], on=_ICRISAT_ON, how="outer")
        keys = pd.MultiIndex.from_frame(merged[_ICRISAT_ON])
        for col in ("State Name", "Dist Name"):      # rows that exist only in the later file
            merged[col] = merged[col].fillna(pd.Series(names[col].reindex(keys).values, index=merged.index))
    if state is not None:
        merged = merged[merged["State Name"].str.lower() == state.lower()]
    C = crop.upper()
    need = [f"{C} AREA (1000 ha)", f"{C} PRODUCTION (1000 tons)", f"{C} YIELD (Kg per ha)"]
    if not set(need) <= set(merged.columns):
        raise ValueError(f"ICRISAT files have no {crop} area/production/yield columns")
    out = pd.DataFrame({"state": merged["State Name"], "district": merged["Dist Name"],
                        "dist_code": merged["Dist Code"].astype(int), "year": merged["Year"].astype(int),
                        "area_kha": merged[need[0]], "production_kt": merged[need[1]],
                        "yield_kg_ha": merged[need[2]]})
    optional = {"NITROGEN PER HA OF GCA (Kg per ha)": "n_kg_ha",
                "PHOSPHATE PER HA OF GCA (Kg per ha)": "p_kg_ha",
                "POTASH PER HA OF GCA (Kg per ha)": "k_kg_ha",
                "NITROGEN PER HA OF NCA (Kg per ha)": "n_kg_ha_nca",
                "TOTAL PER HA OF GCA (Kg per ha)": "npk_kg_ha",
                f"{C} IRRIGATED AREA (1000 ha)": "irrigated_area_kha",
                f"{C} HARVEST PRICE (Rs per Quintal)": "harvest_price"}
    for src, dst in optional.items():
        if src in merged.columns:
            out[dst] = merged[src].values
    num = out.columns.difference(["state", "district", "dist_code", "year"])
    out[num] = out[num].apply(pd.to_numeric, errors="coerce").where(lambda x: x >= 0)
    out.loc[~(out["area_kha"] > 0), ["area_kha", "production_kt", "yield_kg_ha"]] = np.nan
    out.loc[~(out["yield_kg_ha"] > 0), "yield_kg_ha"] = np.nan
    if "harvest_price" in out:
        out.loc[~(out["harvest_price"] > 0), "harvest_price"] = np.nan
    if "irrigated_area_kha" in out:
        out["irrigated_share"] = (out["irrigated_area_kha"] / out["area_kha"]).clip(0, 1)
    out["yield_q_ha"] = out["yield_kg_ha"] / 100.0
    out = out.dropna(subset=["yield_kg_ha"])
    out = out[out["area_kha"] > min_area_kha].sort_values(["state", "district", "year"])
    return _write(out.reset_index(drop=True), processed, f"icrisat_{crop}.csv", save)


def load_district_weather(raw=RAW, processed=PROCESSED, save=True, steps=12, step_days=15):
    """Season summaries for every cached district weather file
    (data/raw/weather_districts/<dist_code>.csv): dist_code, year (= season), rain_mm,
    et0_mm, t_mean, late_t_max, late_rain_mm, heat_days."""
    files = sorted((Path(raw) / DISTRICT_WEATHER).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no files in {Path(raw) / DISTRICT_WEATHER} -- run "
                                "scripts/download_data.py district_weather")
    parts = []
    for f in files:
        try:
            summary = season_summary(build_seasons(read_power(f), steps, step_days))
        except ValueError:          # no complete season in this file
            continue
        parts.append(summary.rename(columns={"season": "year"}).assign(dist_code=int(f.stem)))
    out = pd.concat(parts, ignore_index=True)
    return _write(out, processed, "district_weather.csv", save)


def build_fit_table(crop="wheat", min_area_kha=10.0, raw=RAW, processed=PROCESSED, save=True):
    """The panel the yield model is fitted on: district-year yield, N/P/K per ha, irrigated
    share and harvest price (ICRISAT) joined with that district's season weather (rain,
    temperature, heat-stress days).

    Weather comes from the per-district cache when it exists; otherwise from the named
    ``SITES`` (a handful of district headquarters), which keeps the pipeline usable before
    the district download has been run."""
    ic = load_icrisat(crop, min_area_kha=min_area_kha, raw=raw, processed=processed, save=save)
    cache = Path(raw) / DISTRICT_WEATHER
    if cache.exists() and any(cache.glob("*.csv")):
        table = ic.merge(load_district_weather(raw, processed, save), on=["dist_code", "year"])
    else:
        parts = []
        for site, meta in SITES.items():
            rows = ic[(ic["district"] == meta["district"]) & (ic["state"] == meta["state"])]
            if rows.empty or not (Path(raw) / f"weather_{site}.csv").exists():
                continue
            wx = season_summary(load_seasons(site, raw=raw, processed=processed, save=save))
            parts.append(rows.merge(wx, left_on="year", right_on="season").drop(columns="season"))
        if not parts:
            raise FileNotFoundError("no ICRISAT district has matching weather in "
                                    f"{raw} -- run scripts/download_data.py weather district_weather")
        table = pd.concat(parts, ignore_index=True)
    return _write(table.reset_index(drop=True), processed, "yield_fit_table.csv", save)


# -------------------------------------------------------------------- FAO yield/pesticides
def _fao_raw_series(folder, area, item):
    """(yield hg/ha, pesticide tonnes, mean temperature, cropland 1000 ha) by year from the
    original source files in data/raw/fao/, each ``None`` when its file is not there."""
    def read(name, area_col, value_col, year_col="Year", where=None, how="first"):
        path = Path(folder) / name
        if not path.exists():
            return None
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        df = df[df[area_col] == area]
        for col, value in (where or {}).items():
            df = df[df[col] == value]
        series = pd.to_numeric(df[value_col], errors="coerce").groupby(df[year_col].astype(int)).agg(how)
        return series.dropna() if len(series) else None
    return (read("yield.csv", "Area", "Value", where={"Item": item}),
            read("pesticides.csv", "Area", "Value"),
            read("temp.csv", "country", "avg_temp", year_col="year", how="mean"),
            read("cropland.csv", "Area", "Value"))


def load_fao_pesticides(area="India", item="Wheat", raw=RAW, processed=PROCESSED, save=True):
    """Kaggle 'crop yield prediction dataset' (FAO + World Bank) -> one row per year for
    ``area`` and ``item``.

    * **De-duplication.** ``yield_df.csv`` repeats every Area-Item-Year once per temperature
      station of the country; the rows are collapsed to one, with the temperature averaged
      and the number of repeats kept in ``n_temp_rows``.
    * **Missing years.** A year absent from ``yield_df.csv`` only because one source had no
      row for it is restored from the original files in ``raw/fao/`` when they are there
      (``source`` says which rows these are; their rainfall stays missing).
    * **Pesticide measure.** Pesticide use is national tonnage of active ingredient for all
      crops. If ``raw/fao/cropland.csv`` (FAOSTAT land use: Area, Year, Value in 1000 ha)
      exists it is converted to kg per ha of cropland; otherwise to an index relative to the
      series mean. ``pesticide_x`` holds the chosen measure and ``normalisation`` names it.
    Yield is converted from hg/ha to q/ha."""
    path = _need(Path(raw) / "yield_df.csv", "run scripts/download_data.py kaggle (needs "
                                              "~/.kaggle/kaggle.json), or save the file there by hand")
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df = df[(df["Area"] == area) & (df["Item"] == item)]
    if df.empty:
        raise ValueError(f"yield_df.csv has no rows for {area} / {item}")
    out = (df.groupby("Year", as_index=False)
             .agg(yield_hg_ha=("hg/ha_yield", "mean"), pesticides_t=("pesticides_tonnes", "mean"),
                  rain_mm_year=("average_rain_fall_mm_per_year", "mean"),
                  avg_temp=("avg_temp", "mean"), n_temp_rows=("avg_temp", "size"))
             .rename(columns={"Year": "year"}))
    out["source"] = "yield_df.csv"

    y_raw, p_raw, t_raw, cropland = _fao_raw_series(Path(raw) / "fao", area, item)
    if y_raw is not None and p_raw is not None and t_raw is not None:
        years = y_raw.index.intersection(p_raw.index).intersection(t_raw.index)
        gaps = [y for y in years if out["year"].min() < y < out["year"].max() and y not in set(out["year"])]
        if gaps:
            extra = pd.DataFrame({"year": gaps, "yield_hg_ha": y_raw.loc[gaps].values,
                                  "pesticides_t": p_raw.loc[gaps].values, "rain_mm_year": np.nan,
                                  "avg_temp": t_raw.loc[gaps].values, "n_temp_rows": 0,
                                  "source": "raw/fao (not in yield_df.csv)"})
            out = pd.concat([out, extra], ignore_index=True).sort_values("year").reset_index(drop=True)
    out["yield_q_ha"] = out["yield_hg_ha"] / 1000.0
    out["pesticide_index"] = out["pesticides_t"] / out["pesticides_t"].mean()
    area_kha = None if cropland is None else out["year"].map(cropland)
    if area_kha is not None and area_kha.notna().all():
        out["pesticide_kg_ha"] = out["pesticides_t"] / area_kha          # t / 1000 ha = kg / ha
        out["pesticide_x"], out["normalisation"] = out["pesticide_kg_ha"], "kg of active ingredient per ha of cropland"
    else:
        out["pesticide_x"], out["normalisation"] = out["pesticide_index"], "index (national tonnage / series mean)"
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
    trend = log.rolling(window, center=True, min_periods=min(window // 2 + 1, len(log))).mean()
    return np.exp(log - trend)


def load_prices(crop="wheat", region=None, raw=RAW, processed=PROCESSED, save=True):
    """Season price table: season, price (Rs/quintal), source, price_index, msp.

    The price of season y is what the farmer gets for the crop harvested in April y+1.
    Priority per season: the mean ICRISAT farm harvest price of ``region``'s districts,
    then the mean Apr-Jun Agmarknet modal price, then the MSP. ``price_index`` is each
    price relative to the trend of *its own* source, so splicing sources with different
    levels does not create artificial jumps. Saved as wheat_prices[_<region>].csv."""
    if crop != "wheat":
        raise ValueError("only wheat prices are wired up")
    table = load_msp(raw, processed, save)
    table["price"] = table["msp"].astype(float)
    table["source"] = "MSP (RBI Handbook)"
    table["price_index"] = price_index(table["price"])
    extra = []
    try:
        mandi = load_agmarknet(raw, processed, save)
        harvest = mandi[mandi["date"].dt.month.isin([4, 5, 6])]
        extra.append((harvest.groupby(harvest["date"].dt.year - 1)["modal_price"].mean(),
                      "Agmarknet modal (Apr-Jun)"))
    except FileNotFoundError:
        pass
    if region is not None:
        try:
            ic = load_icrisat(crop, state=region, raw=raw, processed=processed, save=False)
            if "harvest_price" in ic and ic["harvest_price"].notna().any():
                series = ic.dropna(subset=["harvest_price"]).groupby("year")["harvest_price"].mean()
                extra.append((series, f"ICRISAT farm harvest price ({region.title()} districts)"))
        except FileNotFoundError:
            pass
    table = table.set_index("season")
    for series, source in extra:                 # later entries take priority
        series = series.sort_index()
        index = price_index(series.reset_index(drop=True))
        index.index = series.index
        table = table.reindex(table.index.union(series.index))
        table.loc[series.index, "price"] = series
        table.loc[series.index, "source"] = source
        table.loc[series.index, "price_index"] = index
    table = table.dropna(subset=["price"]).rename_axis("season").reset_index()
    table["price_index"] = table["price_index"].round(4)
    name = "wheat_prices.csv" if region is None else f"wheat_prices_{region.lower()}.csv"
    return _write(table[["season", "price", "source", "price_index", "msp"]], processed, name, save)
