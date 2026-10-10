"""Loader tests run on small synthetic files written to tmp_path, in the same layout as
the real downloads, so they need neither the network nor data/raw."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import pytest
from farmrl import data_loader as dl

KEYS = '"Dist Code","Year","State Code","State Name","Dist Name"'


def write_power(path, start="2000-01-01", end="2003-12-31", fill_on=None):
    days = pd.date_range(start, end)
    rows = ["-BEGIN HEADER-", "NASA/POWER synthetic test file", "-END HEADER-",
            "YEAR,DOY,PRECTOTCORR,T2M,T2M_MAX,T2M_MIN,RH2M,ALLSKY_SFC_SW_DWN"]
    for d in days:
        rain = -999.0 if fill_on is not None and d == pd.Timestamp(fill_on) else 1.0
        rows.append(f"{d.year},{d.dayofyear},{rain},15.0,22.0,8.0,50.0,14.0")
    path.write_text("\n".join(rows))


def write_icrisat(raw):
    (raw / "icrisat_crops.csv").write_text(
        f'{KEYS},"WHEAT AREA (1000 ha)","WHEAT PRODUCTION (1000 tons)","WHEAT YIELD (Kg per ha)"\n'
        '"1","2001","4","Haryana","Hissar","100.0","450.0","4500.0"\n'
        '"1","2002","4","Haryana","Hissar","0.0","0.0","0.0"\n'
        '"2","2001","9","Punjab","Ludhiana","200.0","1000.0","5000.0"\n')
    (raw / "icrisat_fertiliser.csv").write_text(
        f'{KEYS},"NITROGEN PER HA OF GCA (Kg per ha)"\n'
        '"1","2001","4","Haryana","Hissar","120.5"\n'
        '"1","2002","4","Haryana","Hissar","125.0"\n'
        '"2","2001","9","Punjab","Ludhiana","210.0"\n')


def test_weather_parses_header_and_fill_values(tmp_path):
    write_power(tmp_path / "weather_ludhiana.csv", fill_on="2001-02-03")
    daily = dl.load_weather("ludhiana", raw=tmp_path, processed=tmp_path)
    assert len(daily) == 1461 and daily["date"].is_monotonic_increasing
    assert daily["rain_mm"].isna().sum() == 1            # -999 is missing, not rain
    assert np.isclose(daily["et0_mm"].iloc[0], 0.0135 * (15 + 17.8) * 14 / 2.45)
    assert (tmp_path / "weather_daily_ludhiana.csv").exists()


def test_seasons_are_fortnightly_and_drop_incomplete_years(tmp_path):
    write_power(tmp_path / "weather_ludhiana.csv", fill_on="2001-02-03")
    seasons = dl.load_seasons("ludhiana", steps=12, raw=tmp_path, processed=tmp_path)
    # 2000 has a missing rain day, 2003 runs past the end of the file
    assert sorted(seasons["season"].unique()) == [2001, 2002]
    assert (seasons.groupby("season").size() == 12).all()
    assert np.allclose(seasons["rain_mm"], 15.0)
    assert str(seasons["start_date"].iloc[0]) == "2001-11-01"
    summary = dl.season_summary(seasons)
    assert np.allclose(summary["rain_mm"], 180.0)


def test_split_seasons():
    seasons = pd.DataFrame({"season": [2015, 2016, 2017, 2018]})
    assert dl.split_seasons(seasons, 2016) == ([2015, 2016], [2017, 2018])


def test_icrisat_merges_files_and_cleans_zeros(tmp_path):
    write_icrisat(tmp_path)
    df = dl.load_icrisat("wheat", raw=tmp_path, processed=tmp_path / "out")
    assert len(df) == 2                                   # the zero-yield row is dropped
    assert set(df["state"]) == {"Haryana", "Punjab"}
    row = df[df["district"] == "Hissar"].iloc[0]
    assert row["yield_q_ha"] == 45.0 and row["n_kg_ha"] == 120.5
    assert list(dl.load_icrisat("wheat", state="punjab", raw=tmp_path, save=False)["district"]) == ["Ludhiana"]


def test_icrisat_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        dl.load_icrisat(raw=tmp_path, save=False)


def test_fit_table_joins_district_weather(tmp_path):
    write_icrisat(tmp_path)
    write_power(tmp_path / "weather_hisar.csv")
    table = dl.build_fit_table(raw=tmp_path, processed=tmp_path / "out")
    assert list(table["district"]) == ["Hissar"] and list(table["year"]) == [2001]
    assert np.isclose(table["rain_mm"].iloc[0], 180.0)


def test_fao_pesticides(tmp_path):
    (tmp_path / "yield_df.csv").write_text(
        ",Area,Item,Year,hg/ha_yield,average_rain_fall_mm_per_year,pesticides_tonnes,avg_temp\n"
        "0,India,Wheat,1990,21210,1083,75000,25.5\n"
        "1,India,Wheat,1990,21210,1083,75000,26.1\n"      # the dataset repeats country rows
        "2,India,Wheat,1991,22810,1083,25000,25.9\n"
        "3,India,Maize,1990,15180,1083,75000,25.5\n"
        "4,Albania,Wheat,1990,30197,1485,121,16.4\n")
    df = dl.load_fao_pesticides(raw=tmp_path, processed=tmp_path)
    assert list(df["year"]) == [1990, 1991]
    assert np.allclose(df["yield_q_ha"], [21.21, 22.81])
    assert np.allclose(df["pesticide_index"], [1.5, 0.5])
    with pytest.raises(FileNotFoundError):
        dl.load_fao_pesticides(raw=tmp_path / "nowhere", save=False)


def rbi_page(rows):
    body = "".join(f"<tr><td>{y}</td><td>{p}</td><td>{w}</td></tr>" for y, p, w in rows)
    return ("<html><body><table><tr><td><table>"
            "<tr><td>Year</td><td>Paddy common</td><td>Wheat</td></tr>"
            f"{body}</table></td></tr></table></body></html>")


def test_msp_splices_editions_and_prices_are_indexed(tmp_path):
    old = [(f"{y}-{(y + 1) % 100:02d}", 100, 150 + 10 * (y - 1981)) for y in range(1981, 1996)]
    new = [(f"{y}-{(y + 1) % 100:02d}", 200, 300 + 20 * (y - 1993)) for y in range(1993, 2008)]
    new[-1] = (new[-1][0], 200, "-")                      # not yet announced
    (tmp_path / "rbi_msp_foodgrains_2014.html").write_text(rbi_page(old), encoding="utf-8")
    (tmp_path / "rbi_msp_foodgrains_latest.html").write_text(rbi_page(new), encoding="utf-8")
    msp = dl.load_msp(raw=tmp_path, processed=tmp_path)
    assert msp["season"].tolist() == list(range(1981, 2007))
    assert msp.set_index("season").loc[1985, "msp"] == 190       # only in the old edition
    assert msp.set_index("season").loc[1994, "msp"] == 320       # latest edition wins
    prices = dl.load_prices(raw=tmp_path, processed=tmp_path)
    assert set(prices["source"]) == {"MSP (RBI Handbook)"}
    assert prices["price_index"].between(0.8, 1.2).all()


def test_price_index_removes_trend():
    steady = pd.Series(100 * 1.08 ** np.arange(20))
    idx = dl.price_index(steady)
    assert np.allclose(idx.iloc[2:-2], 1.0)               # constant growth -> no deviation
    spiky = steady.copy(); spiky.iloc[10] *= 1.2
    assert dl.price_index(spiky).iloc[10] > 1.1


def test_agmarknet_overrides_msp_for_seasons_it_covers(tmp_path):
    rows = [(f"{y}-{(y + 1) % 100:02d}", 200, 1000) for y in range(1993, 2008)]
    (tmp_path / "rbi_msp_foodgrains_latest.html").write_text(rbi_page(rows), encoding="utf-8")
    (tmp_path / "agmarknet_wheat_punjab.csv").write_text(
        "State,Market,Commodity,Arrival_Date,Min_Price,Max_Price,Modal_Price\n"
        "Punjab,Khanna,Wheat,15/04/2001,1050,1150,1100\n"
        "Punjab,Ludhiana,Wheat,20/05/2001,1250,1350,1300\n"
        "Punjab,Khanna,Wheat,10/12/2001,900,950,0\n")
    monthly = dl.load_agmarknet(raw=tmp_path, processed=tmp_path)
    assert monthly["modal_price"].tolist() == [1100, 1300]       # zero quote dropped
    prices = dl.load_prices(raw=tmp_path, processed=tmp_path).set_index("season")
    assert prices.loc[2000, "price"] == 1200 and prices.loc[2000, "source"].startswith("Agmarknet")
    assert prices.loc[2001, "price"] == 1000                     # no harvest quotes -> MSP


def test_cost_of_cultivation_reference_table():
    costs = dl.load_cost_of_cultivation(save=False)
    latest = dl.latest_costs(costs)
    parts = costs[(costs["year"] == costs["year"].max()) & (costs["group"] == "operational")]
    # the transcribed line items must add up to the report's own total (rounding of 1-2 Rs)
    assert abs(parts["value"].sum() - latest["operational_cost"]) <= 3
    assert latest["insecticides"] > 0 and latest["irrigation_charges"] > 0
