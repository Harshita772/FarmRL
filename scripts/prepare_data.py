"""Build data/processed/ from data/raw/ and fit the yield model.

    python scripts/prepare_data.py
    python scripts/prepare_data.py --config configs/haryana.yaml

Datasets that are not on disk are reported and skipped; the weather, price and ICRISAT
tables are required because the environment and the yield fit cannot run without them.
When a raw file behind one of those is missing (a fresh clone has no data/raw/), the table
already in data/processed/ is reused and the line says so; the yield model is refitted and
recalibrated either way.
The yield model is fitted on the pooled wheat panel when it has the columns for it
(irrigated share and heat-stress days), otherwise with the single-region form
(farmrl.fitting).
"""
import sys, argparse
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from farmrl import data_loader as dl, fitting
from farmrl.utils import load_config

p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
p.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
a = p.parse_args()
cfg = load_config(a.config)
env_cfg, fit_cfg = cfg["env"], cfg["yield_fit"]
site, steps, region = env_cfg["weather_site"], env_cfg["steps_per_season"], env_cfg.get("region")


def optional(label, fn, processed_name=None):
    """``fn()`` from data/raw; when a raw file is missing, the table already in
    data/processed (if ``processed_name`` is given and there), else None."""
    try:
        df = fn()
        print(f"  {label:<28} {len(df):>6} rows")
        return df
    except FileNotFoundError as e:
        path = dl.PROCESSED / processed_name if processed_name else None
        if path is not None and path.exists():
            df = pd.read_csv(path)
            print(f"  {label:<28} {len(df):>6} rows   [reused data/processed/{processed_name}: raw file missing]")
            return df
        print(f"  {label:<28} SKIPPED: {e}")
        return None


def required(build, processed_name):
    """``build()`` from data/raw, or the table already in data/processed when a raw file is
    missing. Returns (table, note for the printout)."""
    try:
        return build(), ""
    except FileNotFoundError as e:
        path = dl.PROCESSED / processed_name
        if not path.exists():
            raise
        return pd.read_csv(path), f"   [reused data/processed/{processed_name}: {str(e).split(' -- ')[0]}]"


print(f"required tables (region: {region}, weather site: {site})")
for name in dl.SITES:                      # every named site: the env site and the map districts
    if name != site and (dl.RAW / f"weather_{name}.csv").exists():
        dl.load_seasons(name, steps=steps)
seasons, note = required(lambda: dl.load_seasons(site, steps=steps), f"weather_fortnightly_{site}.csv")
train, test = dl.split_seasons(seasons, env_cfg["train_end"])
print(f"  weather ({site})           {seasons['season'].nunique()} seasons: "
      f"{len(train)} train ({train[0]}-{train[-1]}), {len(test)} test ({test[0]}-{test[-1]}){note}")
required(dl.load_prices, "wheat_prices.csv")       # national table (MSP), kept for reference
prices, note = required(lambda: dl.load_prices(region=region),
                        f"wheat_prices_{region.lower()}.csv" if region else "wheat_prices.csv")
print(f"  prices                       {len(prices)} seasons{note}")
for source, r in prices.groupby("source")["season"].agg(["min", "max", "count"]).iterrows():
    print(f"      {source}: {r['count']} seasons ({r['min']}-{r['max']})")
panel, note = required(lambda: dl.build_fit_table(min_area_kha=fit_cfg.get("min_area_kha", 0.0)),
                       "yield_fit_table.csv")
print(f"  yield panel                  {len(panel)} district-years, {panel['district'].nunique()} districts, "
      f"{panel['state'].nunique()} states, {panel['year'].min()}-{panel['year'].max()}{note}")

print("optional tables")
for ref in sorted(dl.REFERENCE.glob("cacp_wheat_cost_*.csv")):
    state = ref.stem.replace("cacp_wheat_cost_", "")
    optional(f"cost of cultivation ({state})", lambda s=state: dl.load_cost_of_cultivation(state=s))
fao = optional("FAO yield + pesticides", dl.load_fao_pesticides, "fao_wheat_india.csv")
if fao is not None:
    print(f"      {fao['year'].min()}-{fao['year'].max()}, {int(fao['n_temp_rows'].max())} repeated rows per year "
          f"collapsed; pesticide measure: {fao['normalisation'].iloc[0]}")
optional("Agmarknet mandi prices", dl.load_agmarknet)

model, report = fitting.fit_yield_model(cfg, panel, pest_df=fao)
print()
print("\n".join(fitting.describe(model, report)))
if fao is None:
    print("  pest_loss is the prior: no pesticide table to calibrate it on (yield_df.csv missing)")
out = ROOT / env_cfg["fitted_params"]
model.save(out, source=f"ICRISAT district-level data ({panel['state'].nunique()} states, {panel['year'].min()}-"
                       f"{panel['year'].max()}) + NASA POWER district weather; region {region}")
print(f"saved {out.relative_to(ROOT)}")
