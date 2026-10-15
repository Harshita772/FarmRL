"""Build data/processed/ from data/raw/ and fit the yield model.

    conda run -n TogAI python scripts/prepare_data.py
    conda run -n TogAI python scripts/prepare_data.py --config configs/haryana.yaml

Datasets that are not on disk are reported and skipped; the weather, price and ICRISAT
tables are required because the environment and the yield fit cannot run without them.
The yield model is fitted on the pooled wheat panel when it has the columns for it
(irrigated share and heat-stress days), otherwise with the single-region form
(farmrl.fitting).
"""
import sys, argparse
from pathlib import Path
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


def optional(label, fn):
    try:
        df = fn()
        print(f"  {label:<28} {len(df):>6} rows")
        return df
    except FileNotFoundError as e:
        print(f"  {label:<28} SKIPPED: {e}")
        return None


print(f"required tables (region: {region}, weather site: {site})")
for name in dl.SITES:                      # every named site: the env site and the map districts
    if name != site and (dl.RAW / f"weather_{name}.csv").exists():
        dl.load_seasons(name, steps=steps)
seasons = dl.load_seasons(site, steps=steps)
train, test = dl.split_seasons(seasons, env_cfg["train_end"])
print(f"  weather ({site})           {seasons['season'].nunique()} seasons: "
      f"{len(train)} train ({train[0]}-{train[-1]}), {len(test)} test ({test[0]}-{test[-1]})")
dl.load_prices()                           # national table (MSP), kept for reference
prices = dl.load_prices(region=region)
print(f"  prices                       {len(prices)} seasons")
for source, r in prices.groupby("source")["season"].agg(["min", "max", "count"]).iterrows():
    print(f"      {source}: {r['count']} seasons ({r['min']}-{r['max']})")
panel = dl.build_fit_table(min_area_kha=fit_cfg.get("min_area_kha", 0.0))
print(f"  yield panel                  {len(panel)} district-years, {panel['district'].nunique()} districts, "
      f"{panel['state'].nunique()} states, {panel['year'].min()}-{panel['year'].max()}")

print("optional tables")
for ref in sorted(dl.REFERENCE.glob("cacp_wheat_cost_*.csv")):
    state = ref.stem.replace("cacp_wheat_cost_", "")
    optional(f"cost of cultivation ({state})", lambda s=state: dl.load_cost_of_cultivation(state=s))
fao = optional("FAO yield + pesticides", dl.load_fao_pesticides)
optional("Agmarknet mandi prices", dl.load_agmarknet)

model, report = fitting.fit_yield_model(cfg, panel, pest_df=fao)
print()
print("\n".join(fitting.describe(model, report)))
if fao is None:
    print("  pest_loss is the prior: no pesticide table to fit it on (yield_df.csv missing)")
out = ROOT / env_cfg["fitted_params"]
model.save(out, source=f"ICRISAT district-level data ({panel['state'].nunique()} states, {panel['year'].min()}-"
                       f"{panel['year'].max()}) + NASA POWER district weather; region {region}")
print(f"saved {out.relative_to(ROOT)}")
