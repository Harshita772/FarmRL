"""Build data/processed/ from data/raw/ and fit the yield model.

    conda run -n TogAI python scripts/prepare_data.py

Datasets that are not on disk are reported and skipped; the weather, price and ICRISAT
tables are required because the environment and the yield fit cannot run without them.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from farmrl import data_loader as dl
from farmrl.baselines import recommended_schedule
from farmrl.env import calibrate_yield_scale
from farmrl.utils import load_config
from farmrl.yield_model import YieldModel

cfg = load_config(ROOT / "configs/default.yaml")
site, steps = cfg["env"]["weather_site"], cfg["env"]["steps_per_season"]


def optional(label, fn):
    try:
        df = fn()
        print(f"  {label:<28} {len(df):>6} rows")
        return df
    except FileNotFoundError as e:
        print(f"  {label:<28} SKIPPED: {e}")
        return None


print("required tables")
seasons = dl.load_seasons(site, steps=steps)
train, test = dl.split_seasons(seasons, cfg["env"]["train_end"])
print(f"  weather ({site})           {seasons['season'].nunique()} seasons: "
      f"{len(train)} train ({train[0]}-{train[-1]}), {len(test)} test ({test[0]}-{test[-1]})")
prices = dl.load_prices()
print(f"  prices                       {len(prices)} seasons, sources: {sorted(prices['source'].unique())}")
fit_table = dl.build_fit_table()
print(f"  yield fit table              {len(fit_table)} district-years, "
      f"states: {sorted(fit_table['state'].unique())}")

print("optional tables")
optional("cost of cultivation (CACP)", dl.load_cost_of_cultivation)
fao = optional("FAO yield + pesticides", dl.load_fao_pesticides)
optional("Agmarknet mandi prices", dl.load_agmarknet)

model = YieldModel()
rep = model.fit(fit_table, assumed_irrigation=cfg["yield_fit"]["assumed_irrigation_mm"], pest_df=fao)
scale = calibrate_yield_scale(cfg["env"], model, rep["mean_observed_yield"], recommended_schedule)
out = ROOT / cfg["env"]["fitted_params"]
states = "/".join(sorted(fit_table["state"].unique()))
model.save(out, source=f"ICRISAT district-level data ({states}, {fit_table['year'].min()}-"
                       f"{fit_table['year'].max()}) + NASA POWER season rainfall")
print(f"yield model: R2 = {rep['r2']:.3f}, RMSE = {rep['rmse']:.2f} q/ha, n = {rep['n_obs']}")
for k, v in model.params().items():
    flag = rep["identified"].get(k)
    note = "" if flag is None else ("  (fitted)" if flag else "  (not identified -> prior)")
    print(f"  {k:<10} {v:10.4f}{note}")
if fao is None:
    print("  pest_loss is the prior: no pesticide table to fit it on (yield_df.csv missing)")
print(f"  y_scale = {scale:.3f}: the recommended schedule now averages the observed "
      f"{rep['mean_observed_yield']:.1f} q/ha on the train seasons")
print(f"saved {out.relative_to(ROOT)}")
