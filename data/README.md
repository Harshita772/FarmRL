# Data

`data/raw/` holds downloads (git-ignored), `data/processed/` the cleaned tables built by
`scripts/prepare_data.py`, and `data/reference/` small tables transcribed by hand from
published reports. Nothing in this folder is synthetic.

```bash
conda run -n TogAI python scripts/download_data.py     # fetch what can be fetched, print manual steps for the rest
conda run -n TogAI python scripts/prepare_data.py      # raw -> processed, fit the yield model
```

## What is here (retrieved 10 Oct 2026)

| Dataset | Source and licence | File(s) in `raw/` | Coverage | Used for |
|---|---|---|---|---|
| NASA POWER daily weather (PRECTOTCORR, T2M, T2M_MAX, T2M_MIN, RH2M, ALLSKY_SFC_SW_DWN; community AG) | https://power.larc.nasa.gov/api/temporal/daily/point — NASA open data, free to use with acknowledgement | `weather_ludhiana.csv` (30.90N, 75.85E) | 1 Jan 1985 – 31 Dec 2024 | environment weather (39 seasons) |
| same, district headquarters | same | `weather_hisar.csv`, `weather_gurgaon.csv`, `weather_jind.csv`, `weather_narnaul.csv`, `weather_ambala.csv`, `weather_karnal.csv`, `weather_rohtak.csv` | 1985 – 2024 | season rain for the yield fit |
| ICRISAT District Level Data, apportioned (crops area/production/yield; fertiliser N, P, K) | http://data.icrisat.org/dld/ — free registration, non-commercial research use with citation; **downloaded by hand** | `icrisat_crops.csv`, `icrisat_fertiliser.csv` | **Haryana**, 7 districts (Hissar, Gurgaon, Jind, Mahendragarh/Narnaul, Ambala, Karnal, Rohtak), 2008 – 2017 | yield response fit |
| Wheat MSP by crop year | RBI, Handbook of Statistics on Indian Economy, "Minimum Support Price for Foodgrains According to Crop Year": 2014 edition (https://www.rbi.org.in/Scripts/PublicationsView.aspx?id=15815, dated 15 Sep 2014) and 2026 edition (…id=23848, Table 23, dated 31 Jul 2026) — public RBI publication | `rbi_msp_foodgrains_2014.html`, `rbi_msp_foodgrains_latest.html` | crop years 1981-82 – 2025-26 | price series |
| CACP, *Price Policy for Rabi Crops: Marketing Season 2027-28* (July 2026) | https://cacp.da.gov.in/Home/latestRabiReports — Government of India publication | `cacp_rabi_price_policy_latest.pdf` (63 MB, scanned) | Punjab wheat cost of cultivation 2022-23 – 2024-25 | costs in `configs/default.yaml` |

`reference/cacp_wheat_cost_punjab.csv` is Annex Table 5.6(a) (report p. 189) of that CACP
report, typed in by hand because the PDF has no text layer. A test checks that the line
items add up to the report's own total.

## What is NOT here yet

| Dataset | Why | What to do | Expected file |
|---|---|---|---|
| FAO crop yield + pesticides (Kaggle `patelris/crop-yield-prediction-dataset`) | `~/.kaggle/kaggle.json` is missing | create an API token on kaggle.com (Settings → API), put it in `~/.kaggle/`, run `python scripts/download_data.py kaggle` | `raw/yield_df.csv` |
| ICRISAT for **Punjab**, plus irrigated area, rainfall and farm harvest prices | registration wall; the supplied files are Haryana crops + fertiliser only | export from data.icrisat.org/dld (State = Punjab) | `raw/icrisat_crops.csv`, `icrisat_fertiliser.csv`, `icrisat_irrigation.csv`, `icrisat_rainfall.csv`, `icrisat_prices.csv` |
| DES Cost of Cultivation plot-level series (data.gov.in) | `api.data.gov.in` and `desagri.gov.in` refused connections from our network, and the API needs a key | manual CSV download or `DATA_GOV_IN_API_KEY` + `DATA_GOV_IN_COC_RESOURCE` | `raw/cost_of_cultivation.csv` |
| Agmarknet wheat modal prices, Punjab mandis | same (API key / unreachable) | `DATA_GOV_IN_API_KEY`, or export from agmarknet.gov.in | `raw/agmarknet_wheat_punjab.csv` |

`scripts/download_data.py` prints the exact steps for each. Consequences for the current
results: the pesticide-loss parameter is a prior, not a fit; the yield response comes from
Haryana, not Punjab; the price series is the MSP rather than mandi prices.

## Processed tables

| File | Built by | Content |
|---|---|---|
| `weather_daily_<site>.csv` | `load_weather` | date, rain_mm, t_mean, t_max, t_min, rh, srad, et0_mm (Hargreaves) |
| `weather_fortnightly_<site>.csv` | `load_seasons` | season (sowing year), step 0–11 from 1 Nov, fortnight totals/means |
| `icrisat_wheat.csv` | `load_icrisat` | district-year wheat area, production, yield and N, P, K per ha |
| `yield_fit_table.csv` | `build_fit_table` | the above joined with each district's season weather |
| `wheat_msp.csv`, `wheat_prices.csv` | `load_msp`, `load_prices` | price per season and its index around a 5-year trend |
| `cost_of_cultivation_wheat_punjab.csv` | `load_cost_of_cultivation` | CACP table in long form |
| `fao_wheat_india.csv`, `wheat_mandi_monthly.csv` | `load_fao_pesticides`, `load_agmarknet` | only once the raw files exist |
