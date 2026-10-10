# Data

`data/raw/` holds downloads (git-ignored), `data/processed/` the cleaned tables built by
`scripts/prepare_data.py`, `data/reference/` small tables transcribed by hand from
published reports, and `data/geo/` district boundaries. Nothing in this folder is synthetic.

```bash
conda run -n TogAI python scripts/download_data.py     # fetch what can be fetched, print manual steps for the rest
conda run -n TogAI python scripts/prepare_data.py      # raw -> processed, fit the yield model
```

`download_data.py` takes step names to run only part of it:
`weather kaggle icrisat msp costs prices geo district_weather`.

## What is here (retrieved 10 Oct 2026)

| Dataset | Source and licence | File(s) | Coverage | Used for |
|---|---|---|---|---|
| ICRISAT District Level Data, apportioned: crops (area, production, yield), inputs (N, P, K consumption), crop-wise irrigated area, farm harvest prices | <http://data.icrisat.org/dld/> — free registration, non-commercial research use with citation; **downloaded by hand** | `raw/icrisat_crops.csv`, `raw/icrisat_fertiliser.csv`, `raw/icrisat_irrigation.csv`, `raw/icrisat_prices.csv` | all India: 20 states, 311 districts (1966 boundaries), 2008–2017 (prices: marketing years 2007–2016) | wheat panel for the yield fit; region harvest prices |
| NASA POWER daily weather (PRECTOTCORR, T2M, T2M_MAX, T2M_MIN, RH2M, ALLSKY_SFC_SW_DWN; community AG) | <https://power.larc.nasa.gov/api/temporal/daily/point> — NASA open data, free to use with acknowledgement | `raw/weather_ludhiana.csv` and the other named sites in `data_loader.SITES` (`weather_hisar.csv`, `…gurgaon`, `…jind`, `…narnaul`, `…ambala`, `…karnal`, `…rohtak`) | 1 Jan 1985 – 31 Dec 2024 | environment weather (39 seasons), district map |
| same, at the centroid of every district in the wheat panel | same | `raw/weather_districts/<ICRISAT district code>.csv` (220 files, cached: re-running only fetches missing ones) | 2007 – 2018 | season rain, temperature and heat-stress days for the yield fit |
| Wheat MSP by crop year | RBI, Handbook of Statistics on Indian Economy, "Minimum Support Price for Foodgrains According to Crop Year": 2014 edition (<https://www.rbi.org.in/Scripts/PublicationsView.aspx?id=15815>, dated 15 Sep 2014) and 2026 edition (…id=23848, Table 23, dated 31 Jul 2026) — public RBI publication | `raw/rbi_msp_foodgrains_2014.html`, `raw/rbi_msp_foodgrains_latest.html` | crop years 1981-82 – 2025-26 | price series for seasons without harvest prices; base price |
| CACP, *Price Policy for Rabi Crops: Marketing Season 2027-28* (July 2026) | <https://cacp.da.gov.in/Home/latestRabiReports> — Government of India publication | `raw/cacp_rabi_price_policy_latest.pdf` (63 MB, scanned) | wheat cost of cultivation by state, 2022-23 – 2024-25 | costs in `configs/*.yaml` |
| District boundaries of India (ADM2, simplified geometry) | geoBoundaries gbOpen, release of 12 Dec 2023, via <https://www.geoboundaries.org/api/current/gbOpen/IND/ADM2/> — boundary source Pathways Data Pvt. Ltd. / lgdirectory.gov.in, **Open Data Commons Open Database License (ODbL) 1.0**; attribution details are saved in `geo/india_districts_source.json` | `geo/india_districts.geojson` (8 MB, git-ignored) and `geo/districts_subset.geojson` (the districts around the mapped sites, written by `scripts/make_maps.py`, committed) | 735 districts, boundaries as of 2021 | district centroids for the weather download; district map |

`reference/cacp_wheat_cost_punjab.csv` and `reference/cacp_wheat_cost_haryana.csv` are Annex
Table 5.6(a) of that CACP report (pages 189 and 184), typed in by hand because the PDF has
no text layer. A test checks that the line items add up to the report's own total.

### Notes on the ICRISAT files

- `-1` means "not reported" and becomes missing; a wheat area of 0 is treated the same way.
- **Year convention.** Crop files label a crop year by its first calendar year (2008 = sown
  November 2008, harvested April 2009). The price file is labelled by marketing year: the
  mean Punjab harvest price of "year" Y sits on the MSP of crop year Y−1 in every year. The
  loader therefore assigns a price row of year Y to season Y−1
  (`data_loader.PRICE_YEAR_OFFSET`).
- **Irrigated share** = wheat irrigated area / wheat area, clipped to [0, 1] (150 of 2,071
  district-years report more irrigated than total area). It is missing for West Bengal and
  most of Maharashtra, which drops those rows from the yield fit.
- **Nitrogen** is district consumption per hectare of gross cropped area (all crops, whole
  year), the closest proxy available for the wheat dose.
- **Districts are the 1966 ones.** Each is placed on today's map by name (aliases in
  `src/farmrl/districts.py`); its weather is that of the same-named modern district's
  centroid, which for split districts is only part of the old area.
- The panel keeps district-years with more than 10,000 ha of wheat (`yield_fit.min_area_kha`):
  2,071 rows, 220 districts, 15 states; 1,719 rows in 13 states have everything the fit needs.

### Heat-stress days

Days with a daily maximum above 30 °C between 15 February and 31 March (grain filling),
from NASA POWER `T2M_MAX`; threshold and window are `HEAT_TMAX` / `HEAT_WINDOW` in
`data_loader.py`.

## What is NOT here yet

| Dataset | Why | What to do | Expected file |
|---|---|---|---|
| FAO crop yield + pesticides (Kaggle `patelris/crop-yield-prediction-dataset`) | `~/.kaggle/kaggle.json` is missing | create an API token on kaggle.com (Settings → API), put it in `~/.kaggle/`, run `python scripts/download_data.py kaggle` | `raw/yield_df.csv` |
| DES Cost of Cultivation plot-level series (data.gov.in) | `api.data.gov.in` and `desagri.gov.in` refused connections from our network, and the API needs a key | manual CSV download or `DATA_GOV_IN_API_KEY` + `DATA_GOV_IN_COC_RESOURCE` | `raw/cost_of_cultivation.csv` |
| Agmarknet wheat modal prices, Punjab mandis | same (API key / unreachable) | `DATA_GOV_IN_API_KEY`, or export from agmarknet.gov.in | `raw/agmarknet_wheat_punjab.csv` |
| ICRISAT rainfall | not among the supplied files; NASA POWER is used instead | optional | `raw/icrisat_rainfall.csv` |

`scripts/download_data.py` prints the exact steps for each. Consequences for the current
results: the pesticide-loss parameter is a prior, not a fit, and the pesticide figure is
skipped; harvest prices cover only seasons 2008–2015, so the held-out seasons (2017+) are
priced at the MSP.

## Processed tables

| File | Built by | Content |
|---|---|---|
| `weather_daily_<site>.csv` | `load_weather` | date, rain_mm, t_mean, t_max, t_min, rh, srad, et0_mm (Hargreaves), heat_day |
| `weather_fortnightly_<site>.csv` | `load_seasons` | season (sowing year), step 0–11 from 1 Nov, fortnight totals/means, heat_days |
| `icrisat_wheat.csv` | `load_icrisat` | district-year wheat area, production, yield, N/P/K per ha, irrigated share, harvest price |
| `district_sites.csv` | `districts.load_district_sites` | panel districts with matched boundary name and centroid |
| `district_weather.csv` | `load_district_weather` | per district and season: rain, ET0, mean temperature, late-season Tmax, heat-stress days |
| `yield_fit_table.csv` | `build_fit_table` | the wheat panel joined with district weather: what the yield model is fitted on |
| `wheat_msp.csv`, `wheat_prices.csv` | `load_msp`, `load_prices` | MSP per season and its index around a 5-year trend |
| `wheat_prices_<region>.csv` | `load_prices(region=…)` | region harvest prices where available, MSP otherwise; this is what the environment uses |
| `cost_of_cultivation_wheat_<state>.csv` | `load_cost_of_cultivation` | CACP table in long form |
| `fao_wheat_india.csv`, `wheat_mandi_monthly.csv` | `load_fao_pesticides`, `load_agmarknet` | only once the raw files exist |
