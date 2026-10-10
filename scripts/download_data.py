"""Download the raw datasets into data/raw/.

    conda run -n TogAI python scripts/download_data.py            # everything
    conda run -n TogAI python scripts/download_data.py weather msp  # selected steps

Every step is independent: one that cannot run (missing credentials, site unreachable,
registration wall) prints manual instructions and the script moves on. Nothing is ever
synthesised here -- a dataset is either downloaded or reported as missing.
"""
import os
import re
import sys
import subprocess
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from farmrl.data_loader import DISTRICT_WEATHER, RAW, SITES  # noqa: E402
from farmrl.viz.maps import GEO_DIR, GEO_FILE, GEO_META  # noqa: E402

UA = {"User-Agent": "Mozilla/5.0 (FarmRL coursework data download)"}
POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
POWER_PARAMS = "PRECTOTCORR,T2M,T2M_MAX,T2M_MIN,RH2M,ALLSKY_SFC_SW_DWN"
KAGGLE_DATASET = "patelris/crop-yield-prediction-dataset"
RBI_LISTING = ("https://www.rbi.org.in/Scripts/AnnualPublications.aspx"
               "?head=Handbook%20of%20Statistics%20on%20Indian%20Economy")
RBI_VIEW = "https://www.rbi.org.in/Scripts/PublicationsView.aspx?id={}"
RBI_MSP_2014_ID = 15815     # 2014 edition of the table: crop years 1981-82 .. 2013-14
RBI_MSP_LATEST_ID = 23848   # fallback if the listing page cannot be parsed (2026 edition)
CACP_RABI_URL = "https://cacp.da.gov.in/Home/latestRabiReports"
GEOBOUNDARIES_API = "https://www.geoboundaries.org/api/current/gbOpen/IND/ADM2/"
DATAGOV_API = "https://api.data.gov.in/resource/{}"
# Agmarknet "Variety-wise Daily Market Prices Data of Commodity" on data.gov.in. The API
# refused connections from the network this was written on, so the id is NOT verified;
# override it with DATA_GOV_IN_AGMARKNET_RESOURCE if the catalogue page shows another one.
AGMARKNET_RESOURCE = os.environ.get("DATA_GOV_IN_AGMARKNET_RESOURCE",
                                    "35985678-0d79-46b4-9ed6-6f13308a1d24")

STATUS = {}


def _manual(name, text):
    STATUS[name] = "MANUAL STEP NEEDED"
    print(f"\n[{name}] not downloaded -- manual step needed:\n{text}\n")


def _save(name, path, content):
    path.write_bytes(content)
    STATUS[name] = f"ok -> {path.relative_to(ROOT)} ({len(content) / 1024:.0f} kB)"
    print(f"[{name}] saved {path.relative_to(ROOT)}")


# ----------------------------------------------------------------------------- weather
def weather(start=1985, end=2024, force=False):
    """NASA POWER daily weather for every site in data_loader.SITES (no key needed)."""
    for site, meta in SITES.items():
        out = RAW / f"weather_{site}.csv"
        name = f"weather:{site}"
        if out.exists() and not force:
            STATUS[name] = f"already present -> {out.relative_to(ROOT)}"
            continue
        q = dict(parameters=POWER_PARAMS, community="AG", latitude=meta["lat"],
                 longitude=meta["lon"], start=f"{start}0101", end=f"{end}1231", format="CSV")
        try:
            r = requests.get(POWER_URL, params=q, timeout=300)
            r.raise_for_status()
            if "-END HEADER-" not in r.text:
                raise ValueError("unexpected response (no POWER header)")
            _save(name, out, r.content)
        except Exception as e:  # network errors, HTTP errors, malformed payloads
            _manual(name, f"  request failed: {e}\n  Open {r.url if 'r' in dir() else POWER_URL}\n"
                          f"  in a browser and save the CSV as data/raw/{out.name}")


def _power(lat, lon, start, end, out, tries=4):
    """Fetch one POWER daily point file; returns None on success, else the error text."""
    import time
    q = dict(parameters=POWER_PARAMS, community="AG", latitude=lat, longitude=lon,
             start=f"{start}0101", end=f"{end}1231", format="CSV")
    err = None
    for attempt in range(tries):
        try:
            r = requests.get(POWER_URL, params=q, timeout=300)
            r.raise_for_status()
            if "-END HEADER-" not in r.text:
                raise ValueError("unexpected response (no POWER header)")
            out.write_bytes(r.content)
            return None
        except Exception as e:      # network errors, HTTP 429/5xx, malformed payloads
            err = str(e)
            time.sleep(3 * (attempt + 1))
    return err


def district_weather(start=2007, end=2018, workers=4, force=False):
    """NASA POWER daily weather at the centroid of every district in the yield panel.

    Centroids come from the boundary file in data/geo (farmrl.districts); one file per
    district is cached in data/raw/weather_districts/<dist_code>.csv and never fetched
    twice, so an interrupted download resumes where it stopped. The years only need to
    cover the ICRISAT panel (default 2007-2018), not the 40 years used by the environment.
    """
    from concurrent.futures import ThreadPoolExecutor
    from farmrl.districts import load_district_sites
    try:
        sites = load_district_sites()
    except Exception as e:      # no ICRISAT files or no boundary file yet
        _manual("district_weather", f"  cannot list the panel districts: {e}\n"
                                    "  run the icrisat and geo steps first")
        return
    folder = RAW / DISTRICT_WEATHER
    folder.mkdir(parents=True, exist_ok=True)
    placed = sites[sites["matched"]]
    todo = [r for r in placed.itertuples() if force or not (folder / f"{r.dist_code}.csv").exists()]
    print(f"[district_weather] {len(placed)} districts placed on the map "
          f"({len(sites) - len(placed)} could not be), {len(todo)} to download")

    def fetch(r):
        return r, _power(r.lat, r.lon, start, end, folder / f"{r.dist_code}.csv")

    failed = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (r, err) in enumerate(pool.map(fetch, todo), start=1):
            if err:
                failed.append(f"{r.district} ({r.state}): {err}")
            if i % 25 == 0 or i == len(todo):
                print(f"[district_weather] {i}/{len(todo)} done", flush=True)
    have = len(list(folder.glob("*.csv")))
    STATUS["district_weather"] = f"{have}/{len(placed)} district files in {folder.relative_to(ROOT)}"
    unplaced = sites[~sites["matched"]]
    if len(unplaced):
        print("[district_weather] no boundary match (add an alias in farmrl/districts.py): "
              + ", ".join(f"{r.district} ({r.state})" for r in unplaced.itertuples()))
    if failed:
        _manual("district_weather:failed", "  " + "\n  ".join(failed[:20])
                + "\n  re-run `python scripts/download_data.py district_weather` to retry these")


# ------------------------------------------------------------------------------ kaggle
def kaggle():
    out = RAW / "yield_df.csv"
    if out.exists():
        STATUS["kaggle"] = f"already present -> {out.relative_to(ROOT)}"
        return
    cred = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"
    if not cred.exists() and not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
        _manual("kaggle", f"""  {cred} is missing. To add it:
    1. Sign in at https://www.kaggle.com -> profile picture -> Settings -> API ->
       "Create New Token". A file called kaggle.json is downloaded.
    2. Move it to {cred}
       (PowerShell:  mkdir $HOME\\.kaggle -Force; mv $HOME\\Downloads\\kaggle.json $HOME\\.kaggle\\)
    3. Re-run:  conda run -n TogAI python scripts/download_data.py kaggle
  Or download yield_df.csv by hand from
    https://www.kaggle.com/datasets/{KAGGLE_DATASET}
  and save it as data/raw/yield_df.csv""")
        return
    cmd = [sys.executable, "-m", "kaggle", "datasets", "download", "-d", KAGGLE_DATASET,
           "-f", "yield_df.csv", "-p", str(RAW), "--unzip"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    zipped = RAW / "yield_df.csv.zip"
    if zipped.exists():  # older CLI versions ignore --unzip for single files
        import zipfile
        with zipfile.ZipFile(zipped) as z:
            z.extract("yield_df.csv", RAW)
        zipped.unlink()
    if out.exists():
        STATUS["kaggle"] = f"ok -> {out.relative_to(ROOT)}"
    else:
        _manual("kaggle", f"  kaggle CLI failed:\n{res.stdout}{res.stderr}")


# ----------------------------------------------------------------------------- icrisat
def icrisat():
    """The DLD sits behind a (free) registration, so this only checks what is on disk."""
    import pandas as pd
    crops, fert = RAW / "icrisat_crops.csv", RAW / "icrisat_fertiliser.csv"
    have = [p.name for p in sorted(RAW.glob("icrisat_*.csv"))]
    if crops.exists() and fert.exists():
        states = sorted(pd.read_csv(crops, usecols=["State Name"])["State Name"].unique())
        STATUS["icrisat"] = (f"present (manual download): {', '.join(have)}; "
                             f"{len(states)} states incl. {', '.join(states[:4])}...")
    want = """  1. Register (free) and sign in at http://data.icrisat.org/dld/
  2. Open "District Level Data" -> Apportioned. Pick all states (or at least the wheat
     states), the longest year range offered, and export one CSV per category:
        Crops: Area, Production & Yield          -> data/raw/icrisat_crops.csv
        Fertiliser Consumption (N, P, K)         -> data/raw/icrisat_fertiliser.csv
        Irrigation: Crop-wise Irrigated Area     -> data/raw/icrisat_irrigation.csv
        Rainfall: Monthly / Annual               -> data/raw/icrisat_rainfall.csv
        Prices: Farm Harvest Prices              -> data/raw/icrisat_prices.csv
     Replace existing files of the same name; the loader keeps every state it finds.
  3. Re-run:  conda run -n TogAI python scripts/prepare_data.py"""
    missing = [n for n in ("icrisat_crops.csv", "icrisat_fertiliser.csv", "icrisat_irrigation.csv",
                           "icrisat_rainfall.csv", "icrisat_prices.csv") if not (RAW / n).exists()]
    if missing:
        print(f"\n[icrisat] have: {have or 'nothing'}; missing: {missing}\n{want}\n")
        if "icrisat" not in STATUS:
            STATUS["icrisat"] = "MANUAL STEP NEEDED"
        else:
            STATUS["icrisat"] += f" | still missing: {', '.join(missing)}"


# ---------------------------------------------------------------------------- MSP (RBI)
def msp():
    """Wheat MSP by crop year from the RBI Handbook of Statistics on Indian Economy.

    Two editions of the same table are saved as HTML: the 2014 one (1981-82 onwards) and
    the latest one (1993-94 onwards); data_loader splices them.
    """
    latest_id = RBI_MSP_LATEST_ID
    try:
        listing = requests.get(RBI_LISTING, headers=UA, timeout=90).text
        m = re.search(r"PublicationsView\.aspx\?id=(\d+)[^>]*>(?:(?!PublicationsView).)*?"
                      r"Minimum Support Price for Foodgrains", listing, flags=re.S)
        if m:
            latest_id = int(m.group(1))
    except Exception as e:
        print(f"[msp] could not read the RBI listing ({e}); using table id {latest_id}")
    for tag, pid in (("2014", RBI_MSP_2014_ID), ("latest", latest_id)):
        out = RAW / f"rbi_msp_foodgrains_{tag}.html"
        try:
            r = requests.get(RBI_VIEW.format(pid), headers=UA, timeout=90)
            r.raise_for_status()
            if "Wheat" not in r.text:
                raise ValueError("page does not contain the MSP table")
            _save(f"msp:{tag}", out, r.content)
        except Exception as e:
            _manual(f"msp:{tag}", f"  {e}\n  Open {RBI_VIEW.format(pid)} and save the page as "
                                  f"data/raw/{out.name}")


# -------------------------------------------------------------------- cost of cultivation
def _datagov(resource, filters, out, name, page=1000):
    key = os.environ.get("DATA_GOV_IN_API_KEY")
    if not key:
        return "no DATA_GOV_IN_API_KEY environment variable"
    rows, offset = [], 0
    try:
        while True:
            q = {"api-key": key, "format": "json", "limit": page, "offset": offset}
            q.update({f"filters[{k}]": v for k, v in filters.items()})
            r = requests.get(DATAGOV_API.format(resource), params=q, timeout=120)
            r.raise_for_status()
            recs = r.json().get("records", [])
            rows += recs
            offset += page
            if len(recs) < page:
                break
    except Exception as e:
        return f"API request failed: {e}"
    if not rows:
        return "API returned no records for these filters"
    import pandas as pd
    pd.DataFrame(rows).to_csv(out, index=False)
    STATUS[name] = f"ok -> {out.relative_to(ROOT)} ({len(rows)} rows)"
    print(f"[{name}] saved {len(rows)} rows to {out.relative_to(ROOT)}")
    return None


def costs():
    """CACP Rabi price policy report (PDF, has the Punjab wheat cost break-up) + DES data."""
    pdf = RAW / "cacp_rabi_price_policy_latest.pdf"
    if pdf.exists():
        STATUS["costs:cacp_report"] = f"already present -> {pdf.relative_to(ROOT)}"
    else:
        try:
            r = requests.get(CACP_RABI_URL, headers=UA, timeout=600)
            r.raise_for_status()
            if not r.content.startswith(b"%PDF"):
                raise ValueError("the page did not return a PDF")
            _save("costs:cacp_report", pdf, r.content)
        except Exception as e:
            _manual("costs:cacp_report", f"  {e}\n  Open https://cacp.da.gov.in -> Price Policy "
                    f"Reports -> Rabi (latest) and save the PDF as data/raw/{pdf.name}")
    print("[costs] the report is a scanned PDF; the Punjab wheat table (Annex Table 5.6(a)) is\n"
          "        transcribed by hand in data/reference/cacp_wheat_cost_punjab.csv")

    out = RAW / "cost_of_cultivation.csv"
    if out.exists():
        STATUS["costs:des"] = f"already present -> {out.relative_to(ROOT)}"
        return
    resource = os.environ.get("DATA_GOV_IN_COC_RESOURCE")
    err = (_datagov(resource, {}, out, "costs:des") if resource
           else "no DATA_GOV_IN_COC_RESOURCE environment variable")
    if err:
        _manual("costs:des", f"""  ({err})
  The plot-level DES series needs either a data.gov.in key or a manual download:
    A. Manual: https://www.data.gov.in -> search "Cost of Cultivation of Principal Crops"
       (Directorate of Economics & Statistics), open the wheat / state-wise resource,
       click Download -> CSV and save it as data/raw/cost_of_cultivation.csv
       Alternative source: https://desagri.gov.in -> Statistics -> Cost of Cultivation.
    B. API: register at https://www.data.gov.in (My Account -> API key), then
         setx DATA_GOV_IN_API_KEY <your key>
         setx DATA_GOV_IN_COC_RESOURCE <resource id shown on the dataset's API tab>
       open a new terminal and re-run:  python scripts/download_data.py costs""")


# ----------------------------------------------------------------------------- agmarknet
def prices():
    out = RAW / "agmarknet_wheat_punjab.csv"
    if out.exists():
        STATUS["prices:agmarknet"] = f"already present -> {out.relative_to(ROOT)}"
        return
    err = _datagov(AGMARKNET_RESOURCE, {"State": "Punjab", "Commodity": "Wheat"}, out,
                   "prices:agmarknet")
    if err:
        _manual("prices:agmarknet", f"""  ({err})
    A. API: register at https://www.data.gov.in, copy the key from My Account, then
         setx DATA_GOV_IN_API_KEY <your key>
       open a new terminal and re-run:  python scripts/download_data.py prices
       (resource "Variety-wise Daily Market Prices Data of Commodity", id
        {AGMARKNET_RESOURCE} -- set DATA_GOV_IN_AGMARKNET_RESOURCE if the site lists another).
    B. Manual: https://agmarknet.gov.in -> Price Trends / "Market-wise, Commodity-wise
       Daily Report": Commodity = Wheat, State = Punjab, all markets, the widest date
       range allowed (repeat per year if needed), Export -> Excel/CSV. Concatenate the
       exports and save as data/raw/agmarknet_wheat_punjab.csv. The loader needs a date
       column, a market column and a modal price column (Rs/quintal); names are matched
       loosely ("Arrival_Date"/"Price Date", "Market", "Modal_Price"/"Modal Price").
  Until then the pipeline uses the wheat MSP series (RBI) as the price series.""")


# ------------------------------------------------------------------------------- geo
def geo(force=False):
    """District boundaries (geoBoundaries gbOpen, India ADM2, simplified) for the map."""
    import json
    if GEO_FILE.exists() and not force:
        STATUS["geo"] = f"already present -> {GEO_FILE.relative_to(ROOT)}"
        return
    GEO_DIR.mkdir(parents=True, exist_ok=True)
    try:
        meta = requests.get(GEOBOUNDARIES_API, headers=UA, timeout=90).json()
        url = meta.get("simplifiedGeometryGeoJSON") or meta["gjDownloadURL"]
        r = requests.get(url, headers=UA, timeout=600)
        r.raise_for_status()
        if "FeatureCollection" not in r.text[:2000]:
            raise ValueError("download is not a GeoJSON FeatureCollection")
        keep = ["boundaryID", "boundaryName", "boundaryType", "boundaryYearRepresented",
                "boundarySource", "boundaryLicense", "licenseSource", "boundarySourceURL",
                "sourceDataUpdateDate", "buildDate"]
        GEO_META.write_text(json.dumps({**{k: meta.get(k) for k in keep}, "downloaded_from": url}, indent=2))
        _save("geo", GEO_FILE, r.content)
        print(f"[geo] licence: {meta.get('boundaryLicense')} (source: {meta.get('boundarySource')})")
    except Exception as e:
        _manual("geo", f"""  {e}
  Any district-level GeoJSON of India works (one feature per district, with a name
  property such as shapeName / DISTRICT / dtname / NAME_2). Sources:
    - geoBoundaries: https://www.geoboundaries.org -> India -> ADM2 -> GeoJSON (simplified)
    - datameet: https://github.com/datameet/maps (Districts)
  Save it as {GEO_FILE.relative_to(ROOT)}""")


STEPS = {"weather": weather, "kaggle": kaggle, "icrisat": icrisat, "msp": msp,
         "costs": costs, "prices": prices, "geo": geo, "district_weather": district_weather}

if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    chosen = sys.argv[1:] or list(STEPS)
    unknown = [s for s in chosen if s not in STEPS]
    if unknown:
        sys.exit(f"unknown step(s) {unknown}; choose from {list(STEPS)}")
    for step in chosen:
        STEPS[step]()
    print("\n================ download summary ================")
    for k, v in STATUS.items():
        print(f"  {k:<22} {v}")
