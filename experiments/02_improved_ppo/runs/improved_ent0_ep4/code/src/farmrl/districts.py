"""Where the districts of the yield panel are: a centroid for each one, taken from the
boundary GeoJSON in data/geo, so that weather can be fetched per district.

ICRISAT uses 1966 district names; they are matched to today's boundaries by name
(aliases + fuzzy matching, see ``viz.maps``). The boundary file has no state attribute,
and several district names occur in more than one state, so among the name candidates
the one closest to the state's approximate centre is taken, and a candidate further than
``MAX_DEGREES`` from it is rejected.
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from . import data_loader as dl
from .viz import maps

# Approximate geographic centres (lat, lon), only used to tell same-named districts apart.
STATE_CENTRES = {
    "andhra pradesh": (15.9, 79.7), "assam": (26.2, 92.9), "bihar": (25.8, 85.8),
    "chhattisgarh": (21.3, 81.9), "gujarat": (22.7, 71.6), "haryana": (29.2, 76.3),
    "himachal pradesh": (31.8, 77.2), "jharkhand": (23.6, 85.3), "karnataka": (14.7, 75.9),
    "kerala": (10.5, 76.4), "madhya pradesh": (23.5, 78.5), "maharashtra": (19.5, 76.0),
    "orissa": (20.5, 84.4), "odisha": (20.5, 84.4), "punjab": (30.9, 75.4),
    "rajasthan": (26.6, 73.8), "tamil nadu": (11.1, 78.4), "telangana": (17.9, 79.2),
    "uttar pradesh": (27.0, 80.8), "uttarakhand": (30.1, 79.2), "west bengal": (23.8, 87.9),
    "jammu & kashmir": (33.5, 75.0),
}
MAX_DEGREES = 6.0
# ICRISAT spelling (lower case) -> boundary-file spelling, for names fuzzy matching misses.
# Every target below exists in the geoBoundaries file; extend it when new districts appear.
ALIASES = {
    "24 parganas": "North Twenty Four Parganas",
    "ahmedabad": "Ahmadabad",
    "ahmednagar": "Ahmadnagar",
    "badaun": "Budaun",
    "banaskantha": "Banas Kantha",
    "barabanki": "Bara Banki",
    "beed": "Bid",
    "bhatinda": "Bathinda",
    "bijapur / vijayapura": "Bijapur",
    "bombay": "Mumbai",
    "bulandshahar": "Bulandshahr",
    "buldhana": "Buldana",
    "burdwan": "Barddhaman",
    "champaran": "Purba Champaran",
    "chittorgarh": "Chittaurgarh",
    "cooch behar": "Koch Bihar",
    "dangs": "The Dangs",
    "darjeeling": "Darjiling",
    "dholpur": "Dhaulpur",
    "ferozepur": "Firozpur",
    "ferozpur": "Firozpur",
    "gulbarga / kalaburagi": "Gulbarga",
    "gurugram": "Gurgaon",
    "hissar": "Hisar",
    "hooghly": "Hugli",
    "howrah": "Haora",
    "jalore": "Jalor",
    "jhunjhunu": "Jhunjhunun",
    "kanpur": "Kanpur Nagar",
    "kutch": "Kachchh",
    "mahasu": "Shimla",
    "mahendragarh / narnaul": "Mahendragarh",
    "malda": "Maldah",
    "mehsana": "Mahesana",
    "midnapur": "Paschim Medinipur",
    "mohali": "Sahibzada Ajit Singh Nagar",
    "monghyr": "Munger",
    "mungair": "Munger",
    "muzzafarnagar": "Muzaffarnagar",
    "narnaul": "Mahendragarh",
    "narsinghpur": "Narsimhapur",
    "nawanshahr": "Shahid Bhagat Singh Nagar",
    "nowgong": "Nagaon",
    "palamau": "Palamu",
    "panchmahal": "Panch Mahals",
    "panchmahals": "Panch Mahals",
    "purnea": "Purnia",
    "purulia": "Puruliya",
    "rae-bareily": "Rae Bareli",
    "roopnagar / ropar": "Rupnagar",
    "ropar": "Rupnagar",
    "sabarkantha": "Sabar Kantha",
    "santhal paragana": "Dumka",
    "shahabad": "Bhojpur",
    "shahabad (now part of bhojpur district)": "Bhojpur",
    "sibsagar": "Sivasagar",
    "singhbhum": "Pashchimi Singhbhum",
    "swami madhopur": "Sawai Madhopur",
    "west dinajpur": "Uttar Dinajpur",
    "yeotmal": "Yavatmal",
}


def _centroid(feature):
    ring = max(maps.rings(feature), key=len)
    return float(ring[:, 1].mean()), float(ring[:, 0].mean())     # lat, lon


def _candidates(name, by_name, threshold):
    """Boundary names worth trying for one ICRISAT name ('A / B' is tried as A and as B)."""
    parts = [name] + [p.strip() for p in name.split("/")] if "/" in name else [name]
    out = []
    for part in parts:
        target = ALIASES.get(part.strip().lower(), part.strip())
        for hit, score, _ in process.extract(target, list(by_name), scorer=fuzz.ratio, limit=6,
                                             score_cutoff=threshold):
            out.append((score, hit))
    return sorted(set(out), reverse=True)


def district_sites(panel, geojson=None, threshold=82):
    """One row per district of ``panel`` (needs dist_code, state, district): its matched
    boundary name and centroid. Districts that cannot be placed get NaN coordinates and
    ``matched = False``; nothing is guessed."""
    geojson = geojson or maps.load_geojson()
    key = maps.name_key(geojson)
    by_name = {}
    for f in geojson["features"]:
        by_name.setdefault(str(f["properties"][key]), []).append(f)
    rows = []
    for (code, state, district), _ in panel.groupby(["dist_code", "state", "district"]):
        centre = STATE_CENTRES.get(state.strip().lower())
        best = None
        for score, hit in _candidates(district, by_name, threshold):
            for f in by_name[hit]:
                lat, lon = _centroid(f)
                gap = np.hypot(lat - centre[0], lon - centre[1]) if centre else 0.0
                if gap <= MAX_DEGREES and (best is None or (score, -gap) > (best[0], -best[1])):
                    best = (score, gap, hit, lat, lon)
        rows.append({"dist_code": int(code), "state": state, "district": district,
                     "matched": best is not None, "geo_name": best[2] if best else None,
                     "match_score": best[0] if best else np.nan,
                     "lat": round(best[3], 3) if best else np.nan,
                     "lon": round(best[4], 3) if best else np.nan})
    return pd.DataFrame(rows)


def load_district_sites(crop="wheat", min_area_kha=10.0, raw=dl.RAW, processed=dl.PROCESSED, save=True):
    """Centroids of the districts in the yield panel, saved to data/processed/district_sites.csv."""
    panel = dl.load_icrisat(crop, min_area_kha=min_area_kha, raw=raw, processed=processed, save=False)
    sites = district_sites(panel)
    if save:
        sites.to_csv(processed / "district_sites.csv", index=False)
    return sites
