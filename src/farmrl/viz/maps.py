"""District choropleths: a static matplotlib map and an interactive plotly one.

Boundaries are read from a plain GeoJSON file (no geopandas: polygons are drawn directly
with matplotlib / handed to plotly as-is). District names in the results are matched to
the GeoJSON by fuzzy string matching, helped by ``ALIASES`` for old or alternative
spellings and checked against the site coordinates when those are known.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from matplotlib.path import Path as MplPath
try:
    from rapidfuzz import fuzz, process
except ImportError:
    import difflib
    class _ProcessFallback:
        @staticmethod
        def extract(query, choices, scorer=None, limit=5, score_cutoff=0, **kwargs):
            matches = []
            q = str(query).lower()
            for idx, choice in enumerate(choices):
                score = difflib.SequenceMatcher(None, q, str(choice).lower()).ratio() * 100
                if score >= score_cutoff:
                    matches.append((choice, score, idx))
            matches.sort(key=lambda x: x[1], reverse=True)
            return matches[:limit]
    process = _ProcessFallback()
    fuzz = type("Fuzz", (), {"WRatio": None})

from .. import runs
from . import style

ROOT = Path(__file__).resolve().parents[3]
GEO_DIR = ROOT / "data" / "geo"
GEO_FILE = GEO_DIR / "india_districts.geojson"        # full download (git-ignored)
GEO_META = GEO_DIR / "india_districts_source.json"
GEO_SUBSET = GEO_DIR / "districts_subset.geojson"     # small extract around the data, committed

NAME_KEYS = ["shapeName", "DISTRICT", "District", "district", "dtname", "DTNAME", "NAME_2",
             "name", "NAME"]
# Result name (lower case) -> name to look for in the GeoJSON. Add entries when a new
# dataset uses a spelling the fuzzy matcher cannot resolve.
ALIASES = {
    "hissar": "Hisar",
    "mahendragarh / narnaul": "Mahendragarh",
    "narnaul": "Mahendragarh",
    "gurugram": "Gurgaon",
    "ferozepur": "Firozpur",
    "ropar": "Rupnagar",
    "nawanshahr": "Shahid Bhagat Singh Nagar",
    "mohali": "Sahibzada Ajit Singh Nagar",
}
METRICS = {"ppo_profit_gain": "PPO profit gain (INR/ha)", "profit": "Net profit (INR/ha)",
           "ppo_water_saved_mm": "Irrigation saved by PPO (mm)", "yield_q_ha": "Yield (q/ha)",
           "irrigation_mm": "Irrigation (mm)", "nitrogen_kg": "Nitrogen (kg N/ha)"}


class MapUnavailable(Exception):
    """No boundaries or no per-district results to draw."""


# ----------------------------------------------------------------------------- geometry
def load_geojson(path=None):
    """The boundary file: an explicit path, else the full download, else the committed subset."""
    for p in ([Path(path)] if path else [GEO_FILE, GEO_SUBSET]):
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    raise MapUnavailable(f"no district GeoJSON in {GEO_DIR} -- run "
                         "`python scripts/download_data.py geo`")


def name_key(geojson):
    props = geojson["features"][0]["properties"]
    for k in NAME_KEYS:
        if k in props:
            return k
    raise MapUnavailable(f"GeoJSON has no district-name property (looked for {NAME_KEYS})")


def rings(feature):
    """Outer rings of a Polygon / MultiPolygon feature as (n, 2) lon-lat arrays."""
    g = feature["geometry"]
    polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
    return [np.asarray(poly[0], dtype=float)[:, :2] for poly in polys]


def contains(feature, lon, lat):
    return any(MplPath(r).contains_point((lon, lat)) for r in rings(feature))


def match_districts(names, geojson, coords=None, threshold=80):
    """{result name: GeoJSON feature} via alias -> exact -> fuzzy match.

    ``coords`` ({name: (lat, lon)}) settles ambiguous names: a candidate whose polygon
    contains the point wins. Names that cannot be matched are left out."""
    key = name_key(geojson)
    by_name = {}
    for f in geojson["features"]:
        by_name.setdefault(str(f["properties"][key]), []).append(f)
    out = {}
    for name in names:
        target = ALIASES.get(str(name).strip().lower(), str(name).strip())
        hits = process.extract(target, list(by_name), scorer=fuzz.WRatio, limit=5, score_cutoff=threshold)
        candidates = [f for hit, _, _ in hits for f in by_name[hit]]
        if not candidates:
            continue
        chosen = candidates[0]
        if coords and name in coords:
            lat, lon = coords[name]
            inside = [f for f in candidates if contains(f, lon, lat)]
            chosen = inside[0] if inside else chosen
        out[name] = chosen
    return out


def _signed_area(ring):
    r = np.asarray(ring, dtype=float)[:, :2]
    return float(np.sum(r[:-1, 0] * r[1:, 1] - r[1:, 0] * r[:-1, 1]) / 2)


def for_plotly(feature):
    """Copy of ``feature`` with its rings wound the way plotly's map engine (d3-geo) reads
    them: outer rings clockwise, holes counter-clockwise. GeoJSON (RFC 7946) uses the
    opposite order, and plotly fills such a polygon inside out (the whole globe except
    the district)."""
    g = feature["geometry"]
    polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
    fixed = []
    for poly in polys:
        rings_ = []
        for i, ring in enumerate(poly):
            clockwise = _signed_area(ring) < 0
            want_clockwise = i == 0
            rings_.append(list(ring) if clockwise == want_clockwise else list(ring)[::-1])
        fixed.append(rings_)
    geometry = {"type": g["type"], "coordinates": fixed[0] if g["type"] == "Polygon" else fixed}
    return {**feature, "geometry": geometry}


def neighbourhood(geojson, features, pad=0.8):
    """Features whose first ring overlaps the padded bounding box of ``features``."""
    pts = np.vstack([r for f in features for r in rings(f)])
    (x0, y0), (x1, y1) = pts.min(0) - pad, pts.max(0) + pad
    near = []
    for f in geojson["features"]:
        r = rings(f)[0]
        if r[:, 0].max() >= x0 and r[:, 0].min() <= x1 and r[:, 1].max() >= y0 and r[:, 1].min() <= y1:
            near.append(f)
    return near, (x0, x1, y0, y1)


# ------------------------------------------------------------------------------- data
def district_values(run, metric="ppo_profit_gain", policy=None):
    """(table with one row per district, legend label). For 'gain' metrics ``policy`` is
    the baseline PPO is compared with; otherwise it is the policy whose value is shown."""
    table = run.districts
    if table is None or table.empty:
        raise MapUnavailable("run has no district_summary.csv -- re-run "
                             "`python scripts/run_experiments.py --districts`")
    if metric not in table.columns:
        raise MapUnavailable(f"metric '{metric}' not in district_summary.csv ({list(table.columns)})")
    relative = metric.startswith("ppo_")
    choices = [p for p in table["policy"].unique() if not (relative and p == "ppo")]
    if policy not in choices:
        policy = ("recommended" if "recommended" in choices else choices[0]) if relative else \
                 ("ppo" if "ppo" in choices else choices[0])
    d = table[table["policy"] == policy].copy()
    label = METRICS.get(metric, metric)
    label += f" vs {style.label(policy)}" if relative else f", {style.label(policy)}"
    return d, label


def _prepare(run, metric, policy, geojson):
    if not (hasattr(run, "districts") and run.districts is not None):
        run = runs.get(getattr(run, "path", getattr(run, "name", run)))
    d, label = district_values(run, metric, policy)
    geojson = geojson or load_geojson()
    coords = {r["district"]: (r["lat"], r["lon"]) for _, r in d.iterrows()} if {"lat", "lon"} <= set(d) else None
    matched = match_districts(d["district"], geojson, coords)
    if not matched:
        raise MapUnavailable("none of the districts in the results could be matched to the GeoJSON")
    missing = sorted(set(d["district"]) - set(matched))
    values = d.set_index("district")[metric]
    return run, geojson, matched, values, label, missing


def write_subset(run=None, geojson=None, path=GEO_SUBSET):
    """Save the districts around the data as a small GeoJSON the dashboard can ship with."""
    run, geojson, matched, *_ = _prepare(run, "profit", None, geojson)
    key = name_key(geojson)
    near, _ = neighbourhood(geojson, list(matched.values()))
    for f in near:
        f["id"] = str(f["properties"][key])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"type": "FeatureCollection", "features": near}), encoding="utf-8")
    return Path(path)


# -------------------------------------------------------------------------------- maps
def static_map(run=None, metric="ppo_profit_gain", policy=None, geojson=None):
    """Matplotlib choropleth of ``metric`` over the districts in the run."""
    run, geojson, matched, values, label, missing = _prepare(run, metric, policy, geojson)
    key = name_key(geojson)
    near, (x0, x1, y0, y1) = neighbourhood(geojson, list(matched.values()))
    diverging = values.min() < 0 < values.max()
    lim = max(abs(values.min()), abs(values.max()))
    cmap = style.diverging_cmap() if diverging else plt.get_cmap(style.SEQUENTIAL)
    norm = plt.Normalize(-lim, lim) if diverging else plt.Normalize(values.min(), values.max())

    fig, ax = plt.subplots(figsize=(style.COLUMN, 3.4))
    ax.add_collection(PatchCollection([Polygon(r, closed=True) for f in near for r in rings(f)],
                                      facecolor="#f0efec", edgecolor="white", linewidth=0.5))
    for name, f in matched.items():
        rs = rings(f)
        ax.add_collection(PatchCollection([Polygon(r, closed=True) for r in rs], facecolor=cmap(norm(values[name])),
                                          edgecolor=style.INK_2, linewidth=0.6))
        big = max(rs, key=len)
        cx, cy = big[:, 0].mean(), big[:, 1].mean()
        ax.annotate(f"{f['properties'][key]}\n{values[name] / (1000 if abs(lim) > 5000 else 1):,.1f}"
                    + ("k" if abs(lim) > 5000 else ""), (cx, cy), ha="center", va="center", fontsize=5.5,
                    color=style.INK, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect(1 / np.cos(np.radians((y0 + y1) / 2)))
    ax.axis("off")
    ax.set_title(label)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02).ax.tick_params(labelsize=6)
    if missing:
        ax.text(0, -0.03, "not matched to a boundary: " + ", ".join(missing), transform=ax.transAxes,
                fontsize=5.5, color=style.INK_2)
    return fig


def interactive_map(run=None, metric="ppo_profit_gain", policy=None, geojson=None):
    """Plotly choropleth of ``metric``; hover shows the district's numbers."""
    import plotly.graph_objects as go
    run, geojson, matched, values, label, missing = _prepare(run, metric, policy, geojson)
    key = name_key(geojson)
    near, (x0, x1, y0, y1) = neighbourhood(geojson, list(matched.values()))
    near = [{**for_plotly(f), "id": str(f["properties"][key])} for f in near]
    names = list(matched)
    geo_names = [matched[n]["properties"][key] for n in names]
    diverging = values.min() < 0 < values.max()
    lim = float(max(abs(values.min()), abs(values.max())))
    subset = {"type": "FeatureCollection", "features": near}
    context = [f["properties"][key] for f in near]
    fig = go.Figure()
    fig.add_trace(go.Choropleth(geojson=subset, featureidkey="id", locations=context,
                                z=[0] * len(context), colorscale=[[0, "#f0efec"], [1, "#f0efec"]],
                                showscale=False, marker_line_color="white", marker_line_width=0.6,
                                hoverinfo="skip"))
    fig.add_trace(go.Choropleth(
        geojson=subset, featureidkey="id", locations=geo_names,
        z=[float(values[n]) for n in names], text=names,
        colorscale=[[0, style.DIVERGING[0]], [0.5, style.DIVERGING[1]], [1, style.DIVERGING[2]]] if diverging else style.SEQUENTIAL,
        zmin=-lim if diverging else float(values.min()), zmax=lim if diverging else float(values.max()),
        marker_line_color=style.INK_2, marker_line_width=0.8, colorbar=dict(title=dict(text=label, side="right")),
        hovertemplate="<b>%{text}</b><br>" + label + ": %{z:,.0f}<extra></extra>"))
    fig.update_geos(fitbounds="locations", visible=False, projection_type="mercator")
    fig.update_layout(title=label, margin=dict(l=0, r=0, t=45, b=0), height=560,
                      dragmode="pan", font=style.PLOTLY_LAYOUT["font"])
    if missing:
        fig.add_annotation(text="not matched to a boundary: " + ", ".join(missing), xref="paper",
                           yref="paper", x=0, y=0, showarrow=False, font=dict(size=11, color=style.INK_2))
    return fig


def make_maps(run=None, metric="ppo_profit_gain", policy=None, out_dir=None):
    """Save the static PNG/PDF and the interactive HTML under <run>/maps/. Also refreshes
    the committed boundary subset. Returns the paths written."""
    run = runs.get(run)
    geojson = load_geojson()
    out = Path(out_dir) if out_dir else run.out_dir("maps")
    style.apply()
    fig = static_map(run, metric, policy, geojson)
    paths = []
    for ext in ("png", "pdf"):
        paths.append(out / f"district_{metric}.{ext}")
        fig.savefig(paths[-1], dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(out / f"district_{metric}.html")
    interactive_map(run, metric, policy, geojson).write_html(paths[-1], include_plotlyjs="cdn")
    if GEO_FILE.exists():
        paths.append(write_subset(run, geojson))
    return paths
