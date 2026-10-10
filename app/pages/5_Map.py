"""District choropleth of per-district results."""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402
from farmrl.viz import maps, style  # noqa: E402

run, policies = common.setup("Map")
st.title("District map")

table = run.districts
if table is None:
    st.info("This run has no per-district results. Create them with\n\n"
            f"`python scripts/run_experiments.py --run-name {run.name} --load-model results/runs/{run.name}/model.pt --districts`")
    st.stop()

metrics = [m for m in maps.METRICS if m in table.columns]
c = st.columns(2)
metric = c[0].selectbox("Metric", metrics, format_func=maps.METRICS.get)
relative = metric.startswith("ppo_")
options = [p for p in table["policy"].unique() if not (relative and p == "ppo")]
policy = c[1].selectbox("Compared with" if relative else "Policy", options, format_func=style.label,
                        index=options.index("recommended") if relative and "recommended" in options else 0)
try:
    common.chart(maps.interactive_map(run, metric, policy))
except maps.MapUnavailable as why:
    st.warning(f"The map cannot be drawn: {why}")

st.caption(f"The agent was trained on {run.meta['weather_site'].title()} weather only; each district is the same "
           "policy facing that district's own held-out seasons. Yield response and costs are identical everywhere.")
shown = table[table["policy"] == policy][["district", "state", "profit", "yield_q_ha", "irrigation_mm", "nitrogen_kg",
                                          "ppo_profit_gain", "ppo_water_saved_mm"]]
st.dataframe(shown.round(1), hide_index=True, use_container_width=True)
src = maps.GEO_META
if src.exists():
    import json
    m = json.loads(src.read_text())
    st.caption(f"Boundaries: geoBoundaries {m.get('boundaryType')} ({m.get('boundaryYearRepresented')}), "
               f"source {m.get('boundarySource')}, licence {m.get('boundaryLicense')}.")
