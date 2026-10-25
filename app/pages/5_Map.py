"""District choropleth of per-district results."""
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402
from farmrl.viz import maps, style  # noqa: E402

run, policies = common.setup("Map")
st.title("District map")

table = run.districts
if table is None:
    st.info("This run has no per-district results. Select **v4_haryana** or **v4** in the sidebar Run selector, "
            f"or generate them with:\n\n`python scripts/run_experiments.py --run-name {run.name} "
            f"--load-model results/runs/{run.name}/model.pt --districts`")
    st.stop()

metrics = [m for m in maps.METRICS if m in table.columns]
c = st.columns(2)
metric = c[0].selectbox("Metric", metrics, format_func=maps.METRICS.get)
relative = metric.startswith("ppo_")
options = [p for p in table["policy"].unique() if not (relative and p == "ppo")]
policy = c[1].selectbox("Compared with" if relative else "Policy", options, format_func=style.label,
                        index=options.index("recommended") if relative and "recommended" in options else 0)

maps_dir = run.path / "maps"
html_path = maps_dir / f"district_{metric}.html"
png_path = maps_dir / f"district_{metric}.png"
pdf_path = maps_dir / f"district_{metric}.pdf"

is_default_policy = (relative and policy == "recommended") or (not relative and policy == "ppo")
has_prerendered = is_default_policy and (html_path.exists() or png_path.exists() or pdf_path.exists())

tab_interactive, tab_static = st.tabs(["🗺️ Interactive Choropleth", "📊 Static Publication Figure"])

with tab_interactive:
    # 1. Prefer local pre-rendered interactive HTML choropleth if available and default policy selected
    if is_default_policy and html_path.exists():
        html_content = html_path.read_text(encoding="utf-8")
        components.html(html_content, height=600, scrolling=False)
        st.caption(f"🗺️ Loaded interactive choropleth from `{run.name}/maps/{html_path.name}`")
    else:
        # 2. Dynamic generation fallback
        try:
            fig = maps.interactive_map(run, metric, policy)
            common.chart(fig)
        except maps.MapUnavailable as why:
            st.warning(f"The map cannot be drawn: {why}")
        except ImportError as err:
            st.warning(f"Interactive map requires plotly: {err}. Please view the static publication figure tab.")
        except Exception as ex:
            st.warning(f"Could not render interactive map: {ex}")

with tab_static:
    # 1. Prefer local pre-rendered 300 DPI PNG if available and default policy selected
    if is_default_policy and png_path.exists():
        st.image(str(png_path), caption=f"Static Choropleth: {maps.METRICS.get(metric, metric)}", use_container_width=True)
        col1, col2, _ = st.columns([1, 1, 2])
        if pdf_path.exists():
            col1.download_button(
                label="📥 Download Vector PDF",
                data=pdf_path.read_bytes(),
                file_name=pdf_path.name,
                mime="application/pdf",
                key=f"dl_pdf_{run.name}_{metric}"
            )
        col2.download_button(
            label="📥 Download 300 DPI PNG",
            data=png_path.read_bytes(),
            file_name=png_path.name,
            mime="image/png",
            key=f"dl_png_{run.name}_{metric}"
        )
    else:
        # 2. Dynamic static map generation fallback
        try:
            fig = maps.static_map(run, metric, policy)
            st.pyplot(fig)
        except Exception as ex:
            st.info(f"Static figure not precomputed and could not be generated dynamically: {ex}")

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
