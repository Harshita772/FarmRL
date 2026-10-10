"""Shared pieces of the dashboard: path setup, the sidebar and cached loaders.

Pages read saved runs through ``farmrl.runs`` and draw with ``farmrl.viz.interactive``;
the only place the environment is run is ``farmrl.live`` (the "Simulate" page).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import streamlit as st  # noqa: E402

from farmrl import live, runs  # noqa: E402
from farmrl.viz import style  # noqa: E402


@st.cache_resource(show_spinner=False)
def _load_run(name, stamp):
    return runs.get(name)


@st.cache_resource(show_spinner="Loading the trained agent...")
def _load_agent(name, stamp):
    return live.load_agent(runs.get(name))


def _stamp(name):
    """Changes when the run is rebuilt, so cached objects are refreshed."""
    return (runs.RUNS_DIR / name / "metadata.json").stat().st_mtime


def agent_for(run):
    return _load_agent(run.name, _stamp(run.name))


def setup(title):
    """Page config + sidebar. Returns (run, selected policies); stops the page with
    instructions when no run exists yet."""
    st.set_page_config(page_title=f"FarmRL · {title}", page_icon="🌾", layout="wide")
    names = runs.list_runs()
    if not names:
        st.title("FarmRL dashboard")
        st.warning("No experiment run found in `results/runs/`.")
        st.markdown("Build one with the pipeline, then reload this page:")
        st.code("conda run -n TogAI python scripts/run_all.py --run-name v1\n"
                "# or a fast smoke run\nconda run -n TogAI python scripts/run_all.py --run-name smoke --quick")
        st.stop()
    with st.sidebar:
        st.header("Run")
        name = st.selectbox("Experiment run", names, key="run_name", help="Every folder in results/runs/, newest first")
        run = _load_run(name, _stamp(name))
        meta = run.meta
        st.caption(f"created {meta['created'].replace('T', ' ')}"
                   + (" · **quick run**" if meta.get("quick") else ""))
        with st.expander("Run metadata"):
            st.write({"timesteps": meta["timesteps"], "training seeds": meta["train_seeds"],
                      "rollout seeds per season": meta["rollout_seeds"], "weather site": meta["weather_site"],
                      "train seasons": f"{meta['train_seasons'][0]}–{meta['train_seasons'][-1]}",
                      "test seasons": meta["test_seasons"], "git": (meta.get("git_hash") or "n/a")[:10]})
        available = run.policies(include_aux=True)
        key = f"policies_{name}"
        chosen = st.multiselect("Policies", available, default=run.policies(), key=key, format_func=style.label)
    if not chosen:
        st.info("Select at least one policy in the sidebar.")
        st.stop()
    return run, chosen


def chart(fig):
    st.plotly_chart(fig, use_container_width=True)
