"""How results hold up across seasons, weather types and economic assumptions."""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402
from farmrl.viz import interactive  # noqa: E402

run, policies = common.setup("Robustness")
st.title("Robustness")

split = st.radio("Seasons", ["test", "train"], horizontal=True,
                 format_func={"test": "Held-out seasons", "train": "Training seasons"}.get)
common.chart(interactive.profit_by_season(run, policies, split))
a, b = st.columns(2)
with a:
    common.chart(interactive.dry_wet(run, policies))
with b:
    common.chart(interactive.rain_response(run, policies))

st.header("Sensitivity to prices, costs and penalties")
if run.sensitivity is None:
    st.info("No sensitivity results in this run. Add them with\n\n"
            f"`python scripts/run_experiments.py --run-name {run.name} --load-model results/runs/{run.name}/model.pt --sensitivity`")
else:
    st.markdown("Each panel scales one quantity and re-evaluates every policy on the held-out seasons. "
                "The agent is not retrained, so a flat PPO line next to a falling baseline means its policy still fits.")
    common.chart(interactive.sensitivity(run, [p for p in policies if p in set(run.sensitivity["policy"])]))
