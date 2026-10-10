"""What the trained agent does in any state."""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402
from farmrl import live  # noqa: E402
from farmrl.env import OBS_NAMES  # noqa: E402
from farmrl.viz import interactive  # noqa: E402

run, policies = common.setup("Policy explorer")
st.title("Policy explorer")

st.header("Policy maps")
if run.grids is None:
    st.info("This run has no policy_grids.parquet. Re-run `scripts/run_experiments.py`.")
else:
    st.markdown("Expected dose under the agent's policy as one state variable and the season stage vary; "
                "the other variables are held at their median for that stage.")
    a, b = st.columns(2)
    with a:
        common.chart(interactive.policy_heatmap(run, "soil_moisture", "expected_irrigation_mm"))
    with b:
        common.chart(interactive.policy_heatmap(run, "soil_n", "expected_nitrogen_kg"))
    with st.expander("Other combinations"):
        grid = st.radio("State variable", ["soil_moisture", "soil_n"], horizontal=True)
        col = st.radio("Input", ["expected_nitrogen_kg", "expected_irrigation_mm", "expected_pesticide_level"], horizontal=True)
        common.chart(interactive.policy_heatmap(run, grid, col))

st.header("Ask the agent")
medians = run.state_medians
if medians is None or not run.model_path.exists():
    st.info("This run has no saved model or state medians.")
    st.stop()
st.markdown("Set a state with the sliders; the bars show the probability the agent gives each dose. "
            "Sliders start at the median state for the chosen fortnight.")
n_steps = len(medians)
step = st.slider("Fortnight of the season", 1, n_steps, min(4, n_steps)) - 1
base = medians.set_index("step").loc[step]
# slider ranges in the units the agent sees: (label, scale to real units, max in real units)
ranges = {"soil_moisture": ("Soil moisture (% of capacity)", 100, 100), "soil_n": ("Soil mineral N (kg/ha)", 100, 150),
          "pest": ("Pest pressure (%)", 100, 100), "cum_rain": ("Rain so far (mm)", 200, 300),
          "cum_n": ("Nitrogen applied so far (kg/ha)", 200, 300), "cum_water": ("Rain + irrigation so far (mm)", 500, 800),
          "price_index": ("Price relative to base (%)", 100, 150), "last_rain": ("Rain last fortnight (mm)", 50, 120),
          "last_et0": ("Reference ET last fortnight (mm)", 100, 120), "et_ratio": ("Water demand met so far (%)", 100, 100)}
state = {"stage": step / n_steps}
cols = st.columns(2)
for i, name in enumerate(k for k in OBS_NAMES if k != "stage"):
    label, scale, top = ranges.get(name, (name, 1, 1))
    default = float(min(base[f"obs_{name}"] * scale, top))
    value = cols[i % 2].slider(label, 0.0, float(top), round(default, 1), key=f"{name}_{step}")
    state[name] = value / scale
common.chart(interactive.action_probabilities(live.action_distribution(run, common.agent_for(run), state)))
