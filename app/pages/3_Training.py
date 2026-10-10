"""Learning curves and PPO diagnostics."""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402
from farmrl.viz import interactive  # noqa: E402

run, policies = common.setup("Training")
st.title("Training")
h = run.history
if h is None or h.empty:
    st.info("This run loaded an existing model, so there is no training history. "
            "Re-run `scripts/run_experiments.py` without `--load-model` to record one.")
    st.stop()

last = h[h["iteration"] == h["iteration"].max()]
c = st.columns(4)
c[0].metric("Training seeds", h["train_seed"].nunique())
c[1].metric("Environment steps", f"{int(h['timesteps'].max()):,}")
c[2].metric("Final mean return", f"₹{last['mean_return'].mean():,.0f}/ha")
c[3].metric("Final policy entropy", f"{last['entropy'].mean():.2f} nats")
common.chart(interactive.learning_curve(run))
common.chart(interactive.diagnostics(run))
st.caption("Return is the mean over episodes finished in each iteration, with actions sampled from the policy. "
           "Dashed lines are the two best baselines on the same training seasons.")
with st.expander("Hyperparameters"):
    st.json(run.config["train"])
with st.expander("History table"):
    st.dataframe(h.round(4), hide_index=True, use_container_width=True)
