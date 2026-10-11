"""How results hold up across training seeds, seasons, weather types, assumptions and regions."""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402
from farmrl import runs  # noqa: E402
from farmrl.viz import interactive, style  # noqa: E402

run, policies = common.setup("Robustness")
st.title("Robustness")
money = "{:,.0f}"

# ------------------------------------------------------------------------ training seeds
st.header("Training seeds")
if run.seed_summary is None:
    st.info("This run evaluated one training seed only. Runs made with the current pipeline save "
            "`seed_summary.csv` for every seed.")
else:
    st.markdown("Every PPO training seed is rolled out on the held-out seasons. The other pages show the "
                f"first seed ({run.meta['train_seeds'][0] if run.meta['train_seeds'] else 'loaded model'}); "
                "this is the spread across all of them.")
    a, b = st.columns([3, 2])
    with a:
        common.chart(interactive.seed_profit(run, policies))
    with b:
        table = interactive.seed_table(run)
        st.dataframe(table.style.format({c: money if "INR" in c else "{:.1f}" for c in table.columns[1:]}),
                     hide_index=True, use_container_width=True)

# ---------------------------------------------------------------- where the gain comes from
st.header("Where the gain comes from")
if run.attribution is None:
    st.info("This run has no `attribution.csv` (made before the gain was split by decision).")
else:
    st.markdown("Each decision (nitrogen, water, pests, sale timing) is handed from the rule baseline to the agent "
                "in every possible order; a decision's bar is its average marginal effect on held-out profit "
                "(a Shapley value), so the bars add up to the total gain. Whiskers: std over training seeds.")
    bases = list(dict.fromkeys(run.attribution["baseline_policy"]))
    default = bases.index(run.reference_policy) if run.reference_policy in bases else 0
    base = st.radio("Against", bases, index=default, horizontal=True, format_func=style.label)
    common.chart(interactive.attribution(run, base))
    table = interactive.attribution_table(run)
    st.dataframe(table.style.format({c: "{:+,.0f}" for c in table.columns[1:]}), hide_index=True, use_container_width=True)
    if run.tuning:
        with st.expander("Tuned rule baselines (grid search on the training seasons only)"):
            for name, rec in run.tuning.items():
                st.markdown(f"**{style.label(name + '_tuned')}** — training {rec['objective']} "
                            f"₹{rec['train_score']:,.0f}/ha, against ₹{rec['default_train_score']:,.0f}/ha untuned; "
                            f"{rec['evaluations']:,} settings tried on {len(rec['train_seasons'])} seasons.")
                st.json({"tuned": rec["params"], "untuned": rec["default_params"]}, expanded=False)

# ------------------------------------------------------------------------------- seasons
st.header("Seasons and rainfall")
split = st.radio("Seasons", ["test", "train"], horizontal=True,
                 format_func={"test": "Held-out seasons", "train": "Training seasons"}.get)
common.chart(interactive.profit_by_season(run, policies, split))
a, b = st.columns(2)
with a:
    common.chart(interactive.dry_wet(run, policies, split))
with b:
    common.chart(interactive.rain_response(run, policies))
if run.dry_wet is not None:
    table = interactive.dry_wet_table(run, policies, split)
    st.dataframe(table.style.format({c: money for c in table.columns if "INR" in c or c == "Wet − dry"}),
                 hide_index=True, use_container_width=True)
    st.caption("Dry and wet are seasons below and above the median season rain of all seasons in the run.")

# --------------------------------------------------------------------------- sensitivity
st.header("Sensitivity to prices, costs, the yield model and the assumptions")
if run.sensitivity is None:
    st.info("No sensitivity results in this run. Add them with\n\n"
            f"`python scripts/run_experiments.py --run-name {run.name} --load-model results/runs/{run.name}/model.pt --sensitivity`")
else:
    st.markdown("One quantity at a time is scaled and every policy re-evaluated on the held-out seasons. "
                "Nothing is retrained or retuned, so this shows whether the ranking survives a wrong assumption.")
    common.chart(interactive.sensitivity_gain(run))
    shown = [p for p in policies if p in set(run.sensitivity["policy"])]
    groups = [g for g in run.sensitivity["group"].unique() if g != "baseline"] if "group" in run.sensitivity else [None]
    group = st.radio("Profit of every policy, by group", groups, horizontal=True) if groups != [None] else None
    common.chart(interactive.sensitivity(run, shown, group))
    with st.expander("Table"):
        table = interactive.sensitivity_table(run)
        st.dataframe(table.style.format({c: money for c in table.columns[3:6]} | {"Random, % of PPO": "{:.0f}", "×": "{:g}"}),
                     hide_index=True, use_container_width=True)

# ------------------------------------------------------------------------------- regions
st.header("Other regions")
related = runs.related(run, run.path.parent)
if not related:
    st.info(f"No related run found. A run named `{run.name}_<variant>` (for example the Haryana config, "
            f"`python scripts/run_all.py --run-name {run.name}_haryana --config configs/haryana.yaml`) is compared here.")
else:
    group = [run, *related]
    st.markdown("The same pipeline under another config: its own region plateau, costs, harvest prices and weather.")
    common.chart(interactive.compare_runs(group, policies))
    table = interactive.compare_runs_table(group)
    st.dataframe(table.style.format({"PPO profit (INR/ha)": money, "PPO std over seeds": money,
                                     "Gain over reference": "{:+,.0f}", "Random, % of PPO": "{:.0f}"}, na_rep="–"),
                 hide_index=True, use_container_width=True)
