"""FarmRL dashboard, page "Overview".

    conda run -n TogAI streamlit run app/streamlit_app.py
"""
import streamlit as st

import common
from farmrl.viz import interactive, style

run, policies = common.setup("Overview")

st.title("🌾 FarmRL: learning when to fertilise, irrigate and spray")
st.markdown(
    "A farmer decides every fortnight how much **nitrogen**, **irrigation** and **pesticide** to "
    "apply to a crop without knowing the rest of the season's weather, and at harvest whether to "
    "**sell or store** the grain. FarmRL trains a "
    "reinforcement-learning agent (PPO) on replayed historical seasons and compares it with "
    "rule-based practice on seasons it has never seen.")

k = interactive.headline(run)
if k is None:
    st.info("This run has no trained agent and baseline to compare.")
else:
    base = style.label(k["baseline"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PPO net profit", f"₹{k['profit']:,.0f}/ha", f"{k['profit_gain']:+,.0f} vs {base} ({k['profit_gain_pct']:+.0f}%)")
    c2.metric("Irrigation saved", f"{k['water_saved_mm']:,.0f} mm", f"{k['water_saved_pct']:+.0f}% vs {base}")
    c3.metric("Nitrogen change", f"{k['nitrogen_change_kg']:+,.0f} kg/ha", f"{k['nitrogen_change_pct']:+.0f}% vs {base}",
              delta_color="inverse")
    c4.metric("Yield change", f"{k['yield_gain']:+.1f} q/ha", f"vs {base}", delta_color="off")
    note = "" if k["random_share"] is None else f" A random policy earns {k['random_share']:.0f}% of PPO's profit."
    st.caption(f"Means over the {k['seasons']} held-out seasons, first training seed.{note} Results come from a "
               "simulator calibrated on limited data; see the Data page for what is and is not fitted and the "
               "Robustness page for training seeds, sensitivity and where the gain comes from.")

left, right = st.columns([3, 2])
with left:
    common.chart(interactive.profit_box(run, policies))
with right:
    st.subheader("Held-out seasons")
    st.dataframe(interactive.summary_table(run, policies).style.format(
        {"Profit (INR/ha)": "{:,.0f}", "Profit std": "{:,.0f}", "Yield (q/ha)": "{:.1f}", "N (kg/ha)": "{:.0f}",
         "Irrigation (mm)": "{:.0f}", "Pesticide (doses)": "{:.1f}"}), hide_index=True, use_container_width=True)
common.chart(interactive.inputs_bar(run, policies))

st.header("The decision problem")
env = run.config["env"]
a, b = st.columns(2)
with a:
    st.markdown(f"**Episode.** One {env['crop']} season in {env['steps_per_season']} fortnightly steps, "
                f"replaying real weather at *{run.meta['weather_site']}*.")
    st.markdown("**State** (what the agent sees)")
    st.latex(r"s_t = \big(t/T,\ \theta_t,\ N^{soil}_t,\ p_t,\ \textstyle\sum R,\ \sum n,\ \sum (R+w),\ P/P_0,\ R_{t-1},\ ET_{0,t-1},\ \rho_t\big)")
    st.markdown("**Action** (one level of each input per fortnight)")
    st.latex(rf"n_t \in \{{{', '.join(f'{v:g}' for v in env['nitrogen_levels'])}\}}\ \text{{kg N/ha}},\quad "
             rf"w_t \in \{{{', '.join(f'{v:g}' for v in env['irrigation_levels'])}\}}\ \text{{mm}},\quad "
             rf"q_t \in \{{{', '.join(f'{v:g}' for v in env['pesticide_levels'])}\}}")
    if len(run.meta.get("action_heads", [])) > 3:
        st.markdown(f"plus, at harvest, the months to store the grain: 0 to {run.meta['action_heads'][3] - 1}. "
                    f"The state also carries {', '.join(n.replace('_', ' ') for n in run.meta['obs_names'][11:])}.")
with b:
    st.markdown("**Reward** (input costs and environmental penalties each step, the harvest at the end)")
    st.latex(r"r_t = -\big[c_N n_t + c_W w_t + c_P q_t\big] - \big[\lambda_N \max(0, N^{soil}_{t+1} - N_{safe}) + \lambda_P q_t + \lambda_W w_t\big] + \mathbb{1}[t = T-1]\,(Y P - F)")
    st.markdown("**Yield** (fitted response × water-deficit × pest terms)")
    st.latex(r"Y = y_{max}\big[1 - e^{-\beta_N (N + N_0)}\big]\Big[1 - \beta_W \max\big(0, \tfrac{W - W^*}{100}\big)^2\Big]\big[1 - K_y(1 - \rho_T)\big]\big[1 - L\,D_{pest}\big]")
    ym = run.meta["yield_model"]
    st.caption(f"Parameters used in this run: y_max {ym['y_max'] * ym.get('y_scale', 1):.1f} q/ha, β_N {ym['n_coef']:.4f}, "
               f"W* {ym['w_opt']:.0f} mm, β_W {ym['w_excess']:.3f}, K_y {ym['ky']:.3f}, L {ym['pest_loss']}.")

on = [name for name, flag in (run.meta.get("mechanisms") or {}).items() if flag]
if on:
    st.header("What makes the decisions matter")
    st.markdown("Mechanisms switched on in this run's config (each has its own `enabled` flag; sources and "
                "assumptions are next to the numbers in the config):")
    text = {
        "groundwater": ("**Groundwater.** Pumping costs the energy to lift the water: "
                        "{energy_price_per_kwh:g} INR/kWh at {pump_efficiency:.0%} pump efficiency from a water table "
                        "{depth_m:g} m deep in {depth_year} and {decline_m_per_year:g} m shallower for every year before; "
                        "the lift grows through the season with what has been pumped."),
        "waterlogging": ("**Waterlogging.** Water the topsoil cannot hold stands there (draining {drain_mm_per_day:g} "
                         "mm/day); beyond {aeration_buffer_mm:g} mm the roots lack air and yield falls by the FAO-33 "
                         "factor of the growth period (flowering {ky_period[1]}, grain filling {ky_period[2]})."),
        "pest_resistance": ("**Pest resistance.** Each spray leaves the next one "
                            "{efficacy_decay:.0%} less effective, so sprays have to be saved for when they count."),
        "storage": ("**Sale timing.** At harvest the grain is sold or stored up to {max_months} months: the price "
                    "drifts up {monthly_drift:.0%} a month, reverts to its trend and moves randomly "
                    "(±{monthly_volatility:.0%}); storing costs {cost_per_quintal_month:g} INR/quintal a month, "
                    "interest and {loss_per_month:.1%} of the grain."),
    }
    for name in on:
        st.markdown("- " + text[name].format(**env[name]) if name in text else f"- {name}")
st.caption("Use the pages in the sidebar: simulate a season live, explore the learned policy, "
           "inspect training, robustness, the district map and the underlying data.")
