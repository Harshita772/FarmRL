"""FarmRL dashboard, page "Overview".

    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _APP_DIR.parent
for _p in [str(_APP_DIR), str(_ROOT_DIR / 'src'), str(_ROOT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st

import common
from farmrl.viz import interactive, style

run, policies = common.setup("Overview")

st.title("🌾 FarmRL: Crop Input Management & Market Timing")
st.markdown(
    "Wheat cultivation across Northwest India (Punjab & Haryana) follows a six-month Rabi cycle (November–April). "
    "Farmers make recurring fortnightly decisions on **nitrogen fertilizer**, **tubewell irrigation**, and **chemical pesticide sprays**, "
    "concluding with a harvest market timing choice to **sell immediately** at the seasonal Minimum Support Price (MSP) "
    "or **store grain** for 1–2 months in anticipation of price rises. **FarmRL** empirically evaluates whether reinforcement "
    "learning (PPO & critic-free GRPO) can learn state-contingent input policies that outperform standard government extension calendars, "
    "protecting farmer profit while mitigating groundwater overdraft and pesticide resistance."
)
st.caption(
    "📄 **Paper:** [2-Page Academic Report (PDF)](https://github.com/Harshita772/FarmRL/blob/main/report/farmrl_report.pdf) · "
    "🐙 **GitHub:** [Harshita772/FarmRL](https://github.com/Harshita772/FarmRL) · "
    "🏛️ **IIT Jodhpur — CSL3090: Artificial Intelligence**"
)

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
ym = run.meta["yield_model"]

st.subheader("1. Episode Structure")
st.markdown(
    f"Each episode simulates **one full {env['crop']} season** in **{env['steps_per_season']} fortnightly decision steps** "
    f"(from sowing on 1 November through harvest in late April), replaying real historical weather for *{run.meta['weather_site']}*."
)

st.subheader("2. State Space (What the agent observes)")
st.latex(
    r"s_t = \Big[ \frac{t}{T},\ \theta_t,\ \frac{N^{\text{soil}}_t}{100},\ p_t,\ \frac{R^{\text{cum}}_t}{200},\ "
    r"\frac{N^{\text{cum}}_t}{200},\ \frac{W^{\text{cum}}_t}{500},\ \frac{P_t}{P_0},\ \frac{R_{t-1}}{50},\ "
    r"\frac{\text{ET}_{0,t-1}}{100},\ \rho_t,\ \frac{c^{\text{pump}}_t}{20},\ e^{\text{spray}}_t \Big]"
)
st.markdown("""
- **$t / T$**: Normalized season progression stage ($0$ to $1$ across the 12 fortnights).
- **$\theta_t$**: Available root-zone soil moisture as a fraction of field capacity ($0$ to $1$).
- **$N^{\text{soil}}_t$**: Active mineral nitrogen in the root zone (kg N/ha).
- **$p_t$**: Pest infestation pressure index ($0$ = uninfested, $1$ = heavy outbreak).
- **$R^{\text{cum}}_t, N^{\text{cum}}_t, W^{\text{cum}}_t$**: Cumulative seasonal rainfall, fertilizer nitrogen, and total water received to date.
- **$P_t / P_0$**: Current crop market price index relative to historical trend.
- **$R_{t-1}, \text{ET}_{0,t-1}$**: Prior fortnight weather lags (rainfall and Penman-Monteith reference evapotranspiration).
- **$\rho_t = \text{ET}^{\text{cum}}_a / \text{ET}^{\text{cum}}_c$**: Crop water satisfaction ratio (cumulative actual over potential evapotranspiration).
- **$c^{\text{pump}}_t$**: Marginal tubewell pumping cost (INR/mm), dynamically tracking groundwater lift depth.
- **$e^{\text{spray}}_t$**: Chemical pesticide efficacy remaining (decays by 25% with each spray due to resistance).
""")

st.subheader("3. Action Space (Decisions chosen every fortnight)")
st.latex(
    rf"n_t \in \{{{", ".join(f"{v:g}" for v in env['nitrogen_levels'])}\}}\text{{ kg N/ha}},\quad "
    rf"w_t \in \{{{", ".join(f"{v:g}" for v in env['irrigation_levels'])}\}}\text{{ mm}},\quad "
    rf"q_t \in \{{{", ".join(f"{v:g}" for v in env['pesticide_levels'])}\}}\text{{ doses}}"
)
extra_actions = []
if len(run.meta.get("action_heads", [])) > 3:
    extra_actions.append(f"- **Post-Harvest Grain Storage ($m$)**: At the final harvest step ($t = 11$), the agent chooses whether to sell immediately ($0$ months) or store grain for $1$ to {run.meta['action_heads'][3] - 1} months in anticipation of market price rises.")
st.markdown("""
- **Nitrogen ($n_t$)**: Fertilizer application rate chosen from discrete dosage levels.
- **Irrigation ($w_t$)**: Tubewell irrigation water depth chosen from discrete volume levels.
- **Pesticide ($q_t$)**: Chemical spray dosage ($0, 1, 2$ doses).
""" + ("\n".join(extra_actions) if extra_actions else ""))

st.subheader("4. Reward Function (Economics and Environmental Penalties)")
st.latex(
    r"r_t = -\underbrace{\big(c_N n_t + c_W w_t + c_P q_t + C^{\text{pump}}_t(w_t)\big)}_{\text{Input \& Pumping Costs}} "
    r"- \underbrace{\big(\lambda_N \max(0, N^{\text{soil}}_{t+1} - N_{\text{safe}}) + \lambda_P q_t\big)}_{\text{Environmental Penalties}} "
    r"+ \underbrace{\mathbb{I}_{[t = T-1]} \big(\text{Revenue} - C_{\text{fixed}}\big)}_{\text{Harvest Revenue Net of Fixed Costs}}"
)
st.markdown(r"""
- **$c_N, c_W, c_P$**: Unit input costs calibrated to official CACP tables (₹12.9/kg N, ₹2.86/mm canal water, ₹1,183/pesticide dose).
- **$C^{\text{pump}}_t(w_t)$**: Pumping electricity cost, scaling with depth to the water table ($25\text{ m}$ base lift, dropping $0.5\text{ m/year}$).
- **$\lambda_N, N_{\text{safe}}$**: Penalty weight (₹5/kg) on soil nitrogen exceeding safe root threshold ($N_{\text{safe}} = 80\text{ kg/ha}$) to deter groundwater nitrate leaching.
- **$\lambda_P$**: Penalty (₹100/dose) discouraging excessive toxic chemical spraying.
- **$\text{Revenue} - C_{\text{fixed}}$**: Harvest grain yield $\times$ sale price minus storage costs and fixed non-input labor/machinery expenses ($C_{\text{fixed}} = 29{,}450\text{ INR/ha}$).
""")

st.subheader("5. Biophysical Yield Response Function")
st.latex(
    r"Y = y_{\text{max}} \cdot \underbrace{\big(1 - e^{-\beta_N (N + N_0)}\big)}_{\text{Nitrogen (Mitscherlich)}} \cdot "
    r"\underbrace{\big(1 - K_y(1 - \rho_T)\big)}_{\text{Water Deficit (FAO-33)}} \cdot "
    r"\underbrace{\prod_{g=1}^4 \big(1 - K_{y,g} \bar{\sigma}^{\text{aer}}_g\big)}_{\text{Waterlogging Hypoxia (FAO-66)}} \cdot "
    r"\underbrace{\big(1 - L_p D_{\text{pest}}\big)}_{\text{Pest Damage}} \cdot "
    r"\underbrace{\big(1 - \alpha_H H_{\text{days}}\big)}_{\text{Terminal Heat Stress}}"
)
y_plat = ym['y_max'] * ym.get('y_scale', 1)
n_c = ym['n_coef']
s_n = env.get('soil_n_start', 30)
ky_v = ym['ky']
p_l = ym['pest_loss']

st.markdown(
    f"- **$y_{{\\text{{max}}}}$**: Potential maximum plateau yield ({y_plat:.1f} q/ha), fitted on 1,719 ICRISAT district-years across 13 states.\n"
    f"- **$\\beta_N, N_0$**: Nitrogen response curvature ({n_c:.4f}) and baseline soil nitrogen endowment ({s_n:.0f} kg/ha).\n"
    f"- **$K_y, \\rho_T$**: FAO-33 water-deficit yield factor ({ky_v:.3f}) and full-season evapotranspiration satisfaction ratio.\n"
    "- **$K_{y,g}, \\bar{\\sigma}^{\\text{aer}}_g$**: Stage-specific root hypoxia yield loss due to waterlogging when excess rainfall/irrigation exceeds drainage capacity.\n"
    f"- **$L_p, D_{{\\text{{pest}}}}$**: Maximum fractional pest yield loss ($L_p = {p_l:.3f}$ calibrated) and cumulative season pest damage index.\n"
    "- **$\\alpha_H, H_{\\text{days}}$**: Heat stress penalty coefficient for extreme temperature days during the critical grain filling period."
)

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
