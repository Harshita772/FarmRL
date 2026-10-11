"""Run the environment live with the trained agent and chosen baselines."""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402
from farmrl import live  # noqa: E402
from farmrl.viz import interactive, style  # noqa: E402

run, policies = common.setup("Simulate a season")
st.title("Simulate a season")
st.markdown("Pick a weather year and change prices, costs or penalties. The environment is run "
            "live with the trained agent and the selected baselines. The agent was **not** retrained "
            "for the changed economics, so this shows how its fixed policy copes.")

playable = [p for p in policies if p in live.policies(run, None)]
if not playable:
    st.info("None of the selected policies can be simulated live. Select PPO or a baseline in the sidebar.")
    st.stop()

meta = run.meta
seasons = meta["train_seasons"] + meta["test_seasons"]
c = st.columns([1.2, 1, 1, 1])
season = c[0].selectbox("Weather year (sowing year)", seasons, index=len(seasons) - 1,
                        format_func=lambda s: f"{s}-{str(s + 1)[-2:]}" + ("  · held out" if s in meta["test_seasons"] else ""))
price = c[1].slider("Crop price ×", 0.5, 1.5, 1.0, 0.05)
penalty = c[2].slider("Penalty weights ×", 0.0, 5.0, 1.0, 0.25)
seed = c[3].number_input("Random seed (pests, stored-grain price)", 0, 999, 0)
c = st.columns(3)
n_cost = c[0].slider("Nitrogen cost ×", 0.25, 4.0, 1.0, 0.25)
w_cost = c[1].slider("Irrigation cost ×", 0.25, 10.0, 1.0, 0.25)
p_cost = c[2].slider("Pesticide cost ×", 0.25, 4.0, 1.0, 0.25)

agent = common.agent_for(run)
steps = live.simulate(run, agent, season, playable, seed=int(seed), price=price, nitrogen_cost=n_cost,
                      irrigation_cost=w_cost, pesticide_cost=p_cost, penalty=penalty)

first = next(iter(steps.values()))
st.subheader(f"Season {season}-{str(season + 1)[-2:]}: {first['rain_mm'].sum():.0f} mm of rain, "
             f"price ₹{first['price'].iloc[0]:,.0f}/quintal")
best = max(steps, key=lambda p: steps[p]["final_profit"].iloc[0])
cards = st.columns(len(steps))
for card, (name, d) in zip(cards, steps.items()):
    profit, ref = d["final_profit"].iloc[0], steps[best]["final_profit"].iloc[0]
    card.metric(style.label(name), f"₹{profit:,.0f}/ha",
                "best" if name == best else f"{profit - ref:+,.0f} vs {style.label(best)}",
                delta_color="off" if name == best else "normal")
    months = int(d["store_months"].sum())
    sale = f" · stored {months} mo, sold at ₹{d['final_sale_price'].iloc[0]:,.0f}" if months else ""
    card.caption(f"yield {d['final_yield'].iloc[0]:.1f} q/ha · N {d['nitrogen_kg'].sum():.0f} kg · "
                 f"water {d['irrigation_mm'].sum():.0f} mm · spray {d['pesticide_level'].sum():.0f}{sale}")

st.caption("Press ▶ Play or drag the slider to step through the fortnights.")
common.chart(interactive.season_player(steps))
common.chart(interactive.action_bars(steps))
with st.expander("Step table"):
    name = st.selectbox("Policy", list(steps), format_func=style.label)
    cols = ["step", "date", "rain_mm", "soil_moisture", "soil_n", "pest", "nitrogen_kg", "irrigation_mm",
            "pesticide_level", "store_months", "input_cost", "pumping_cost", "aeration_stress", "spray_efficacy",
            "penalty", "reward", "cum_profit"]
    st.dataframe(steps[name][cols].assign(date=lambda d: d["date"].dt.strftime("%d %b %Y")).round(2),
                 hide_index=True, use_container_width=True)
