"""Plotly charts for the dashboard. Each function returns a ``plotly.graph_objects.Figure``
built from a saved run or from step tables; there is no Streamlit code in here, so the
charts can be tested and reused on their own.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import style

INPUTS = ["nitrogen_kg", "irrigation_mm", "pesticide_level"]
AGENT = "ppo"


def _layout(fig, **kw):
    fig.update_layout(**{**style.PLOTLY_LAYOUT, **kw})
    fig.update_xaxes(gridcolor=style.GRID, linecolor=style.AXIS)
    fig.update_yaxes(gridcolor=style.GRID, linecolor=style.AXIS)
    return fig


def _rgba(hex_color, alpha):
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def _test(run, policies=None):
    eps = run.episodes[run.episodes["split"] == "test"]
    return eps[eps["policy"].isin(policies or run.policies())]


# ---------------------------------------------------------------------------- overview
def headline(run, baseline=None):
    """Numbers for the KPI cards: PPO against one baseline on the held-out seasons."""
    means = _test(run).groupby("policy")[["profit", "yield_q_ha"] + INPUTS].mean()
    if AGENT not in means.index or len(means) < 2:
        return None
    others = means.drop(index=AGENT)
    base = baseline if baseline in others.index else ("recommended" if "recommended" in others.index
                                                      else others["profit"].idxmax())
    a, b = means.loc[AGENT], means.loc[base]
    pct = lambda new, old: (new - old) / abs(old) * 100 if old else float("nan")
    return {"baseline": base, "profit": a["profit"], "profit_gain": a["profit"] - b["profit"],
            "profit_gain_pct": pct(a["profit"], b["profit"]),
            "water_saved_mm": b["irrigation_mm"] - a["irrigation_mm"],
            "water_saved_pct": -pct(a["irrigation_mm"], b["irrigation_mm"]),
            "nitrogen_change_kg": a["nitrogen_kg"] - b["nitrogen_kg"],
            "nitrogen_change_pct": pct(a["nitrogen_kg"], b["nitrogen_kg"]),
            "yield_gain": a["yield_q_ha"] - b["yield_q_ha"], "seasons": int(_test(run)["season"].nunique())}


def summary_table(run, policies=None, split="test"):
    s = run.summary[(run.summary["split"] == split) & run.summary["policy"].isin(policies or run.policies())]
    cols = {"policy": "Policy", "profit": "Profit (INR/ha)", "profit_std": "Profit std", "yield_q_ha": "Yield (q/ha)",
            "nitrogen_kg": "N (kg/ha)", "irrigation_mm": "Irrigation (mm)", "pesticide_level": "Pesticide (doses)",
            "episodes": "Episodes"}
    out = s.sort_values("profit", ascending=False)[list(cols)].rename(columns=cols)
    out["Policy"] = out["Policy"].map(style.label)
    return out.reset_index(drop=True)


def profit_box(run, policies=None):
    eps = _test(run, policies)
    order = eps.groupby("policy")["profit"].mean().sort_values(ascending=False).index
    colors = style.policy_colors(order)
    fig = go.Figure()
    for p in order:
        fig.add_trace(go.Box(y=eps.loc[eps["policy"] == p, "profit"], name=style.label(p), marker_color=colors[p],
                             boxmean=True, hovertemplate="%{y:,.0f} INR/ha<extra></extra>"))
    return _layout(fig, title="Net profit on held-out seasons", yaxis_title="INR per hectare", showlegend=False)


def inputs_bar(run, policies=None):
    means = _test(run, policies).groupby("policy")[INPUTS].mean()
    order = _test(run, policies).groupby("policy")["profit"].mean().sort_values(ascending=False).index
    colors = style.policy_colors(order)
    fig = make_subplots(rows=1, cols=3, subplot_titles=[style.INPUT_LABELS[c] for c in INPUTS])
    for j, col in enumerate(INPUTS, start=1):
        fig.add_trace(go.Bar(x=[style.label(p) for p in order], y=means.loc[order, col],
                             marker_color=[colors[p] for p in order], showlegend=False,
                             hovertemplate="%{x}: %{y:.1f}<extra></extra>"), row=1, col=j)
    return _layout(fig, title="Inputs used per season (held-out mean)", height=380, margin=dict(l=50, r=20, t=90, b=45))


# --------------------------------------------------------------------------- simulator
def season_player(steps):
    """Animated timeline for ``{policy: step table}``: rain, soil moisture, pest pressure
    and cumulative profit, revealed fortnight by fortnight (play button and slider)."""
    names = list(steps)
    colors = style.policy_colors(names)
    first = steps[names[0]]
    x = [pd.Timestamp(d).strftime("%d %b") for d in first["date"]]
    n = len(first)
    rows = [("Rain (mm)", None), ("Soil moisture (% of capacity)", "soil_moisture_end"),
            ("Pest pressure (%)", "pest"), ("Cumulative profit (thousand INR/ha)", "cum_profit")]
    scale = {"soil_moisture_end": 100, "pest": 100, "cum_profit": 1 / 1000}
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        subplot_titles=[r[0] for r in rows], row_heights=[0.2, 0.25, 0.2, 0.35])

    def traces(k):
        out = [go.Bar(x=x[:k], y=first["rain_mm"][:k], marker_color="#86b6ef", showlegend=False,
                      hovertemplate="%{y:.0f} mm<extra></extra>")]
        for _, col in rows[1:]:
            for p in names:
                out.append(go.Scatter(x=x[:k], y=steps[p][col][:k] * scale[col], mode="lines+markers",
                                      name=style.label(p), legendgroup=p, showlegend=col == "cum_profit",
                                      line=dict(color=colors[p], width=2.5), marker=dict(size=6),
                                      hovertemplate=style.label(p) + ": %{y:.1f}<extra></extra>"))
        return out

    where = [(1, 1)] + [(r, 1) for r in (2, 3, 4) for _ in names]
    for tr, (r, c) in zip(traces(n), where):
        fig.add_trace(tr, row=r, col=c)
    fig.frames = [go.Frame(data=traces(k), name=str(k), traces=list(range(len(where)))) for k in range(1, n + 1)]
    fig.update_xaxes(categoryorder="array", categoryarray=x, range=[-0.5, n - 0.5])
    lo = min(float(s["cum_profit"].min()) for s in steps.values()) / 1000
    hi = max(float(s["cum_profit"].max()) for s in steps.values()) / 1000
    fig.update_yaxes(range=[0, max(5.0, float(first["rain_mm"].max()) * 1.15)], row=1, col=1)
    fig.update_yaxes(range=[0, 105], row=2, col=1)
    fig.update_yaxes(range=[0, 105], row=3, col=1)
    fig.update_yaxes(range=[lo - 5, hi + 5], row=4, col=1)
    play = dict(type="buttons", direction="left", x=0, y=-0.08, xanchor="left", yanchor="top", buttons=[
        dict(label="▶ Play", method="animate", args=[None, dict(frame=dict(duration=450, redraw=True), fromcurrent=True, transition=dict(duration=0))]),
        dict(label="❚❚ Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])
    slider = dict(active=n - 1, x=0.18, len=0.82, y=-0.06, currentvalue=dict(prefix="fortnight starting "),
                  steps=[dict(label=x[k - 1], method="animate",
                              args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=True))]) for k in range(1, n + 1)])
    return _layout(fig, height=820, updatemenus=[play], sliders=[slider], margin=dict(l=55, r=20, t=60, b=110),
                   legend=dict(orientation="h", yanchor="bottom", y=1.03, x=0))


def action_bars(steps):
    """Inputs applied each fortnight, one row per input, grouped by policy."""
    names = list(steps)
    colors = style.policy_colors(names)
    x = [pd.Timestamp(d).strftime("%d %b") for d in steps[names[0]]["date"]]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=[style.INPUT_LABELS[c] for c in INPUTS])
    for i, col in enumerate(INPUTS, start=1):
        for p in names:
            fig.add_trace(go.Bar(x=x, y=steps[p][col], name=style.label(p), legendgroup=p, showlegend=i == 1,
                                 marker_color=colors[p], hovertemplate=style.label(p) + ": %{y:g}<extra></extra>"),
                          row=i, col=1)
    return _layout(fig, barmode="group", height=560, title="Inputs applied per fortnight", margin=dict(l=50, r=20, t=100, b=45))


# ---------------------------------------------------------------------- policy explorer
def policy_heatmap(run, grid, column):
    g = run.grids[run.grids["grid"] == grid]
    z = g.pivot(index="value", columns="step", values=column)
    dates = run.rollouts.groupby("step")["date"].first()
    k = 100 if grid == "soil_moisture" else 1
    ylabel = "soil moisture (% of capacity)" if grid == "soil_moisture" else "soil mineral N (kg/ha)"
    label = style.INPUT_LABELS[column.replace("expected_", "")]
    fig = go.Figure(go.Heatmap(z=z.values, x=[pd.Timestamp(dates.loc[s]).strftime("%d %b") for s in z.columns],
                               y=z.index * k, colorscale=style.SEQUENTIAL, colorbar=dict(title=label),
                               hovertemplate="fortnight %{x}<br>" + ylabel + " %{y:.0f}<br>" + label + " %{z:.1f}<extra></extra>"))
    return _layout(fig, title=f"{label} chosen by the agent", xaxis_title="fortnight starting", yaxis_title=ylabel, height=420)


def action_probabilities(distribution):
    """Bar chart per input of the agent's action probabilities (from live.action_distribution)."""
    fig = make_subplots(rows=1, cols=len(distribution), subplot_titles=[style.INPUT_LABELS[k] for k in distribution])
    for j, (name, (levels, probs)) in enumerate(distribution.items(), start=1):
        fig.add_trace(go.Bar(x=[f"{v:g}" for v in levels], y=probs, marker_color=style.INPUT_COLORS[name], showlegend=False,
                             text=[f"{p:.0%}" for p in probs], textposition="outside",
                             hovertemplate="%{x}: %{y:.1%}<extra></extra>"), row=1, col=j)
        fig.update_yaxes(range=[0, 1.15], tickformat=".0%", row=1, col=j)
        fig.update_xaxes(type="category", row=1, col=j)
    return _layout(fig, height=380, title="Probability of each dose in this state", margin=dict(l=50, r=20, t=90, b=45))


# ------------------------------------------------------------------------------ training
def _band(fig, x, mean, std, color, name, row=None, col=None):
    kw = {} if row is None else {"row": row, "col": col}
    fig.add_trace(go.Scatter(x=list(x) + list(x[::-1]), y=list(mean + std) + list((mean - std)[::-1]), fill="toself",
                             fillcolor=_rgba(color, 0.18), line=dict(width=0), hoverinfo="skip", showlegend=False), **kw)
    fig.add_trace(go.Scatter(x=x, y=mean, mode="lines", line=dict(color=color, width=2.5), name=name,
                             showlegend=row is None, hovertemplate="%{y:,.2f}<extra></extra>"), **kw)


def learning_curve(run):
    h = run.history
    g = h.groupby("timesteps")["mean_return"]
    colors = style.policy_colors(run.policies())
    fig = go.Figure()
    _band(fig, g.mean().index.to_numpy(), g.mean().to_numpy() / 1000, g.std().fillna(0).to_numpy() / 1000,
          colors.get(AGENT, style.PALETTE[0]), f"PPO (mean ± std, {h['train_seed'].nunique()} seeds)")
    train = run.episodes[(run.episodes["split"] == "train") & run.episodes["policy"].isin(run.policies())]
    ref = train[train["policy"] != AGENT].groupby("policy")["reward"].mean().sort_values(ascending=False).head(2)
    for name, value in ref.items():
        fig.add_hline(y=value / 1000, line=dict(color=colors[name], dash="dash", width=1.5),
                      annotation_text=style.label(name), annotation_position="bottom right")
    return _layout(fig, title="Learning curve (training seasons)", xaxis_title="environment steps",
                   yaxis_title="mean episode return (thousand INR/ha)", height=430)


def diagnostics(run):
    h = run.history
    panels = [(c, t) for c, t in [("entropy", "Policy entropy"), ("value_loss", "Value loss"),
                                  ("policy_loss", "Policy loss"), ("kl", "Approximate KL")] if c in h.columns]
    fig = make_subplots(rows=1, cols=len(panels), subplot_titles=[t for _, t in panels])
    for j, (col, _) in enumerate(panels, start=1):
        g = h.groupby("timesteps")[col]
        _band(fig, g.mean().index.to_numpy(), g.mean().to_numpy(), g.std().fillna(0).to_numpy(), style.PALETTE[0], col, row=1, col=j)
    return _layout(fig, height=370, title="Training diagnostics (mean ± std over seeds)", showlegend=False,
                   margin=dict(l=50, r=20, t=90, b=45))


# ---------------------------------------------------------------------------- robustness
def profit_by_season(run, policies=None, split="test"):
    eps = run.episodes[(run.episodes["split"] == split) & run.episodes["policy"].isin(policies or run.policies())]
    by = eps.groupby(["policy", "season"])[["profit", "rain_mm"]].mean().reset_index()
    colors = style.policy_colors(by["policy"].unique())
    fig = go.Figure()
    for p, d in by.groupby("policy", sort=False):
        fig.add_trace(go.Scatter(x=d["season"], y=d["profit"], mode="lines+markers", name=style.label(p),
                                 line=dict(color=colors[p], width=2.5), customdata=d["rain_mm"],
                                 hovertemplate=style.label(p) + " %{x}: %{y:,.0f} INR/ha (rain %{customdata:.0f} mm)<extra></extra>"))
    return _layout(fig, title=f"Profit by season ({split})", xaxis_title="season (sowing year)", yaxis_title="INR per hectare",
                   xaxis=dict(dtick=1 if by["season"].nunique() < 15 else None))


def dry_wet(run, policies=None):
    """Mean profit in seasons drier / wetter than the median, all seasons pooled."""
    eps = run.episodes[run.episodes["policy"].isin(policies or run.policies())].copy()
    median = eps.groupby("season")["rain_mm"].first().median()
    eps["type"] = np.where(eps["rain_mm"] > median, "wetter than median", "drier than median")
    by = eps.groupby(["policy", "type"])["profit"].mean().unstack()
    order = by.mean(axis=1).sort_values(ascending=False).index
    fig = go.Figure()
    for kind, color in (("drier than median", style.DRY), ("wetter than median", style.WET)):
        if kind in by:
            fig.add_trace(go.Bar(x=[style.label(p) for p in order], y=by.loc[order, kind], name=kind, marker_color=color,
                                 hovertemplate="%{x}, " + kind + ": %{y:,.0f}<extra></extra>"))
    return _layout(fig, barmode="group", title=f"Profit in dry and wet seasons (median rain {median:.0f} mm)",
                   yaxis_title="INR per hectare")


def rain_response(run, policies=None):
    eps = run.episodes[run.episodes["policy"].isin(policies or run.policies())]
    by = eps.groupby(["policy", "season", "split"])[["irrigation_mm", "rain_mm"]].mean().reset_index()
    colors = style.policy_colors(by["policy"].unique())
    fig = go.Figure()
    for p, d in by.groupby("policy", sort=False):
        fig.add_trace(go.Scatter(x=d["rain_mm"], y=d["irrigation_mm"], mode="markers", name=style.label(p),
                                 marker=dict(color=colors[p], size=9, line=dict(color="white", width=1),
                                             symbol=np.where(d["split"] == "test", "diamond", "circle")),
                                 customdata=d[["season", "split"]],
                                 hovertemplate=style.label(p) + " %{customdata[0]} (%{customdata[1]}): %{y:.0f} mm irrigation, %{x:.0f} mm rain<extra></extra>"))
    return _layout(fig, title="Irrigation applied vs season rain (diamonds = held-out seasons)",
                   xaxis_title="season rain (mm)", yaxis_title="irrigation (mm)")


def sensitivity(run, policies=None):
    s = run.sensitivity
    s = s[s["policy"].isin(policies or run.policies())]
    factors = [f for f in s["factor"].unique() if f != "baseline"]
    base = s[s["factor"] == "baseline"].set_index("policy")["profit"]
    colors = style.policy_colors(s["policy"].unique())
    fig = make_subplots(rows=1, cols=len(factors), shared_yaxes=True,
                        subplot_titles=[f.replace("_", " ") + " ×" for f in factors])
    for j, f in enumerate(factors, start=1):
        d = s[s["factor"] == f]
        for p in base.index:
            pts = pd.concat([d[d["policy"] == p][["multiplier", "profit"]],
                             pd.DataFrame({"multiplier": [1.0], "profit": [base[p]]})]).sort_values("multiplier")
            fig.add_trace(go.Scatter(x=pts["multiplier"], y=pts["profit"], mode="lines+markers", name=style.label(p),
                                     legendgroup=p, showlegend=j == 1, line=dict(color=colors[p], width=2),
                                     hovertemplate=style.label(p) + " ×%{x}: %{y:,.0f}<extra></extra>"), row=1, col=j)
    fig.update_yaxes(title_text="profit (INR/ha)", row=1, col=1)
    return _layout(fig, height=440, title="Sensitivity: held-out profit when one price, cost or penalty is scaled",
                   margin=dict(l=50, r=20, t=110, b=45))


# ---------------------------------------------------------------------------------- data
def line(df, x, y, title, ytitle, color=None, group=None):
    fig = go.Figure()
    if group:
        for c, (name, d) in zip(style.PALETTE * 4, df.groupby(group)):
            fig.add_trace(go.Scatter(x=d[x], y=d[y], mode="lines+markers", name=str(name), line=dict(color=c, width=2)))
    else:
        fig.add_trace(go.Scatter(x=df[x], y=df[y], mode="lines", line=dict(color=color or style.PALETTE[0], width=2.5), showlegend=False))
    return _layout(fig, title=title, xaxis_title=x.replace("_", " "), yaxis_title=ytitle, height=380)


def bars(x, y, title, ytitle, colors=None, horizontal=False):
    trace = go.Bar(x=y if horizontal else x, y=x if horizontal else y, orientation="h" if horizontal else "v",
                   marker_color=colors or style.PALETTE[0], showlegend=False)
    return _layout(go.Figure(trace), title=title, **({"xaxis_title": ytitle} if horizontal else {"yaxis_title": ytitle}), height=380)


def scatter(df, x, y, title, xtitle, ytitle, group=None):
    fig = go.Figure()
    groups = df.groupby(group) if group else [("", df)]
    for c, (name, d) in zip(style.PALETTE * 4, groups):
        fig.add_trace(go.Scatter(x=d[x], y=d[y], mode="markers", name=str(name), showlegend=bool(group),
                                 marker=dict(color=c, size=8, line=dict(color="white", width=1))))
    return _layout(fig, title=title, xaxis_title=xtitle, yaxis_title=ytitle, height=380)
