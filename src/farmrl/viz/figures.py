"""Report figures. One function per figure: ``f(run, **options) -> matplotlib Figure``.

``run`` is a run name, a run directory, a ``farmrl.runs.Run`` or ``None`` (newest run).
A function raises ``SkipFigure`` when the data it needs is not there; ``make_all``
turns that into a warning and carries on. Nothing here hardcodes policies, seasons or
numbers: everything is read from the run.
"""
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import data_loader as dl
from .. import runs
from ..yield_model import YieldModel
from . import style

INPUTS = ["nitrogen_kg", "irrigation_mm", "pesticide_level"]
AGENT = "ppo"


class SkipFigure(Exception):
    """The figure cannot be drawn from the data available (not an error)."""


# ----------------------------------------------------------------------------- helpers
def _test_episodes(run, aux=False):
    eps = run.episodes
    return eps[(eps["split"] == "test") & eps["policy"].isin(run.policies(include_aux=aux))]


def _order(eps):
    """Policies sorted by mean profit, best first."""
    return eps.groupby("policy")["profit"].mean().sort_values(ascending=False).index.tolist()


def _need_agent(run):
    if AGENT not in run.policies():
        raise SkipFigure(f"run has no '{AGENT}' policy")


def _baseline(run, baseline=None):
    """Comparison policy: the one asked for, else 'recommended', else the best baseline."""
    others = [p for p in run.policies() if p != AGENT]
    if baseline in others:
        return baseline
    if not others:
        raise SkipFigure("run has no baseline policy")
    return "recommended" if "recommended" in others else _order(_test_episodes(run)[lambda d: d["policy"] != AGENT])[0]


def _season_rain(run):
    eps = run.episodes
    return eps.groupby("season")["rain_mm"].first()


def _mean_steps(run, policy, season):
    """Step table for one policy and season, averaged over seeds."""
    r = run.rollouts
    d = r[(r["policy"] == policy) & (r["season"] == season)]
    if d.empty:
        raise SkipFigure(f"no rollout for {policy} in season {season}")
    cols = INPUTS + ["rain_mm", "soil_moisture", "pest", "cum_profit", "crop_growth"]
    out = d.groupby("step")[cols].mean().reset_index()
    out["date"] = d.groupby("step")["date"].first().values
    return out


def _date_ticks(ax, steps, every=2):
    ax.set_xticks(steps["step"][::every])
    ax.set_xticklabels([pd.Timestamp(d).strftime("%d %b") for d in steps["date"][::every]])


def _extreme_seasons(run):
    rain = _season_rain(run).loc[run.meta["test_seasons"]]
    return int(rain.idxmin()), int(rain.idxmax())


def _kinr(x, _=None):
    return f"{x / 1000:.0f}"


# -------------------------------------------------------------------------- a) learning
def learning_curve(run=None):
    """Mean episode return per PPO iteration, mean +/- std over training seeds."""
    run = runs.get(run)
    h = run.history
    if h is None or h.empty:
        raise SkipFigure("no train_history.csv (model was loaded, not trained)")
    g = h.groupby("timesteps")["mean_return"]
    x, mean, std = g.mean().index / 1000, g.mean() / 1000, g.std().fillna(0) / 1000
    n_seeds = h["train_seed"].nunique()
    colors = style.policy_colors(run.policies())
    fig, ax = plt.subplots(figsize=(style.COLUMN, 2.4))
    ax.fill_between(x, mean - std, mean + std, color=colors[AGENT], alpha=0.2, lw=0)
    ax.plot(x, mean, color=colors[AGENT], label=f"PPO, mean ± std of {n_seeds} seed{'s' * (n_seeds > 1)}")
    train = run.episodes[(run.episodes["split"] == "train") & run.episodes["policy"].isin(run.policies())]
    ref = train[train["policy"] != AGENT].groupby("policy")["reward"].mean().sort_values(ascending=False).head(2)
    for name, value in ref.items():
        ax.axhline(value / 1000, color=colors[name], lw=1, ls="--", label=style.label(name))
    ax.set(xlabel="environment steps (thousands)", ylabel="mean episode return\n(thousand INR/ha)",
           title="PPO learning curve (training seasons)")
    ax.legend(loc="lower right")
    return fig


# ------------------------------------------------------------------------ b) comparison
def policy_comparison(run=None):
    """Box plot of profit per policy on the test seasons, with yield and inputs alongside."""
    run = runs.get(run)
    eps = _test_episodes(run)
    order = _order(eps)
    colors = style.policy_colors(order)
    fig = plt.figure(figsize=(style.FULL, 2.9))
    gs = fig.add_gridspec(2, 4, width_ratios=[2.3, 0.15, 1, 1], hspace=0.75, wspace=0.55)
    ax = fig.add_subplot(gs[:, 0])
    data = [eps.loc[eps["policy"] == p, "profit"] / 1000 for p in order]
    box = ax.boxplot(data, patch_artist=True, widths=0.55, medianprops={"color": style.INK, "lw": 1},
                     flierprops={"marker": "o", "ms": 2, "mfc": style.MUTED, "mec": "none"},
                     whiskerprops={"color": style.INK_2, "lw": 0.7}, capprops={"color": style.INK_2, "lw": 0.7})
    for patch, p in zip(box["boxes"], order):
        patch.set(facecolor=colors[p], edgecolor=style.INK_2, linewidth=0.6)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels([style.label(p) for p in order], rotation=20, ha="right")
    ax.axhline(0, color=style.AXIS, lw=0.8)
    n_seasons = eps["season"].nunique()
    ax.set(ylabel="net profit (thousand INR/ha)", title=f"Profit on {n_seasons} held-out seasons")
    ax.grid(axis="x", visible=False)

    means = eps.groupby("policy")[["yield_q_ha"] + INPUTS].mean().loc[order]
    panels = [("yield_q_ha", "Yield (q/ha)")] + [(c, style.INPUT_LABELS[c]) for c in INPUTS]
    for (col, title), cell in zip(panels, [gs[0, 2], gs[0, 3], gs[1, 2], gs[1, 3]]):
        a = fig.add_subplot(cell)
        a.bar(range(len(order)), means[col], color=[colors[p] for p in order], width=0.7)
        a.set_xticks(range(len(order)))
        a.set_xticklabels([style.label(p) for p in order], rotation=45, ha="right", fontsize=5)
        a.set_title(title, fontsize=7, fontweight="normal")
        a.grid(axis="x", visible=False)
    return fig


# -------------------------------------------------------------------------- c) timeline
def season_timeline(run=None, season=None, policy=AGENT):
    """What one policy did through one season, above the rain and soil moisture it faced.
    Default season: the held-out season with median rainfall."""
    run = runs.get(run)
    if season is None:
        rain = _season_rain(run).loc[run.meta["test_seasons"]].sort_values()
        season = int(rain.index[len(rain) // 2])
    s = _mean_steps(run, policy, int(season))
    fig, axes = plt.subplots(5, 1, figsize=(style.COLUMN, 4.6), sharex=True,
                             gridspec_kw={"height_ratios": [1.1, 1.1, 0.8, 0.8, 0.8], "hspace": 0.35})
    axes[0].bar(s["step"], s["rain_mm"], color="#86b6ef", width=0.7)
    axes[0].set_ylabel("rain\n(mm)")
    axes[1].plot(s["step"], s["soil_moisture"] * 100, color=style.PALETTE[0], marker="o", ms=2.5)
    axes[1].set_ylabel("soil moisture\n(% capacity)")
    axes[1].set_ylim(0, 105)
    for ax, col in zip(axes[2:], INPUTS):
        ax.bar(s["step"], s[col], color=style.INPUT_COLORS[col], width=0.7)
        ax.set_ylabel(style.INPUT_LABELS[col].replace(" (", "\n("))
    _date_ticks(axes[-1], s)
    axes[-1].set_xlabel("fortnight starting")
    total = run.episodes[(run.episodes["policy"] == policy) & (run.episodes["season"] == season)]
    axes[0].set_title(f"{style.label(policy)}, season {season}: {s['rain_mm'].sum():.0f} mm rain, "
                      f"profit {total['profit'].mean() / 1000:.1f}k INR/ha")
    fig.align_ylabels(axes)
    return fig


# -------------------------------------------------------------------------- d) heatmaps
def policy_heatmaps(run=None):
    """Agent's expected irrigation over (soil moisture x stage) and nitrogen over
    (soil N x stage), other state variables at their per-stage medians."""
    run = runs.get(run)
    grids = run.grids
    if grids is None or grids.empty:
        raise SkipFigure("no policy_grids.parquet in the run")
    panels = [("soil_moisture", "expected_irrigation_mm", "Irrigation (mm)", 100, "soil moisture (% of capacity)"),
              ("soil_n", "expected_nitrogen_kg", "Nitrogen (kg N/ha)", 1, "soil mineral N (kg/ha)")]
    fig, axes = plt.subplots(1, 2, figsize=(style.FULL, 2.6))
    dates = run.rollouts.groupby("step")["date"].first()
    for ax, (grid, col, title, k, ylabel) in zip(axes, panels):
        g = grids[grids["grid"] == grid]
        if g.empty:
            raise SkipFigure(f"policy grid '{grid}' missing")
        z = g.pivot(index="value", columns="step", values=col)
        im = ax.imshow(z.values, origin="lower", aspect="auto", cmap=style.SEQUENTIAL,
                       extent=[-0.5, z.shape[1] - 0.5, z.index.min() * k, z.index.max() * k])
        ticks = list(z.columns[::2])
        ax.set_xticks(ticks)
        ax.set_xticklabels([pd.Timestamp(dates.loc[t]).strftime("%d %b") for t in ticks])
        ax.set(xlabel="fortnight starting", ylabel=ylabel, title=f"{title} chosen by the agent")
        ax.grid(False)
        fig.colorbar(im, ax=ax, pad=0.02).ax.tick_params(labelsize=6)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------------ e) dry vs wet
def dry_vs_wet(run=None, baseline=None):
    """Agent and one baseline, step by step, in the driest and wettest held-out season."""
    run = runs.get(run)
    _need_agent(run)
    base = _baseline(run, baseline)
    colors = style.policy_colors([AGENT, base])
    fig, axes = plt.subplots(4, 2, figsize=(style.FULL, 4.6), sharex="col", sharey="row",
                             gridspec_kw={"hspace": 0.3, "wspace": 0.08})
    for j, (season, tag) in enumerate(zip(_extreme_seasons(run), ["Driest", "Wettest"])):
        a, b = _mean_steps(run, AGENT, season), _mean_steps(run, base, season)
        axes[0, j].bar(a["step"], a["rain_mm"], color="#86b6ef", width=0.7)
        axes[0, j].set_title(f"{tag} held-out season: {season} ({a['rain_mm'].sum():.0f} mm rain)")
        for i, col in enumerate(INPUTS, start=1):
            axes[i, j].bar(a["step"] - 0.2, a[col], width=0.38, color=colors[AGENT], label=style.label(AGENT))
            axes[i, j].bar(b["step"] + 0.2, b[col], width=0.38, color=colors[base], label=style.label(base))
            axes[i, 0].set_ylabel(style.INPUT_LABELS[col].replace(" (", "\n("))
        _date_ticks(axes[3, j], a)
        axes[3, j].set_xlabel("fortnight starting")
    axes[0, 0].set_ylabel("rain\n(mm)")
    axes[1, 1].legend(loc="upper right")
    fig.align_ylabels(axes[:, 0])
    return fig


# ------------------------------------------------------------------- f) profit vs inputs
def profit_vs_resources(run=None):
    """Mean profit against mean water, nitrogen and pesticide use: one point per policy."""
    run = runs.get(run)
    means = _test_episodes(run).groupby("policy")[["profit"] + INPUTS].mean()
    colors = style.policy_colors(means.index)
    fig, axes = plt.subplots(1, 3, figsize=(style.FULL, 2.3), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, col in zip(axes, ["irrigation_mm", "nitrogen_kg", "pesticide_level"]):
        for p, row in means.iterrows():
            ax.scatter(row[col], row["profit"] / 1000, s=42, color=colors[p], edgecolor="white", lw=1, zorder=3)
            ax.annotate(style.label(p), (row[col], row["profit"] / 1000), xytext=(5, 4),
                        textcoords="offset points", fontsize=6, color=style.INK_2)
        ax.set_xlabel(style.INPUT_LABELS[col].lower() + " per season")
        ax.margins(x=0.22, y=0.15)
    axes[0].set_ylabel("net profit (thousand INR/ha)")
    fig.suptitle("Profit against resource use, held-out seasons", fontsize=8, fontweight="bold", y=1.02)
    return fig


# ----------------------------------------------------------------------- g) yield model
def yield_surface(run=None, fit_table=None):
    """Fitted yield response over nitrogen x season water, with the fitted records on top."""
    run = runs.get(run)
    path = Path(fit_table or dl.PROCESSED / "yield_fit_table.csv")
    if not path.exists() or not run.yield_params_path.exists():
        raise SkipFigure(f"{path.name} or the run's fitted_params.yaml is missing")
    data = pd.read_csv(path).dropna(subset=["n_kg_ha", "rain_mm", "yield_q_ha"])
    region = str(run.config["env"].get("region", "")).lower()
    if "state" in data and region in set(data["state"].str.lower()):
        data = data[data["state"].str.lower() == region]      # the surface is this region's plateau
    if "irrigated_share" in data:
        data = data[data["irrigated_share"].fillna(1.0) > 0.8]  # the surface is for an irrigated crop
    if data.empty:
        raise SkipFigure("no irrigated records for the run's region in the fit table")
    model = YieldModel.load(run.yield_params_path)
    model.y_scale = 1.0                     # compare with the records, not the simulator's potential
    irrigation = run.config.get("yield_fit", {}).get("assumed_irrigation_mm", 0.0)
    water = data["rain_mm"] + irrigation
    n = np.linspace(0, max(280, data["n_kg_ha"].max() * 1.05), 120)
    w = np.linspace(min(250, water.min() * 0.9), max(650, water.max() * 1.05), 120)
    N, W = np.meshgrid(n, w)
    Z = model.potential(N, W)
    lo, hi = min(Z.min(), data["yield_q_ha"].min()), max(Z.max(), data["yield_q_ha"].max())
    fig, ax = plt.subplots(figsize=(style.COLUMN, 2.8))
    cf = ax.contourf(N, W, Z, levels=np.linspace(lo, hi, 13), cmap=style.SEQUENTIAL)
    ax.scatter(data["n_kg_ha"], water, c=data["yield_q_ha"], cmap=style.SEQUENTIAL, vmin=lo, vmax=hi,
               s=14, edgecolor="white", lw=0.6)
    fit = run.meta.get("yield_fit") or {}
    pooled = "state_effects" in fit                  # pooled R2 is over all states, not these points
    r2 = f", R² = {fit['r2']:.2f}" if "r2" in fit and not pooled else ""
    states = "/".join(sorted(data["state"].unique())) if "state" in data else ""
    ax.set(xlabel="nitrogen applied (kg N/ha)", ylabel="season water: rain + irrigation (mm)",
           title=f"Fitted yield response{', ' + states if states else ''}\n({len(data)} district-years shown{r2})")
    ax.grid(False)
    fig.colorbar(cf, ax=ax, label="yield (q/ha)", pad=0.02, format="%.0f")
    return fig


def pesticide_response(run=None, table=None):
    """Pesticide use and yield in the national FAO series: the raw relation, and yield with
    the year trend and weather controls removed against pesticide use, with the calibrated
    loss curve. Skipped when the FAO table is not there."""
    run = runs.get(run)
    files = sorted(dl.PROCESSED.glob("fao_*.csv")) if table is None else [Path(table)]
    if not files or not files[0].exists():
        raise SkipFigure("no FAO pesticide table in data/processed (yield_df.csv not downloaded)")
    d = pd.read_csv(files[0])
    try:
        cal = YieldModel().calibrate_pesticide(d)
    except ValueError as why:
        raise SkipFigure(str(why))
    label = files[0].stem.replace("fao_", "").replace("_", ", ").title()
    fig, axes = plt.subplots(1, 2, figsize=(style.FULL, 2.6), gridspec_kw={"wspace": 0.38})
    ax = axes[0]
    sc = ax.scatter(cal["x"], cal["observed"], c=cal["year"], cmap=style.SEQUENTIAL, s=20,
                    edgecolor=style.INK_2, lw=0.4)
    fig.colorbar(sc, ax=ax, label="year", pad=0.02)
    ax.set(xlabel="pesticide use (series mean = 1)", ylabel="yield (q/ha)", title=f"{label}: raw series")
    ax = axes[1]
    grid = np.linspace(cal["x"].min() * 0.9, cal["x"].max() * 1.05, 100)
    level = cal["partial_yield"] / (1 - cal["estimate"] * np.exp(-cal["k_fixed"] * cal["x"]))
    ax.scatter(cal["x"], cal["partial_yield"], c=cal["year"], cmap=style.SEQUENTIAL, s=20,
               edgecolor=style.INK_2, lw=0.4)
    ax.plot(grid, level.mean() * (1 - cal["estimate"] * np.exp(-cal["k_fixed"] * grid)), color=style.PALETTE[1],
            label=f"calibrated loss L = {cal['estimate']:.2f} ± {cal['std_err']:.2f}")
    used = run.meta["yield_model"].get("pest_loss")
    ax.set(xlabel="pesticide use (series mean = 1)", ylabel="yield at the mean year
and weather (q/ha)",
           title=f"After controls ({', '.join(cal['controls'])}); n = {cal['n_obs']}")
    ax.legend(loc="upper left", title=None if used is None else f"this run used L = {used:.2f}", title_fontsize=6.5)
    return fig


# ----------------------------------------------------------------------- h) diagnostics
def training_diagnostics(run=None):
    """Policy entropy, value loss and approximate KL per iteration (mean +/- std over seeds)."""
    run = runs.get(run)
    h = run.history
    if h is None or h.empty:
        raise SkipFigure("no train_history.csv (model was loaded, not trained)")
    panels = [("entropy", "Policy entropy (nats)"), ("value_loss", "Value loss"), ("kl", "Approximate KL")]
    panels = [p for p in panels if p[0] in h.columns]
    fig, axes = plt.subplots(1, len(panels), figsize=(style.FULL, 2.0), gridspec_kw={"wspace": 0.35})
    for ax, (col, title) in zip(np.atleast_1d(axes), panels):
        g = h.groupby("timesteps")[col]
        x, m, s = g.mean().index / 1000, g.mean(), g.std().fillna(0)
        ax.fill_between(x, m - s, m + s, color=style.PALETTE[0], alpha=0.2, lw=0)
        ax.plot(x, m, color=style.PALETTE[0])
        ax.set(xlabel="environment steps (thousands)", title=title)
    return fig


# ---------------------------------------------------------------------- i) season gains
def season_gain(run=None, min_share=0.5):
    """PPO profit minus each baseline's, per held-out season, coloured by season type.
    Baselines earning less than ``min_share`` of PPO's mean profit are left out so the
    scale stays readable."""
    run = runs.get(run)
    _need_agent(run)
    eps = _test_episodes(run)
    by = eps.pivot_table(index="season", columns="policy", values="profit")
    keep = [p for p in by.columns if p != AGENT and by[p].mean() >= min_share * by[AGENT].mean()]
    if not keep:
        keep = [p for p in _order(eps) if p != AGENT][:1]
    if not keep:
        raise SkipFigure("run has no baseline policy")
    rain = _season_rain(run)
    median = rain.median()
    wet = rain.loc[by.index] > median
    fig, axes = plt.subplots(1, len(keep), figsize=(style.FULL, 2.3), sharey=True, squeeze=False,
                             gridspec_kw={"wspace": 0.08})
    for ax, base in zip(axes[0], keep):
        gain = (by[AGENT] - by[base]) / 1000
        ax.bar(range(len(gain)), gain, width=0.7, color=[style.WET if w else style.DRY for w in wet])
        ax.axhline(0, color=style.AXIS, lw=0.8)
        ax.set_xticks(range(len(gain)))
        ax.set_xticklabels(gain.index, rotation=45, ha="right")
        ax.set_title(f"vs {style.label(base)}\nmean {gain.mean():+.1f}k INR/ha", fontsize=7.5)
        ax.set_xlabel("held-out season (sowing year)")
        ax.grid(axis="x", visible=False)
    axes[0, 0].set_ylabel("PPO profit gain\n(thousand INR/ha)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (style.DRY, style.WET)]
    fig.legend(handles, [f"season drier than median ({median:.0f} mm)", "wetter than median"],
               loc="upper center", ncol=2, fontsize=6.5, bbox_to_anchor=(0.5, -0.08))
    return fig


FIGURES = {
    "learning_curve": learning_curve,
    "policy_comparison": policy_comparison,
    "season_timeline": season_timeline,
    "policy_heatmaps": policy_heatmaps,
    "dry_vs_wet": dry_vs_wet,
    "profit_vs_resources": profit_vs_resources,
    "yield_surface": yield_surface,
    "pesticide_response": pesticide_response,
    "training_diagnostics": training_diagnostics,
    "season_gain": season_gain,
}


def make_all(run=None, out_dir=None, formats=("png", "pdf"), only=None, **options):
    """Draw every figure and save it under <run>/figures/. Returns {name: path or reason}.
    ``options`` (e.g. ``season=2019``) are passed to the functions that accept them."""
    import inspect
    run = runs.get(run)
    out = Path(out_dir) if out_dir else run.out_dir("figures")
    out.mkdir(parents=True, exist_ok=True)
    style.apply()
    report = {}
    for name, fn in FIGURES.items():
        if only and name not in only:
            continue
        kw = {k: v for k, v in options.items() if k in inspect.signature(fn).parameters and v is not None}
        try:
            fig = fn(run, **kw)
        except SkipFigure as why:
            warnings.warn(f"figure '{name}' skipped: {why}")
            report[name] = f"skipped: {why}"
            continue
        for ext in formats:
            fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)
        report[name] = str(out / f"{name}.{formats[0]}")
    return report
