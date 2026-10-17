"""Animated season: a stylised field per policy, driven by saved rollouts.

Each panel shows a field whose crop height follows crop growth, whose soil colour follows
soil moisture, with pest dots scaled by pest pressure and rain streaks in rainy
fortnights; below it, the inputs applied in the current fortnight and the profit so far.
``compare`` adds a second policy on the same season, side by side.
"""
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation
from matplotlib.colors import to_rgb

from .. import runs
from . import style

INPUTS = ["nitrogen_kg", "irrigation_mm", "pesticide_level"]
SOIL_DRY, SOIL_WET = np.array(to_rgb("#dcc7a1")), np.array(to_rgb("#5b4632"))
LEAF, RIPE = np.array(to_rgb("#2e9b4f")), np.array(to_rgb("#d9a520"))
SKY, RAIN, PEST = "#eef5fd", "#2a78d6", "#e34948"
STATE = ["crop_growth", "soil_moisture_end", "pest", "cum_profit"]


def episode_steps(run, policy, season, seed=0):
    r = run.rollouts
    d = r[(r["policy"] == policy) & (r["season"] == season)]
    if d.empty:
        raise ValueError(f"run '{run.name}' has no rollout for policy '{policy}' in season {season}")
    seed = seed if seed in set(d["seed"]) else d["seed"].min()
    return d[d["seed"] == seed].sort_values("step").reset_index(drop=True)


def best_season(run, policy="ppo", compare=None):
    """Held-out season where ``policy`` beats ``compare`` by most (or, alone, earns most)."""
    eps = run.episodes[run.episodes["split"] == "test"]
    by = eps.pivot_table(index="season", columns="policy", values="profit")
    score = by[policy] - by[compare] if compare in by else by[policy]
    return int(score.idxmax())


def _frames(steps, sub):
    """Interpolate the slowly varying state between fortnights for smooth motion."""
    start = {"crop_growth": 0.0, "soil_moisture_end": steps["soil_moisture"].iloc[0],
             "pest": steps["pest"].iloc[0], "cum_profit": 0.0}
    after = steps[STATE].copy()
    after["pest"] = steps["pest"].shift(-1).fillna(steps["pest"].iloc[-1])
    before = after.shift(1)
    before.iloc[0] = pd.Series(start)
    frames = []
    for i in range(len(steps)):
        for k in range(1, sub + 1):
            f = k / sub
            row = {c: before.iloc[i][c] + f * (after.iloc[i][c] - before.iloc[i][c]) for c in STATE}
            if i == len(steps) - 1:
                row["cum_profit"] = before.iloc[i]["cum_profit"] if k < sub else after.iloc[i]["cum_profit"]
            frames.append({"i": i, "phase": f, **row})
    return frames


def _draw_field(ax, step, frame, max_rain, rng_xy, title, color):
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0.26), 1, 0.74, color=SKY, zorder=0))
    moisture = float(np.clip(frame["soil_moisture_end"], 0, 1))
    ax.add_patch(plt.Rectangle((0, 0), 1, 0.26, color=SOIL_DRY + moisture * (SOIL_WET - SOIL_DRY), zorder=1))
    ax.text(0.015, 0.03, f"soil moisture {moisture:.0%}", fontsize=6.5, color="white" if moisture > 0.45 else style.INK_2, zorder=5)

    growth = float(np.clip(frame["crop_growth"], 0, 1))
    stage = (frame["i"] + frame["phase"]) / step["n_steps"]
    ripeness = np.clip((stage - 0.72) / 0.25, 0, 1)
    plant = LEAF + ripeness * (RIPE - LEAF)
    height = 0.03 + 0.5 * growth
    xs = np.linspace(0.05, 0.95, 19)
    for j, x in enumerate(xs):
        h = height * (0.92 + 0.08 * np.sin(j * 1.7))
        ax.plot([x, x], [0.26, 0.26 + h], color=plant, lw=2.2, solid_capstyle="round", zorder=3)
        for frac, side in ((0.45, -1), (0.7, 1)):
            if h > 0.08:
                ax.plot([x, x + side * 0.022], [0.26 + frac * h, 0.26 + frac * h + 0.03], color=plant, lw=1.4, zorder=3)
        if stage > 0.6 and h > 0.2:
            ax.plot([x, x], [0.26 + h, 0.26 + h + 0.045], color=RIPE, lw=3.2, solid_capstyle="round", zorder=4)

    n_pests = int(round(float(np.clip(frame["pest"], 0, 1)) * len(rng_xy)))
    if n_pests:
        px, py = rng_xy[:n_pests, 0], 0.27 + rng_xy[:n_pests, 1] * max(height, 0.04)
        ax.scatter(px, py, s=9, color=PEST, zorder=6, linewidths=0)
    if step["rain_mm"] > 1.0:
        n_drops = int(np.clip(60 * step["rain_mm"] / max(max_rain, 1.0), 6, 60))
        drops = np.random.default_rng(1000 + frame["i"]).random((n_drops, 2))
        dy = (drops[:, 1] - 0.35 * frame["phase"]) % 1.0
        y = 0.3 + 0.68 * dy
        ax.plot([drops[:, 0], drops[:, 0] - 0.008], [y, y - 0.035], color=RAIN, lw=0.9, alpha=0.75, zorder=2)
        ax.text(0.985, 0.94, f"rain {step['rain_mm']:.0f} mm", ha="right", fontsize=6.5, color=RAIN, zorder=7)
    ax.add_patch(plt.Rectangle((0, 0.985), 1, 0.015, color=color, zorder=8, clip_on=False))
    ax.set_title(title, fontsize=9, fontweight="bold", loc="left", pad=4)
    profit = frame["cum_profit"]
    ax.text(0.015, 0.94, f"profit so far  {profit:+,.0f} INR/ha", fontsize=7.5, fontweight="bold",
            color=style.INK if profit >= 0 else "#b3261e", zorder=7)
    ax.text(0.015, 0.865, f"pest pressure {frame['pest']:.0%}", fontsize=6.5, color=style.INK_2, zorder=7)


def _draw_actions(ax, step, maxima):
    ax.clear()
    ax.set_xlim(0, 1.32)
    ax.set_ylim(-0.6, len(INPUTS) - 0.4)
    ax.axis("off")
    for y, col in enumerate(reversed(INPUTS)):
        frac = step[col] / maxima[col] if maxima[col] else 0.0
        ax.barh(y, 1.0, height=0.55, color="#f0efec", zorder=1)
        ax.barh(y, frac, height=0.55, color=style.INPUT_COLORS[col], zorder=2)
        ax.text(-0.02, y, style.INPUT_LABELS[col].split(" (")[0], ha="right", va="center", fontsize=6.5, color=style.INK_2)
        unit = style.INPUT_LABELS[col].split("(")[1].rstrip(")").replace("kg N/ha", "kg/ha")
        ax.text(1.03, y, f"{step[col]:g} {unit}", va="center", fontsize=6.5, color=style.INK)


def animate_season(run=None, season=None, policy="ppo", compare=None, seed=0, sub=4, hold=6):
    """Build the animation. Returns (FuncAnimation, figure, season)."""
    run = runs.get(run)
    season = int(season) if season is not None else best_season(run, policy, compare)
    names = [policy] + ([compare] if compare else [])
    data = {p: episode_steps(run, p, season, seed) for p in names}
    colors = style.policy_colors(names)
    n_steps = len(data[policy])
    frames = {p: _frames(d, sub) for p, d in data.items()}
    maxima = {c: float(run.rollouts[c].max()) for c in INPUTS}
    max_rain = float(run.rollouts["rain_mm"].max())
    pest_xy = np.random.default_rng(7).random((36, 2)) * [0.9, 1.0] + [0.05, 0.0]

    style.apply()
    fig = plt.figure(figsize=(3.6 * len(names) + 0.4, 4.0), dpi=110)
    gs = fig.add_gridspec(2, len(names), height_ratios=[3.2, 1], hspace=0.12, wspace=0.28,
                          left=0.11 if len(names) == 1 else 0.09, right=0.93, top=0.84, bottom=0.04)
    field = [fig.add_subplot(gs[0, k]) for k in range(len(names))]
    bars = [fig.add_subplot(gs[1, k]) for k in range(len(names))]
    header = fig.text(0.5, 0.955, "", ha="center", fontsize=10, fontweight="bold")
    total = n_steps * sub + hold

    def update(k):
        k = min(k, n_steps * sub - 1)
        for ax_f, ax_b, p in zip(field, bars, names):
            fr = frames[p][k]
            step = {**data[p].iloc[fr["i"]].to_dict(), "n_steps": n_steps}
            _draw_field(ax_f, step, fr, max_rain, pest_xy, style.label(p), colors[p])
            _draw_actions(ax_b, step, maxima)
        i = frames[policy][k]["i"]
        date = pd.Timestamp(data[policy]["date"].iloc[i])
        header.set_text(f"Season {season}-{str(season + 1)[-2:]}  ·  fortnight {i + 1}/{n_steps}  ·  from {date:%d %b %Y}")
        if k == n_steps * sub - 1:
            finals = "   ".join(f"{style.label(p)}: {data[p]['final_yield'].iloc[0]:.1f} q/ha" for p in names)
            header.set_text(f"Season {season}-{str(season + 1)[-2:]} harvest  ·  {finals}")
        return []

    anim = animation.FuncAnimation(fig, update, frames=total, interval=160, blit=False)
    return anim, fig, season


def save_animation(run=None, season=None, policy="ppo", compare=None, out_dir=None, fps=6, demo_path=None, **kw):
    """Write the GIF (and MP4 when ffmpeg is installed) under <run>/animations/.
    ``demo_path`` receives a copy of the GIF. Returns the list of files written."""
    run = runs.get(run)
    anim, fig, season = animate_season(run, season, policy, compare, **kw)
    out = Path(out_dir) if out_dir else run.out_dir("animations")
    stem = f"season_{season}_{policy}" + (f"_vs_{compare}" if compare else "")
    paths = [out / f"{stem}.gif"]
    anim.save(paths[0], writer=animation.PillowWriter(fps=fps), dpi=110)
    if animation.writers.is_available("ffmpeg"):
        paths.append(out / f"{stem}.mp4")
        anim.save(paths[-1], writer=animation.FFMpegWriter(fps=fps, bitrate=1800), dpi=110)
    plt.close(fig)
    if demo_path:
        Path(demo_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(paths[0], demo_path)
        paths.append(Path(demo_path))
    return paths
