"""Comparative-study summaries: one table, one figure and one card per study.

A study is a folder ``experiments/<study>/`` whose ``runs/`` holds run folders made by
``scripts/run_experiments.py``. For every run this reads the held-out profit of each PPO
(or other agent) training seed, the run's own ``threshold_tuned`` and ``oracle`` rows, and
the Shapley split of the agent's gain over ``threshold_tuned``. Gaps are agent minus
reference, so a negative gap means the agent is below it.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import runs as runs_mod

REFERENCE = "threshold_tuned"
DECISIONS = ["nitrogen", "water", "pests", "sale"]


def _test(df):
    return df[df["split"] == "test"]


def run_row(path, label=None, oracle_from=None):
    """One results row for a run folder (see module docstring)."""
    run = runs_mod.Run(path)
    meta, summary = run.meta, _test(run.summary).set_index("policy")
    seeds = _test(run.seed_summary).sort_values("train_seed")
    profit, reward = seeds["profit"].to_numpy(float), seeds["reward"].to_numpy(float)
    ref = summary.loc[REFERENCE] if REFERENCE in summary.index else None
    oracle = summary.loc["oracle"] if "oracle" in summary.index else None
    oracle_src = "own"
    if oracle is None and oracle_from is not None:
        other = runs_mod.Run(oracle_from)
        table = _test(other.summary).set_index("policy")
        oracle, oracle_src = (table.loc["oracle"] if "oracle" in table.index else None), other.name
    use = {k: float(seeds[k].mean()) for k in ("nitrogen_kg", "irrigation_mm", "pesticide_level") if k in seeds}
    row = {"run": run.name, "config": label or run.name, "folder": str(Path(path).resolve()), "algo": meta.get("algo", "ppo"),
           "seeds": len(profit), "profit_mean": profit.mean(), "profit_std": profit.std(ddof=1) if len(profit) > 1 else np.nan,
           **{f"agent_{k}": v for k, v in use.items()}, "reward_mean": reward.mean(), "profit_min": profit.min(), "profit_max": profit.max(),
           "tuned_threshold_profit": None if ref is None else float(ref["profit"]),
           "tuned_threshold_reward": None if ref is None else float(ref["reward"]),
           "oracle_profit": None if oracle is None else float(oracle["profit"]),
           "oracle_reward": None if oracle is None else float(oracle["reward"]), "oracle_source": oracle_src}
    forecast = (run.config["env"].get("forecast") or {})
    row.update({"pest_loss": (meta.get("yield_model") or {}).get("pest_loss"),
                "forecast_noise_mm": forecast.get("noise_mm") if forecast.get("enabled") else None,
                "tuned_threshold_sprays": None if ref is None else float(ref["pesticide_level"]),
                "oracle_sprays": None if oracle is None else float(oracle["pesticide_level"])})
    for key in ("recommended_tuned", "recommended", "threshold", "random"):
        row[f"{key}_profit"] = float(summary.loc[key, "profit"]) if key in summary.index else None
    if ref is not None:
        gap = profit - ref["profit"]
        row.update(gap_threshold_mean=gap.mean(), gap_threshold_std=gap.std(ddof=1) if len(gap) > 1 else np.nan,
                   seeds_ahead_of_threshold=int((gap > 0).sum()))
    if oracle is not None:
        gap = profit - oracle["profit"]
        row.update(gap_oracle_mean=gap.mean(), gap_oracle_std=gap.std(ddof=1) if len(gap) > 1 else np.nan,
                   share_of_oracle=float(profit.mean() / oracle["profit"]))
        if ref is not None:
            row["threshold_share_of_oracle"] = float(ref["profit"] / oracle["profit"])
    att = run.attribution
    if att is not None and REFERENCE in set(att["baseline_policy"]):
        a = att[att["baseline_policy"] == REFERENCE].groupby("decision", sort=False)["contribution"].mean()
        row.update({f"shapley_{d}": float(a.get(d, np.nan)) for d in DECISIONS})
    return row


def study_table(entries, oracle_from=None):
    """DataFrame with one row per ``(run folder, label)`` entry that exists, in order."""
    rows = [run_row(Path(p), label, oracle_from) for p, label in entries if (Path(p) / "metadata.json").exists()]
    return pd.DataFrame(rows)


def _k(x, signed=False):
    return "–" if x is None or pd.isna(x) else (f"{x:+,.0f}" if signed else f"{x:,.0f}")


def _pm(m, s, signed=False):
    if m is None or pd.isna(m):
        return "–"
    return f"{_k(m, signed)} ± {_k(s)}" if not pd.isna(s) else _k(m, signed)


def markdown_table(df):
    """The results table as Markdown (INR/ha, held-out seasons, mean ± std over training seeds)."""
    head = ["Configuration", "Seeds", "Profit", "Gap to threshold_tuned", "Gap to oracle", "% of oracle",
            "Shapley N / W / P / S", "N kg · water mm · sprays"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in df.itertuples(index=False):
        d = r._asdict()
        shap = " / ".join(_k(d.get(f"shapley_{k}"), True) for k in DECISIONS)
        share = d.get("share_of_oracle")
        lines.append("| " + " | ".join([
            str(d["config"]), str(d["seeds"]), _pm(d["profit_mean"], d["profit_std"]),
            _pm(d.get("gap_threshold_mean"), d.get("gap_threshold_std"), True),
            _pm(d.get("gap_oracle_mean"), d.get("gap_oracle_std"), True),
            "–" if share is None or pd.isna(share) else f"{100 * share:.1f}%", shap,
            " · ".join(_k(d.get(f"agent_{k}")) if k != "pesticide_level" else
                       ("–" if d.get(f"agent_{k}") is None else f"{d[f'agent_{k}']:.2f}")
                       for k in ("nitrogen_kg", "irrigation_mm", "pesticide_level"))]) + " |")
    return "\n".join(lines)


def reference_table(df):
    """Profit of the reference policies the gaps are measured against, per run."""
    cols = [("tuned_threshold_profit", "threshold_tuned"), ("recommended_tuned_profit", "recommended_tuned"),
            ("recommended_profit", "recommended"), ("random_profit", "random"), ("oracle_profit", "oracle")]
    lines = ["| Run | " + " | ".join(n for _, n in cols) + " | threshold_tuned % of oracle |", "|" + "---|" * (len(cols) + 2)]
    for r in df.itertuples(index=False):
        d = r._asdict()
        share = d.get("threshold_share_of_oracle")
        lines.append("| " + " | ".join([str(d["config"])] + [_k(d.get(c)) for c, _ in cols]
                                       + ["–" if share is None or pd.isna(share) else f"{100 * share:.1f}%"]) + " |")
    return "\n".join(lines)


def findings(df):
    """Plain statements read straight off the table, one block per environment (rows that
    share an oracle value share an environment; the oracle and the rule differ between them)."""
    out = []
    if df.empty:
        return ["No runs found."]
    groups = list(df.groupby("oracle_profit", sort=False)) if df["oracle_profit"].notna().any() else [(None, df)]
    for oracle, g in groups:
        best = g.loc[g["profit_mean"].idxmax()]
        thr = g["tuned_threshold_profit"].iloc[0]
        tag = f"Environment with oracle {oracle:,.0f} INR/ha: " if oracle is not None else ""
        line = f"{tag}best agent **{best['config']}** at {best['profit_mean']:,.0f} ± {best['profit_std']:,.0f} INR/ha"
        if oracle is not None:
            line += f" ({100 * best['profit_mean'] / oracle:.1f}% of the oracle); `threshold_tuned` {thr:,.0f} ({100 * thr / oracle:.1f}%)"
        out.append(line + ".")
        ahead = g[g["gap_threshold_mean"] > 0]
        out.append("Ahead of its own `threshold_tuned` on average: "
                   + (", ".join(f"{c} ({n} of {m} seeds)" for c, n, m in zip(ahead["config"], ahead["seeds_ahead_of_threshold"], ahead["seeds"]))
                      if len(ahead) else "none") + ".")
    return out


def write_card(study_dir, title, question, setup, df, extra=()):
    """experiments/<study>/README.md and results.csv / results.md."""
    d = Path(study_dir)
    df.to_csv(d / "results.csv", index=False)
    table = markdown_table(df)
    (d / "results.md").write_text(table + "\n", encoding="utf-8")
    text = [f"# {title}", "", f"**Question.** {question}", "", "## Setup", "", *[f"- {s}" for s in setup], "",
            "Every configuration is trained with 5 seeds and scored on the same held-out seasons (2017–2023, "
            "30 pest-growth replicates each). Profit is INR/ha before environmental penalties; ± is the standard "
            "deviation over training seeds. A gap is agent minus reference (negative = below). The Shapley columns "
            "split the agent's gain over `threshold_tuned` between nitrogen, water, pesticide and sale.", "",
            "## Results", "", table, "", "### Reference policies", "", reference_table(df), "",
            "## Readings off the table", "", *[f"- {s}" for s in findings(df)], ""]
    text += list(extra)
    here = d.resolve()
    links = []
    for r, folder in zip(df["run"], df["folder"]):
        rel = Path(folder).relative_to(here) if str(folder).startswith(str(here)) else Path("..") / Path(folder).relative_to(here.parent)
        links.append(f"- [`{r}`]({rel.as_posix()}/README.md)")
    text += ["## Runs", "", *links, ""]
    (d / "README.md").write_text("\n".join(text), encoding="utf-8")
    return d / "README.md"


# ---------------------------------------------------------------------------- the figure
def summary_figure(df, path, title="Held-out profit by configuration"):
    """One dot-and-whisker figure for all studies: the agent's mean ± std over seeds (dot),
    its own ``threshold_tuned`` (diamond) and the oracle (open square) on the same row."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .utils import COLORS, set_plot_style

    set_plot_style()
    studies = list(dict.fromkeys(df["study"]))
    fig, ax = plt.subplots(figsize=(8.2, 0.34 * len(df) + 2.6))
    y = np.arange(len(df))[::-1]
    for yi, r in zip(y, df.itertuples(index=False)):
        d = r._asdict()
        color = COLORS[studies.index(d["study"]) % len(COLORS)]
        if d.get("tuned_threshold_profit") is not None and not pd.isna(d["tuned_threshold_profit"]):
            ax.plot(d["tuned_threshold_profit"], yi, marker="D", ms=5, color="#52514e", ls="none", zorder=2)
        if d.get("oracle_profit") is not None and not pd.isna(d["oracle_profit"]):
            ax.plot(d["oracle_profit"], yi, marker="s", ms=6, mfc="none", mec="#52514e", ls="none", zorder=2)
        std = 0 if pd.isna(d["profit_std"]) else d["profit_std"]
        ax.errorbar(d["profit_mean"], yi, xerr=std, fmt="o", ms=6, color=color, ecolor=color, elinewidth=1.6,
                    capsize=3, zorder=3, mec="white", mew=0.8)
        ax.annotate(f"{d['profit_mean']:,.0f}", (d["profit_mean"], yi), xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=7, color="#52514e")
    ax.set_yticks(y, [f"{c}" for c in df["config"]])
    ax.set_xlabel("held-out profit (INR/ha), 2017–2023")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    handles = [plt.Line2D([], [], marker="o", color=COLORS[i % len(COLORS)], ls="none", label=s) for i, s in enumerate(studies)]
    handles += [plt.Line2D([], [], marker="D", color="#52514e", ls="none", label="threshold_tuned (same environment)"),
                plt.Line2D([], [], marker="s", mfc="none", mec="#52514e", ls="none", label="hindsight oracle")]
    fig.legend(handles=handles, loc="lower center", fontsize=7, ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def sweep_figure(df, xcol, path, xlabel, title, baseline=None, baseline_label="no forecast"):
    """Held-out profit of the agent, the tuned rule and the oracle against ``xcol`` (top), the
    agent's gap to the rule with its seed std (middle) and sprays a season (bottom, when present).
    ``baseline`` (a one-row frame, e.g. the run without a forecast) is drawn as a dashed line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .utils import COLORS, set_plot_style

    set_plot_style()
    d = df.sort_values(xcol)
    x = d[xcol].to_numpy(float)
    panels = 3 if d["tuned_threshold_sprays"].notna().any() and xcol == "pest_loss" else 2
    fig, axes = plt.subplots(panels, 1, figsize=(6.6, 2.7 * panels), sharex=True)
    ax = axes[0]
    ax.fill_between(x, d["profit_mean"] - d["profit_std"], d["profit_mean"] + d["profit_std"], color=COLORS[0], alpha=0.18, lw=0)
    ax.plot(x, d["profit_mean"], "o-", color=COLORS[0], label="PPO (mean ± std over seeds)")
    ax.plot(x, d["tuned_threshold_profit"], "D-", color=COLORS[2], label="threshold_tuned")
    ax.plot(x, d["oracle_profit"], "s--", color="#52514e", mfc="none", label="hindsight oracle")
    if baseline is not None and len(baseline):
        ax.axhline(float(baseline["profit_mean"].iloc[0]), color=COLORS[0], ls=":", lw=1.4, label=baseline_label)
    ax.set(ylabel="held-out profit (INR/ha)", title=title)
    ax.legend(fontsize=7, loc="best")
    ax = axes[1]
    gap = d["profit_mean"] - d["tuned_threshold_profit"]
    ax.axhline(0, color="#52514e", lw=1)
    ax.errorbar(x, gap, yerr=d["profit_std"].fillna(0), fmt="o-", color=COLORS[0], capsize=3)
    for xi, gi in zip(x, gap):
        ax.annotate(f"{gi:+,.0f}", (xi, gi), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7, color="#52514e")
    ax.set(ylabel="PPO − threshold_tuned\n(INR/ha)")
    if panels == 3:
        ax = axes[2]
        ax.plot(x, d["agent_pesticide_level"], "o-", color=COLORS[0], label="PPO")
        ax.plot(x, d["tuned_threshold_sprays"], "D-", color=COLORS[2], label="threshold_tuned")
        ax.plot(x, d["oracle_sprays"], "s--", color="#52514e", mfc="none", label="oracle")
        ax.set(ylabel="spray doses per season")
        ax.legend(fontsize=7)
    axes[-1].set_xlabel(xlabel)
    axes[-1].set_xticks(x)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path
