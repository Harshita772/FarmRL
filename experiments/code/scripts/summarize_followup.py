"""Run cards, results tables, study cards and figures for the follow-up studies 06, 07 and 09
(08 and 10 write their own cards from scripts/transfer_eval.py and scripts/oracle_breakdown.py).

    python scripts/summarize_followup.py --experiments ../ --all
"""
import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
import summarize_study as base
from farmrl import runcard, study

PEST = [("0p115", "pest_loss = 0.115 (calibrated)"), ("0p2", "pest_loss = 0.2"), ("0p3", "pest_loss = 0.3"),
        ("0p35", "pest_loss = 0.35"), ("0p5", "pest_loss = 0.5")]
BEST = ("02_improved_ppo", "BEST", "improved PPO, no forecast (study 2 best)")

STUDIES2 = {
    "06_pest_loss_sweep": dict(
        title="Study 6 — Pest-loss sweep: when does PPO beat the rule?",
        question="As pests become more costly, where does adaptive control start to pay? Pest loss is swept from the "
                 "calibrated 0.115 to 0.5.",
        setup=["At every value the whole pipeline is rerun: `y_scale` recalibrated, the threshold rule retuned on the "
               "training seasons, PPO retrained (study-2 best: improved bundle, ent_coef 0.003, epochs 10, 1M steps, "
               "5 seeds) and the hindsight oracle recomputed.",
               "The rule's pest cut-offs are searched on a fine grid (every pair of cut-offs on a 0.02 step) at every "
               "value, so the rule is not handicapped by a coarse spray grid (an earlier concern about study 5).",
               "The last table column gives sprays per season."],
        runs=[(f"pest_{t}", l) for t, l in PEST], include=[],
        figure=("pest_loss", "pest_sweep.png", "pest_loss", "Held-out profit against pest loss", None)),
    "07_forecast_information": dict(
        title="Study 7 — Does a rain forecast help PPO close the oracle gap?",
        question="If the observation includes a noisy forecast of the next fortnight's rain, does PPO close the gap to the "
                 "oracle as the forecast improves?",
        setup=["The observation gains one entry: the rain of the fortnight about to be played plus Gaussian noise "
               "(sd 0, 10 and 25 mm), floored at 0, drawn once per season.",
               "Improved PPO as in study 2 (ent_coef 0.003, epochs 10, 1M steps, 5 seeds). The rules do not use the "
               "forecast, so the tuned rules of study 1 are reused; the oracle is recomputed.",
               "Comparison row: the same agent without a forecast (study 2 best)."],
        runs=[("forecast_sd0mm", "forecast noise 0 mm (perfect)"), ("forecast_sd10mm", "forecast noise 10 mm"),
              ("forecast_sd25mm", "forecast noise 25 mm")], include=[BEST],
        figure=("forecast_noise_mm", "forecast_sweep.png", "forecast noise, sd of the rain forecast error (mm)",
                "Held-out profit against forecast quality", "no forecast")),
    "09_combined_action_settings": dict(
        title="Study 9 — Combining the best action settings",
        question="Does the improved PPO with a finer water grid and the joint nitrogen x water head, together, beat the rule?",
        setup=["All runs use the improved bundle (ent_coef 0.003, epochs 10, 1M steps, 5 seeds).",
               "`fine4_joint`: water grid {0, 25, 50, 75} mm and the joint head. `fine6_joint`: grid {0, 15, 30, 45, 60, 75} "
               "mm and the joint head. `fine6_factored`: the same 6-level grid with separate heads, to isolate the head.",
               "On each grid the rule is retuned and maps its doses onto the grid (its 37.5 mm becomes the nearest level, ties "
               "upward), so it is handicapped relative to the standard grid; the last table below therefore also gives "
               "the gap to the **standard-grid** `threshold_tuned` of study 1."],
        runs=[("fine4_joint", "improved PPO, grid 0/25/50/75, joint head"),
              ("fine6_joint", "improved PPO, grid 0/15/30/45/60/75, joint head"),
              ("fine6_factored", "improved PPO, grid 0/15/30/45/60/75, factored heads")],
        include=[base.S1, BEST, ("03_action_structure", "fine_water", "improved PPO, grid 0/25/50/75, factored (study 3)"),
                 ("03_action_structure", "joint_nw", "improved PPO, standard grid, joint head (study 3)")],
        figure=None, standard_rule=True),
}
PPO_REF = ("02_improved_ppo", "BEST", "improved PPO with critic (study 2 best)")
STUDIES2["11_grpo_vs_ppo"] = dict(
    title="Study 11 — GRPO vs PPO: does the critic matter?",
    question="Is PPO's learned value function needed, or does a critic-free agent that takes its advantage from a group "
             "of episodes in the same season do as well?",
    setup=["The GRPO agent (`agents/grpo.py`, written from scratch) uses the same policy network and action heads as the "
           "improved PPO (observation normalisation, orthogonal init, storage-head mask, factored heads) with the value head "
           "removed. Each update draws B training seasons and, for each, G episodes with the same weather and prices and "
           "different pest-growth seeds, collected with the current policy.",
           "Advantage, `episode` mode: (episode return − group mean) / (group std + 0.01 in reward units, about 100 INR/ha), "
           "applied to every step of the episode. `stepwise` mode: the return-to-go from each step, normalised within the group "
           "at that step. Loss: PPO's clipped ratio plus an entropy bonus (ent_coef 0.003, 10 epochs, batch 64); a KL penalty "
           "to the collecting policy exists (`kl_coef`) and is off.",
           "Budget: 1M environment steps per seed as for PPO. B x G is about 168 episodes (about 2,000 steps) per update, as "
           "PPO's batch: G = 8 uses B = 21 seasons, G = 16 uses B = 10.",
           "Rule tuning and oracle are those of study 1 (same environment). Reference: the best improved PPO of study 2 "
           "(ent_coef 0.003, epochs 10), not retrained.",
           "The learning curves show the mean reward of the training episodes (profit minus penalties) against environment "
           "steps; PPO's curve is read from its saved training history, which logs reward and not profit."],
    runs=[("grpo_g8", "GRPO, group 8, episode advantage"), ("grpo_g16", "GRPO, group 16, episode advantage"),
          ("grpo_g8_stepwise", "GRPO, group 8, per-step advantage")],
    include=[PPO_REF], figure=None, extras="grpo")
STUDIES2["12_layout_plus_forecast"] = dict(
    title="Study 12 — The best action layout combined with a rain forecast",
    question="The 4-level water grid with the joint head (study 9) and a noisy next-fortnight rain forecast (study 7) each helped "
             "PPO on their own. Do the two add up?",
    setup=["Improved PPO (ent_coef 0.003, epochs 10, 1M steps, 5 seeds) with the water grid {0, 25, 50, 75} mm, the joint "
           "nitrogen x water head and the rain forecast in the observation at noise sd 0, 10 and 25 mm.",
           "The rule is retuned on the 4-level grid (reusing the tuning of study 9's `fine4_joint`, which is the same grid with the "
           "same dynamics) and the oracle is recomputed in each forecast environment.",
           "Comparison rows: layout only without a forecast (study 9), the study-2 best agent, and the forecast-only runs of study 7 "
           "(standard grid, separate heads). The table below the results lines these up by forecast noise."],
    runs=[("layout_fc0mm", "layout + forecast noise 0 mm (perfect)"), ("layout_fc10mm", "layout + forecast noise 10 mm"),
          ("layout_fc25mm", "layout + forecast noise 25 mm")],
    include=[("09_combined_action_settings", "fine4_joint", "layout only, no forecast (study 9)"), BEST,
             ("07_forecast_information", "forecast_sd0mm", "forecast only, noise 0 mm (study 7)"),
             ("07_forecast_information", "forecast_sd10mm", "forecast only, noise 10 mm (study 7)"),
             ("07_forecast_information", "forecast_sd25mm", "forecast only, noise 25 mm (study 7)")],
    figure=("forecast_noise_mm", "layout_forecast_sweep.png", "forecast noise, sd of the rain forecast error (mm)",
            "Held-out profit: 4-level grid + joint head + forecast", "layout only, no forecast"),
    extras="layout_forecast")
STUDIES2["08_transfer_and_shift"] = dict(cards_only=True)      # its card comes from scripts/transfer_eval.py
ANALYSIS2 = {}          # written by hand once the results are in


def learning_curve_figure(df, path):
    """Mean training-episode reward against environment steps, mean and std over seeds, for every row."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    import numpy as np
    from farmrl.utils import COLORS, set_plot_style
    set_plot_style()
    grid = np.arange(20_000, 1_000_001, 10_000)
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for i, r in enumerate(df.itertuples()):
        h = pd.read_csv(Path(r.folder) / "train_history.csv")
        curves = []
        for _, g in h.groupby("train_seed"):
            g = g.sort_values("timesteps")
            y = np.interp(grid, g["timesteps"], g["mean_return"].rolling(10, min_periods=1).mean())
            y[grid > g["timesteps"].max()] = np.nan            # no curve beyond what the run trained
            curves.append(y)
        curves = np.array(curves)
        m, sd = np.nanmean(curves, 0), np.nanstd(curves, 0)
        ax.fill_between(grid, m - sd, m + sd, color=COLORS[i % len(COLORS)], alpha=0.15, lw=0)
        ax.plot(grid, m, color=COLORS[i % len(COLORS)], label=r.config, ls="-" if i else "--")
    ax.set(xlabel="environment steps", ylabel="mean training-episode reward (INR/ha)\nprofit minus penalties, 10-update smoothing",
           title="Learning curves, mean ± std over 5 seeds")
    ax.set_ylim(bottom=max(ax.get_ylim()[0], 60_000))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e6:g}M" if v else "0"))
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def grpo_extras(d, df):
    """Markdown lines: learning-curve figure, seed variance and wall-clock tables of study 11."""
    learning_curve_figure(df, d / "learning_curves.png")
    out = ["## Learning curves", "", "![Learning curves](learning_curves.png)", "",
           "## Seed variance", "", "| Configuration | Mean | Std over seeds | Min | Max | Range |", "|---|---|---|---|---|---|"]
    out += [f"| {r.config} | {r.profit_mean:,.0f} | {r.profit_std:,.0f} | {r.profit_min:,.0f} | {r.profit_max:,.0f} | "
            f"{r.profit_max - r.profit_min:,.0f} |" for r in df.itertuples()]
    out += ["", "## Wall-clock time", ""]
    probe = d / "timing" / "time_probe.csv"
    rows = {}
    if probe.exists():
        for r in pd.read_csv(probe).itertuples():
            rows[r.agent] = r.minutes_per_1M
        out += ["Training time per seed for 1M steps, from a probe that trained each agent for 100k steps in 4 processes at once "
                "(1 torch thread each) and scaled up; evaluation is not included.", "",
                "| Agent | Minutes per 1M steps (probe) | Measured in the run (mean over seeds) |", "|---|---|---|"]
        names = {"improved PPO with critic (study 2 best)": "improved PPO", "GRPO, group 8, episode advantage": "GRPO, group 8",
                 "GRPO, group 16, episode advantage": "GRPO, group 16", "GRPO, group 8, per-step advantage": "GRPO, group 8, stepwise"}
        for r in df.itertuples():
            h = pd.read_csv(Path(r.folder) / "train_history.csv")
            measured = f"{h.groupby('train_seed')['wall_s'].max().mean() / 60:.1f} min" if "wall_s" in h else "not logged"
            out.append(f"| {r.config} | {rows.get(names.get(r.config, ''), float('nan')):.1f} | {measured} |")
    else:
        out.append("(timing probe not run)")
    out.append("")
    return out


def layout_forecast_extras(df):
    """Markdown: layout and forecast effects lined up by forecast noise (study 12)."""
    by = df.set_index("run")
    best = [r for r in by.index if r.startswith("improved_ent")][0]
    cells = [("none", best, "fine4_joint"), ("0 mm", "forecast_sd0mm", "layout_fc0mm"),
             ("10 mm", "forecast_sd10mm", "layout_fc10mm"), ("25 mm", "forecast_sd25mm", "layout_fc25mm")]
    out = ["## Layout and forecast, side by side", "",
           "Held-out profit (INR/ha), mean ± std over 5 seeds. The left column is the standard water grid with separate heads, "
           "the middle column the 4-level grid with the joint head. Each row's own tuned rule and oracle are in the table above.", "",
           "| Forecast noise | Standard grid, separate heads | 4-level grid + joint head | Layout effect | Forecast effect on the layout agent |",
           "|---|---|---|---|---|"]
    base = by.loc["fine4_joint", "profit_mean"]
    for name, left, right in cells:
        if left not in by.index or right not in by.index:
            continue
        a, b = by.loc[left], by.loc[right]
        out.append(f"| {name} | {a['profit_mean']:,.0f} ± {a['profit_std']:,.0f} | {b['profit_mean']:,.0f} ± {b['profit_std']:,.0f} | "
                   f"{b['profit_mean'] - a['profit_mean']:+,.0f} | {b['profit_mean'] - base:+,.0f} |")
    return out + [""]


def write_global_index(exp):
    """experiments/INDEX.md: one row per run of every study."""
    from farmrl import runs as runs_mod
    oracle_from = exp / base.S1[0] / "runs" / base.S1[1]
    rows = ["| Study | Run | Algorithm | Overrides | Profit | Gap to threshold_tuned | Gap to oracle |", "|" + "---|" * 7]
    for meta in sorted(exp.glob("[0-9][0-9]_*/runs/*/metadata.json"), key=lambda m: (m.parents[2].name, m.parent.name)):
        run_dir = meta.parent
        row = study.run_row(run_dir, None, oracle_from if oracle_from.exists() else None)
        m = runs_mod.Run(run_dir).meta
        rows.append(f"| {run_dir.parents[1].name} | [`{run_dir.name}`]({run_dir.parents[1].name}/runs/{run_dir.name}/README.md) | "
                    f"{m.get('algo', 'ppo')} | {', '.join(f'`{o}`' for o in m.get('overrides', [])) or '–'} | "
                    f"{study._pm(row['profit_mean'], row['profit_std'])} | "
                    f"{study._pm(row.get('gap_threshold_mean'), row.get('gap_threshold_std'), True)} | "
                    f"{study._pm(row.get('gap_oracle_mean'), row.get('gap_oracle_std'), True)} |")
    (exp / "INDEX.md").write_text("# Experiment runs\n\nOne row per run in `experiments/`; held-out profit in INR/ha, mean ± std over "
                                  "the training seeds. Gaps are agent minus reference (negative = below); the oracle of study 1 is used "
                                  "where a run has none of its own. Each run folder has its own run card.\n\n"
                                  + "\n".join(rows) + "\n", encoding="utf-8")
    print(f"index: {len(rows) - 2} runs -> {exp / 'INDEX.md'}")


def summarize(exp, name):
    spec = STUDIES2[name]
    d = exp / name
    if spec.get("cards_only"):
        for p in sorted((d / "runs").glob("*/metadata.json")):
            runcard.write_card(p.parent)
        if list((d / "runs").glob("*/metadata.json")):
            runcard.write_index(d / "runs")
        return None
    (d / "runs").mkdir(parents=True, exist_ok=True)
    for p in sorted((d / "runs").glob("*/metadata.json")):
        try:
            runcard.write_card(p.parent)
        except Exception:
            print(f"run card failed for {p.parent.name}:\n{traceback.format_exc()}")
    if list((d / "runs").glob("*/metadata.json")):
        runcard.write_index(d / "runs")
    entries = base.entries_for(exp, name, spec)
    oracle_from = exp / base.S1[0] / "runs" / base.S1[1]
    df = study.study_table([(p, l) for p, l, _ in entries], oracle_from if oracle_from.exists() else None)
    if df.empty:
        print(f"{name}: no finished runs yet")
        return None
    owned = {str(p.resolve()) for p, _, own in entries if own}
    df["included"] = ~df["folder"].isin(owned)
    df.insert(0, "study", name)
    extra = []
    if spec.get("figure") and (~df["included"]).sum() >= 2:
        xcol, fname, xlabel, title, base_label = spec["figure"]
        own, inc = df[~df["included"]], df[df["included"]]
        study.sweep_figure(own, xcol, d / fname, xlabel, title, baseline=inc if base_label else None,
                           baseline_label=base_label or "")
        extra += ["## Figure", "", f"![{title}]({fname})", ""]
    if spec.get("standard_rule"):
        std = df[df["run"] == base.S1[1]]["tuned_threshold_profit"].iloc[0]
        lines = [f"## Against the standard-grid rule ({std:,.0f} INR/ha)", "", "| Configuration | Profit | Gap to standard-grid `threshold_tuned` |", "|---|---|---|"]
        lines += [f"| {r.config} | {study._pm(r.profit_mean, r.profit_std)} | {r.profit_mean - std:+,.0f} |" for r in df.itertuples()]
        extra += lines + [""]
    if spec.get("extras") == "grpo":
        extra += grpo_extras(d, df)
    if spec.get("extras") == "layout_forecast":
        extra += layout_forecast_extras(df)
    if ANALYSIS2.get(name):
        extra += ["## Analysis", "", *[f"{t}\n" for t in ANALYSIS2[name]]]
    study.write_card(d, spec["title"], spec["question"], spec["setup"], df, extra=extra)
    print(f"{name}: {len(df)} rows -> {d / 'README.md'}")
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiments", default=str(ROOT.parent))
    p.add_argument("--study", choices=list(STUDIES2))
    p.add_argument("--all", action="store_true")
    p.add_argument("--index", action="store_true", help="only rewrite experiments/INDEX.md")
    a = p.parse_args()
    exp = Path(a.experiments).resolve()
    if a.index:
        write_global_index(exp)
    elif a.all:
        frames = [summarize(exp, n) for n in STUDIES2]
        done = [f[~f["included"]] for f in frames if f is not None]
        if done:
            pd.concat(done, ignore_index=True).to_csv(exp / "results_followup.csv", index=False)
        write_global_index(exp)
    elif a.study:
        summarize(exp, a.study)
        write_global_index(exp)
    else:
        p.error("give --study NAME or --all")


if __name__ == "__main__":
    main()
