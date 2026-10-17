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
STUDIES2["08_transfer_and_shift"] = dict(cards_only=True)      # its card comes from scripts/transfer_eval.py
ANALYSIS2 = {}          # written by hand once the results are in


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
    a = p.parse_args()
    exp = Path(a.experiments).resolve()
    if a.all:
        frames = [summarize(exp, n) for n in STUDIES2]
        done = [f[~f["included"]] for f in frames if f is not None]
        if done:
            pd.concat(done, ignore_index=True).to_csv(exp / "results_followup.csv", index=False)
    elif a.study:
        summarize(exp, a.study)
    else:
        p.error("give --study NAME or --all")


if __name__ == "__main__":
    main()
