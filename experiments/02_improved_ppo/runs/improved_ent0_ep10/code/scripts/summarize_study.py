"""Build the run cards, results table, study card and the summary figure for the studies.

    python scripts/summarize_study.py --experiments ../ --study 02_improved_ppo
    python scripts/summarize_study.py --experiments ../ --all        # every study + summary figure

``--experiments`` is the folder that holds the study folders (the parent of ``code/``).
"""
import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
from farmrl import runcard, study

S1 = ("01_oracle_bound", "baseline_v4_oracle", "PPO v4 (200k steps, original settings)")
E = [("0.01", "0p01"), ("0.003", "0p003"), ("0", "0")]
S2_RUNS = [(f"improved_ent{tag}_ep{ep}", f"improved PPO, ent_coef={ent}, epochs={ep}")
           for ep in (10, 4) for ent, tag in E]

STUDIES = {
    "01_oracle_bound": dict(
        title="Study 1 — Hindsight oracle: how much is there to gain?",
        question="Is PPO underperforming, or is the tuned threshold rule already close to the best possible?",
        setup=["Environment: `configs/default.yaml` (Punjab, Ludhiana weather, all four mechanisms on).",
               "Agent: the original PPO (200k steps, default hyperparameters), 5 training seeds, reproducing run v4.",
               "Oracle: for each held-out season, a CEM search over the 12-step action plan (N, water, pesticide per "
               "fortnight and the months to store) with that season's weather and price known in advance, "
               "maximising the training reward and averaged over 6 pest-growth seeds; the plan is then replayed on the "
               "30 evaluation seeds. Search: population 240, 40 iterations, then coordinate-ascent polishing."],
        runs=[(S1[1], S1[2])], include=[]),
    "02_improved_ppo": dict(
        title="Study 2 — Improved PPO and a small sweep",
        question="Do running observation/return normalisation, linear LR decay, orthogonal init, gamma = 1, a masked "
                 "storage head and 1M steps close the gap, and how sensitive is it to ent_coef and epochs?",
        setup=["Bundle `train.variant: improved`: obs_norm, ret_norm, lr_decay, ortho_init, mask_storage, "
               "gamma = 1.0, total_timesteps = 1,000,000.",
               "Sweep: ent_coef in {0, 0.003, 0.01} x epochs in {4, 10}; everything else as v4.",
               "The tuned rule baselines are reused from study 1 (same environment)."],
        runs=S2_RUNS, include=[S1]),
    "03_action_structure": dict(
        title="Study 3 — Action structure",
        question="Does a joint softmax over nitrogen x water, or a finer water grid, help?",
        setup=["Base: the best configuration of study 2 (same bundle, ent_coef and epochs).",
               "`joint_nw`: one softmax over the 9 nitrogen x water pairs; pesticide and storage keep their own heads.",
               "`fine_water`: water grid {0, 25, 50, 75} mm instead of {0, 37.5, 75}. The rule baselines are retuned on the "
               "new grid and map their doses onto it (their 37.5 mm becomes 50 mm); the oracle is recomputed on the grid."],
        runs=[("joint_nw", "improved PPO, joint N x water head"), ("fine_water", "improved PPO, water grid 0/25/50/75")],
        include=[("02_improved_ppo", "BEST", "improved PPO, factored heads (study 2 best)")]),
    "04_other_algorithms": dict(
        title="Study 4 — Other algorithms",
        question="Does a different learner, or a direct search over the rule's parameters, do better than PPO?",
        setup=["`dqn_branching`: dueling Q-network with one branch per input, double-DQN target averaged over the active "
               "branches, 1M steps, gamma = 1, lr 5e-4, epsilon 1 -> 0.05 over half the run.",
               "`cem_threshold`: cross-entropy search (population 64, 40 iterations, 20% elite) over the ten parameters of "
               "the threshold rule, scored on the training seasons with the training reward.",
               "Same seasons, seeds and reused rule tuning as study 2."],
        runs=[("dqn_branching", "branching dueling DQN, 1M steps"), ("cem_threshold", "CEM over threshold-rule parameters")],
        include=[S1, ("02_improved_ppo", "BEST", "improved PPO (study 2 best)")]),
    "05_pest_loss_prior": dict(
        title="Study 5 — Does PPO spray when pests cost more?",
        question="With the old prior pest_loss = 0.35 (calibrated value 0.115), does the best PPO learn to spray?",
        setup=["Best study-2 configuration retrained with `pest_loss = 0.35`; `y_scale` recalibrated so the recommended "
               "calendar still averages the observed yield; rule baselines retuned and oracle recomputed in this environment.",
               "The pesticide head is kept; the last column of the table gives the mean sprays per season."],
        runs=[("improved_best_pest035", "improved PPO, pest_loss = 0.35")],
        include=[("02_improved_ppo", "BEST", "improved PPO, pest_loss = 0.115 (study 2 best)")]),
}


def best_run(exp):
    from farmrl.study import run_row
    rows = [run_row(p) for p, _ in [(exp / "02_improved_ppo" / "runs" / n, l) for n, l in S2_RUNS]
            if (p / "metadata.json").exists()]
    return max(rows, key=lambda r: r["profit_mean"])["run"] if rows else None


def entries_for(exp, name, spec):
    """[(run folder, label, owned)] for a study, includes first."""
    out = []
    for study_name, run, label in spec["include"]:
        run = best_run(exp) if run == "BEST" else run
        if run:
            out.append((exp / study_name / "runs" / run, label, False))
    out += [(exp / name / "runs" / run, label, True) for run, label in spec["runs"]]
    return out


def summarize(exp, name):
    spec = STUDIES[name]
    d = exp / name
    (d / "runs").mkdir(parents=True, exist_ok=True)
    for p in sorted((d / "runs").glob("*/metadata.json")):
        try:
            runcard.write_card(p.parent)
        except Exception:
            print(f"run card failed for {p.parent.name}:\n{traceback.format_exc()}")
    if list((d / "runs").glob("*/metadata.json")):
        runcard.write_index(d / "runs")
    entries = entries_for(exp, name, spec)
    oracle_from = exp / S1[0] / "runs" / S1[1]
    df = study.study_table([(p, l) for p, l, _ in entries], oracle_from if oracle_from.exists() else None)
    if df.empty:
        print(f"{name}: no finished runs yet")
        return None
    owned = {str(p.resolve()) for p, _, own in entries if own}
    df["included"] = ~df["folder"].isin(owned)
    df.insert(0, "study", name)
    study.write_card(d, spec["title"], spec["question"], spec["setup"], df)
    print(f"{name}: {len(df)} rows -> {d / 'README.md'}")
    return df


def overall(exp, frames):
    df = pd.concat([f[~f["included"]] for f in frames if f is not None], ignore_index=True)
    df.to_csv(exp / "results_all.csv", index=False)
    study.summary_figure(df, exp / "summary.png")
    text = ["# Comparative study: is PPO underperforming, or is the rule already near the best possible?", "",
            "Held-out profit (INR/ha, seasons 2017–2023, mean ± std over 5 training seeds). Every study card lists "
            "its setup, run folders and the full numbers; every run folder has its own run card, config, logs and the "
            "exact code that produced it.", "", "![Summary](summary.png)", "",
            "Dot: the agent (± std over seeds). Diamond: `threshold_tuned` in the same environment. Open square: "
            "hindsight oracle in the same environment.", "", study.markdown_table(df), "", "## Studies", ""]
    text += [f"- [{STUDIES[n]['title']}]({n}/README.md)" for n in STUDIES if (exp / n / "README.md").exists()]
    text += ["", "## Layout", "",
             "- `code/` — self-contained copy of the code the experiments use (`src/farmrl`, `scripts`, `configs`, "
             "`data`, `tests`); the repository's own `src/` is untouched.",
             "- `<study>/runs/<run>/` — one folder per run (run card, tables, models, config and code snapshot).",
             "- `<study>/logs/` — the stdout/stderr of every run.",
             "- `run_all_studies.sh` — the sequential driver that produced everything.", ""]
    (exp / "README.md").write_text("\n".join(text), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiments", default=str(ROOT.parent))
    p.add_argument("--study", choices=list(STUDIES))
    p.add_argument("--all", action="store_true")
    a = p.parse_args()
    exp = Path(a.experiments).resolve()
    if a.all:
        overall(exp, [summarize(exp, n) for n in STUDIES])
    elif a.study:
        summarize(exp, a.study)
    else:
        p.error("give --study NAME or --all")


if __name__ == "__main__":
    main()
