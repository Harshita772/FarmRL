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

ANALYSIS = {'01_oracle_bound': ['The reproduction of v4 gives 86,135 ± 179, not the 86,584 ± 153 recorded in `results/runs/v4`. The agent code is bit-identical to the original for the same seed (5 PPO iterations compared on this server, returns equal to four decimals) and the tuned baselines match v4 exactly (87,232 and 81,072), so the difference comes from the platform (v4 was trained on Windows with torch 2.5, these runs on Linux with torch 2.6), not from the code. Numbers in this folder should be compared with each other, not with the v4 card.', "The hindsight oracle reaches 89,654 INR/ha. `threshold_tuned` earns 97.3% of it (2,422 below) and PPO 96.1% (3,518 below). PPO's shortfall to the rule (1,096, 0 of 5 seeds ahead) is about 45% of the rule's own distance to the oracle.", "The oracle knows the whole season's weather and price in advance, so no policy that reacts to the season as it unfolds can reach it. The tuned rule is therefore within 2.7% of a ceiling it cannot attain, and the room above it that any learned policy could use is smaller than 2,422 INR/ha. Both readings hold: the rule is already near the best possible, and PPO is still a little below it.", 'Caveat: the oracle plan is open-loop in the pest noise (averaged over 6 seeds) and its search is heuristic (CEM plus coordinate ascent), so 89,654 is a lower bound on the true foresight optimum.'], '02_improved_ppo': ['No configuration is distinguishable from the original agent: the six improved variants span 85,820–86,200 against 86,135 ± 179 for the original, and the best one (ent_coef 0.003, epochs 10) is +65 INR/ha, well inside the seed-to-seed std of 270–550. All six are behind `threshold_tuned` in every seed (0 of 5 seeds ahead), by 1,032–1,412.', 'The improvements (running observation and return normalisation, linear LR decay, orthogonal init, gamma = 1, the storage-head mask and five times more steps) are standard fixes for optimisation problems, so the remaining gap to the rule is unlikely to be an optimisation problem. The Shapley split points the same way for every configuration: the loss against the rule is in the water decisions (−1,005 to −1,456); nitrogen, pesticide and sale contribute almost nothing.', 'ent_coef and epochs move the mean by at most about 380 INR/ha, less than the seed spread of the worst-performing settings, so these runs do not support a claim about which value is better.'], '03_action_structure': ['A joint softmax over nitrogen x water gives 86,250 ± 194, +50 over the best factored head (86,200 ± 542): no measurable change, so head independence is not what limits PPO here.', "The finer water grid is the one configuration ahead of its own `threshold_tuned` (+196 ± 286, 4 of 5 seeds), but that comparison is not like for like: on the {0, 25, 50, 75} grid the rule's 37.5 mm 'half' irrigation becomes 50 mm and it cannot use 25 mm, so the rule falls from 87,232 to 86,390. In absolute terms PPO gains +387 over the best factored run (86,587 vs 86,200, about one seed std) and is still 645 below the standard-grid rule. The evidence that a finer grid helps PPO is weak."], '04_other_algorithms': ["Branching DQN reaches 83,617 ± 1,741 (seeds from 80,954 to 85,545), about 2,600 below the best PPO and far less stable. CEM over the threshold rule's ten parameters reaches 84,178 ± 192, 3,054 below the grid-tuned rule of the same family.", "CEM's shortfall is almost entirely the pesticide decision (Shapley −3,088): it settles on about 3.5 sprays a season where the tuned rule sprays never. With 64 candidates x 40 iterations and 2 replicates per candidate, the pest cut-offs are hard to separate from noise, so the search did not converge to the 'never spray' corner that the block-wise grid search found. This says the grid-searched rule is a strong baseline, not that CEM is a poor method in general."], '05_pest_loss_prior': ["With pest_loss = 0.35, PPO learns to spray: 3.59 doses a season against 0 at the calibrated 0.115. It earns 86,410 ± 492 (95.4% of the oracle's 90,561) and is ahead of its own tuned threshold rule by +2,664 ± 492 in all 5 seeds; the rule earns only 83,746 here (92.5% of the oracle), and the whole advantage is the pest decision (Shapley +4,057).", "So when pests matter, PPO adapts the spray decision where the rule's tuned spray cut-offs are poor. y_scale was recalibrated for the new loss; ky was not."]}

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
    study.write_card(d, spec["title"], spec["question"], spec["setup"], df,
                     extra=["## Analysis", "", *[t + "\n" for t in ANALYSIS[name]]])
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
    text += ["", "## Conclusion", "",
             "- **The rule is already near the best possible.** A hindsight oracle that knows each held-out season's weather and "
             "price in advance earns 89,654 INR/ha; the grid-tuned threshold rule earns 97.3% of that, and no policy that reacts to "
             "the season as it unfolds can reach the oracle.",
             "- **PPO is slightly below the rule, and this is not an optimisation problem.** Seven PPO variants (original, six "
             "improved/sweep configurations, a joint head) sit at 85.8–86.3k, 1.0–1.4k below the rule in every seed; normalisation, "
             "LR decay, orthogonal init, gamma = 1, a masked storage head and five times more steps changed nothing measurable. "
             "The shortfall is in the water decisions.",
             "- **A finer water grid is the only change with a hint of benefit** (+387 over the best factored run, about one seed "
             "std; ahead of its own, handicapped, rule in 4 of 5 seeds).",
             "- **DQN and CEM over the rule's parameters are worse** (83.6k and 84.2k).",
             "- **When pests cost more (pest_loss 0.35), PPO does learn to spray** and beats the tuned rule by 2.7k.",
             "- Caveats: 7 held-out seasons; the v4 reproduction is 450 INR/ha below the original v4 run because of the platform "
             "(identical code, Windows/torch 2.5 vs Linux/torch 2.6); the oracle is a heuristic search, hence a lower bound on the "
             "foresight optimum.", "", "## Layout", "",
             "- `code/` — self-contained copy of the code the experiments use (`src/farmrl`, `scripts`, `configs`, "
             "`data`, `tests`); the repository's own `src/` is untouched.",
             "- `<study>/runs/<run>/` — one folder per run (run card, tables, models, config and code snapshot).",
             "- `<study>/logs/` — the stdout/stderr of every run.",
             "- `run_all_studies.sh` — the sequential driver that produced everything.", ""]
    (exp / "README.md").write_text("\n".join(text), encoding="utf-8")
    from farmrl import runs as runs_mod
    rows = ["| Study | Run | Algorithm | Overrides | Profit | Gap to threshold_tuned | Gap to oracle |", "|" + "---|" * 7]
    for r in df.itertuples(index=False):
        meta = runs_mod.Run(r.folder).meta
        rows.append(f"| {r.study} | [`{r.run}`]({r.study}/runs/{r.run}/README.md) | {meta.get('algo', 'ppo')} | "
                    f"{', '.join(f'`{o}`' for o in meta.get('overrides', [])) or '–'} | "
                    f"{study._pm(r.profit_mean, r.profit_std)} | {study._pm(r.gap_threshold_mean, r.gap_threshold_std, True)} | "
                    f"{study._pm(r.gap_oracle_mean, r.gap_oracle_std, True)} |")
    (exp / "INDEX.md").write_text("# Experiment runs\n\nOne row per run in `experiments/`; held-out profit in INR/ha, mean ± std over "
                                  "5 training seeds. Each run folder has its own run card.\n\n" + "\n".join(rows) + "\n",
                                  encoding="utf-8")


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
