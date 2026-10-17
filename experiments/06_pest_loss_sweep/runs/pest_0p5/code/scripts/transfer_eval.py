"""Cross-region transfer and stress tests of trained agents against tuned rules.

    python scripts/transfer_eval.py --out ../08_transfer_and_shift/transfer \\
        --punjab-run ../02_improved_ppo/runs/improved_ent0p003_ep10 \\
        --punjab-rules ../01_oracle_bound/runs/baseline_v4_oracle \\
        --haryana-run ../08_transfer_and_shift/runs/haryana_native

Scenarios (held-out seasons, 30 pest-growth replicates each):
  * Punjab -> Haryana and Haryana -> Punjab: the PPO seeds and the threshold rule tuned in one
    region are replayed unchanged in the other, next to the rule retuned there, the PPO trained
    there and the hindsight oracle there.
  * Shifted Punjab (+1 C, +2 C, 20% less rain, 2x price volatility, +2 C and 20% less rain, see
    farmrl.shifts): the Punjab-trained PPO seeds and the Punjab-tuned rule unchanged, the rule
    retuned on the shifted training seasons, and the oracle on the shifted held-out seasons.
Nothing about PPO is retrained here.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import torch

from farmrl import oracle as oracle_mod
from farmrl import rollout, runs, shifts, tuning
from farmrl.agents import build_agent
from farmrl.baselines import threshold_policy
from farmrl.experiment import env_maker
from farmrl.yield_model import YieldModel

POLICY_LABEL = {"ppo_fixed": "PPO trained elsewhere, unchanged", "ppo_native": "PPO trained here",
                "rule_fixed": "rule tuned elsewhere, unchanged", "rule_retuned": "rule tuned here",
                "oracle": "hindsight oracle"}


def seed_policies(run, env):
    """{seed: policy} for every PPO training seed saved in ``run``; ``env`` only fixes the shapes."""
    out = {}
    for path in sorted((run.path / "models").glob("seed_*.pt")):
        agent = build_agent(env, run.config["train"])
        agent.load(path)
        out[int(path.stem.split("_")[1])] = (lambda a: lambda obs, e: a.predict(obs, e))(agent)
    return out


def rule_of(run_dir):
    rec = json.loads((Path(run_dir) / "baseline_tuning.json").read_text())["threshold"]
    return threshold_policy(rec["params"]), rec["params"]


def plans_of(run_dir):
    return oracle_mod.plan_policy(json.loads((Path(run_dir) / "oracle_plans.json").read_text()))


def evaluate(env, rows, name, seed_groups, single, seeds):
    """Roll everything out on ``env`` and append result rows. ``seed_groups``: {label: {seed: policy}}
    (reported as mean and std over seeds); ``single``: {label: policy}."""
    policies = dict(single)
    for label, group in seed_groups.items():
        policies.update({f"{label}#{s}": p for s, p in group.items()})
    eps = rollout.episodes(rollout.run_many(env, policies, seeds=seeds))
    per = eps.groupby("policy", sort=False)[["profit", "reward"]].mean()
    for label in single:
        rows.append({"scenario": name, "policy": label, "profit_mean": per.loc[label, "profit"],
                     "profit_std": np.nan, "reward_mean": per.loc[label, "reward"], "n_seeds": 1})
    for label, group in seed_groups.items():
        p = per.loc[[f"{label}#{s}" for s in group]]
        rows.append({"scenario": name, "policy": label, "profit_mean": p["profit"].mean(),
                     "profit_std": p["profit"].std(ddof=1), "reward_mean": p["reward"].mean(), "n_seeds": len(group)})


def table_md(df):
    oracle = df.set_index("policy").loc["oracle", "profit_mean"]
    retuned = df.set_index("policy")["profit_mean"].get("rule_retuned", np.nan)
    lines = ["| Policy | Profit (INR/ha) | % of oracle | Gap to rule tuned here |", "|---|---|---|---|"]
    for r in df.itertuples():
        std = "" if pd.isna(r.profit_std) else f" ± {r.profit_std:,.0f}"
        gap = "–" if pd.isna(retuned) or r.policy in ("oracle", "rule_retuned") else f"{r.profit_mean - retuned:+,.0f}"
        lines.append(f"| {POLICY_LABEL[r.policy]} | {r.profit_mean:,.0f}{std} | {100 * r.profit_mean / oracle:.1f}% | {gap} |")
    return "\n".join(lines)


def figure(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from farmrl.utils import COLORS, set_plot_style
    set_plot_style()
    scenarios = list(dict.fromkeys(df["scenario"]))
    style = {"oracle": ("s", "none", "#52514e"), "rule_retuned": ("D", "none", COLORS[2]),
             "rule_fixed": ("D", COLORS[2], COLORS[2]), "ppo_native": ("o", "none", COLORS[0]),
             "ppo_fixed": ("o", COLORS[0], COLORS[0])}
    fig, ax = plt.subplots(figsize=(8.2, 0.62 * len(scenarios) + 2.2))
    y = np.arange(len(scenarios))[::-1]
    offsets = {"ppo_fixed": 0.24, "ppo_native": 0.12, "rule_fixed": -0.04, "rule_retuned": -0.16, "oracle": -0.28}
    for yi, sc in zip(y, scenarios):
        g = df[df["scenario"] == sc].set_index("policy")
        o = g.loc["oracle", "profit_mean"]
        for pol, r in g.iterrows():
            marker, face, edge = style[pol]
            xerr = 0 if pd.isna(r["profit_std"]) else 100 * r["profit_std"] / o
            ax.errorbar(100 * r["profit_mean"] / o, yi + offsets[pol], xerr=xerr, fmt=marker, ms=6, mfc=face, mec=edge,
                        ecolor=edge, capsize=2, ls="none")
    ax.axvline(100, color="#c3c2b7", lw=1)
    ax.set_yticks(y, scenarios)
    ax.set_xlabel("held-out profit, % of the hindsight oracle in that scenario")
    ax.set_title("Transfer and stress tests")
    ax.grid(axis="y", visible=False)
    handles = [plt.Line2D([], [], marker=style[k][0], mfc=style[k][1], mec=style[k][2], color=style[k][2], ls="none", label=POLICY_LABEL[k])
               for k in ("ppo_fixed", "ppo_native", "rule_fixed", "rule_retuned", "oracle")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True)
    p.add_argument("--punjab-run", required=True, help="run whose PPO seeds were trained on Punjab")
    p.add_argument("--punjab-rules", required=True, help="run with the Punjab rule tuning and oracle plans")
    p.add_argument("--haryana-run", required=True, help="run trained, tuned and oracled on Haryana")
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--eval-seeds", type=int, default=30)
    p.add_argument("--tune-seeds", type=int, default=3)
    p.add_argument("--tune-rounds", type=int, default=4)
    p.add_argument("--oracle-cfg", default=None, help="JSON overriding the oracle search settings (for smoke tests)")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    log = lambda m: print(m, flush=True)

    pj, pj_rules, hr = runs.Run(a.punjab_run), runs.Run(a.punjab_rules), runs.Run(a.haryana_run)
    oracle_cfg = json.loads(a.oracle_cfg) if a.oracle_cfg else (pj.config.get("oracle") or None)
    pj_cfg, hr_cfg = pj.config["env"], hr.config["env"]
    pj_ym, hr_ym = YieldModel.from_config(pj.yield_params_path), YieldModel.from_config(hr.yield_params_path)
    rows = []

    # ---------------------------------------------------------------- across regions
    log("Punjab -> Haryana")
    make = env_maker(hr_ym)
    env = make(hr_cfg, "test")
    evaluate(env, rows, "Punjab → Haryana",
             {"ppo_fixed": seed_policies(pj, env), "ppo_native": seed_policies(hr, env)},
             {"rule_fixed": rule_of(a.punjab_rules)[0], "rule_retuned": rule_of(a.haryana_run)[0],
              "oracle": plans_of(a.haryana_run)}, a.eval_seeds)
    log("Haryana -> Punjab")
    make = env_maker(pj_ym)
    env = make(pj_cfg, "test")
    evaluate(env, rows, "Haryana → Punjab",
             {"ppo_fixed": seed_policies(hr, env), "ppo_native": seed_policies(pj, env)},
             {"rule_fixed": rule_of(a.haryana_run)[0], "rule_retuned": rule_of(a.punjab_rules)[0],
              "oracle": plans_of(a.punjab_rules)}, a.eval_seeds)

    # ---------------------------------------------------------------- shifted Punjab
    scenarios = {"Punjab, as trained": {}, **shifts.SCENARIOS}
    for name, spec in scenarios.items():
        log(f"shift: {name}")
        cfg_env, weather, prices = shifts.apply(pj_cfg, spec)
        make = env_maker(pj_ym, weather, prices)
        folder = out / "shifts" / name.replace(" ", "_").replace("%", "pct").replace("°", "").replace("+", "plus")
        folder.mkdir(parents=True, exist_ok=True)
        if spec:
            rec = tuning.tune(make(cfg_env, "train"), "threshold", seeds=a.tune_seeds, objective="reward",
                              max_rounds=a.tune_rounds, log=log)
            (folder / "baseline_tuning.json").write_text(json.dumps({"threshold": rec}, indent=2))
            retuned = threshold_policy(rec["params"])
            test_env = make(cfg_env, "test")
            plans = oracle_mod.solve_all(cfg_env, test_env.seasons, pj_ym, oracle_cfg, weather, prices,
                                         parallel=a.parallel, log=log)
            (folder / "oracle_plans.json").write_text(json.dumps({str(k): v for k, v in plans.items()}))
            oracle = oracle_mod.plan_policy(plans)
        else:
            retuned, oracle = rule_of(a.punjab_rules)[0], plans_of(a.punjab_rules)
            test_env = make(cfg_env, "test")
        evaluate(test_env, rows, name, {"ppo_fixed": seed_policies(pj, test_env)},
                 {"rule_fixed": rule_of(a.punjab_rules)[0], "rule_retuned": retuned, "oracle": oracle}, a.eval_seeds)

    df = pd.DataFrame(rows)
    df.to_csv(out / "transfer_results.csv", index=False)
    figure(df, out / "transfer_summary.png")
    text = ["# Study 8 — Transfer across regions and stress tests", "",
            "**Question.** Does one learned policy adapt to a new region or shifted conditions better than a rule that "
            "has to be retuned for each setting?", "",
            "Nothing is retrained here. `PPO trained elsewhere` and `rule tuned elsewhere` are the Punjab-tuned agent "
            "and rule (or the Haryana ones for the reverse direction) applied unchanged. `Rule tuned here` is the "
            "threshold rule retuned on the training seasons of the scenario (the rule's best case); `PPO trained here` "
            "is the agent trained on that region. The oracle knows each held-out season in advance. Held-out seasons "
            "2017–2023, 30 pest replicates, 5 PPO training seeds (± is std over seeds).", "",
            "Shifted conditions are assumptions, not data: " + " ".join(shifts.__doc__.split("\n\n")[1].split()), "",
            "![Transfer and stress tests](transfer_summary.png)", ""]
    for sc in dict.fromkeys(df["scenario"]):
        text += [f"## {sc}", "", table_md(df[df["scenario"] == sc]), ""]
    (out / "README.md").write_text("\n".join(text), encoding="utf-8")
    log(f"written {out / 'README.md'}")


if __name__ == "__main__":
    main()
