"""Which information is worth the most? Oracles that know only part of the season.

    python scripts/oracle_breakdown.py --out ../10_oracle_information_breakdown/breakdown \\
        --base-run ../01_oracle_bound/runs/baseline_v4_oracle \\
        --ppo-run ../02_improved_ppo/runs/improved_ent0p003_ep10

Four plan-searching oracles, each choosing one fixed 12-step plan per held-out season and then
replayed on the true season:
  full     knows the season's weather and its price (the oracle of study 1)
  weather  knows the season's weather, only the average price of the training seasons
  price    knows the season's price, only the weather of ten training seasons spread over the record
  neither  knows neither: the best single plan for the average price and those ten weather seasons
The tuned rules and the PPO seeds are replayed on the same seasons and replicates.
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
from farmrl import rollout, runs
from farmrl.agents import build_agent
from farmrl.baselines import tuned_policies
from farmrl.experiment import env_maker
from farmrl.yield_model import YieldModel

LABEL = {"oracle_full": "oracle: weather and price known", "oracle_weather": "oracle: weather known, average price",
         "oracle_price": "oracle: price known, typical weather", "oracle_none": "oracle: neither (best fixed plan)",
         "threshold_tuned": "threshold_tuned", "recommended_tuned": "recommended_tuned", "ppo": "PPO"}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True)
    p.add_argument("--base-run", required=True, help="run with the config, rule tuning and the oracle of study 1")
    p.add_argument("--ppo-run", required=True, help="run whose PPO seeds are replayed")
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--eval-seeds", type=int, default=30)
    p.add_argument("--sample", type=int, default=10, help="training seasons the 'typical weather' is spread over")
    p.add_argument("--oracle-cfg", default=None, help="JSON overriding the oracle search settings (smoke tests)")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    log = lambda m: print(m, flush=True)

    base, ppo_run = runs.Run(a.base_run), runs.Run(a.ppo_run)
    cfg = base.config
    ym = YieldModel.from_config(base.yield_params_path)
    make = env_maker(ym)
    env = make(cfg["env"], "test")
    oracle_cfg = json.loads(a.oracle_cfg) if a.oracle_cfg else (cfg.get("oracle") or None)
    plans = oracle_mod.solve_info(cfg["env"], env.seasons, ym, oracle_mod.KINDS, oracle_cfg,
                                  parallel=a.parallel, sample=a.sample, log=log)
    (out / "plans.json").write_text(json.dumps({k: {str(s): v for s, v in d.items()} for k, d in plans.items()}))

    policies = {f"oracle_{k}": oracle_mod.plan_policy(plans[k]) for k in oracle_mod.KINDS}
    policies.update(tuned_policies(base.tuning))
    seed_policies = {}
    for path in sorted((ppo_run.path / "models").glob("seed_*.pt")):
        agent = build_agent(env, ppo_run.config["train"])
        agent.load(path)
        seed_policies[int(path.stem.split("_")[1])] = (lambda ag: lambda obs, e: ag.predict(obs, e))(agent)
    policies.update({f"ppo#{s}": f for s, f in seed_policies.items()})
    eps = rollout.episodes(rollout.run_many(env, policies, seeds=a.eval_seeds))
    eps.to_csv(out / "episodes.csv", index=False)
    per = eps.groupby("policy", sort=False)[["profit", "reward"]].mean()
    ppo_rows = per.loc[[f"ppo#{s}" for s in seed_policies]]

    rows = []
    for name in ["oracle_full", "oracle_weather", "oracle_price", "oracle_none", "threshold_tuned", "recommended_tuned"]:
        rows.append({"policy": name, "profit_mean": per.loc[name, "profit"], "profit_std": np.nan, "reward_mean": per.loc[name, "reward"]})
    rows.append({"policy": "ppo", "profit_mean": ppo_rows["profit"].mean(), "profit_std": ppo_rows["profit"].std(ddof=1),
                 "reward_mean": ppo_rows["reward"].mean()})
    df = pd.DataFrame(rows)
    full = df.set_index("policy").loc["oracle_full", "profit_mean"]
    none = df.set_index("policy").loc["oracle_none", "profit_mean"]
    df["pct_of_full_oracle"] = 100 * df["profit_mean"] / full
    df["share_of_information_gain"] = (df["profit_mean"] - none) / (full - none)
    df.to_csv(out / "results.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from farmrl.utils import COLORS, set_plot_style
    set_plot_style()
    order = ["oracle_full", "oracle_weather", "oracle_price", "oracle_none", "threshold_tuned", "recommended_tuned", "ppo"]
    d = df.set_index("policy").loc[order]
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    y = np.arange(len(order))[::-1]
    colors = ["#52514e"] * 4 + [COLORS[2], COLORS[2], COLORS[0]]
    for yi, (pol, r), c in zip(y, d.iterrows(), colors):
        ax.errorbar(r["profit_mean"], yi, xerr=0 if pd.isna(r["profit_std"]) else r["profit_std"], fmt="s" if pol.startswith("oracle") else "o",
                    ms=7, color=c, mfc="none" if pol.startswith("oracle") else c, capsize=3)
        ax.annotate(f"{r['profit_mean']:,.0f}", (r["profit_mean"], yi), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=7, color="#52514e")
    ax.set_yticks(y, [LABEL[o] for o in order])
    ax.set_xlabel("held-out profit (INR/ha), 2017–2023")
    ax.set_title("What the oracle needs to know")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out / "oracle_breakdown.png", dpi=200)
    plt.close(fig)

    lines = ["| Policy | Profit (INR/ha) | % of full oracle |", "|---|---|---|"]
    for pol, r in d.iterrows():
        std = "" if pd.isna(r["profit_std"]) else f" ± {r['profit_std']:,.0f}"
        lines.append(f"| {LABEL[pol]} | {r['profit_mean']:,.0f}{std} | {r['pct_of_full_oracle']:.1f}% |")
    text = ["# Study 10 — What does the oracle need to know?", "",
            "**Question.** Does the remaining headroom above the tuned rule come from weather uncertainty or from price uncertainty?", "",
            "Four plan-searching oracles, each choosing one fixed 12-step plan per held-out season and then replayed on the true "
            "season: **full** knows the season's weather and price; **weather** knows the weather but only the average price of "
            "the training seasons; **price** knows the price but only the weather of ten training seasons spread over the record; "
            "**neither** is the best single plan for the average price and those ten seasons.", "",
            "Held-out seasons 2017–2023, 30 pest replicates; PPO is the study-2 best configuration, mean ± std over 5 seeds. "
            "The oracles are heuristic searches, so each is a lower bound on what its information allows.", "",
            "![What the oracle needs to know](oracle_breakdown.png)", "", "\n".join(lines), "",
            f"Value of information over the best fixed plan ({none:,.0f}): knowing the weather adds "
            f"{df.set_index('policy').loc['oracle_weather', 'profit_mean'] - none:+,.0f}, knowing the price adds "
            f"{df.set_index('policy').loc['oracle_price', 'profit_mean'] - none:+,.0f}, knowing both adds {full - none:+,.0f} INR/ha.", ""]
    (out / "README.md").write_text("\n".join(text), encoding="utf-8")
    log(f"written {out / 'README.md'}")


if __name__ == "__main__":
    main()
