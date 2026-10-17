"""Self-describing run folders.

Every run in results/runs/<name>/ gets, next to the tables the dashboard reads:

    README.md   the run card: what was run, with which methods and data, and what came out
    params/     copies of every parameter file the run used (config, fitted yield model,
                tuned baselines, price index)
    code/       a snapshot of the code that produced it (src/farmrl, scripts, configs,
                requirements.txt) plus code/SNAPSHOT.md saying where the snapshot came from

and results/runs/INDEX.md lists all runs side by side. ``snapshot_code`` is called by the
experiment runner at run time; for runs made before that existed, ``snapshot_from_git``
rebuilds the code from the nearest commit and the card says so.
"""
import json
import shutil
import subprocess
import tarfile
import io
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from . import runs

ROOT = Path(__file__).resolve().parents[2]
CODE_PARTS = ["src/farmrl", "scripts", "configs", "requirements.txt"]
PARAM_FILES = ["config_snapshot.yaml", "fitted_params.yaml", "baseline_tuning.json", "prices.csv"]
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".ipynb_checkpoints")


# ------------------------------------------------------------------------------- code
def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except Exception:
        return None


def git_state():
    """Commit hash and whether the working tree differs from it (tracked code only)."""
    head = (_git("rev-parse", "HEAD") or "").strip() or None
    dirty = _git("status", "--porcelain", "--", *CODE_PARTS)
    return {"git_hash": head, "dirty": bool(dirty and dirty.strip()),
            "changed_files": [l[3:] for l in (dirty or "").splitlines() if l.strip()]}


def snapshot_code(run_dir):
    """Copy the code as it is on disk now into <run>/code/ (done at run time)."""
    out = Path(run_dir) / "code"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for part in CODE_PARTS:
        src = ROOT / part
        if src.is_dir():
            shutil.copytree(src, out / part, ignore=IGNORE)
        elif src.exists():
            (out / part).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out / part)
    state = git_state()
    lines = ["# Code snapshot", "",
             f"Copied from the working tree when the run was made ({datetime.now():%Y-%m-%d %H:%M}).", "",
             f"- commit: `{state['git_hash']}`",
             f"- uncommitted changes to the code at that time: {'yes' if state['dirty'] else 'no'}"]
    lines += [f"  - `{f}`" for f in state["changed_files"][:50]]
    lines += ["", f"Contents: {', '.join(f'`{p}`' for p in CODE_PARTS)}. This is the exact code of the run."]
    (out / "SNAPSHOT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def snapshot_from_git(run_dir, commit, reason):
    """Rebuild <run>/code/ from a commit, for runs made before snapshots existed."""
    out = Path(run_dir) / "code"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    blob = subprocess.run(["git", "archive", "--format=tar", commit, *CODE_PARTS], cwd=ROOT,
                          capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        tar.extractall(out, filter="data")
    subject = (_git("log", "-1", "--format=%h %ad %s", "--date=format:%Y-%m-%d %H:%M", commit) or "").strip()
    text = ["# Code snapshot (reconstructed)", "",
            "This run was made before runs saved their own code, so this folder was rebuilt from git",
            "afterwards and is **not guaranteed to be the exact code of the run**.", "",
            f"- taken from commit: `{subject}`", f"- why this commit: {reason}", "",
            "The parameters in `../params/` *are* the ones the run used; where code and parameters",
            "disagree, the parameters (and `../metadata.json`) describe what actually ran."]
    (out / "SNAPSHOT.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    return out


def copy_params(run_dir):
    out = Path(run_dir) / "params"
    out.mkdir(exist_ok=True)
    copied = []
    for name in PARAM_FILES:
        if (Path(run_dir) / name).exists():
            shutil.copy2(Path(run_dir) / name, out / name)
            copied.append(name)
    return copied


# ------------------------------------------------------------------------------- card
def _money(x):
    return "–" if pd.isna(x) else f"{x:,.0f}"


def _table(df):
    cols = list(df.columns)
    rows = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    rows += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows)


def _results_table(run):
    s = run.summary
    s = s[(s["split"] == "test") & s["policy"].isin(run.policies())].sort_values("profit", ascending=False)
    cols = {"policy": "Policy", "profit": "Profit (INR/ha)", "profit_std": "Std", "yield_q_ha": "Yield (q/ha)",
            "nitrogen_kg": "N (kg/ha)", "irrigation_mm": "Irrigation (mm)", "pesticide_level": "Pesticide (doses)"}
    out = s[[c for c in cols if c in s]].rename(columns=cols)
    for c in ("Profit (INR/ha)", "Std"):
        if c in out:
            out[c] = out[c].map(_money)
    for c in ("Yield (q/ha)", "N (kg/ha)", "Irrigation (mm)", "Pesticide (doses)"):
        if c in out:
            out[c] = out[c].map(lambda v: f"{v:.1f}" if "Yield" in c or "Pest" in c else f"{v:.0f}")
    from .viz import style
    out["Policy"] = out["Policy"].map(style.label)
    return out


def _mechanisms(cfg_env):
    names = ["groundwater", "waterlogging", "pest_resistance", "storage"]
    rows = []
    for n in names:
        block = cfg_env.get(n)
        if block is None:
            rows.append((n, "not in this version of the simulator"))
            continue
        on = block.get("enabled", False)
        keys = ", ".join(f"{k} = {v}" for k, v in block.items() if k != "enabled" and not isinstance(v, (list, dict)))
        rows.append((n, ("**on**: " + keys) if on else "off"))
    return rows


def describe_run(run):
    """The run card as Markdown text."""
    m, cfg = run.meta, run.config
    env, train = cfg["env"], cfg.get("train", {})
    fit = m.get("yield_fit") or {}
    ym = m.get("yield_model", {})
    out = [f"# Run `{run.name}`", ""]
    region = (m.get("region") or env.get("region") or "?").title()
    mech_on = [k for k, v in (m.get("mechanisms") or {}).items() if v]
    form = fit.get("model", "single-region fit (nitrogen + rain)") if fit else "prior parameters"
    out += [f"**{region}**, weather of {m.get('weather_site', env.get('weather_site', '?')).title()}; yield model: {form}; "
            f"simulator mechanisms: {', '.join(mech_on) if mech_on else 'none beyond the base water/N/pest model'}; "
            f"{'**quick smoke run**; ' if m.get('quick') else ''}created {m['created'].replace('T', ' ')}.", ""]

    # ---------------------------------------------------------------- results
    out += ["## Results (held-out seasons)", ""]
    try:
        out += [_table(_results_table(run)), ""]
    except Exception as why:                         # very old runs
        out += [f"(summary table unavailable: {why})", ""]
    seeds = run.seed_summary
    if seeds is not None and "split" in seeds:
        t = seeds[seeds["split"] == "test"]["profit"]
        out += [f"The agent (`ppo` rows) across its {len(t)} training seeds: mean {t.mean():,.0f}, std {t.std():,.0f}, "
                f"range {t.min():,.0f}–{t.max():,.0f} INR/ha. The table shows seed {m['train_seeds'][0]}.", ""]
    att = run.attribution
    if att is not None and run.reference_policy in set(att["baseline_policy"]):
        a = att[att["baseline_policy"] == run.reference_policy].groupby("decision", sort=False)["contribution"].mean()
        out += [f"Gain over `{run.reference_policy}` split by decision (Shapley, mean over seeds): "
                + ", ".join(f"{k} {v:+,.0f}" for k, v in a.items()) + " INR/ha.", ""]

    # ---------------------------------------------------------------- methods
    out += ["## Methods", "", "### Data", ""]
    files = m.get("data_files", [])
    raw = sorted({Path(f["path"]).parts[2] if len(Path(f["path"]).parts) > 2 else f["path"] for f in files
                  if f["path"].startswith("data/raw")})
    out += [f"- Raw inputs present: {', '.join(f'`{r}`' for r in raw) if raw else 'not recorded'} "
            f"({len(files)} files listed with size and date in `metadata.json`).",
            f"- Seasons: {len(m['train_seasons'])} training ({m['train_seasons'][0]}–{m['train_seasons'][-1]}), "
            f"{len(m['test_seasons'])} held out ({m['test_seasons'][0]}–{m['test_seasons'][-1]}); sowing years, "
            f"{env.get('steps_per_season')} fortnights from 1 November.",
            f"- Prices: season price index from `params/prices.csv` around a base of {env.get('base_price')} INR/quintal."
            if (run.path / "prices.csv").exists() else f"- Prices: base {env.get('base_price')} INR/quintal.", ""]

    out += ["### Yield model", ""]
    if fit:
        out.append(f"- Form: {form}; {fit.get('n_obs', '?')} records"
                   + (f", {fit.get('n_states')} states" if fit.get("n_states") else "") + ".")
        out.append(f"- In-sample R² {fit.get('r2', float('nan')):.3f}, RMSE {fit.get('rmse', float('nan')):.2f} q/ha.")
        h = fit.get("holdout")
        if h:
            out.append(f"- Held-out years ≥ {h['from_year']}: R² {h['r2_test']:.3f} (state means alone {h['r2_test_state_means_only']:.3f}).")
        ident = fit.get("identified", {})
        if ident:
            out.append("- Identified by the data: " + ", ".join(k for k, v in ident.items() if v)
                       + "; reset to neutral/prior: " + (", ".join(k for k, v in ident.items() if not v) or "none") + ".")
        pest = fit.get("pesticide")
        status = fit.get("pest_loss_status") or (pest or {}).get("status") or "prior (no pesticide data)"
        out.append(f"- Pesticide loss: {ym.get('pest_loss', float('nan')):.3f} ({status})"
                   + (f"; {pest['quality_note']}" if pest and pest.get("quality_note") else "") + ".")
    out.append("- Parameters used: " + ", ".join(f"{k} = {v:.4g}" for k, v in ym.items()) + " (`params/fitted_params.yaml`).")
    out.append("")

    out += ["### Simulator", "", "| Mechanism | Setting |", "|---|---|"]
    out += [f"| {n} | {s} |" for n, s in _mechanisms(env)]
    costs = env.get("costs", {})
    out += ["", "Costs (INR): " + ", ".join(f"{k} = {v}" for k, v in costs.items())
            + "; penalties: " + ", ".join(f"{k} = {v}" for k, v in env.get("penalties", {}).items())
            + f". Actions: N {env.get('nitrogen_levels')} kg/ha, water {env.get('irrigation_levels')} mm, "
              f"pesticide {env.get('pesticide_levels')}"
            + (f", sale/storage heads {m.get('action_heads')}" if m.get("action_heads") else "") + ".", ""]

    out += ["### Agent and training", ""]
    algo = m.get("algo", "ppo")
    what = {"ppo": "PPO (own implementation, `code/src/farmrl/agents/ppo.py`)",
            "dqn": "branching dueling DQN (own implementation, `code/src/farmrl/agents/dqn.py`)",
            "cem_threshold": "cross-entropy search over the threshold rule's parameters "
                             "(`code/src/farmrl/agents/cem.py`)"}[algo]
    budget = (f"{m['timesteps']:,} steps per seed" if algo != "cem_threshold"
              else f"{(train.get('cem') or {})} per seed")
    out.append(f"- {what}, {budget}, "
               f"training seeds {m.get('train_seeds') or '(loaded model: ' + str(m.get('loaded_model')) + ')'}.")
    out.append("- Hyperparameters: " + ", ".join(f"{k} = {v}" for k, v in train.items() if not isinstance(v, dict)) + ".")
    if m.get("overrides"):
        out.append("- Config overrides given on the command line: " + ", ".join(f"`{o}`" for o in m["overrides"]) + ".")
    if m.get("yield_overrides"):
        out.append(f"- Yield-model overrides (y_scale recalibrated afterwards): {m['yield_overrides']}.")
    if m.get("compute"):
        out.append(f"- Compute: {m['compute']}.")
    if m.get("obs_names"):
        out.append(f"- Observation ({len(m['obs_names'])}): {', '.join(m['obs_names'])}.")
    out.append("")

    out += ["### Baselines and evaluation", ""]
    out.append(f"- Policies: {', '.join(run.policies())}; auxiliary (not ranked): {', '.join(m.get('aux_policies', [])) or 'none'}.")
    if run.tuning:
        for name, rec in run.tuning.items():
            out.append(f"- `{name}_tuned`: grid search on the training seasons only ({rec.get('evaluations', '?')} settings, "
                       f"objective {rec.get('objective')}): {rec.get('params')}.")
    if m.get("oracle"):
        orc = (cfg.get("oracle") or {})
        out.append("- `oracle`: hindsight oracle, the best fixed 12-step plan per held-out season found by CEM "
                   f"with that season's weather and price known in advance (`oracle_plans.json`; settings {orc or 'default'}).")
    out.append(f"- Each policy replayed on every season with {m['rollout_seeds']} pest-growth seeds shared across policies.")
    extra = [n for n, f in (("per-district evaluation", "district_summary.csv"), ("sensitivity analysis", "sensitivity.csv"),
                            ("gain attribution", "attribution.csv"), ("per-seed evaluation", "seed_summary.csv"))
             if (run.path / f).exists()]
    out.append(f"- Also computed: {', '.join(extra) if extra else 'nothing beyond the main rollouts'}.")
    out.append("")

    # ---------------------------------------------------------------- provenance & files
    out += ["## Provenance", ""]
    snap = run.path / "code" / "SNAPSHOT.md"
    first = snap.read_text(encoding="utf-8").splitlines()[0] if snap.exists() else "no code snapshot"
    out.append(f"- Code: `code/` — {first.lstrip('# ').lower()} (details in `code/SNAPSHOT.md`).")
    out.append(f"- Commit recorded at run time: `{(m.get('git_hash') or '?')[:10]}`"
               + ("" if "git_dirty" not in m else f" ({'with' if m['git_dirty'] else 'without'} uncommitted code changes)") + ".")
    out.append(f"- Software: {m.get('versions')}.")
    out += ["", "## Files", ""]
    groups = {"figures": "report figures (PNG 300 dpi + PDF)", "animations": "season animations",
              "maps": "district maps", "models": "one model per training seed", "params": "parameter copies",
              "code": "code snapshot", "tensorboard": "TensorBoard logs"}
    for d, what in groups.items():
        if (run.path / d).is_dir():
            n = sum(1 for p in (run.path / d).rglob("*") if p.is_file())
            out.append(f"- `{d}/` — {what} ({n} files)")
    tables = sorted(p.name for p in run.path.iterdir() if p.is_file() and p.suffix in {".csv", ".parquet", ".json", ".yaml", ".pt"})
    out.append("- tables and model: " + ", ".join(f"`{t}`" for t in tables))
    out += ["", f"_Card generated {datetime.now():%Y-%m-%d %H:%M} by `farmrl.runcard`; regenerate with "
                f"`python scripts/make_runcard.py --run {run.name}`._", ""]
    return "\n".join(out)


def write_card(run):
    run = runs.get(run)
    copy_params(run.path)
    path = run.path / "README.md"
    path.write_text(describe_run(run), encoding="utf-8")
    return path


def write_index(runs_dir=None):
    """results/runs/INDEX.md: one line per run."""
    root = Path(runs_dir or runs.RUNS_DIR)
    rows = []
    for name in sorted(runs.list_runs(root), key=lambda n: runs.get(n, root).meta["created"]):
        r = runs.get(name, root)
        m = r.meta
        s = r.summary[(r.summary["split"] == "test") & r.summary["policy"].isin(r.policies())].set_index("policy")
        ppo = s["profit"].get("ppo")
        others = s.drop(index=["ppo", "oracle"], errors="ignore")["profit"]
        best = others.idxmax() if len(others) else None
        fit = m.get("yield_fit") or {}
        mech = [k for k, v in (m.get("mechanisms") or {}).items() if v]
        rows.append({"Run": f"[`{name}`]({name}/README.md)", "Created": m["created"][:16].replace("T", " "),
                     "Region": (m.get("region") or r.config["env"].get("region", "")).title(),
                     "Yield model": "pooled panel" if "state_effects" in fit else "single region",
                     "pest_loss": f"{m['yield_model'].get('pest_loss', float('nan')):.2f}",
                     "Mechanisms": ", ".join(mech) or "–",
                     "Train seeds": len(m.get("train_seeds") or []),
                     "PPO profit": _money(ppo), "Best other policy": f"{best} ({_money(others.max())})" if best else "–",
                     "PPO − best": f"{ppo - others.max():+,.0f}" if best and ppo is not None else "–"})
    text = ["# Experiment runs", "",
            "One folder per run. Each has a `README.md` run card (methods, parameters, results, provenance),",
            "`params/` (every parameter file used), `code/` (the code that produced it) and the tables the",
            "dashboard reads. Profits are held-out seasons, INR/ha, PPO's first training seed.", "",
            _table(pd.DataFrame(rows)) if rows else "(no runs)", "",
            f"_Generated {datetime.now():%Y-%m-%d %H:%M} by `python scripts/make_runcard.py --all`._", ""]
    path = root / "INDEX.md"
    path.write_text("\n".join(text), encoding="utf-8")
    return path
