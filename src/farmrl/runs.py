"""Access to saved experiment runs in results/runs/<name>/.

A run directory is self-contained: model, config snapshot, fitted yield parameters,
step-level rollouts and the tables derived from them. Figures, animations, maps and the
dashboard only ever read runs through this module.
"""
import json
from functools import cached_property
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "results" / "runs"

HOW_TO = ("No experiment run found in results/runs/. Create one with\n"
          "    python scripts/run_all.py --run-name <name>        (whole pipeline)\n"
          "    python scripts/run_experiments.py --run-name <name> (simulation outputs only)")


class Run:
    """Lazy reader for one run directory. Optional tables are ``None`` when absent."""

    def __init__(self, path):
        self.path = Path(path)
        self.name = self.path.name
        if not (self.path / "metadata.json").exists():
            raise FileNotFoundError(f"{self.path} is not a run directory (no metadata.json)")

    def __fspath__(self):
        return str(self.path)

    def _csv(self, name, **kw):
        p = self.path / name
        return pd.read_csv(p, **kw) if p.exists() else None

    @cached_property
    def meta(self):
        return json.loads((self.path / "metadata.json").read_text())

    @cached_property
    def config(self):
        return yaml.safe_load((self.path / "config_snapshot.yaml").read_text())

    @cached_property
    def rollouts(self):
        return pd.read_parquet(self.path / "rollouts.parquet")

    @cached_property
    def episodes(self):
        return self._csv("episodes.csv")

    @cached_property
    def summary(self):
        return self._csv("summary.csv")

    @cached_property
    def history(self):
        return self._csv("train_history.csv")

    @cached_property
    def grids(self):
        p = self.path / "policy_grids.parquet"
        return pd.read_parquet(p) if p.exists() else None

    @cached_property
    def state_medians(self):
        return self._csv("state_medians.csv")

    @cached_property
    def districts(self):
        return self._csv("district_summary.csv")

    @cached_property
    def sensitivity(self):
        return self._csv("sensitivity.csv")

    @cached_property
    def seed_episodes(self):
        return self._csv("seed_episodes.csv")

    @cached_property
    def seed_summary(self):
        return self._csv("seed_summary.csv")

    @cached_property
    def attribution(self):
        return self._csv("attribution.csv")

    @cached_property
    def dry_wet(self):
        return self._csv("dry_wet.csv")

    @cached_property
    def tuning(self):
        """Tuned rule-baseline parameters ({family: record}), or None for older runs."""
        p = self.path / "baseline_tuning.json"
        return json.loads(p.read_text()) if p.exists() else None

    @property
    def reference_policy(self):
        """The rule baseline PPO is measured against: the tuned one that did best in
        training; 'recommended' for runs made before the baselines were tuned."""
        ref = self.meta.get("reference_policy")
        if ref:
            return ref
        return "recommended" if "recommended" in self.meta["policies"] else None

    @property
    def model_path(self):
        return self.path / "model.pt"

    @property
    def yield_params_path(self):
        return self.path / "fitted_params.yaml"

    def policies(self, include_aux=False):
        """Policy names in the run; auxiliary ones (e.g. the untrained agent) on request."""
        aux = set(self.meta.get("aux_policies", []))
        return [p for p in self.meta["policies"] if include_aux or p not in aux]

    def out_dir(self, name):
        d = self.path / name
        d.mkdir(parents=True, exist_ok=True)
        return d


def list_runs(runs_dir=None):
    """Names of all run directories, newest first."""
    root = Path(runs_dir or RUNS_DIR)
    if not root.exists():
        return []
    metas = [p for p in root.glob("*/metadata.json")]
    metas.sort(key=lambda p: json.loads(p.read_text()).get("created", ""), reverse=True)
    return [p.parent.name for p in metas]


def related(run, runs_dir=None):
    """Runs made as variants of ``run`` (other region, other data): those named
    ``<run name>_<something>`` in the same folder, e.g. v3_haryana next to v3."""
    run = get(run, runs_dir)
    names = [n for n in list_runs(run.path.parent) if n.startswith(run.name + "_")]
    return [Run(run.path.parent / n) for n in sorted(names)]


def get(run=None, runs_dir=None):
    """Resolve a ``Run`` from a Run, a path, a name, or ``None`` for the newest run."""
    if isinstance(run, Run) or type(run).__name__ == "Run" or (hasattr(run, "path") and hasattr(run, "meta")):
        return run
    if hasattr(run, "path") and hasattr(run, "districts"):
        return run
    root = Path(runs_dir or RUNS_DIR)
    if run is None:
        names = list_runs(root)
        if not names:
            raise FileNotFoundError(HOW_TO)
        return Run(root / names[0])
    # Ensure run is converted to string path if a Run or PathLike object was passed
    run_str = str(getattr(run, "path", getattr(run, "name", run)))
    path = Path(run_str)
    return Run(path if (path / "metadata.json").exists() else root / run_str)
