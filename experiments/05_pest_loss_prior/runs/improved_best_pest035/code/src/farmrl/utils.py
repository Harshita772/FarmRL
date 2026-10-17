import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "results" / "figures"


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


# Named bundles of training settings, switched on with ``train.variant: <name>``. Keys the
# user sets explicitly (``--set train.<key>=...``) win over the bundle.
TRAIN_VARIANTS = {
    "improved": {"obs_norm": True, "ret_norm": True, "lr_decay": True, "ortho_init": True,
                 "mask_storage": True, "gamma": 1.0, "total_timesteps": 1_000_000},
}


def apply_overrides(cfg, sets=()):
    """Apply ``key.sub=value`` overrides (value parsed as YAML) to a config in place and
    then expand ``train.variant``. Returns the list of dotted keys that were set."""
    explicit = []
    for item in sets or ():
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override '{item}' needs the form a.b.c=value")
        node = cfg
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
        explicit.append(key.strip())
    variant = (cfg.get("train") or {}).get("variant")
    if variant:
        if variant not in TRAIN_VARIANTS:
            raise ValueError(f"unknown train.variant '{variant}' (known: {list(TRAIN_VARIANTS)})")
        for k, v in TRAIN_VARIANTS[variant].items():
            if f"train.{k}" not in explicit:
                cfg["train"][k] = v
    return explicit


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def save_fig(fig, name, dpi=300):
    """Save a matplotlib figure to results/figures/<name>.png at print resolution."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


# Categorical colours, always used in this order (colour-blind-safe set); one hue per
# entity across all figures, e.g. PPO is always POLICY_COLORS["ppo"].
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
POLICY_COLORS = {"ppo": COLORS[0], "recommended": COLORS[1], "threshold": COLORS[2],
                 "random": COLORS[3], "no_input": "#898781"}


def set_plot_style():
    """Shared matplotlib look: thin marks, light grid, no top/right spines."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "font.size": 9, "axes.titlesize": 10,
        "axes.titleweight": "bold", "axes.labelsize": 9, "legend.fontsize": 8,
        "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#c3c2b7", "axes.grid": True, "grid.color": "#e1e0d9",
        "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 2.0,
        "lines.markersize": 5, "xtick.color": "#52514e", "ytick.color": "#52514e",
        "axes.labelcolor": "#52514e", "text.color": "#0b0b0b",
        "axes.prop_cycle": mpl.cycler(color=COLORS),
    })
