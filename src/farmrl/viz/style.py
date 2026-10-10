"""One visual style for every figure, animation, map and dashboard chart.

Colours are tied to the policy *name*, never to its rank or position, so a policy keeps
its colour in every figure and when others are filtered out. The palette is a fixed,
colour-blind-safe categorical order; policies not listed below (new baselines) take the
next free slot in alphabetical order, which is stable from run to run.
"""
import matplotlib as mpl

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#ffffff"

_FIXED = {"ppo": PALETTE[0], "recommended": PALETTE[1], "threshold": PALETTE[2],
          "random": PALETTE[3], "no_input": MUTED, "ppo_untrained": PALETTE[6]}
LABELS = {"ppo": "PPO", "ppo_untrained": "PPO (untrained)", "no_input": "No inputs",
          "recommended": "Recommended", "threshold": "Threshold rules", "random": "Random"}

DRY, WET = PALETTE[1], PALETTE[0]            # season type: below / above median rain
INPUT_COLORS = {"nitrogen_kg": PALETTE[5], "irrigation_mm": PALETTE[0], "pesticide_level": PALETTE[7]}
INPUT_LABELS = {"nitrogen_kg": "Nitrogen (kg N/ha)", "irrigation_mm": "Irrigation (mm)",
                "pesticide_level": "Pesticide (dose)"}
SEQUENTIAL = "Blues"                          # magnitude: one hue, light to dark
DIVERGING = ["#eb6834", "#f0efec", "#2a78d6"]  # loss - neutral grey - gain

# Two-column LaTeX article: one column is ~3.4 in wide, the full text width ~7.0 in.
COLUMN, FULL = 3.4, 7.0


def policy_colors(policies):
    """{policy: colour} for the given names, fixed for known policies."""
    spare = [c for c in PALETTE if c not in _FIXED.values()]
    unknown = sorted(p for p in set(policies) if p not in _FIXED)
    extra = {p: spare[i % len(spare)] for i, p in enumerate(unknown)}
    return {p: _FIXED.get(p, extra.get(p)) for p in policies}


def label(policy):
    return LABELS.get(policy, policy.replace("_", " ").capitalize())


def apply():
    """Set matplotlib defaults for print-sized figures."""
    mpl.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "font.size": 7.5, "axes.titlesize": 8, "axes.titleweight": "bold", "axes.labelsize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": AXIS,
        "axes.linewidth": 0.6, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "axes.axisbelow": True, "lines.linewidth": 1.4, "lines.markersize": 4,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.labelcolor": INK_2, "text.color": INK,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
    })


def diverging_cmap():
    return mpl.colors.LinearSegmentedColormap.from_list("farmrl_div", DIVERGING)


# Plotly counterpart, used by the dashboard and the interactive map.
PLOTLY_LAYOUT = dict(
    template="simple_white", font=dict(family="Inter, Arial, sans-serif", size=13, color=INK),
    colorway=PALETTE, margin=dict(l=50, r=20, t=50, b=45), hoverlabel=dict(font_size=12),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    xaxis=dict(gridcolor=GRID, linecolor=AXIS), yaxis=dict(gridcolor=GRID, linecolor=AXIS),
)
