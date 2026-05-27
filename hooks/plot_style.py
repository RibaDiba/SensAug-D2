import matplotlib as mpl
import matplotlib.pyplot as plt


# Colorblind-friendly palette (Okabe-Ito inspired)
COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]


def apply_style():
    """Set matplotlib rcParams for clean, publication-quality plots."""
    mpl.rcParams.update({
        # Font
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,

        # Figure
        "figure.facecolor": "white",
        "figure.dpi": 100,
        "savefig.dpi": 150,
        "figure.autolayout": True,

        # Lines
        "lines.linewidth": 2.0,
        "lines.markersize": 4,

        # Grid
        "axes.grid": True,
        "grid.color": "#cccccc",
        "grid.linestyle": "--",
        "grid.alpha": 0.5,

        # Axes / spines
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,

        # Color cycle
        "axes.prop_cycle": mpl.cycler(color=COLORS),

        # Legend
        "legend.frameon": True,
        "legend.framealpha": 0.85,
        "legend.edgecolor": "#cccccc",
        "legend.fontsize": 11,
        "legend.fancybox": True,

        # Ticks
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
    })
