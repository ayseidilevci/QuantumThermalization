"""Shared figure style for the study notebooks (same palette as the optimizer notebook)."""

import matplotlib.pyplot as plt

# Categorical slots, always in this order; grays for references and ink.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
LAYER_COLORS = {1: SLOTS[0], 2: SLOTS[1], 3: SLOTS[2]}
LAYER_MARKERS = {1: "o", 2: "s", 3: "^"}
KIND_COLORS = {"noiseless": SLOTS[0], "noisy": SLOTS[1]}
INK, INK_DIM, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK_DIM, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK_DIM, "ytick.color": INK_DIM,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "lines.linewidth": 2, "lines.markersize": 7, "font.size": 10, "axes.titlesize": 11,
        "axes.titlelocation": "left", "legend.frameon": False,
    })


def beta_label(b):
    return "β → 0" if b < 1e-6 else f"β = {b:g}"
