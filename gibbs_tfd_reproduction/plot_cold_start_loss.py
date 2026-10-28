"""SPSA loss at beta = 5 on Tuna-17 (qubits 2, 4, 5, 8) over 40 steps, one figure per run:

  cold          cold start, all angles 0.1 (parallel_20261022_215422, continued in parallel_20261023_114448)
  reproduction  warm-start chain 0 -> 0.5 -> 1 -> 5 (parallel_20261019_175824, continued in _154440)
  intermediate  warm-start chain 1 -> 2 -> 3 -> 4 -> 5 (parallel_20261022_140225, continued in _114417)

Steps 1-20 are the original run, steps 21-40 its continuation from each run's saved SPSA state.
Loss per step = mean of SPSA's +/- evaluation pair. Thin lines: the individual runs; thick line: their
mean, with a trailing WINDOW-step running mean. Dashed line: device cost of the best (noiseless-optimal)
angles from the reference tomographies (the reproduction continuation had none, so the original run's).

    python gibbs_tfd_reproduction/plot_cold_start_loss.py
Writes experiments/2spin_cold_start/loss_cold_start.png, loss_b5_reproduction.png and loss_b5_intermediate.png.
Needs convergence.json from analyze_convergence.py for each run.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"


def run_dir(name):
    """A run directory, under experiments/<experiment>/."""
    return next(EXP.glob(f"*/{name}"))
WINDOW = 5
MUTED, GRID, SURF = "#5B6475", "#E4E3DF", "#FCFCFB"
BLUE, VIOLET, AQUA = "#2a78d6", "#6250d6", "#1baf7a"
LINESTYLES = ["-", "--", "-.", ":"]
RUNS = {
    "cold": ("parallel_20261023_114448", None, AQUA, "loss_cold_start.png",
             "Cold start at β = 5 on Tuna-17 (qubits 2, 4, 5, 8; SPSA from all angles 0.1)"),
    "reproduction": ("parallel_20261023_154440", "parallel_20261019_175824", BLUE, "loss_b5_reproduction.png",
                     "β = 5 on Tuna-17, warm-start chain 0 → 0.5 → 1 → 5 (qubits 2, 4, 5, 8)"),
    "intermediate": ("parallel_20261023_114417", None, VIOLET, "loss_b5_intermediate.png",
                     "β = 5 on Tuna-17, warm-start chain 1 → 2 → 3 → 4 → 5 (qubits 2, 4, 5, 8)"),
}


def running_mean(y, w=WINDOW):
    c = np.cumsum(np.insert(y, 0, 0.0))
    return np.array([(c[i + 1] - c[max(0, i + 1 - w)]) / (i + 1 - max(0, i + 1 - w)) for i in range(len(y))])


def beta5(run):
    conv = json.loads((run_dir(run) / "convergence.json").read_text())["betas"]
    return next(v for k, v in conv.items() if float(k) == 5.0)


def main(name):
    run, ref_run, color, out_name, title = RUNS[name]
    rec = beta5(run)
    runs = np.array([r["loss"] for r in rec["replicas"]])
    refs = rec["device_cost_best_angles"] or beta5(ref_run)["device_cost_best_angles"]
    best = np.mean([c for c, _ in refs])
    steps = np.arange(1, runs.shape[1] + 1)

    fig, ax = plt.subplots(figsize=(8.4, 4.4), facecolor=SURF)
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)

    for i, y in enumerate(runs):
        ax.plot(steps, y, color=color, lw=0.9, alpha=0.45, ls=LINESTYLES[i % len(LINESTYLES)],
                label=f"run {i + 1} (per step)")
    ax.plot(steps, running_mean(runs.mean(axis=0)), color=color, lw=2.4,
            label=f"mean of {len(runs)} independent runs ({WINDOW}-step running mean)")
    ax.axhline(best, color=MUTED, lw=1.2, ls="--", label="best angles, measured on the device"
               + ("" if rec["device_cost_best_angles"] else " (original run)"))
    lo, hi = ax.get_ylim()
    ax.set_ylim(min(lo, best - 0.15), hi)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.set(xlabel="SPSA step", ylabel="loss", xlim=(0.5, steps[-1] + 0.5))
    ax.set_title(title, fontsize=10, loc="left")
    ax.legend(fontsize=7.8, frameon=False, loc="upper right")
    fig.tight_layout()
    out = EXP / "2spin_cold_start" / out_name
    fig.savefig(out, dpi=170, facecolor=SURF)
    print("wrote", out)
    print(f"  best angles' device cost {best:.3f}; last-5 mean loss per run",
          [round(float(y[-5:].mean()), 3) for y in runs])


if __name__ == "__main__":
    for run_name in RUNS:
        main(run_name)
