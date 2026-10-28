"""Our 2+2 warm-start chain on Tuna-17 versus Sagastizabal et al. (npj Quantum Inf. 7, 130).

The chain is continuous: the original run went beta ~ 0 -> 0.5 -> 1 (parallel_20261019_175824)
and the follow-up started each replica from its own beta = 1 angles and went 2 -> 3 -> 4 -> 5
(parallel_20261022_140225); 20 SPSA steps per beta, 4 replicas, qubits 2, 4, 5, 8.

The paper defines the fidelity as F = Tr sqrt(rho_G^1/2 rho_exp rho_G^1/2), without the square
used elsewhere in this repository; its values (99 %, 92 %, 75 % at beta = 0, 1, 5) are compared
here with our data in that same convention (the square root of our F, per replica).

    python gibbs_tfd_reproduction/plot_vs_paper.py
Writes experiments/2spin_cold_start/chain_vs_paper.png and chain_vs_paper_no_cold.png, and
experiments/2spin_reproduction_chain/reproduction_vs_paper.png and reproduction_vs_paper_015.png (beta ~ 0, 1, 5 only, the temperatures the paper reports).
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tfd_study import model  # noqa: E402

EXP = ROOT / "experiments"


def run_dir(name):
    """A run directory, under experiments/<experiment>/."""
    return next(EXP.glob(f"*/{name}"))
PARTS = [("parallel_20261019_175824", ["1e-08", "0.5", "1"]),
         ("parallel_20261022_140225", ["2.0", "3.0", "4.0", "5.0"])]
PAPER = {0.0: 0.99, 1.0: 0.92, 5.0: 0.75}
MUTED, GRID, SURF, INK = "#5B6475", "#E4E3DF", "#FCFCFB", "#14213D"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
COLD = "parallel_20261022_215422"   # cold start straight at beta = 5 (no chain), same qubits and settings


def main(cold_start=True, reproduction_only=False, paper_betas=False):
    beta, F_mean, F_std, S_mean, S_std = [], [], [], [], []
    parts = [(PARTS[0][0], PARTS[0][1] + ["5"])] if reproduction_only else PARTS
    if paper_betas:     # only the temperatures the paper reports (the chain still ran through beta = 0.5)
        parts = [(PARTS[0][0], ["1e-08", "1", "5"])]
    for run, keys in parts:
        tomo = json.loads((run_dir(run) / "state.json").read_text())["exp"]["tomo"]
        for k in keys:
            f = np.array([v["A"]["fid"] for name, v in tomo.items() if name.startswith("replica") and name.endswith(f"beta={k}")])
            beta.append(0.0 if float(k) < 1e-3 else float(k))
            F_mean.append(f.mean()); F_std.append(f.std())
            S_mean.append(np.sqrt(f).mean()); S_std.append(np.sqrt(f).std())

    fig, ax = plt.subplots(figsize=(7.6, 4.8), facecolor=SURF)
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)

    ax.errorbar(beta, S_mean, yerr=S_std, fmt="o-", color=BLUE, ms=7, lw=2, capsize=3, mec=SURF, mew=0.8,
                label=("this work (warm-start chain 0 → 0.5 → 1 → 5; β = 0.5 not shown)" if paper_betas else
                       "this work (warm-start chain 0 → 0.5 → 1 → 5)" if reproduction_only else "warm start chain"))
    if cold_start:
        cold_tomo = json.loads((run_dir(COLD) / "state.json").read_text())["exp"]["tomo"]
        cold = np.sqrt([v["A"]["fid"] for name, v in cold_tomo.items() if name.startswith("replica")])
        ax.errorbar(5.15, cold.mean(), yerr=cold.std(), fmt="D", color=AQUA, ms=7, capsize=3, mec=SURF, mew=0.8,
                    label="cold start")
        ax.annotate(f"{cold.mean():.2f}", (5.15, cold.mean()), textcoords="offset points", xytext=(0, -16),
                    ha="center", fontsize=8, color=AQUA)
        print(f"cold start beta=5: sqrtF = {cold.mean():.3f} ± {cold.std():.3f} (runs: {np.round(cold, 3).tolist()})")
    pb = sorted(PAPER)
    ax.plot(pb, [PAPER[b] for b in pb], "*", color=ORANGE, ms=17, mec=INK, mew=0.6,
            label="Sagastizabal et al. 2021")
    for b, s_ in zip(beta, S_mean):
        below_star = b in PAPER and PAPER[b] < s_
        ax.annotate(f"{s_:.2f}", (b, s_), textcoords="offset points",
                    xytext=(18, -3) if b > 4 and reproduction_only else (0, 9 if below_star else -16), ha="center",
                    fontsize=8, color=BLUE)
    for b in pb:
        ax.annotate(f"{PAPER[b]:.2f}", (b, PAPER[b]), textcoords="offset points", xytext=(13, 4),
                    fontsize=8, color=ORANGE)
    ax.set(xlabel="inverse temperature β", ylabel=r"fidelity $F=\mathrm{Tr}\sqrt{\rho_G^{1/2}\rho_{\mathrm{exp}}\rho_G^{1/2}}$", ylim=(0.45, 1.03),
           title=("Reproduction on Tuna-17 (qubits 2, 4, 5, 8; 4 independent runs, 20 SPSA steps per β)"
                  if reproduction_only else
                  "2+2 warm-start chain on Tuna-17 (qubits 2, 4, 5, 8; 4 independent runs, 20 SPSA steps per β)"))
    ax.title.set_fontsize(10)
    ticks = [0, 1, 5] if paper_betas else [0, 0.5, 1, 5] if reproduction_only else [0, 0.5, 1, 2, 3, 4, 5]
    ax.set_xticks(ticks, ["≈0" if t == 0 else f"{t:g}" for t in ticks])
    ax.set_xlim(-0.25, 5.45)
    ax.legend(fontsize=7.8, frameon=False, loc="lower left")
    fig.tight_layout()
    out = EXP / ("2spin_reproduction_chain" if reproduction_only or paper_betas else "2spin_cold_start") / ("reproduction_vs_paper_015.png" if paper_betas else "reproduction_vs_paper.png" if reproduction_only else
               "chain_vs_paper.png" if cold_start else "chain_vs_paper_no_cold.png")
    fig.savefig(out, dpi=170, facecolor=SURF)
    print("wrote", out)
    for b, f, fs, s in zip(beta, F_mean, F_std, S_mean):
        print(f"beta={b:<4g} F={f:.3f}±{fs:.3f}  sqrtF={s:.3f}  paper={PAPER.get(b, '-')}")


if __name__ == "__main__":
    main()
    main(cold_start=False)
    main(cold_start=False, reproduction_only=True)
    main(cold_start=False, reproduction_only=True, paper_betas=True)
