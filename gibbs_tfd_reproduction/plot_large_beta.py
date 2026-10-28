"""Why the application stops at beta = 5: theory and the Tuna-17 cold starts at beta = 10, 20, 50, 100.

Figure 1 (results/large_beta_theory.png), exact, 2 spins:
  (a) fidelity of the Gibbs state at beta with the ground state (beta = 1000), and the ground-state
      population, for beta = 0.1 ... 100
  (b) weight beta^-varsigma of the cost term that couples A and B
Figure 2 (results/large_beta_hardware.png), Tuna-17, qubits 12, 14, 15, 16, one cold-start run per beta
(20 SPSA steps from all angles 0.1):
  (a) fidelity of rho_A and rho_B, and the noiseless fidelity of the same angles, versus beta
  (b) the six largest elements |rho_ij| of rho_A, measured at each beta against the exact state
Figure 3 (results/large_beta_manhattan.png): exact rho_A (the same for every beta >= 10) next to the
measured rho_A at beta = 10, 20, 50, 100.
Fidelities in the paper's convention F = Tr sqrt(rho_G^1/2 rho rho_G^1/2).

    python gibbs_tfd_reproduction/plot_large_beta.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from qiskit.quantum_info import DensityMatrix, state_fidelity  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "gibbs_tfd_reproduction"))
from run_hardware_parallel import VARSIGMA, gibbs_target  # noqa: E402
from tfd_study.plot_manhattan import MAGNITUDE_CMAP, plot_bars  # noqa: E402

R = ROOT / "experiments" / "2spin_large_beta"
RUNS = {10: "parallel_20261023_181044", 20: "parallel_20261023_181058",
        50: "parallel_20261023_181113", 100: "parallel_20261023_181128"}
MUTED, GRID, SURF, INK = "#5B6475", "#E4E3DF", "#FCFCFB", "#14213D"
BLUE, VIOLET, AQUA, ORANGE = "#2a78d6", "#6250d6", "#1baf7a", "#eb6834"
RAMP = ["#86b6ef", "#2a78d6", "#104281", "#0b0b0b"]          # beta = 10, 20, 50, 100 (light -> dark)
GROUND = 1000.0


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def paper_fid(a, b):
    return float(np.sqrt(state_fidelity(DensityMatrix(a), DensityMatrix(b), validate=False)))


def rho(entry, system="A"):
    return np.array(entry[system]["rho"][0]) + 1j * np.array(entry[system]["rho"][1])


def run_data(b):
    st = json.loads((R / RUNS[b] / "state.json").read_text())
    e = st["exp"]["tomo"][f"replica0/beta={float(b)}"]
    noiseless = np.sqrt(st["exp"]["replicas"][0]["per_beta"][str(float(b))]["fid_exact_angles"])
    return rho(e, "A"), np.sqrt(e["A"]["fid"]), np.sqrt(e["B"]["fid"]), noiseless


def theory():
    betas = np.logspace(-1, 2, 200)
    g_inf = gibbs_target(GROUND)
    fid = [paper_fid(gibbs_target(b), g_inf) for b in betas]
    pop = [np.linalg.eigvalsh(gibbs_target(b)).max() for b in betas]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.9), facecolor=SURF)
    for ax in (a, b):
        style(ax)
        ax.set_xscale("log")
        ax.axvline(5, color=ORANGE, lw=1.2, ls="--")
    a.plot(betas, fid, color=BLUE, lw=2.2, label="fidelity with the ground state")
    a.plot(betas, pop, color=VIOLET, lw=2.2, ls="-.", label="ground-state population")
    a.text(5.4, 0.42, "β = 5", color=ORANGE, fontsize=8.5)
    a.annotate(f"{paper_fid(gibbs_target(5), g_inf):.3f}", (5, paper_fid(gibbs_target(5), g_inf)),
               textcoords="offset points", xytext=(8, -16), ha="left", fontsize=8, color=BLUE)
    a.set(xlabel="inverse temperature β", ylabel="exact value", ylim=(0.2, 1.03),
          title="(a) Gibbs state of system A against its ground state")
    a.legend(fontsize=8, frameon=False, loc="lower right")
    b.plot(betas, betas ** -VARSIGMA, color=BLUE, lw=2.2)
    for x in (5, 10, 20, 50, 100):
        b.plot(x, x ** -VARSIGMA, "o", color=BLUE, ms=6, mec=SURF, mew=0.8)
        b.annotate(f"{x ** -VARSIGMA:.1e}" if x > 5 else f"{x ** -VARSIGMA:.2f}", (x, x ** -VARSIGMA),
                   textcoords="offset points", xytext=(6, 4), fontsize=7.5, color=INK)
    b.set_yscale("log")
    b.set(xlabel="inverse temperature β", ylabel=r"$\beta^{-\varsigma}$",
          title=r"(b) weight of the A–B coupling term in the cost ($\varsigma$ = 1.57)")
    for ax in (a, b):
        ax.title.set_fontsize(9.5)
    fig.tight_layout()
    out = R / "large_beta_theory.png"
    fig.savefig(out, dpi=170, facecolor=SURF)
    print("wrote", out)


ELEMENTS = [((0, 0), r"$\rho_{00,00}$"), ((1, 1), r"$\rho_{01,01}$"), ((2, 2), r"$\rho_{10,10}$"),
            ((3, 3), r"$\rho_{11,11}$"), ((1, 2), r"$|\rho_{01,10}|$"), ((0, 1), r"$|\rho_{00,01}|$")]


def hardware():
    data = {b: run_data(b) for b in RUNS}
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.2), facecolor=SURF, gridspec_kw={"width_ratios": [1, 1.25]})
    style(a)
    style(b)
    x = np.arange(len(RUNS))
    a.plot(x, [d[3] for d in data.values()], "D--", color=MUTED, ms=6, lw=1.3, label="same angles without noise")
    a.plot(x, [d[1] for d in data.values()], "o-", color=BLUE, ms=7.5, lw=2, mec=SURF, mew=0.8,
           label="Tuna-17, system A")
    a.plot(x, [d[2] for d in data.values()], "s-", color=AQUA, ms=6.5, lw=1.6, mec=SURF, mew=0.8,
           label="Tuna-17, system B")
    for xi, d in zip(x, data.values()):
        a.annotate(f"{d[1]:.2f}", (xi, d[1]), textcoords="offset points", xytext=(0, -17), ha="center",
                   fontsize=8, color=BLUE)
    a.set_xticks(x, [f"{k:g}" for k in RUNS])
    a.set(xlabel="inverse temperature β", ylim=(0.5, 1.02), xlim=(-0.45, len(RUNS) - 0.6),
          ylabel=r"fidelity $F=\mathrm{Tr}\sqrt{\rho_G^{1/2}\rho\,\rho_G^{1/2}}$",
          title="(a) fidelity after a cold start (one run per β)")
    a.legend(fontsize=8, frameon=False, loc="lower left")

    exact = gibbs_target(GROUND)
    width = 0.15
    xe = np.arange(len(ELEMENTS))
    b.bar(xe - 2 * width, [abs(exact[i, j]) for (i, j), _ in ELEMENTS], width, color="#c8cdd6", edgecolor=MUTED,
          lw=0.6, hatch="///", label="exact (every β ≥ 10)")
    for k, (bb, d) in enumerate(data.items()):
        b.bar(xe + (k - 1) * width, [abs(d[0][i, j]) for (i, j), _ in ELEMENTS], width, color=RAMP[k],
              label=f"Tuna-17, β = {bb:g}")
    b.set_xticks(xe, [lab for _, lab in ELEMENTS], fontsize=9)
    b.set(ylabel=r"$|\rho_{ij}|$ of system A", ylim=(0, 0.5),
          title="(b) largest density-matrix elements: exact vs measured")
    b.legend(fontsize=7.8, frameon=False, ncol=3, loc="upper right")
    for ax in (a, b):
        ax.title.set_fontsize(9.5)
    fig.suptitle("Cold starts at β = 10, 20, 50, 100 on Tuna-17 (qubits 12, 14, 15, 16; 20 SPSA steps)",
                 fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = R / "large_beta_hardware.png"
    fig.savefig(out, dpi=170, facecolor=SURF)
    print("wrote", out)
    for bb, d in data.items():
        print(f"beta={bb:<4g} F_A={d[1]:.3f} F_B={d[2]:.3f} noiseless={d[3]:.3f}  "
              + "  ".join(f"{abs(d[0][i, j]):.3f}" for (i, j), _ in ELEMENTS))
    print("exact            " + "  ".join(f"{abs(exact[i, j]):.3f}" for (i, j), _ in ELEMENTS))


def manhattan():
    data = {b: run_data(b) for b in RUNS}
    exact = gibbs_target(GROUND)
    norm = Normalize(0, max([np.abs(exact).max()] + [np.abs(d[0]).max() for d in data.values()]))
    fig = plt.figure(figsize=(22, 5.2), facecolor=SURF)
    axes = [fig.add_subplot(1, 5, k + 1, projection="3d") for k in range(5)]
    plot_bars(axes[0], exact, "Exact Gibbs state\n(the same for every β ≥ 10)", norm)
    for ax, (bb, d) in zip(axes[1:], data.items()):
        plot_bars(ax, d[0], f"Tuna-17, β = {bb:g}\nF = {d[1]:.3f}", norm)
    for ax in axes:
        ax.title.set_fontsize(11)
        ax.set_zlabel("")
        ax.set_zticks([0, 0.1, 0.2, 0.3, 0.4])
        ax.tick_params(axis="z", labelsize=7)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=MAGNITUDE_CMAP)
    sm.set_array([])
    fig.subplots_adjust(left=0.0, right=0.93, top=0.80, bottom=0.04, wspace=0.02)
    cax = fig.add_axes([0.945, 0.25, 0.008, 0.5])
    fig.colorbar(sm, cax=cax).set_label(r"$|\rho_{ij}|$")
    fig.suptitle("System A on Tuna-17 (qubits 12, 14, 15, 16) after a cold start at β = 10, 20, 50, 100, "
                 r"against the exact state;  F = Tr$\sqrt{\rho_G^{1/2}\rho\,\rho_G^{1/2}}$",
                 fontsize=13, x=0.01, y=0.98, ha="left")
    out = R / "large_beta_manhattan.png"
    fig.savefig(out, dpi=140, facecolor=SURF)
    print("wrote", out)


if __name__ == "__main__":
    theory()
    hardware()
    manhattan()
