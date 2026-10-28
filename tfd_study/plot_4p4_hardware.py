"""4+4 realization on Tuna-17: fidelity versus beta in the paper's convention F = Tr sqrt(...),
the square root of the qiskit fidelity stored in the runs.

Series (p = 1, varsigma = 1.57, beta = 0.5, 1, 2, 5; 40 SPSA steps per beta):
  Tuna-17, SPSA          the hardware run tfd_n4p1_20261022_053620 (one run), rho_A after SPSA
  Tuna-17, best angles   noiseless-optimal angles measured on the device at the start and end of the run
  ideal simulator        same plan on a noiseless simulator with shots (3 independent runs)
  best possible          noiseless fidelity of the noiseless-optimal one-layer angles
  mixed-state baseline   fidelity of the maximally mixed state I/16 with the Gibbs state

    python -m tfd_study.plot_4p4_hardware
Writes experiments/4p4_spsa_old_qubits/tfd_n4p1_20261022_053620/fidelity_paper_convention.png and prints a table of every
backend (also the QX emulator) for the report.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tfd_study import model  # noqa: E402

R = Path(__file__).resolve().parents[1] / "experiments" / "4p4_spsa_old_qubits"
HW = "tfd_n4p1_20261022_053620"
SIMS = {"ideal simulator": "tfd-sim-ideal_n4p1_20261021_151839",
        "QX emulator": "tfd-qx-emulator_n4p1_20261021_155516"}
BETAS = [0.5, 1.0, 2.0, 5.0]
N, P = 4, 1
MUTED, GRID, SURF, INK = "#5B6475", "#E4E3DF", "#FCFCFB", "#14213D"
BLUE, AQUA = "#2a78d6", "#1baf7a"


def state(run):
    return json.loads((R / run / "state.json").read_text())


def replicas(run, b):
    tomo = state(run)["exp"]["tomo"]
    return np.sqrt([v["A"]["fid"] for k, v in tomo.items() if k.startswith("replica") and k.endswith(f"beta={b}")])


def reference(run, b, when):
    return float(np.sqrt(state(run)["exp"]["tomo"][f"reference_{when}/beta={b}"]["A"]["fid"]))


def best_possible(b):
    theta = state(HW)["cfg"]["reference_angles"][str(b)]
    return float(np.sqrt(model.exact_fidelity(N, P, np.array(theta), b)))


def mixed_baseline(b):
    return float(np.sqrt(model.fidelity(np.eye(2 ** N) / 2 ** N, b, N)))


def main():
    hw = [replicas(HW, b)[0] for b in BETAS]
    ref_s = [reference(HW, b, "start") for b in BETAS]
    ref_e = [reference(HW, b, "end") for b in BETAS]
    best = [best_possible(b) for b in BETAS]
    base = [mixed_baseline(b) for b in BETAS]
    sims = {name: [replicas(run, b) for b in BETAS] for name, run in SIMS.items()}

    print("beta                       " + "  ".join(f"{b:>11g}" for b in BETAS))
    print("best possible (p = 1)      " + "  ".join(f"{x:>11.3f}" for x in best))
    for name, vals in sims.items():
        print(f"{name:27s}" + "  ".join(f"{v.mean():>5.3f}±{v.std():.3f}" if len(v) > 1 else f"{v.mean():>11.3f}"
                                         for v in vals))
    print("Tuna-17, SPSA              " + "  ".join(f"{x:>11.3f}" for x in hw))
    print("Tuna-17, best angles start " + "  ".join(f"{x:>11.3f}" for x in ref_s))
    print("Tuna-17, best angles end   " + "  ".join(f"{x:>11.3f}" for x in ref_e))
    print("mixed-state baseline       " + "  ".join(f"{x:>11.3f}" for x in base))

    fig, ax = plt.subplots(figsize=(7.6, 4.8), facecolor=SURF)
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)

    x = np.arange(len(BETAS))      # equally spaced: beta = 0.5, 1, 2, 5
    ax.plot(x, best, "--", color=MUTED, lw=1.3, label="best possible (one layer, no noise)")
    ax.plot(x, base, ":", color=MUTED, lw=1.6, label="maximally mixed state")
    ideal = sims["ideal simulator"]
    ax.errorbar(x - 0.09, [v.mean() for v in ideal], yerr=[v.std() for v in ideal], fmt="D-", color=AQUA, ms=6,
                lw=1.6, capsize=3, mec=SURF, mew=0.8, label="ideal simulator (3 runs)")
    ax.vlines(x + 0.09, ref_s, ref_e, color=BLUE, lw=2.2, alpha=0.5)
    ax.plot(x + 0.09, ref_s, "^", color=BLUE, ms=6, alpha=0.6, mec=SURF, mew=0.6,
            label="Tuna-17, best angles (start → end of run)")
    ax.plot(x + 0.09, ref_e, "v", color=BLUE, ms=6, alpha=0.6, mec=SURF, mew=0.6)
    ax.plot(x + 0.03, hw, "o-", color=BLUE, ms=7.5, lw=2, mec=SURF, mew=0.8, label="Tuna-17, SPSA (1 run)")
    for xi, h in zip(x, hw):
        ax.annotate(f"{h:.2f}", (xi + 0.03, h), textcoords="offset points",
                    xytext=(-14, -4) if xi < 2 else (12, -12) if xi == 2 else (0, -16),
                    ha="right" if xi < 2 else "left" if xi == 2 else "center",
                    fontsize=8, color=BLUE)
    ax.set_xticks(x, [f"{b:g}" for b in BETAS])
    ax.set(xlabel="inverse temperature β", ylim=(0.2, 1.02), xlim=(-0.35, len(BETAS) - 0.6),
           ylabel=r"fidelity $F=\mathrm{Tr}\sqrt{\rho_G^{1/2}\rho_{\mathrm{exp}}\rho_G^{1/2}}$",
           title="4+4 on Tuna-17 (qubits 0, 2, 5, 9 | 1, 4, 8, 12; one layer, 40 SPSA steps per β)")
    ax.title.set_fontsize(9.5)
    ax.legend(fontsize=7.8, frameon=False, loc="lower left")
    fig.tight_layout()
    out = R / HW / "fidelity_paper_convention.png"
    fig.savefig(out, dpi=170, facecolor=SURF)
    print("wrote", out)


if __name__ == "__main__":
    main()
