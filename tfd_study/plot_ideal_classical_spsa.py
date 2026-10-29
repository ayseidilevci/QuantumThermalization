"""Fidelity versus beta for 3+3 and 4+4 on the user's qubits: ideal simulation, the classically
optimized angles measured on Tuna-17, and full hardware optimization (SPSA on Tuna-17).

Series (p = 1, varsigma = 1.57, beta = 0.5, 1, 2, 5), in the paper's convention F = Tr sqrt(...):
  ideal simulation        noiseless fidelity of the classically optimized (noiseless-optimal) angles
  classical angles, HW    those angles prepared on Tuna-17 (run_mitigated, 27/28 Oct):
                            3+3 mitig-hw_n3_20261027_200737 (twirl, 2 repeats)
                            4+4 mitig-hw_n4_20261028_054813 (native, 3 repeats)
  SPSA on HW              angles optimized on Tuna-17 (run_hardware, 28/29 Oct):
                            3+3 tfd_n3p1_20261028_084445 (2 replicas)
                            4+4 tfd_n4p1_20261028_104418 (1 replica)
Each hardware point is the mean of F_A and F_B over all repeats/replicas; the bar spans the
smallest to the largest of those values. Readout correction is per-qubit in both hardware series,
so they are corrected the same way.

    python -m tfd_study.plot_ideal_classical_spsa
Writes experiments/3p3_4p4_user_qubits_comparison/ideal_classical_spsa.png and prints the table.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tfd_study import model  # noqa: E402

R = Path(__file__).resolve().parents[1] / "experiments"
OUT = R / "3p3_4p4_user_qubits_comparison"
RUNS = {3: {"mitig": "3p3_mitigated/mitig-hw_n3_20261027_200737", "spsa": "3p3_spsa_user_qubits/tfd_n3p1_20261028_084445",
            "qubits": "8, 12, 15 | 11, 14, 16", "mitig_note": "twirl, 2 repeats", "spsa_note": "2 replicas"},
        4: {"mitig": "4p4_mitigated/mitig-hw_n4_20261028_054813", "spsa": "4p4_spsa_user_qubits/tfd_n4p1_20261028_104418",
            "qubits": "4, 8, 12, 15 | 7, 11, 14, 16", "mitig_note": "native, 3 repeats",
            "spsa_note": "1 replica"}}
BETAS = [0.5, 1.0, 2.0, 5.0]
P = 1
MUTED, GRID, SURF, INK = "#5B6475", "#E4E3DF", "#FCFCFB", "#14213D"
BLUE, ORANGE = "#2a78d6", "#d9822b"


def state(run):
    return json.loads((R / run / "state.json").read_text())


def classical_hw(n, b):
    res = state(RUNS[n]["mitig"])["exp"]["results"].values()
    return np.sqrt([r["corrections"]["perqubit"][s]["fid"] for r in res if r["beta"] == b for s in "AB"])


def spsa_hw(n, b):
    tomo = state(RUNS[n]["spsa"])["exp"]["tomo"]
    return np.sqrt([v[s]["fid"] for k, v in tomo.items()
                    if k.startswith("replica") and k.endswith(f"beta={b}") for s in "AB"])


def ideal(n, b):
    theta = state(RUNS[n]["spsa"])["cfg"]["reference_angles"][str(b)]
    return float(np.sqrt(model.exact_fidelity(n, P, np.array(theta), b)))


def panel(ax, n):
    x = np.arange(len(BETAS))      # equally spaced: beta = 0.5, 1, 2, 5
    ideal_v = [ideal(n, b) for b in BETAS]
    series = [("classical angles on Tuna-17", RUNS[n]["mitig_note"], classical_hw, ORANGE, "s", -0.06),
              ("full optimization on real hardware", RUNS[n]["spsa_note"], spsa_hw, BLUE, "o", 0.06)]

    print(f"\n{n}+{n}  beta            " + "  ".join(f"{b:>15g}" for b in BETAS))
    print(f"{'ideal simulation':26s}" + "  ".join(f"{v:>15.3f}" for v in ideal_v))
    ax.plot(x, ideal_v, "--", color=MUTED, lw=1.4, label="ideal simulation")
    for name, note, fn, color, marker, dx in series:
        vals = [fn(n, b) for b in BETAS]
        mean = np.array([v.mean() for v in vals])
        lo, hi = mean - [v.min() for v in vals], [v.max() for v in vals] - mean
        print(f"{name:28s}" + "  ".join(f"{v.mean():.3f} [{v.min():.2f},{v.max():.2f}]" for v in vals))
        ax.errorbar(x + dx, mean, yerr=[lo, hi], fmt=marker + "-", color=color, ms=7, lw=2, capsize=3,
                    elinewidth=1.2, mec=SURF, mew=0.8, label=name)
        for xi, m in zip(x, mean):
            ax.annotate(f"{m:.2f}", (xi + dx, m), textcoords="offset points",
                        xytext=(-9 if dx < 0 else 9, 0), ha="right" if dx < 0 else "left", va="center",
                        fontsize=7.5, color=INK)

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_facecolor(SURF)
    ax.set_xticks(x, [f"{b:g}" for b in BETAS])
    ax.set(xlabel="inverse temperature β", xlim=(-0.45, len(BETAS) - 0.55), ylim=(0.4, 1.02))
    ax.set_title(f"{n}+{n}   (qubits {RUNS[n]['qubits']})", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=7.8, frameon=False, loc="lower left")


def main():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), facecolor=SURF, sharey=True)
    for ax, n in zip(axes, (3, 4)):
        panel(ax, n)
    axes[0].set_ylabel(r"fidelity $F=\mathrm{Tr}\sqrt{\rho_G^{1/2}\rho\,\rho_G^{1/2}}$  (mean of A and B)")
    fig.suptitle("Ideal simulation vs classical angles on Tuna-17 vs full optimization on real hardware (one layer)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    out = OUT / "ideal_classical_spsa.png"
    fig.savefig(out, dpi=170, facecolor=SURF)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
