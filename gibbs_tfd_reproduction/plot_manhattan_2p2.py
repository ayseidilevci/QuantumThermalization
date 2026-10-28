"""Manhattan plots of the 2+2 reproduction chain on Tuna-17 against the exact Gibbs state.

Top row: exact Gibbs state of system A. Bottom row: measured rho_A, the mean of the four independent
runs' reconstructed density matrices at the end of SPSA (20 steps). Bar height and shade are |rho_ij|
(light = small, dark = large), as in tfd_study/plot_manhattan.py. Fidelities in the paper's
convention F = Tr sqrt(rho_G^1/2 rho rho_G^1/2): the mean over runs, and the fidelity of the
averaged matrix shown.

    python gibbs_tfd_reproduction/plot_manhattan_2p2.py [run_dir]
Writes <run_dir>/manhattan_2p2.png (default run: parallel_20261019_175824).
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "gibbs_tfd_reproduction"))
from run_hardware_parallel import fidelities, gibbs_target  # noqa: E402
from tfd_study.plot_manhattan import MAGNITUDE_CMAP, plot_bars  # noqa: E402

SURF = "#FCFCFB"
KEYS = [("1e-08", "β ≈ 0"), ("0.5", "β = 0.5"), ("1", "β = 1"), ("5", "β = 5")]


def rho(entry):
    return np.array(entry["A"]["rho"][0]) + 1j * np.array(entry["A"]["rho"][1])


def lowest_cost_run(run, key):
    """Index of the run with the lowest device cost at its final angles (from analyze_convergence.py);
    a choice that needs no knowledge of the exact Gibbs state."""
    rec = json.loads((run / "convergence.json").read_text())["betas"][key]
    costs = [r["device_cost_final"][0] for r in rec["replicas"]]
    return rec["replicas"][int(np.argmin(costs))]["id"]


def main(run, best=False, by_fidelity=False):
    tomo = json.loads((run / "state.json").read_text())["exp"]["tomo"]
    fig = plt.figure(figsize=(18, 8.6), facecolor=SURF)
    axes = [[fig.add_subplot(2, len(KEYS), r * len(KEYS) + c + 1, projection="3d") for c in range(len(KEYS))]
            for r in range(2)]
    for c, (key, label) in enumerate(KEYS):
        beta = float(key)
        runs = [rho(v) for k, v in tomo.items() if k.startswith("replica") and k.endswith(f"beta={key}")]
        if best:
            i = (int(np.argmax(f_runs_all := [fidelities(rho(tomo[f"replica{j}/beta={key}"]), beta)[1]
                                              for j in range(len(runs))]))
                 if by_fidelity else lowest_cost_run(run, key))
            measured = rho(tomo[f"replica{i}/beta={key}"])
            f_sel = fidelities(measured, beta)[1]
        else:
            measured = np.mean(runs, axis=0)
        exact = gibbs_target(beta)
        f_runs = [fidelities(r, beta)[1] for r in runs]
        f_mean_matrix = fidelities(measured, beta)[1]
        norm = Normalize(0, max(np.abs(exact).max(), np.abs(measured).max()))
        plot_bars(axes[0][c], exact, f"{label}: exact Gibbs state", norm)
        if best:
            title = f"{label}: Tuna-17, best of {len(runs)} runs (run {i + 1})\nF = {f_sel:.3f}"
        else:
            title = f"{label}: Tuna-17 (mean of {len(runs)} runs)\nF = {np.mean(f_runs):.3f} ± {np.std(f_runs):.3f}"
        plot_bars(axes[1][c], measured, title, norm)
        for ax in (axes[0][c], axes[1][c]):
            ax.title.set_fontsize(9.5)
            ax.set_zlabel("")
    sm = plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=MAGNITUDE_CMAP)
    sm.set_array([])
    fig.subplots_adjust(left=0.0, right=0.9, top=0.86 if best else 0.9, bottom=0.03, wspace=0.05, hspace=0.18)
    cax = fig.add_axes([0.92, 0.3, 0.012, 0.4])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(r"$|\rho_{ij}|$, relative to the largest element at that β")
    fig.suptitle("2+2 reproduction chain on Tuna-17 (qubits 2, 4, 5, 8; 20 SPSA steps per β) vs the exact Gibbs state;"
                 r"  F = Tr$\sqrt{\rho_G^{1/2}\rho\,\rho_G^{1/2}}$"
                 + (("\nBottom row: the run with the highest fidelity at each β (best case, selected by fidelity)"
                    if by_fidelity else "\nBottom row: the run with the lowest measured cost at each β") if best else ""),
                 fontsize=12, x=0.01, ha="left")
    out = run / (("manhattan_2p2_best_fidelity.png" if by_fidelity else "manhattan_2p2_best.png")
                 if best else "manhattan_2p2.png")
    fig.savefig(out, dpi=150, facecolor=SURF)
    print("wrote", out)
    for key, label in KEYS:
        beta = float(key)
        runs = [rho(v) for k, v in tomo.items() if k.startswith("replica") and k.endswith(f"beta={key}")]
        print(f"{label}: per-run F = {[round(fidelities(r, beta)[1], 3) for r in runs]}")


if __name__ == "__main__":
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "experiments" / "2spin_reproduction_chain" / "parallel_20261019_175824"
    main(run_dir)
    main(run_dir, best=True)
    main(run_dir, best=True, by_fidelity=True)
