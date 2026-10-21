"""Density matrices of system A from the full-plan simulation runs, next to the exact Gibbs state.

For each size n, the latest results/tfd-sim-{ideal,uniform,coherent}_n{n}p1_* run
is loaded (see compare_sim_runs.py). Each row is one beta; the columns are the
exact Gibbs state and the rho_A that replica 0's 3^n-setting tomography
reconstructed on each simulator. The Gibbs state of H_A is real, so Re(rho) is
shown; the largest |Im(rho)| of each panel is printed under it.

    python -m tfd_study.plot_density      # writes figures/density_n{3,4}.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tfd_study import model
from tfd_study.compare_sim_runs import SIMS, latest_run, rho
from tfd_study.plotting import apply_style

HERE = Path(__file__).resolve().parent


def basis_labels(n):
    """Bitstring A_n..A_1 of each basis index (little-endian, as in model.py)."""
    return [format(i, f"0{n}b") for i in range(2 ** n)]


def plot_size(n):
    runs = {s: latest_run(s, n) for s in SIMS}
    if not all(run for run, _ in runs.values()):
        print(f"{n}+{n}: no complete simulation runs, skipped")
        return None
    betas = runs["ideal"][1]["cfg"]["betas"]
    columns = [("exact Gibbs", None)] + [(SIMS[s][2], s) for s in SIMS]

    panels = []
    for b in betas:
        row = [(model.gibbs_state(n, b), None)]
        for s in SIMS:
            t = runs[s][1]["exp"]["tomo"][f"replica0/beta={b}"]
            row.append((rho(t, "A"), t["A"]["fid"]))
        panels.append(row)
    vmax = max(np.abs(r.real).max() for row in panels for r, _ in row)

    fig, axes = plt.subplots(len(betas), len(columns), figsize=(3.3 * len(columns), 3.4 * len(betas)),
                             squeeze=False, constrained_layout=True)
    labels = basis_labels(n)
    ticks = range(0, 2 ** n, max(1, 2 ** n // 8))
    for i, (b, row) in enumerate(zip(betas, panels)):
        for j, ((r, f), (name, _)) in enumerate(zip(row, columns)):
            ax = axes[i, j]
            im = ax.imshow(r.real, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.grid(False)
            ax.set_xticks(list(ticks), [labels[k] for k in ticks], rotation=90, fontsize=7)
            ax.set_yticks(list(ticks), [labels[k] for k in ticks], fontsize=7)
            stats = f"max |Im ρ| = {np.abs(r.imag).max():.3f}"
            if f is not None:
                stats = f"F = {f:.3f}   " + stats
            ax.set_title((name + "\n" if i == 0 else "") + stats, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"β = {b:g}", fontsize=11)
    fig.colorbar(im, ax=axes, shrink=0.5, label="Re ρ_A")
    fig.suptitle(f"{n}+{n}, p = 1: ρ_A from tomography (replica 0) vs the exact Gibbs state", x=0.02, ha="left")
    out = HERE / "figures" / f"density_n{n}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"{n}+{n}: wrote {out.relative_to(HERE.parent)}  (runs: {', '.join(r.name for r, _ in runs.values())})")
    return out


def main():
    apply_style()
    for n in (3, 4):
        plot_size(n)


if __name__ == "__main__":
    main()
