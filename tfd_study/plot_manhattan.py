"""Manhattan plots of a run's tomographies: bar height and colour are |rho_ij| (one hue,
light = small, dark = large), laid out like gibbs_tfd_reproduction/plot_3spin.py.

For every beta: the exact Gibbs state, rho_A measured at the noiseless-optimal reference
angles (start of the run) and rho_A measured after SPSA (one replica), each with its
fidelity to the Gibbs state. The Gibbs state is real, so each title also gives the largest
|Im rho_ij| (the phase information the colour no longer shows).

    python -m tfd_study.plot_manhattan results/tfd_n3p1_<tag> [--replica 0]
Writes <run>/manhattan_beta_<beta>.png and <run>/manhattan_all.png.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402

from tfd_study import model  # noqa: E402

# One hue, light (small |rho_ij|) to dark (large).
MAGNITUDE_CMAP = LinearSegmentedColormap.from_list("light_to_dark", ["#EEF3FA", "#8FB3E3", "#2A5DA8", "#14213D"])


def rho(entry, system="A"):
    return np.array(entry[system]["rho"][0]) + 1j * np.array(entry[system]["rho"][1])


def label(r, text):
    return text + "\n" + rf"max $|\mathrm{{Im}}\,\rho_{{ij}}|$ = {np.abs(r.imag).max():.3f}"


def panels(state, beta, replica):
    n, tomo = state["cfg"]["n"], state["exp"]["tomo"]
    ref, rep = tomo[f"reference_start/beta={beta}"], tomo[f"replica{replica}/beta={beta}"]
    return [(model.gibbs_state(n, beta), r"Exact Gibbs state $\rho_\beta$"),
            (rho(ref), rf"Tuna-17, reference angles: $F={ref['A']['fid']:.3f}$"),
            (rho(rep), rf"Tuna-17, SPSA (replica {replica}): $F={rep['A']['fid']:.3f}$")]


def plot_bars(ax, r, title, norm):
    """Manhattan plot as in plot_3spin.plot_manhattan_matrix, coloured by |rho_ij|."""
    dim = r.shape[0]
    xs, ys = np.meshgrid(np.arange(dim), np.arange(dim), indexing="ij")
    heights = np.abs(r).ravel()
    width = 0.78
    ax.bar3d(xs.ravel(), ys.ravel(), np.zeros_like(heights), width, width, heights,
             color=MAGNITUDE_CMAP(norm(heights)), shade=True, edgecolor="black", linewidth=0.12)
    labels = [format(i, f"0{int(np.log2(dim))}b") for i in range(dim)]
    ax.set_xticks(np.arange(dim) + width / 2, labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(np.arange(dim) + width / 2, labels, fontsize=7)
    ax.set_xlabel("Column basis state")
    ax.set_ylabel("Row basis state")
    ax.set_zlabel(r"$|\rho_{ij}|$")
    ax.set_zlim(0, norm.vmax * 1.08)
    ax.set_title(title, pad=12)
    ax.view_init(elev=27, azim=-55)


def draw_row(fig, axes, row, n):
    norm = Normalize(0, max(float(np.abs(r).max()) for r, _ in row))
    for ax, (r, title) in zip(axes, row):
        plot_bars(ax, r, label(r, title), norm)
    return norm


def magnitude_colorbar(fig, axes, norm):
    sm = plt.cm.ScalarMappable(norm=norm, cmap=MAGNITUDE_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.04)
    cbar.set_label(r"$|\rho_{ij}|$")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--replica", type=int, default=0)
    args = ap.parse_args()
    state = json.loads((args.run_dir / "state.json").read_text())
    cfg = state["cfg"]
    n, s, betas = cfg["n"], cfg["varsigma"], cfg["betas"]

    for b in betas:
        fig = plt.figure(figsize=(20, 7), constrained_layout=True)
        axes = [fig.add_subplot(1, 3, k + 1, projection="3d") for k in range(3)]
        norm = draw_row(fig, axes, panels(state, b, args.replica), n)
        magnitude_colorbar(fig, axes, norm)
        fig.suptitle(rf"{n}+{n} Gibbs-state tomography on Tuna-17: $\beta={b:g}$, $\varsigma={s:g}$, p = {cfg['p']}",
                     fontsize=15)
        out = args.run_dir / f"manhattan_beta_{b:g}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"wrote {out}")

    fig = plt.figure(figsize=(20, 6.2 * len(betas)), constrained_layout=True)
    axes = [fig.add_subplot(len(betas), 3, k + 1, projection="3d") for k in range(3 * len(betas))]
    for i, b in enumerate(betas):
        row = [(r, rf"$\beta={b:g}$: " + t) for r, t in panels(state, b, args.replica)]
        norm = draw_row(fig, axes[3 * i:3 * i + 3], row, n)
    magnitude_colorbar(fig, axes, norm)
    fig.suptitle(rf"{n}+{n} on Tuna-17, $\varsigma={s:g}$, p = {cfg['p']}: exact vs measured $\rho_A$", fontsize=16)
    out = args.run_dir / "manhattan_all.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
