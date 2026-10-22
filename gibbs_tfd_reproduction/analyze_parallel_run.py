"""Plot and tabulate a run_hardware_parallel.py run (works on partial runs too).

  python gibbs_tfd_reproduction/analyze_parallel_run.py results/parallel_<tag>
  python gibbs_tfd_reproduction/analyze_parallel_run.py results/parallel_<tag> \
      --previous results/hardware_20261019_001940     # overlay the single chain run

Writes <run dir>/analysis.png and <run dir>/analysis.txt. Fidelities are qiskit's
state_fidelity, F = (Tr sqrt(...))^2. The paper (Sagastizabal et al. 2021) reports
Tr sqrt(...), so its values are drawn squared to put them on the same scale.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# categorical slots 1-4 of the dataviz skill's validated default palette, fixed order
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

# paper's reported fidelities (Tr sqrt convention) at beta = 0, 1, 5
PAPER = {1e-08: 0.99, 1.0: 0.92, 5.0: 0.75}


def load(run_dir):
    st = json.loads((run_dir / "state.json").read_text())
    return st["cfg"], st["exp"]


def collect(cfg, exp):
    rows = []
    for beta in cfg["betas"]:
        reps = [exp["tomo"].get(f"replica{r}/beta={beta}") for r in range(cfg["replicas"])]
        ref_s = exp["tomo"].get(f"reference_start/beta={beta}")
        ref_e = exp["tomo"].get(f"reference_end/beta={beta}")
        exact = [rep["per_beta"][str(beta)]["fid_exact_angles"]
                 for rep in exp["replicas"] if str(beta) in rep["per_beta"]]
        rows.append({
            "beta": beta,
            "A": [r["A"]["fid"] for r in reps if r], "B": [r["B"]["fid"] for r in reps if r],
            "ref_start": ref_s, "ref_end": ref_e, "exact": exact,
            "sim_exact": (ref_s or ref_e or {}).get("fid_exact_angles"),
        })
    return rows


def previous_run(path):
    if path is None or not (path / "state.json").exists():
        return {}
    st = json.loads((path / "state.json").read_text())
    return {float(b): e for b, e in st["betas"].items() if e.get("done")}


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--previous", type=Path, default=Path("results/hardware_20261019_001940"),
                   help="single chain run (run_hardware.py) to overlay; system A only")
    args = p.parse_args()

    cfg, exp = load(args.run_dir)
    rows = collect(cfg, exp)
    prev = previous_run(args.previous)
    n_tomo, n_total = len(exp["tomo"]), len(cfg["betas"]) * (cfg["replicas"] + 2)
    x = np.arange(len(rows))
    labels = ["≈0" if r["beta"] < 1e-3 else f"{r['beta']:g}" for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), facecolor=SURFACE)
    rng = np.random.default_rng(0)

    for ax, system in zip(axes[:2], "AB"):
        style(ax, f"System {system}: fidelity to the Gibbs state on Tuna-17", "fidelity (qiskit F)")
        for i, r in enumerate(rows):
            vals = r[system]
            if vals:
                ax.scatter(i + rng.uniform(-0.12, 0.12, len(vals)), vals, s=22, color=BLUE,
                           alpha=0.35, linewidths=0, zorder=2)
                ax.errorbar(i, np.mean(vals), yerr=np.std(vals) if len(vals) > 1 else None,
                            fmt="o", ms=8, color=BLUE, capsize=5, lw=2, zorder=3,
                            label=f"SPSA from 0.1, replicas: mean ± std" if i == 0 else None)
            for key, dx, face, lab in (("ref_start", 0.24, ORANGE, "reference: simulator angles (start)"),
                                       ("ref_end", 0.38, "none", "reference: simulator angles (end)")):
                if r[key]:
                    ax.scatter(i + dx, r[key][system]["fid"], marker="s", s=60, facecolors=face,
                               edgecolors=ORANGE, linewidths=2, zorder=3, label=lab)
            if system == "A" and r["beta"] in prev:
                ax.scatter(i - 0.28, prev[r["beta"]]["fid_final"], marker="D", s=55, color=AQUA,
                           edgecolors=SURFACE, linewidths=1.5, zorder=3,
                           label="single chain run, night 1 (system A)")
            if r["beta"] in PAPER:
                ax.scatter(i, PAPER[r["beta"]] ** 2, marker="^", s=70, color=YELLOW,
                           edgecolors=INK_2, linewidths=1, zorder=3, label="paper (squared to qiskit F)")
        ax.set_xticks(x, labels)
        ax.set_xlabel("β", color=INK_2)
        ax.set_ylim(0.45, 1.02)

    ax = axes[2]
    style(ax, "Angle quality: same angles without noise", "fidelity (qiskit F, exact)")
    for i, r in enumerate(rows):
        if r["exact"]:
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(r["exact"])), r["exact"], s=22, color=BLUE,
                       alpha=0.35, linewidths=0, zorder=2)
            ax.errorbar(i, np.mean(r["exact"]), yerr=np.std(r["exact"]) if len(r["exact"]) > 1 else None,
                        fmt="o", ms=8, color=BLUE, capsize=5, lw=2, zorder=3,
                        label="SPSA replicas: mean ± std" if i == 0 else None)
        if r["sim_exact"]:
            ax.scatter(i + 0.28, r["sim_exact"], marker="s", s=60, color=ORANGE, zorder=3,
                       label="simulator optimum (best the cost allows)" if i == 0 else None)
        if r["beta"] in prev:
            ax.scatter(i - 0.28, prev[r["beta"]]["fid_final_exact"], marker="D", s=55, color=AQUA,
                       edgecolors=SURFACE, linewidths=1.5, zorder=3,
                       label="single chain run, night 1" if i == 0 else None)
    ax.set_xticks(x, labels)
    ax.set_xlabel("β", color=INK_2)
    ax.set_ylim(0.8, 1.005)

    # one legend per panel, without duplicate entries
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        seen = dict(zip(l, h))
        ax.legend(seen.values(), seen.keys(), fontsize=7.5, frameon=False, loc="lower left",
                  labelcolor=INK_2)

    status = "complete" if n_tomo >= n_total else f"partial: {n_tomo} of {n_total} tomographies"
    fig.suptitle(f"Variational TFD/Gibbs preparation on Tuna-17 — {cfg['replicas']} SPSA replicas, "
                 f"chain from 0.1, {cfg['tomography']}-setting tomography ({status})",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = args.run_dir / "analysis.png"
    fig.savefig(out, dpi=140, facecolor=SURFACE)
    plt.close(fig)

    # table
    f = lambda v: "   -  " if v is None else f"{v:.3f}"
    ms = lambda v: "      -      " if not v else (f"{np.mean(v):.3f} ± {np.std(v):.3f}" if len(v) > 1 else f"{v[0]:.3f}        ")
    lines = [f"run {args.run_dir.name} ({status}); qiskit fidelity F = (Tr sqrt)^2",
             "",
             "beta   replicas F_A (n)      replicas F_B        ref start A/B   ref end A/B     "
             "night-1 A   angles exact (replicas)   sim optimum exact"]
    for r in rows:
        rs, re = r["ref_start"], r["ref_end"]
        lines.append(
            f"{labels[rows.index(r)]:<6} {ms(r['A'])} ({len(r['A'])})  {ms(r['B'])}   "
            f"{f(rs and rs['A']['fid'])}/{f(rs and rs['B']['fid'])}   "
            f"{f(re and re['A']['fid'])}/{f(re and re['B']['fid'])}   "
            f"{f(prev.get(r['beta'], {}).get('fid_final'))}       {ms(r['exact'])}             {f(r['sim_exact'])}")
    text = "\n".join(lines)
    (args.run_dir / "analysis.txt").write_text(text + "\n")
    print(text)
    print(f"\nwrote {out} and {args.run_dir / 'analysis.txt'}")


if __name__ == "__main__":
    main()
