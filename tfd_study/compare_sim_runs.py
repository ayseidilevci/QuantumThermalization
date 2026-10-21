"""Compare full-plan simulation runs (run_hardware.py --simulate) with the ideal simulator.

For each size n, the latest results/tfd-sim-{ideal,uniform,coherent}_n{n}p1_* run is
loaded. All three ran the same plan through the same scheduler and analysis
(readout calibration, kicks, SPSA, 3^n-setting tomography), so differences come
from the simulated device alone:
  ideal     shot noise only
  uniform   the uniform Tuna-17 model of noise_fit.json (what the plan was designed with)
  coherent  the per-coupler model with coherent CZ phase errors (noise_fit_coherent.json)

    python -m tfd_study.compare_sim_runs      # writes figures/sim_vs_ideal.png, results/sim_vs_ideal.txt
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info import DensityMatrix, state_fidelity

from tfd_study import model
from tfd_study.plotting import INK_DIM, SLOTS, apply_style

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
SIMS = {"ideal": (SLOTS[0], "o", "ideal (shots only)"),
        "uniform": (SLOTS[1], "s", "uniform noise model"),
        "coherent": (SLOTS[2], "^", "coherent per-coupler model")}


def latest_run(sim, n):
    runs = sorted((ROOT / "results").glob(f"tfd-sim-{sim}_n{n}p1_*"))
    for run in reversed(runs):
        state = json.loads((run / "state.json").read_text())
        if len(state["exp"]["tomo"]) >= len(state["cfg"]["betas"]) * (state["cfg"]["replicas"] + 2):
            return run, state
    return None, None


def rho(t, system="A"):
    return np.array(t[system]["rho"][0]) + 1j * np.array(t[system]["rho"][1])


def fid(a, b):
    return float(state_fidelity(DensityMatrix(a), DensityMatrix(b), validate=False))


def summarize(state):
    cfg, exp = state["cfg"], state["exp"]
    n, betas = cfg["n"], cfg["betas"]
    out = {"betas": betas, "ref": [], "ref_end": [], "rep": [], "rep_std": [], "angles": [], "ref_angles": []}
    for b in betas:
        ref, end = exp["tomo"][f"reference_start/beta={b}"], exp["tomo"][f"reference_end/beta={b}"]
        reps = [t for k, t in exp["tomo"].items() if k.startswith("replica") and k.endswith(f"beta={b}")]
        out["ref"].append(np.mean([ref["A"]["fid"], ref["B"]["fid"]]))
        out["ref_end"].append(np.mean([end["A"]["fid"], end["B"]["fid"]]))
        f = [np.mean([t["A"]["fid"], t["B"]["fid"]]) for t in reps]
        out["rep"].append(np.mean(f))
        out["rep_std"].append(np.std(f))
        out["angles"].append(np.mean([t["fid_exact_angles"] for t in reps]))
        out["ref_angles"].append(ref["fid_exact_angles"])
    # SPSA loss per step, all betas in sequence, averaged over replicas (mean of the +/- pair)
    traces = [np.concatenate([np.mean(rep["per_beta"][str(b)]["spsa"]["costs"], axis=1) for b in betas])
              for rep in exp["replicas"]]
    out["loss"] = np.mean(traces, axis=0)
    out["steps_per_beta"] = cfg["iters"]
    out["exact_loss_ref"] = [model.exact_loss(cfg["reference_angles"][str(b)], n, cfg["p"], b, cfg["varsigma"])
                             for b in betas]
    return out


def main():
    apply_style()
    sizes = [n for n in (3, 4) if all(latest_run(s, n)[0] for s in SIMS)]
    fig, axes = plt.subplots(len(sizes), 3, figsize=(15, 4.4 * len(sizes)), squeeze=False)
    lines = []
    for row, n in zip(axes, sizes):
        runs = {s: latest_run(s, n) for s in SIMS}
        summ = {s: summarize(st) for s, (_, st) in runs.items()}
        ideal_state = runs["ideal"][1]
        betas = summ["ideal"]["betas"]
        x = np.arange(len(betas))

        # 1: fidelity to the Gibbs state of the replicas' tomographies
        ax = row[0]
        ax.plot(x, summ["ideal"]["ref_angles"], color=INK_DIM, ls="--", lw=1.5, label="noiseless optimum (exact)")
        for s, (c, m, lab) in SIMS.items():
            d = summ[s]
            ax.errorbar(x, d["rep"], yerr=d["rep_std"], color=c, marker=m, capsize=3, label=lab)
        ends = sorted((summ[s]["rep"][-1], s) for s in SIMS)        # end labels, spread apart
        ys = [ends[0][0]]
        for v, _ in ends[1:]:
            ys.append(max(v, ys[-1] + 0.035))
        for (v, s), y in zip(ends, ys):
            ax.annotate(f"{v:.2f}", (x[-1], v), xytext=(x[-1] + 0.12, y), textcoords="data",
                        va="center", fontsize=9, color=INK_DIM)
        ax.set_xticks(x, [f"{b:g}" for b in betas])
        ax.set_xlabel("β")
        ax.set_ylabel("F(ρ, Gibbs), mean of A and B")
        ax.set_title(f"{n}+{n}: SPSA result, measured by tomography")
        ax.set_ylim(0.35, 1.0)
        ax.set_xlim(-0.3, len(betas) - 0.4)

        # 2: SPSA loss trajectory
        ax = row[1]
        k = summ["ideal"]["steps_per_beta"]
        for s, (c, m, lab) in SIMS.items():
            ax.plot(summ[s]["loss"], color=c, lw=1.5, label=lab)
        for i, (b, e) in enumerate(zip(betas, summ["ideal"]["exact_loss_ref"])):
            ax.hlines(e, i * k, (i + 1) * k - 1, color=INK_DIM, ls="--", lw=1.5)
            ax.axvline(i * k, color=INK_DIM, lw=0.6, alpha=0.4)
            ax.text(i * k + 1, 0.98, f"β={b:g}", transform=ax.get_xaxis_transform(), va="top", fontsize=8, color=INK_DIM)
        ax.set_xlabel("SPSA step (all β in sequence)")
        ax.set_ylabel("measured loss (mean over replicas)")
        ax.set_title(f"{n}+{n}: loss during optimisation")

        # 3: how far the noisy states are from the ideal ones, at identical angles
        ax = row[2]
        for s, (c, m, lab) in SIMS.items():
            if s == "ideal":
                continue
            st = runs[s][1]
            f = [np.mean([fid(rho(st["exp"]["tomo"][f"reference_start/beta={b}"], sy),
                              rho(ideal_state["exp"]["tomo"][f"reference_start/beta={b}"], sy)) for sy in "AB"])
                 for b in betas]
            ax.plot(x, f, color=c, marker=m, label=lab)
            for xi, fi in zip(x, f):
                ax.annotate(f"{fi:.2f}", (xi, fi), xytext=(0, 8 if s == "uniform" else -14), textcoords="offset points", ha="center",
                            fontsize=8, color=INK_DIM)
        ax.set_xticks(x, [f"{b:g}" for b in betas])
        ax.set_xlabel("β")
        ax.set_ylabel("F(ρ_noisy, ρ_ideal), mean of A and B")
        ax.set_title(f"{n}+{n}: state damage at the same (reference) angles")
        ax.set_ylim(0.35, 1.02)

        lines.append(f"{n}+{n}  layout {ideal_state['cfg']['layout']}")
        lines.append(f"{'beta':<6}" + "".join(f"{s + ' F_rep':>20}{'F(angles)':>11}" for s in SIMS)
                     + f"{'F_ref ideal/unif/coh':>26}{'exact optimum':>15}")
        for i, b in enumerate(betas):
            refs = "/".join(f"{summ[s]['ref'][i]:.3f}" for s in SIMS)
            lines.append(f"{b:<6g}" + "".join(f"{summ[s]['rep'][i]:>13.3f} ± {summ[s]['rep_std'][i]:.3f}"
                                              f"{summ[s]['angles'][i]:>11.3f}" for s in SIMS)
                         + f"{refs:>26}{summ['ideal']['ref_angles'][i]:>15.3f}")
        lines.append("  runs: " + ", ".join(runs[s][0].name for s in SIMS))
        lines.append("")
    handles, labels = axes[0][0].get_legend_handles_labels()
    labels = [l.replace("noiseless optimum (exact)", "exact noiseless optimum (fidelity / loss)") for l in labels]
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_fig = HERE / "figures" / "sim_vs_ideal.png"
    fig.savefig(out_fig, dpi=130)
    text = "\n".join(lines)
    (HERE / "results" / "sim_vs_ideal.txt").write_text(text + "\n")
    print(text)
    print(f"wrote {out_fig}")


if __name__ == "__main__":
    main()
