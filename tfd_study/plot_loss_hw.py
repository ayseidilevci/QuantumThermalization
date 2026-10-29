"""Loss C_s(beta) along the full hardware optimization (SPSA on Tuna-17), 3+3 and 4+4 on the user's
qubits, one panel per (n, beta).

Per panel:
  measured, run r           SPSA's loss per step on Tuna-17 (mean of its +/- pair), faint raw and a
                            trailing WINDOW-step running mean
  same angles, no noise     exact loss of SPSA's iterate after each step (where the optimizer is in the
                            noiseless landscape)
  ideal simulation          exact loss of the classically optimized angles: the noiseless minimum
  classical angles, Tuna-17 loss of those angles measured on the device, from the all-X and all-Z
                            settings of the reference tomographies at the start and end of the same run
                            (readout-corrected with the calibration in use at the time)

Runs: tfd_n3p1_20261028_084445 (2 replicas), tfd_n4p1_20261028_104418 (1 replica).

    python -m tfd_study.plot_loss_hw
Writes experiments/3p3_4p4_user_qubits_comparison/loss_hw_3p3_4p4.png and prints the final losses.
"""

import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tfd_study import model  # noqa: E402

R = Path(__file__).resolve().parents[1] / "experiments"
OUT = R / "3p3_4p4_user_qubits_comparison"
RUNS = {3: "3p3_spsa_user_qubits/tfd_n3p1_20261028_084445", 4: "4p4_spsa_user_qubits/tfd_n4p1_20261028_104418"}
BETAS = [0.5, 1.0, 2.0, 5.0]
P, WINDOW, MAX_CIRCUITS = 1, 5, 5
MUTED, GRID, SURF, INK = "#5B6475", "#E4E3DF", "#FCFCFB", "#14213D"
REPLICA = ["#2a78d6", "#1baf7a"]
ORANGE = "#d9822b"


def running_mean(y, w=WINDOW):
    c = np.cumsum(np.insert(y, 0, 0.0))
    return np.array([(c[i + 1] - c[max(0, i + 1 - w)]) / min(w, i + 1) for i in range(len(y))])


def measured_reference_losses(run, n, s):
    """{(when, beta): loss} of the reference tomographies, from their all-X and all-Z settings."""
    jobs = [json.loads(line) for line in (R / run / "raw_jobs.jsonl").read_text().splitlines()]
    cals = [(datetime.fromisoformat(j["timestamp"]), j["counts"]) for j in jobs if j["task"]["kind"] == "cal"]
    settings = model.tomography_settings(n)
    ix, iz = settings.index("X" * 2 * n), settings.index("Z" * 2 * n)
    chunks = {}
    for j in jobs:
        t = j["task"]
        if t["kind"] == "tomo" and t["label"].startswith("reference"):
            chunks.setdefault(t["label"], {}).setdefault(j["chunk"], j)
    out = {}
    for label, by_chunk in chunks.items():
        def probs(i):
            j = by_chunk[i // MAX_CIRCUITS]
            when = datetime.fromisoformat(j["timestamp"])
            cal = max((c for c in cals if c[0] <= when), key=lambda c: c[0])[1]
            mats = model.readout_matrices(*(model.counts_to_probs(c, 2 * n) for c in cal), 2 * n)
            return model.correct_readout(model.counts_to_probs(j["counts"][i % MAX_CIRCUITS], 2 * n), mats)
        beta = float(label.split("beta=")[1])
        out[(label.split("/")[0].split("_")[1], beta)] = model.loss(model.loss_terms(probs(iz), probs(ix), n), beta, s)
    return out


def main():
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.2), facecolor=SURF)
    for row, n in enumerate((3, 4)):
        st = json.loads((R / RUNS[n] / "state.json").read_text())
        cfg, reps = st["cfg"], st["exp"]["replicas"]
        s = cfg["varsigma"]
        refs = measured_reference_losses(RUNS[n], n, s)
        print(f"\n{n}+{n}   beta: ideal min | classical on HW start/end | runs: measured (last {WINDOW}) / exact")
        for col, b in enumerate(BETAS):
            ax = axes[row, col]
            ideal = model.exact_loss(np.array(cfg["reference_angles"][str(b)]), n, P, b, s)
            ax.axhline(ideal, ls="--", color=MUTED, lw=1.3, label="ideal simulation (minimum)")
            hw_ref = [refs[(w, b)] for w in ("start", "end")]
            ax.axhspan(min(hw_ref), max(hw_ref), color=ORANGE, alpha=0.18, lw=0)
            ax.axhline(np.mean(hw_ref), color=ORANGE, lw=1.6, label="classical angles on Tuna-17 (start–end)")
            line = f"  β={b:<4g} {ideal:8.3f} | {hw_ref[0]:8.3f} / {hw_ref[1]:8.3f} |"
            for rep in reps:
                sp = rep["per_beta"][str(b)]["spsa"]
                meas = np.mean(sp["costs"], axis=1)
                exact = [model.exact_loss(np.array(x), n, P, b, s) for x in sp["iterates"]]
                k = np.arange(1, len(meas) + 1)
                c = REPLICA[rep["id"]]
                tag = f"run {rep['id'] + 1}"
                ax.plot(k, meas, color=c, lw=0.8, alpha=0.3)
                ax.plot(k, running_mean(meas), color=c, lw=2, label=f"{tag}: measured on Tuna-17")
                ax.plot(k, exact, color=c, lw=1.3, ls=":", label=f"{tag}: same angles, no noise")
                line += f"  {meas[-WINDOW:].mean():8.3f} / {exact[-1]:8.3f}"
            print(line)

            for sp_ in ("top", "right"):
                ax.spines[sp_].set_visible(False)
            ax.spines[["left", "bottom"]].set_color(GRID)
            ax.tick_params(colors=MUTED, labelsize=8.5)
            ax.grid(True, axis="y", color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            ax.set_facecolor(SURF)
            ax.set_title(f"{n}+{n}   β = {b:g}", fontsize=10, color=INK, loc="left")
            ax.set_xlim(0, cfg["iters"] + 1)
            if row == 1:
                ax.set_xlabel("SPSA step")
            if col == 0:
                ax.set_ylabel(r"loss $C_s(\beta)$")
    fig.suptitle(f"Loss along the full optimization on real hardware (running mean over {WINDOW} steps; "
                 "faint: every step)", fontsize=11, color=INK, x=0.01, ha="left")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, fontsize=8.5, frameon=False, loc="lower center")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out = OUT / "loss_hw_3p3_4p4.png"
    fig.savefig(out, dpi=160, facecolor=SURF)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
