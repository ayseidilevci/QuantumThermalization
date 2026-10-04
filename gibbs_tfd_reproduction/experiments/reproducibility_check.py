"""Reproducibility check for experiment.py's actual pipeline (not the
noiseless shortcut in verify_exact_cost.py / beta_zero_sanity_check.ipynb).

For each beta in BETAS, runs the same shot-based CostFunction (cost_function.py)
against the same NoisySimulator backend experiment.py uses, from N_TRIALS
independent random nonzero initial guesses, then reconstructs rho_A via real
state tomography (tomography.py) and reports the fidelity to the true Gibbs
state each trial converges to -- i.e. does this pipeline reliably land on a
good answer regardless of where the optimizer starts, or does it sometimes
get stuck?

For each beta, saves three things to plots/:
  - exact_rho_A_beta{beta}.png       the true (analytic) Gibbs-state rho_A
  - avg_tomography_beta{beta}.png    the experimental result: elementwise
                                      average of all N_TRIALS reconstructed
                                      rho_A's, same manhattan-plot style
  - fidelities_beta{beta}.png        bar chart of all N_TRIALS individual
                                      trial fidelities

This is slow: each trial is a real COBYLA optimization over a shot-sampled
cost function (~8-13s observed) plus a tomography reconstruction (~0.2s), so
N_TRIALS=25 at one beta takes roughly 4-6 minutes. Multiply by len(BETAS)
for the total.

Run from this folder:  python reproducibility_check.py
"""

import sys
import time

sys.path.insert(0, '.')  # run from this folder

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from qiskit.quantum_info import DensityMatrix, state_fidelity
from scipy.optimize import minimize

from qi_utilities.device_simulation.simulators import NoisySimulator
from cost_function import CostFunction
from tomography import run_tomography, reconstruct_density_matrix, plot_density_matrix
from ground_truth import GibbsState, default_subsystem_hamiltonian

PLOTS_DIR = Path("plots")

VARSIGMA = 1.57
INITIAL_LAYOUT = [0, 2, 3, 5]
N_TRIALS = 25
BETAS = [1e-8, 1, 5]
SHOTS = 8192
FIDELITY_THRESHOLD = 0.95
RNG_SEED = 42


def plot_fidelities(fidelities, beta, save_path):
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.bar(range(len(fidelities)), fidelities, color="tab:blue")
    ax.axhline(1.0, color="tab:green", linestyle="--", linewidth=1, label="fidelity = 1")
    lo = min(fidelities)
    ax.set_ylim(max(0.0, lo - 0.05), 1.05)
    ax.set_xlabel("trial (random init)")
    ax.set_ylabel("fidelity to true rho_A")
    ax.set_title(f"Fidelity across {len(fidelities)} random initializations, beta={beta}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def random_nonzero_x0(rng):
    # each component in [-pi,-0.1] U [0.1,pi] -- nonzero on purpose, so we're
    # testing recovery from a genuinely displaced starting point each time
    magnitude = rng.uniform(0.1, np.pi, size=4)
    sign = rng.choice([-1, 1], size=4)
    return magnitude * sign


def run_one_trial(backend, beta, x0, target):
    cf = CostFunction(varsigma=VARSIGMA, backend=backend, initial_layout=INITIAL_LAYOUT)
    cf.circuit_pdf = None  # disable cost_function.py's per-call circuit-diagram side effect

    result = minimize(cf.cost_function, x0=x0, args=(beta,), method='COBYLA',
                       options={'maxiter': 100, 'rhobeg': 0.5})

    corrected, settings = run_tomography(backend, INITIAL_LAYOUT, *result.x,
                                          subsystem_qubits=(0, 1), shots=SHOTS)
    rho_A = reconstruct_density_matrix(corrected, settings, subsystem_qubits=(0, 1))

    # reconstruct_density_matrix is a plain linear (Pauli-expansion) inversion
    # with no positive-semi-definite projection, so shot noise can push a
    # reconstruction slightly non-physical (small negative eigenvalues) --
    # validate=False so that doesn't hard-crash the sweep; min_eig flags it.
    min_eig = float(np.linalg.eigvalsh(rho_A).min())
    fidelity = state_fidelity(DensityMatrix(rho_A), target, validate=False)
    return result.x, fidelity, rho_A, min_eig


def main():
    backend = NoisySimulator("Starmon-7", ideal_simulation=True)
    H_AA = default_subsystem_hamiltonian()
    gibbs = GibbsState(H_AA)
    rng = np.random.default_rng(RNG_SEED)

    overall_t0 = time.time()

    PLOTS_DIR.mkdir(exist_ok=True)

    for beta in BETAS:
        target = DensityMatrix(gibbs.gibbs_density_matrix(beta))
        fidelities = []
        rho_As = []

        print(f"\n=== beta={beta} ({N_TRIALS} trials) ===")
        for trial in range(N_TRIALS):
            x0 = random_nonzero_x0(rng)
            t0 = time.time()
            angles, fidelity, rho_A, min_eig = run_one_trial(backend, beta, x0, target)
            dt = time.time() - t0
            fidelities.append(fidelity)
            rho_As.append(rho_A)
            flag = "" if fidelity >= FIDELITY_THRESHOLD else "  <-- below threshold"
            nonphys = f"  [non-physical: min eig={min_eig:.3f}]" if min_eig < -0.01 else ""
            print(f"  trial {trial:2d}  x0={np.round(x0, 2)}  "
                  f"angles={np.round(angles, 3)}  fidelity={fidelity:.4f}  "
                  f"({dt:.1f}s){flag}{nonphys}")

        fidelities = np.array(fidelities)
        n_below = int((fidelities < FIDELITY_THRESHOLD).sum())
        print(f"\nbeta={beta} summary: "
              f"min={fidelities.min():.4f}  mean={fidelities.mean():.4f}  "
              f"max={fidelities.max():.4f}  std={fidelities.std():.4f}")
        print(f"  trials below {FIDELITY_THRESHOLD}: {n_below}/{N_TRIALS}"
              + ("  <-- NOT reproducible across random starts" if n_below else
                 "  -- reliably converges regardless of starting point"))

        # Average tomography: the elementwise-mean reconstructed rho_A across
        # all N_TRIALS runs, plotted the same manhattan-plot way as a single
        # run (tomography.py's plot_density_matrix). Note this is a different
        # number from the mean of the per-trial fidelities above -- it's the
        # fidelity of the *averaged state*, which also reflects how much the
        # individual reconstructions' phases/off-diagonals cancel each other
        # out when they don't agree trial-to-trial.
        rho_avg = np.mean(rho_As, axis=0)
        fid_avg = state_fidelity(DensityMatrix(rho_avg), target, validate=False)
        print(f"  averaged over {N_TRIALS} trials: fidelity(avg rho_A, true Gibbs) = {fid_avg:.4f}")

        # 1) actual solution: the exact, analytic Gibbs-state rho_A (no noise,
        # no optimizer, no tomography -- just the target itself)
        exact_path = PLOTS_DIR / f"exact_rho_A_beta{beta}.png"
        plot_density_matrix(
            gibbs.gibbs_density_matrix(beta),
            title=f"Exact (analytic) true rho_A, beta={beta}",
            save_path=str(exact_path),
        )

        # 2) experimental result: the average reconstructed rho_A over all
        # N_TRIALS runs of the real shot-based pipeline, same plot style so
        # it's directly comparable to (1) side by side.
        tomo_path = PLOTS_DIR / f"avg_tomography_beta{beta}.png"
        plot_density_matrix(
            rho_avg,
            title=f"Experimental: average rho_A over {N_TRIALS} random starts, beta={beta}\n"
                  f"(fidelity of average = {fid_avg:.4f}, mean per-trial fidelity = {fidelities.mean():.4f})",
            save_path=str(tomo_path),
        )

        # 3) all N_TRIALS individual trial fidelities, one bar chart per beta
        fid_path = PLOTS_DIR / f"fidelities_beta{beta}.png"
        plot_fidelities(fidelities, beta, str(fid_path))

        print(f"  saved: {exact_path.name}, {tomo_path.name}, {fid_path.name}")

    print(f"\ntotal time: {time.time() - overall_t0:.1f}s")


if __name__ == "__main__":
    main()
