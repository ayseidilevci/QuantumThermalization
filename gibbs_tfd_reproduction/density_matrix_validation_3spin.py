"""Validate the three-spin TFD circuit by comparing reduced density matrices.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace, state_fidelity

from cost_function_3spin import CostFunction3Spin
from tfd_circuit_3spin import TFD3SpinCircuit
from validate_cost_function_3spin import (
    PAPER_BETAS,
    SIGMA_VALUES,
    exact_gibbs_state,
    three_spin_hamiltonian,
)

# Match the 17-point beta grid used by the Xi validation by default.
DEFAULT_PLOT_BETAS = (0.001, 0.01, 0.1, 1.0, 5.0)
P_LAYERS = 1  # Deliberately fixed: this test does not change the ansatz.
N_QUBITS_A = 3
N_QUBITS_B = 3


def reduced_state_a(state: Statevector) -> DensityMatrix:
    """Trace out B (q3,q4,q5), leaving rho_A on q0,q1,q2."""
    return partial_trace(state, [3, 4, 5])


def trace_distance_from_matrices(rho_a: np.ndarray, rho_b: np.ndarray) -> float:
    """Compute D(rho_a, rho_b) = 1/2 ||rho_a-rho_b||_1 using NumPy.

    The difference of two density matrices is Hermitian. Symmetrizing the
    difference removes tiny floating-point anti-Hermitian components; for a
    Hermitian matrix, the trace norm is the sum of the absolute eigenvalues.
    This avoids relying on Qiskit versions that do not export trace_distance.
    """
    difference = np.asarray(rho_a, dtype=complex) - np.asarray(rho_b, dtype=complex)
    difference = 0.5 * (difference + difference.conj().T)
    eigenvalues = np.linalg.eigvalsh(difference)
    return float(0.5 * np.sum(np.abs(eigenvalues)))


def compare_density_matrices(rho_a: DensityMatrix, rho_gibbs: DensityMatrix) -> dict[str, float]:
    """Compute complementary diagnostics for rho_A versus the exact Gibbs state."""
    a = np.asarray(rho_a.data, dtype=complex)
    target = np.asarray(rho_gibbs.data, dtype=complex)
    difference = a - target

    return {
        "gibbs_fidelity": float(state_fidelity(rho_a, rho_gibbs)),
        "trace_distance": trace_distance_from_matrices(a, target),
        "max_abs_matrix_element_difference": float(np.max(np.abs(difference))),
        "frobenius_norm_difference": float(np.linalg.norm(difference, ord="fro")),
        "max_abs_imaginary_part_rho_a": float(np.max(np.abs(a.imag))),
        "max_abs_imaginary_part_rho_gibbs": float(np.max(np.abs(target.imag))),
    }


def optimize_and_evaluate(
    betas: tuple[float, ...],
    h_eigendecomposition,
    maxiter: int,
    rhobeg: float,
    tol: float,
):
    """Optimize p=1 independently for each sigma, warm-starting across beta."""
    results: dict[float, dict[float, dict]] = {}
    matrices: dict[str, np.ndarray] = {}

    for varsigma in SIGMA_VALUES:
        print(f"\nOptimizing p=1 with varsigma={varsigma:g}")
        model = CostFunction3Spin(varsigma=varsigma, p_layers=P_LAYERS, periodic=False)
        x0 = np.zeros(4, dtype=float)
        sigma_results: dict[float, dict] = {}

        for beta in betas:
            opt = model.minimize_cost(
                x0=x0,
                beta=float(beta),
                method="COBYLA",
                maxiter=maxiter,
                rhobeg=rhobeg,
                tol=tol,
            )
            params = np.asarray(opt.x, dtype=float)
            state = TFD3SpinCircuit(params, p_layers=P_LAYERS, periodic=False).get_state_vector()
            rho_a = reduced_state_a(state)
            rho_gibbs = exact_gibbs_state(float(beta), h_eigendecomposition)
            metrics = compare_density_matrices(rho_a, rho_gibbs)

            record = {
                **metrics,
                "beta": float(beta),
                "varsigma": float(varsigma),
                "cost": float(opt.fun),
                "nfev": int(getattr(opt, "nfev", 0)),
                "optimizer_success": bool(opt.success),
                "optimizer_message": str(opt.message),
                "params": params.tolist(),
            }
            sigma_results[float(beta)] = record

            matrices[f"rho_gibbs_beta_{beta:.12g}"] = np.asarray(rho_gibbs.data)
            matrices[f"rho_a_sigma_{varsigma:.12g}_beta_{beta:.12g}"] = np.asarray(rho_a.data)
            matrices[f"abs_diff_sigma_{varsigma:.12g}_beta_{beta:.12g}"] = np.abs(
                np.asarray(rho_a.data) - np.asarray(rho_gibbs.data)
            )

            print(
                f"  beta={beta:.6g}: F_Gibbs={metrics['gibbs_fidelity']:.8f}, "
                f"D_trace={metrics['trace_distance']:.8g}, "
                f"max|Delta rho_ij|={metrics['max_abs_matrix_element_difference']:.8g}, "
                f"nfev={record['nfev']}"
            )
            x0 = params.copy()

        results[float(varsigma)] = sigma_results

    return results, matrices


def save_heatmap(
    beta: float,
    result_by_sigma: dict[float, dict[float, dict]],
    matrices: dict[str, np.ndarray],
    output_dir: Path,
):
    """Save exact/reduced/absolute-difference heatmaps for both sigma values."""
    target_key = f"rho_gibbs_beta_{beta:.12g}"
    rho_target = np.asarray(matrices[target_key])
    rho_reduced = {}
    abs_diffs = {}
    for sigma in SIGMA_VALUES:
        rho_reduced[sigma] = np.asarray(matrices[f"rho_a_sigma_{sigma:.12g}_beta_{beta:.12g}"])
        abs_diffs[sigma] = np.asarray(matrices[f"abs_diff_sigma_{sigma:.12g}_beta_{beta:.12g}"])

    # Use a shared diverging range for real parts of the density matrices.
    matrix_max = max(
        float(np.max(np.abs(rho_target.real))),
        *(float(np.max(np.abs(rho.real))) for rho in rho_reduced.values()),
    )
    diff_max = max(float(np.max(d.real)) for d in abs_diffs.values())
    if diff_max < 1e-15:
        diff_max = 1e-15

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.3), constrained_layout=True)
    col_titles = (
        r"Exact Gibbs state: Re$(\rho_\beta)$",
        r"Circuit reduced state: Re$(\rho_A)$",
        r"Elementwise difference: $|\rho_A-\rho_\beta|$",
    )
    images = [None, None, None]

    for row, sigma in enumerate(SIGMA_VALUES):
        images[0] = axes[row, 0].imshow(
            rho_target.real, cmap="RdBu_r", vmin=-matrix_max, vmax=matrix_max, interpolation="nearest"
        )
        images[1] = axes[row, 1].imshow(
            rho_reduced[sigma].real, cmap="RdBu_r", vmin=-matrix_max, vmax=matrix_max, interpolation="nearest"
        )
        images[2] = axes[row, 2].imshow(
            abs_diffs[sigma], cmap="magma", vmin=0.0, vmax=diff_max, interpolation="nearest"
        )

        metrics = result_by_sigma[sigma][float(beta)]
        axes[row, 1].text(
            0.03, 0.97,
            f"F={metrics['gibbs_fidelity']:.6f}\nD={metrics['trace_distance']:.4g}",
            transform=axes[row, 1].transAxes,
            ha="left", va="top", fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
        for col in range(3):
            axes[row, col].set_xlabel("Column index")
            axes[row, col].set_xticks(range(8))
            axes[row, col].set_yticks(range(8))
        axes[row, 0].set_ylabel(fr"$\varsigma={sigma:g}$")

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title)
        fig.colorbar(images[col], ax=axes[:, col], shrink=0.82, pad=0.02)

    fig.suptitle(fr"Three-spin Gibbs density-matrix comparison, $\beta={beta:g}$ (fixed $p=1$)", fontsize=14)
    path = output_dir / f"density_matrix_beta_{beta:.6g}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"  saved heatmap: {path}")


def write_outputs(results, matrices, betas, plot_betas, output_dir: Path, maxiter: int):
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "density_matrix_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        columns = [
            "varsigma", "beta", "gibbs_fidelity", "trace_distance",
            "max_abs_matrix_element_difference", "frobenius_norm_difference",
            "max_abs_imaginary_part_rho_a", "max_abs_imaginary_part_rho_gibbs",
            "cost", "nfev", "optimizer_success", "param_1", "param_2", "param_3", "param_4",
        ]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for sigma in SIGMA_VALUES:
            for beta in betas:
                record = results[float(sigma)][float(beta)]
                row = {key: record[key] for key in columns if key in record}
                row.update({f"param_{i + 1}": value for i, value in enumerate(record["params"])})
                writer.writerow(row)

    summary = {
        "p_layers": P_LAYERS,
        "parameters_per_layer": 4,
        "betas": [float(b) for b in betas],
        "plotted_betas": [float(b) for b in plot_betas],
        "maxiter": int(maxiter),
        "sigma_results": {},
        "interpretation": {
            "gibbs_fidelity": "Squared Uhlmann fidelity between reduced rho_A and exact rho_beta; 1 is perfect.",
            "trace_distance": "One-half trace norm of rho_A-rho_beta; 0 is perfect and larger is more distinguishable.",
            "matrix_difference": "Elementwise absolute value of complex matrix difference |rho_A-rho_beta|.",
        },
    }
    for sigma in SIGMA_VALUES:
        records = [results[float(sigma)][float(beta)] for beta in betas]
        summary["sigma_results"][str(sigma)] = {
            "mean_gibbs_fidelity": float(np.mean([r["gibbs_fidelity"] for r in records])),
            "mean_trace_distance": float(np.mean([r["trace_distance"] for r in records])),
            "max_trace_distance": float(np.max([r["trace_distance"] for r in records])),
            "mean_max_abs_matrix_element_difference": float(
                np.mean([r["max_abs_matrix_element_difference"] for r in records])
            ),
        }

    json_path = output_dir / "density_matrix_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    npz_path = output_dir / "density_matrices.npz"
    np.savez_compressed(npz_path, **matrices)

    # Only generate heatmaps for beta values present in the optimized grid.
    actual_plot_betas = []
    for requested in plot_betas:
        matched = min(betas, key=lambda b: abs(float(b) - requested)) if betas else None
        if matched is not None and np.isclose(float(matched), requested, rtol=1e-7, atol=1e-12):
            if float(matched) not in actual_plot_betas:
                actual_plot_betas.append(float(matched))

    for beta in actual_plot_betas:
        save_heatmap(beta, results, matrices, output_dir)

    print("\n=== Density-matrix validation summary ===")
    for sigma in SIGMA_VALUES:
        item = summary["sigma_results"][str(sigma)]
        print(
            f"varsigma={sigma:g}: mean Gibbs fidelity={item['mean_gibbs_fidelity']:.8f}, "
            f"mean trace distance={item['mean_trace_distance']:.8g}, "
            f"max trace distance={item['max_trace_distance']:.8g}"
        )
    print("\nSaved:")
    print(f"  {csv_path}")
    print(f"  {json_path}")
    print(f"  {npz_path}")
    for beta in actual_plot_betas:
        print(f"  {output_dir / f'density_matrix_beta_{beta:.6g}.png'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--betas", nargs="+", type=float, default=None,
        help="optional beta grid; by default use the paper's 17-point grid",
    )
    parser.add_argument(
        "--plot-betas", nargs="+", type=float, default=list(DEFAULT_PLOT_BETAS),
        help="beta values for heatmap plots (must also be present in --betas, if supplied)",
    )
    parser.add_argument("--maxiter", type=int, default=200, help="COBYLA iteration limit per beta")
    parser.add_argument("--rhobeg", type=float, default=0.5, help="COBYLA initial trust-region radius")
    parser.add_argument("--tol", type=float, default=0.02, help="COBYLA stopping tolerance")
    parser.add_argument("--outdir", type=Path, default=Path("density_matrix_validation_3spin"))
    args = parser.parse_args()

    if args.maxiter < 1:
        parser.error("--maxiter must be >= 1")
    betas = tuple(sorted(set(float(b) for b in (args.betas if args.betas else PAPER_BETAS))))
    if not betas or any((not np.isfinite(b) or b < 0) for b in betas):
        parser.error("betas must be finite, non-negative numbers")
    if any(b == 0 for b in betas):
        parser.error("beta=0 is not supported by the current cost model; use a small positive beta")

    missing = [b for b in args.plot_betas if not any(np.isclose(b, bb, rtol=1e-7, atol=1e-12) for bb in betas)]
    if missing:
        # Do not silently plot a neighboring beta: tell the user what to include.
        parser.error(f"plot beta(s) {missing} are not in the optimized beta grid; include them with --betas")

    print("=== Three-spin / six-qubit density-matrix validation ===")
    print(f"Ansatz depth fixed at p={P_LAYERS}; trainable angles = 4")
    print(f"Betas ({len(betas)}): {betas}")
    print(f"Heatmap betas: {tuple(args.plot_betas)}")
    print("Metrics: reduced Gibbs fidelity and trace distance")
    h_eigendecomposition = three_spin_hamiltonian(g=1.0, periodic=False)

    results, matrices = optimize_and_evaluate(
        betas=betas,
        h_eigendecomposition=h_eigendecomposition,
        maxiter=args.maxiter,
        rhobeg=args.rhobeg,
        tol=args.tol,
    )
    write_outputs(results, matrices, betas, tuple(args.plot_betas), args.outdir, args.maxiter)


if __name__ == "__main__":
    main()

