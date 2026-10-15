"""Three-qubit Pauli tomography for the reduced Gibbs state in the 3-spin TFD ansatz.

* local finite-shot simulations using Qiskit's Statevector probabilities;
* a backend adapter for executing the same 27 measurement circuits on a Qiskit
  backend, including a compatible Quantum Inspire backend;
* linear-inversion tomography, projection onto a physical density matrix, and
  comparison with the exact Gibbs state;
* heatmaps and optional circuit diagram export.

Run from the project root:
    python -m gibbs_tfd_reproduction.tomography_3spin

This script's default mode is a shot-based ideal simulation, NOT real hardware.
To use hardware, call tomography_counts_from_backend() with an authenticated
backend object from your project's provider setup.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import DensityMatrix, Statevector, state_fidelity

from circuit_visualization import print_circuit_text, save_circuit_diagram
from cost_function_3spin import CostFunction3Spin
from tfd_circuit_3spin import TFD3SpinCircuit
from validate_cost_function_3spin import exact_gibbs_state, three_spin_hamiltonian

N_QUBITS = 6
N_A = 3
P_LAYERS = 1  # Keep the paper's ansatz structure unchanged: four angles.
SIGMA_VALUES = (1.0, 1.57)
DEFAULT_BETAS = (0.001, 0.01, 0.1, 1.0, 5.0)
AXES = "XYZ"
PAULI_LABELS = "IXYZ"
PAULI = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}
TOMOGRAPHY_SETTINGS = tuple(itertools.product(AXES, repeat=N_A))


def _basis_rotated_circuit(ansatz_circuit, setting: tuple[str, str, str], *, measurements: bool):
    """Rotate A's qubits from the requested X/Y/Z basis to the Z measurement basis.

    `setting[q]` is the basis used to measure q in subsystem A (q0=A1, q1=A2,
    q2=A3). Subsystem B is left unmeasured and is traced out statistically.
    """
    circuit = QuantumCircuit(N_QUBITS, N_A if measurements else 0)
    circuit.compose(ansatz_circuit, qubits=list(range(N_QUBITS)), inplace=True)
    for qubit, axis in enumerate(setting):
        if axis == "X":
            circuit.h(qubit)
        elif axis == "Y":
            circuit.sdg(qubit)
            circuit.h(qubit)
        elif axis != "Z":
            raise ValueError(f"Unsupported Pauli measurement axis {axis!r}")
    if measurements:
        for qubit in range(N_A):
            circuit.measure(qubit, qubit)
    return circuit


def _counts_to_outcome_vector(counts: dict) -> np.ndarray:
    """Convert Qiskit 3-bit count strings to outcomes 0..7.

    Classical bits c0,c1,c2 are mapped from q0,q1,q2. Qiskit prints the string
    most-significant bit first, but converting it to an integer preserves the
    convention that q0 is outcome bit 0.
    """
    result = np.zeros(2**N_A, dtype=np.int64)
    for key, count in counts.items():
        text = str(key).replace(" ", "")
        outcome = int(text, 16) if text.startswith("0x") else int(text, 2)
        if not 0 <= outcome < 2**N_A:
            raise ValueError(f"Unexpected tomography outcome {key!r}; expected 3 classical bits")
        result[outcome] += int(count)
    return result


def sample_statevector_counts(ansatz_circuit, shots: int, rng: np.random.Generator):
    """Generate finite-shot counts for all 27 settings from an ideal statevector.

    This is useful for testing the tomography implementation without Aer or
    hardware. It includes shot noise, but not gate noise or readout error.
    """
    if shots < 1:
        raise ValueError("shots must be positive")
    counts_by_setting: dict[tuple[str, str, str], np.ndarray] = {}
    for setting in TOMOGRAPHY_SETTINGS:
        circuit = _basis_rotated_circuit(ansatz_circuit, setting, measurements=False)
        state = Statevector.from_instruction(circuit)
        # q0, q1, q2 are the three least-significant qubits in Qiskit's
        # statevector indexing. Sum over all B outcomes to obtain A's marginal.
        probabilities = np.zeros(2**N_A, dtype=float)
        for basis_index, amplitude in enumerate(state.data):
            probabilities[basis_index & 0b111] += float(abs(amplitude) ** 2)
        probabilities = np.clip(probabilities, 0.0, None)
        probabilities /= probabilities.sum()
        counts_by_setting[setting] = rng.multinomial(int(shots), probabilities)
    return counts_by_setting


def tomography_counts_from_backend(ansatz_circuit, backend, shots: int = 4096, optimization_level: int = 1):
    """Execute the 27 three-qubit Pauli tomography settings on a Qiskit backend.

    Pass an already-authenticated backend from the project's existing provider
    setup. Only q0, q1, q2 are measured; q3, q4, q5 remain unmeasured. Backend
    results are converted to the same outcome-vector format as the local mode.
    """
    if shots < 1:
        raise ValueError("shots must be positive")
    circuits = [
        _basis_rotated_circuit(ansatz_circuit, setting, measurements=True)
        for setting in TOMOGRAPHY_SETTINGS
    ]
    compiled = transpile(circuits, backend=backend, optimization_level=optimization_level)
    job = backend.run(compiled, shots=int(shots))
    result = job.result()
    counts_by_setting = {}
    for index, setting in enumerate(TOMOGRAPHY_SETTINGS):
        counts_by_setting[setting] = _counts_to_outcome_vector(result.get_counts(index))
    return counts_by_setting


def reconstruct_linear_inversion(counts_by_setting: dict[tuple[str, str, str], np.ndarray]):
    """Reconstruct rho_A = 1/8 sum_P <P>P from 27 local Pauli settings.

    Returns (rho_raw, expectation_values). Linear inversion is informationally
    complete, but finite-shot noise can yield a matrix with small negative
    eigenvalues; call project_to_physical_density_matrix before using it as a
    physical state in fidelity calculations.
    """
    expectation_values: dict[str, float] = {}
    rho = np.zeros((2**N_A, 2**N_A), dtype=complex)

    for axes in itertools.product(PAULI_LABELS, repeat=N_A):
        setting = tuple(axis if axis != "I" else "Z" for axis in axes)
        counts = np.asarray(counts_by_setting[setting], dtype=float)
        total = float(counts.sum())
        if total <= 0:
            raise ValueError(f"No shots were recorded for setting {setting}")
        expectation = 0.0
        for outcome, count in enumerate(counts):
            sign = 1
            for qubit, axis in enumerate(axes):
                if axis != "I" and ((outcome >> qubit) & 1):
                    sign *= -1
            expectation += sign * count / total

        label = "".join(reversed(axes))  # Qiskit-style label: leftmost acts on q2.
        expectation_values[label] = float(expectation)
        operator = np.kron(np.kron(PAULI[axes[2]], PAULI[axes[1]]), PAULI[axes[0]])
        rho += expectation * operator / (2**N_A)

    rho = 0.5 * (rho + rho.conj().T)
    # The identity expectation is exactly one in count-based reconstruction.
    rho /= np.trace(rho).real
    return rho, expectation_values


def project_to_physical_density_matrix(rho: np.ndarray) -> np.ndarray:
    """Project a Hermitian estimate to the nearest PSD trace-one matrix in Frobenius norm.

    This eigenvalue-simplex projection is not maximum-likelihood tomography. It
    provides a transparent, physical estimate for fidelity and trace-distance
    diagnostics while retaining the raw linear-inversion estimate separately.
    """
    hermitian = 0.5 * (np.asarray(rho, dtype=complex) + np.asarray(rho, dtype=complex).conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    ordered = np.sort(eigenvalues)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    indices = np.arange(1, len(ordered) + 1)
    valid = np.where(ordered - cumulative / indices > 0)[0]
    if len(valid) == 0:
        projected_values = np.full_like(eigenvalues, 1.0 / len(eigenvalues))
    else:
        rho_index = valid[-1]
        threshold = cumulative[rho_index] / (rho_index + 1)
        projected_values = np.maximum(eigenvalues - threshold, 0.0)
    projected_values /= projected_values.sum()
    projected = (eigenvectors * projected_values) @ eigenvectors.conj().T
    return 0.5 * (projected + projected.conj().T)


def trace_distance(rho_a: np.ndarray, rho_b: np.ndarray) -> float:
    difference = np.asarray(rho_a, dtype=complex) - np.asarray(rho_b, dtype=complex)
    difference = 0.5 * (difference + difference.conj().T)
    return float(0.5 * np.sum(np.abs(np.linalg.eigvalsh(difference))))


def evaluate_tomography(rho_raw: np.ndarray, rho_exact: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    rho_physical = project_to_physical_density_matrix(rho_raw)
    exact_matrix = np.asarray(rho_exact, dtype=complex)
    linear_eigenvalues = np.linalg.eigvalsh(0.5 * (rho_raw + rho_raw.conj().T))
    metrics = {
        "fidelity_physical_estimate": float(state_fidelity(DensityMatrix(rho_physical), DensityMatrix(exact_matrix))),
        "trace_distance_physical_estimate": trace_distance(rho_physical, exact_matrix),
        "trace_distance_raw_linear_inversion": trace_distance(rho_raw, exact_matrix),
        "max_abs_element_difference_physical_estimate": float(np.max(np.abs(rho_physical - exact_matrix))),
        "frobenius_norm_difference_physical_estimate": float(np.linalg.norm(rho_physical - exact_matrix, ord="fro")),
        "min_eigenvalue_raw_linear_inversion": float(linear_eigenvalues.min()),
        "purity_physical_estimate": float(np.trace(rho_physical @ rho_physical).real),
    }
    return rho_physical, metrics


def _save_heatmap(beta: float, rho_exact: np.ndarray, reconstructed: dict[float, np.ndarray],
                  records: dict[float, dict[str, float]], outdir: Path):
    differences = {sigma: np.abs(rho - rho_exact) for sigma, rho in reconstructed.items()}
    matrix_max = max(float(np.max(np.abs(rho_exact.real))),
                     *(float(np.max(np.abs(rho.real))) for rho in reconstructed.values()))
    diff_max = max(float(d.max()) for d in differences.values())
    diff_max = max(diff_max, 1e-12)

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.3), constrained_layout=True)
    titles = (r"Exact Gibbs state: Re$(\rho_\beta)$",
              r"Tomographic estimate: Re$(\hat\rho_A)$",
              r"Elementwise difference: $|\hat\rho_A-\rho_\beta|$")
    mappables = [None, None, None]
    for row, sigma in enumerate(SIGMA_VALUES):
        mappables[0] = axes[row, 0].imshow(rho_exact.real, cmap="RdBu_r", vmin=-matrix_max,
                                          vmax=matrix_max, interpolation="nearest")
        mappables[1] = axes[row, 1].imshow(reconstructed[sigma].real, cmap="RdBu_r", vmin=-matrix_max,
                                          vmax=matrix_max, interpolation="nearest")
        mappables[2] = axes[row, 2].imshow(differences[sigma], cmap="magma", vmin=0.0,
                                          vmax=diff_max, interpolation="nearest")
        metric = records[sigma]
        axes[row, 1].text(0.03, 0.97,
            f"F={metric['fidelity_physical_estimate']:.5f}\nD={metric['trace_distance_physical_estimate']:.4g}",
            transform=axes[row, 1].transAxes, ha="left", va="top", fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"})
        axes[row, 0].set_ylabel(fr"$\varsigma={sigma:g}$")
        for col in range(3):
            axes[row, col].set_xticks(range(8))
            axes[row, col].set_yticks(range(8))
            axes[row, col].set_xlabel("Column index")
    for col, title in enumerate(titles):
        axes[0, col].set_title(title)
        fig.colorbar(mappables[col], ax=axes[:, col], shrink=0.82, pad=0.02)
    fig.suptitle(fr"Three-spin tomography versus exact Gibbs state, $\beta={beta:g}$", fontsize=14)
    path = outdir / f"tomography_beta_{beta:.6g}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"  saved tomography heatmap: {path}")



def _save_city_plot(beta: float, rho_exact: np.ndarray, reconstructed: dict[float, np.ndarray], outdir: Path):
    """Save phase-coloured 3D Manhattan plots for 8x8 density matrices.

    Bar height encodes |rho[i,j]| and bar colour encodes arg(rho[i,j]) in
    [-pi, pi], matching the visual convention of the supplied two-qubit figure.
    Each figure compares the exact Gibbs state with the reconstructed reduced
    state for varsigma=1 and varsigma=1.57.
    """
    from matplotlib import cm
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    states = [
        (r"Exact Gibbs state $\rho_\beta$", rho_exact),
        (r"Tomography $\hat{\rho}_A$, $\varsigma=1$", reconstructed[1.0]),
        (r"Tomography $\hat{\rho}_A$, $\varsigma=1.57$", reconstructed[1.57]),
    ]

    # Computational basis labels for the three-qubit subsystem.
    basis_labels = [format(i, "03b") for i in range(8)]
    x, y = np.meshgrid(np.arange(8), np.arange(8))
    x, y = x.ravel(), y.ravel()
    dx = dy = 0.62

    phase_cmap = cm.twilight
    phase_norm = Normalize(vmin=-np.pi, vmax=np.pi)
    phase_mappable = ScalarMappable(norm=phase_norm, cmap=phase_cmap)
    phase_mappable.set_array([])

    fig = plt.figure(figsize=(17, 6.4), constrained_layout=True)

    for col, (title, rho) in enumerate(states, start=1):
        ax = fig.add_subplot(1, 3, col, projection="3d")
        magnitudes = np.abs(rho).ravel()
        phases = np.angle(rho).ravel()
        colors = phase_cmap(phase_norm(phases))

        # All bars rise from zero; height is magnitude and colour is phase.
        ax.bar3d(
            x, y, np.zeros_like(magnitudes),
            dx, dy, magnitudes,
            color=colors, shade=False, zsort="average",
            edgecolor="black", linewidth=0.12,
        )
        ax.set_title(title, fontsize=11, pad=12)
        ax.set_xlabel("Column basis state")
        ax.set_ylabel("Row basis state")
        ax.set_zlabel(r"$|\rho_{ij}|$")
        ticks = np.arange(8) + dx / 2
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(basis_labels, rotation=35, ha="right", fontsize=8)
        ax.set_yticklabels(basis_labels, fontsize=8)
        ax.set_zlim(0, max(float(magnitudes.max()) * 1.12, 1e-8))
        ax.view_init(elev=28, azim=-55)

    colorbar = fig.colorbar(
        phase_mappable, ax=fig.axes, shrink=0.72, pad=0.04, aspect=25,
        ticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
    )
    colorbar.ax.set_yticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
    colorbar.set_label("Phase of density-matrix element")

    fig.suptitle(
        fr"Three-spin Gibbs-state tomography, $\beta={beta:g}$",
        fontsize=15,
    )
    output_path = outdir / f"tomography_city_beta_{beta:.6g}.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved phase-coloured 3D Manhattan plots: {output_path}")

def run_local_tomography(betas: Iterable[float], shots: int, seed: int, maxiter: int,
                         rhobeg: float, tol: float, output_dir: Path,
                         save_circuits: bool = False, print_circuits: bool = False):
    output_dir.mkdir(parents=True, exist_ok=True)
    circuit_dir = output_dir / "circuits"
    rng = np.random.default_rng(seed)
    h_eigendecomposition = three_spin_hamiltonian(g=1.0, periodic=False)
    results: list[dict] = []
    matrices: dict[str, np.ndarray] = {}
    counts_json: dict[str, dict[str, dict[str, list[int]]]] = {}

    for sigma in SIGMA_VALUES:
        model = CostFunction3Spin(varsigma=sigma, p_layers=P_LAYERS, periodic=False)
        x0 = np.zeros(4, dtype=float)
        print(f"\nOptimizing fixed p=1 ansatz with varsigma={sigma:g}")
        for beta in sorted(float(b) for b in betas):
            opt = model.minimize_cost(x0=x0, beta=beta, method="COBYLA", maxiter=maxiter,
                                      rhobeg=rhobeg, tol=tol)
            params = np.asarray(opt.x, dtype=float)
            ansatz = TFD3SpinCircuit(params, p_layers=P_LAYERS, periodic=False).as_qiskit_circuit()
            if print_circuits:
                print(f"\nCircuit: varsigma={sigma:g}, beta={beta:g}, params={params}")
                print_circuit_text(ansatz)
            if save_circuits:
                ext = circuit_dir / f"circuit_beta_{beta:.6g}_sigma_{sigma:g}.png"
                save_circuit_diagram(ansatz, ext, dpi=180)

            counts_by_setting = sample_statevector_counts(ansatz, shots=shots, rng=rng)
            rho_raw, exp_values = reconstruct_linear_inversion(counts_by_setting)
            rho_exact = np.asarray(exact_gibbs_state(beta, h_eigendecomposition).data, dtype=complex)
            rho_physical, metrics = evaluate_tomography(rho_raw, rho_exact)
            row = {
                "varsigma": sigma, "beta": beta, "p_layers": P_LAYERS,
                "shots_per_setting": shots, "number_of_settings": len(TOMOGRAPHY_SETTINGS),
                "total_samples": shots * len(TOMOGRAPHY_SETTINGS),
                "cost": float(opt.fun), "nfev": int(getattr(opt, "nfev", 0)),
                "optimizer_success": bool(opt.success), "optimizer_message": str(opt.message),
                "param_1": float(params[0]), "param_2": float(params[1]),
                "param_3": float(params[2]), "param_4": float(params[3]),
                **metrics,
            }
            results.append(row)
            matrix_key = f"sigma_{sigma:g}_beta_{beta:.12g}"
            matrices[f"rho_exact_{matrix_key}"] = rho_exact
            matrices[f"rho_tomography_raw_{matrix_key}"] = rho_raw
            matrices[f"rho_tomography_physical_{matrix_key}"] = rho_physical
            matrices[f"rho_tomography_abs_difference_{matrix_key}"] = np.abs(rho_physical - rho_exact)
            settings_json = {"".join(setting): counts_by_setting[setting].tolist()
                             for setting in TOMOGRAPHY_SETTINGS}
            sigma_counts = counts_json.setdefault(f"sigma_{sigma:g}", {})
            sigma_counts[f"beta_{beta:.12g}"] = settings_json
            # Add the expectation values for reproducibility/debugging, too.
            matrices[f"pauli_expectations_{matrix_key}"] = np.array(
                [exp_values[k] for k in sorted(exp_values)], dtype=float)

            print(
                f"  beta={beta:g}: F_tomo={metrics['fidelity_physical_estimate']:.6f}, "
                f"D_tomo={metrics['trace_distance_physical_estimate']:.6f}, "
                f"D_raw={metrics['trace_distance_raw_linear_inversion']:.6f}, "
                f"min_eig(raw)={metrics['min_eigenvalue_raw_linear_inversion']:.3e}, "
                f"nfev={row['nfev']}"
            )
            x0 = params.copy()  # warm-start as in the earlier statevector tests

    # Assemble heatmaps and tabular outputs.
    by_beta_reconstruction = {}
    by_beta_records = {}
    for beta in sorted(float(b) for b in betas):
        exact_key = f"rho_exact_sigma_1_beta_{beta:.12g}"
        rho_exact = matrices[exact_key]
        reconstructed = {}
        recs = {}
        for sigma in SIGMA_VALUES:
            key = f"sigma_{sigma:g}_beta_{beta:.12g}"
            reconstructed[sigma] = matrices[f"rho_tomography_physical_{key}"]
            recs[sigma] = next(r for r in results if r["varsigma"] == sigma and np.isclose(r["beta"], beta))
        _save_heatmap(beta, rho_exact, reconstructed, recs, output_dir)
        _save_city_plot(beta, rho_exact, reconstructed, output_dir)

    csv_path = output_dir / "tomography_metrics.csv"
    if results:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
    npz_path = output_dir / "tomography_matrices.npz"
    np.savez_compressed(npz_path, **matrices)
    counts_path = output_dir / "tomography_counts.json"
    counts_path.write_text(json.dumps(counts_json, indent=2), encoding="utf-8")
    summary = {
        "p_layers": P_LAYERS, "parameters_per_layer": 4,
        "shots_per_setting": shots, "num_pauli_settings": len(TOMOGRAPHY_SETTINGS),
        "betas": sorted(float(b) for b in betas), "seed": seed,
        "mode": "ideal_statevector_sampled_counts",
        "warning": "This is shot-based ideal simulation, not hardware tomography; it includes sampling noise only.",
        "per_sigma": {},
    }
    for sigma in SIGMA_VALUES:
        rows = [r for r in results if r["varsigma"] == sigma]
        summary["per_sigma"][str(sigma)] = {
            "mean_fidelity": float(np.mean([r["fidelity_physical_estimate"] for r in rows])),
            "mean_trace_distance": float(np.mean([r["trace_distance_physical_estimate"] for r in rows])),
            "max_trace_distance": float(np.max([r["trace_distance_physical_estimate"] for r in rows])),
        }
    summary_path = output_dir / "tomography_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== Tomography summary (ideal statevector sampled counts) ===")
    for sigma in SIGMA_VALUES:
        item = summary["per_sigma"][str(sigma)]
        print(f"varsigma={sigma:g}: mean F={item['mean_fidelity']:.6f}, "
              f"mean D={item['mean_trace_distance']:.6f}, max D={item['max_trace_distance']:.6f}")
    print("\nSaved:")
    for path in (csv_path, npz_path, counts_path, summary_path):
        print(f"  {path}")
    if save_circuits:
        print(f"  circuit diagrams: {circuit_dir}")
    return results, matrices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--betas", nargs="+", type=float, default=list(DEFAULT_BETAS),
                        help="inverse temperatures (default: 0.1, 0.316..., 1, 5)")
    parser.add_argument("--shots", type=int, default=4096, help="shots per Pauli basis setting")
    parser.add_argument("--seed", type=int, default=1234, help="random seed for local shot simulation")
    parser.add_argument("--maxiter", type=int, default=200, help="COBYLA iteration limit per beta")
    parser.add_argument("--rhobeg", type=float, default=0.5)
    parser.add_argument("--tol", type=float, default=0.02)
    parser.add_argument("--outdir", type=Path, default=Path("tomography_3spin"))
    parser.add_argument("--save-circuits", action="store_true", help="save one PNG circuit diagram per optimized beta/sigma")
    parser.add_argument("--print-circuits", action="store_true", help="print ASCII circuit diagrams to the terminal")
    args = parser.parse_args()
    if args.shots < 1 or args.maxiter < 1:
        parser.error("--shots and --maxiter must be positive")
    if any(not np.isfinite(b) or b <= 0 for b in args.betas):
        parser.error("all beta values must be finite and positive")
    betas = tuple(sorted(set(float(b) for b in args.betas)))
    print("=== Three-spin / six-qubit measurement tomography ===")
    print("This run uses p=1, four angles per circuit, and measures subsystem A only.")
    print(f"Tomography settings: {len(TOMOGRAPHY_SETTINGS)} (all X/Y/Z combinations for 3 qubits)")
    print(f"Shots per setting: {args.shots}; settings per beta/sigma: {len(TOMOGRAPHY_SETTINGS)}")
    print(f"Total measurement samples per beta/sigma: {args.shots * len(TOMOGRAPHY_SETTINGS):,}")
    run_local_tomography(betas=betas, shots=args.shots, seed=args.seed, maxiter=args.maxiter,
                         rhobeg=args.rhobeg, tol=args.tol, output_dir=args.outdir,
                         save_circuits=args.save_circuits, print_circuits=args.print_circuits)


if __name__ == "__main__":
    main()

