"""Three-spin TFD extension experiment.

Optimization uses the existing CostFunction3Spin implementation. For each beta,
the ansatz and all 27 measured tomography circuits are saved; tomography
circuits are transpiled before saving/submission so the saved version is the
one sent to the backend.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile, qasm3
from qiskit.quantum_info import (
    DensityMatrix, Statevector, partial_trace, state_fidelity,
)
from qiskit.visualization import circuit_drawer

from .cost_function_3spin import CostFunction3Spin
from .tfd_circuit_3spin import TFD3SpinCircuit
from .validate_cost_function_3spin import exact_gibbs_state, three_spin_hamiltonian
from .tomography_3spin import plot_gibbs_tomography_comparison


VARSIGMA = 2.05
DEFAULT_BETAS = (0.00001, 1.0, 5.0)
SHOTS_PER_SETTING = 4096
BACKEND_NAME = "QX emulator"
OUTPUT_ROOT = Path("outputs/three_spin")
CIRCUIT_DIR = OUTPUT_ROOT / "circuits"
CONVERGENCE_DIR = OUTPUT_ROOT / "cost_convergence"
TOMOGRAPHY_DIR = OUTPUT_ROOT / "tomography"
RESULTS_DIR = OUTPUT_ROOT / "results"
PAULI_SETTINGS = tuple("".join(x) for x in product("XYZ", repeat=3))
PAULI_MATRICES = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.diag([1, -1]).astype(complex),
}


def safe_beta(beta: float) -> str:
    return f"{beta:g}".replace(".", "p").replace("-", "m")


def make_backend():
    """Return the Quantum Inspire backend selected for this experiment."""
    from qiskit_quantuminspire.qi_provider import QIProvider

    provider = QIProvider()
    return provider.get_backend(name=BACKEND_NAME)


def matrix_trace_distance(rho_a, rho_b) -> float:
    """Trace distance, implemented locally for compatibility with older Qiskit."""
    delta = np.asarray(rho_a, dtype=complex) - np.asarray(rho_b, dtype=complex)
    singular_values = np.linalg.svd(delta, compute_uv=False)
    return float(0.5 * np.sum(singular_values))


def exact_cost_bound(beta: float) -> float:
    """Exact spectral lower bound of the engineered cost operator."""
    return float(CostFunction3Spin.cost_lower_bound(beta, VARSIGMA, periodic=False))


def naive_cost_bound(beta: float) -> float:
    """Loose term-wise lower bound, obtained by minimizing each term separately."""
    if beta == 0:
        return -6.0
    return -6.0 - 4.0 * VARSIGMA - 6.0 * beta ** (-VARSIGMA)


def save_circuit(circuit: QuantumCircuit, stem: str) -> None:
    """Save the submitted circuit as QASM 3, PNG, and readable text."""
    CIRCUIT_DIR.mkdir(parents=True, exist_ok=True)
    (CIRCUIT_DIR / f"{stem}.qasm").write_text(qasm3.dumps(circuit), encoding="utf-8")
    (CIRCUIT_DIR / f"{stem}.txt").write_text(
        str(circuit.draw(output="text", fold=-1)), encoding="utf-8"
    )
    fig = circuit_drawer(circuit, output="mpl", fold=-1, idle_wires=False)
    fig.savefig(CIRCUIT_DIR / f"{stem}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def make_tomography_circuit(ansatz: QuantumCircuit, setting: str) -> QuantumCircuit:
    """Rotate subsystem A into the requested X/Y/Z basis and measure all qubits."""
    qc = ansatz.copy()
    qc.name = f"tomo_{setting}"
    for q, basis in enumerate(setting):
        if basis == "X":
            qc.h(q)
        elif basis == "Y":
            qc.sdg(q)
            qc.h(q)
    qc.measure_all()
    return qc


def _bits_for_A(bitstring: str) -> str:
    # Qiskit displays classical bit 5 ... bit 0; q0 is the rightmost bit.
    bits = bitstring.replace(" ", "")[::-1]
    return "".join(bits[q] for q in range(3))


def _expectation_from_counts(counts: dict, setting: str, pauli: str) -> float:
    """Estimate <pauli> using the counts measured in `setting`."""
    total = sum(counts.values())
    if total == 0:
        raise ValueError("Backend returned no shots.")
    if any(p != "I" and setting[q] != p for q, p in enumerate(pauli)):
        raise ValueError(f"Setting {setting} cannot measure Pauli string {pauli}.")
    value = 0.0
    for bitstring, count in counts.items():
        bits = _bits_for_A(bitstring)
        parity = sum(
            1 for q, p in enumerate(pauli) if p != "I" and bits[q] == "1"
        )
        value += (-1.0 if parity % 2 else 1.0) * count
    return value / total


def reconstruct_density_matrix(counts_by_setting: dict[str, dict]) -> DensityMatrix:
    """Linear-inversion reconstruction of the three-qubit subsystem-A state."""
    rho = np.zeros((8, 8), dtype=complex)
    for chars in product("IXYZ", repeat=3):
        pauli = "".join(chars)
        if pauli == "III":
            expectation = 1.0
        else:
            setting = "".join("Z" if p == "I" else p for p in pauli)
            expectation = _expectation_from_counts(
                counts_by_setting[setting], setting, pauli
            )
        matrix = np.kron(np.kron(PAULI_MATRICES[pauli[2]],
                                 PAULI_MATRICES[pauli[1]]),
                         PAULI_MATRICES[pauli[0]])
        rho += expectation * matrix / 8.0
    return DensityMatrix(rho)


def project_to_physical_density_matrix(rho) -> DensityMatrix:
    """Hermitize, clip negative eigenvalues, and renormalize trace."""
    matrix = rho.data if isinstance(rho, DensityMatrix) else np.asarray(rho, dtype=complex)
    matrix = (matrix + matrix.conj().T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    if eigenvalues.sum() <= 0:
        raise ValueError("Tomography reconstruction could not be projected to a state.")
    eigenvalues /= eigenvalues.sum()
    physical = (eigenvectors * eigenvalues) @ eigenvectors.conj().T
    return DensityMatrix(physical)


def plot_cost_convergence(beta: float, history: np.ndarray) -> None:
    CONVERGENCE_DIR.mkdir(parents=True, exist_ok=True)
    exact_bound = exact_cost_bound(beta)
    naive_bound = naive_cost_bound(beta)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(1, len(history) + 1), history, marker=".", ms=3,
            linewidth=1, label="Optimizer evaluations")
    ax.axhline(exact_bound, linestyle="--", linewidth=1.4,
               label=f"Exact spectral lower bound = {exact_bound:.4f}")
    ax.axhline(naive_bound, linestyle=":", linewidth=1.4,
               label=f"Term-wise lower bound = {naive_bound:.4f}")
    ax.set_xlabel("Function evaluation")
    ax.set_ylabel(r"$C_{\varsigma}(\beta)$")
    ax.set_title(f"Three-spin cost convergence, beta={beta:g}, varsigma={VARSIGMA:g}")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(CONVERGENCE_DIR / f"cost_beta_{safe_beta(beta)}.png", dpi=180)
    plt.close(fig)


def run_experiment(
    betas=DEFAULT_BETAS, shots: int = SHOTS_PER_SETTING, maxiter: int = 200,
    rhobeg: float = 0.5, tol: float = 0.02, seed: int = 1234,
    local_sampling: bool = False,
):
    # Start each run with a clean output tree so files from previous runs
    # (including results for betas not selected this time) cannot remain.
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    for folder in (CIRCUIT_DIR, CONVERGENCE_DIR, TOMOGRAPHY_DIR, RESULTS_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    backend = None if local_sampling else make_backend()
    h_eigendecomposition = three_spin_hamiltonian(g=1.0, periodic=False)
    rng = np.random.default_rng(seed)
    optimizer = CostFunction3Spin(varsigma=VARSIGMA, periodic=False)
    rows = []
    x0 = np.zeros(4, dtype=float)

    print("Three-spin TFD experiment")
    print(f"varsigma = {VARSIGMA}")
    print(f"backend = {'local statevector sampler' if local_sampling else BACKEND_NAME}")
    print(f"betas = {tuple(betas)}")
    print(f"output root = {OUTPUT_ROOT.resolve()}\n")

    for beta in betas:
        beta = float(beta)
        result = optimizer.minimize_cost(
            x0=x0, beta=beta, maxiter=maxiter, rhobeg=rhobeg, tol=tol
        )
        params = np.asarray(result.x, dtype=float)
        history = np.asarray(optimizer.cost_history, dtype=float).copy()
        x0 = params.copy()

        ansatz = TFD3SpinCircuit(params).as_qiskit_circuit()
        save_circuit(ansatz, f"beta_{safe_beta(beta)}_ansatz")
        plot_cost_convergence(beta, history)

        rho_gibbs = exact_gibbs_state(beta, h_eigendecomposition)
        rho_ideal_a = partial_trace(Statevector(ansatz), [3, 4, 5])
        ideal_fidelity = float(state_fidelity(rho_ideal_a, rho_gibbs))

        counts_by_setting = {}
        for setting in PAULI_SETTINGS:
            circuit = make_tomography_circuit(ansatz, setting)
            if local_sampling:
                # Apply basis changes, then sample the ideal statevector.
                rotated = ansatz.copy()
                for q, basis in enumerate(setting):
                    if basis == "X":
                        rotated.h(q)
                    elif basis == "Y":
                        rotated.sdg(q)
                        rotated.h(q)
                probabilities = Statevector(rotated).probabilities()
                samples = rng.multinomial(shots, probabilities)
                counts = {
                    format(i, "06b")[::-1]: int(n)
                    for i, n in enumerate(samples) if n
                }
                submitted = circuit
            else:
                submitted = transpile(circuit, backend=backend, optimization_level=1)
            # Save the exact transpiled circuit that will be sent to the backend.
            save_circuit(submitted, f"beta_{safe_beta(beta)}_tomo_{setting}")
            if not local_sampling:
                job = backend.run(submitted, shots=shots)
                counts = job.result().get_counts(0)
            counts_by_setting[setting] = counts

        rho_raw = reconstruct_density_matrix(counts_by_setting)
        rho_physical = project_to_physical_density_matrix(rho_raw)
        tomography_fidelity = float(state_fidelity(rho_physical, rho_gibbs))
        infidelity = 1.0 - tomography_fidelity
        distance = matrix_trace_distance(rho_physical.data, rho_gibbs.data)

        comparison_path = TOMOGRAPHY_DIR / f"gibbs_comparison_beta_{safe_beta(beta)}.png"
        plot_gibbs_tomography_comparison(
            rho_gibbs, rho_physical, beta=beta, sigma=VARSIGMA,
            output_path=comparison_path, show=False,
        )

        np.savez_compressed(
            RESULTS_DIR / f"beta_{safe_beta(beta)}_density_matrices.npz",
            rho_gibbs=rho_gibbs.data,
            rho_tomography_raw=rho_raw.data,
            rho_tomography_physical=rho_physical.data,
            rho_ideal_a=rho_ideal_a.data,
        )

        row = {
            "beta": beta,
            "varsigma": VARSIGMA,
            "final_cost": float(history[-1]),
            "exact_spectral_lower_bound": exact_cost_bound(beta),
            "termwise_lower_bound": naive_cost_bound(beta),
            "evaluations": int(len(history)),
            "optimizer_success": bool(result.success),
            "optimizer_message": str(result.message),
            "ideal_statevector_gibbs_fidelity": ideal_fidelity,
            "tomography_gibbs_fidelity": tomography_fidelity,
            "tomography_infidelity": infidelity,
            "tomography_trace_distance": distance,
            "shots_per_setting": int(shots),
            "backend": "local statevector sampler" if local_sampling else BACKEND_NAME,
        }
        rows.append(row)
        print(
            f"beta={beta:g}: cost={row['final_cost']:.6f}; "
            f"exact bound={row['exact_spectral_lower_bound']:.6f}; "
            f"term-wise bound={row['termwise_lower_bound']:.6f}; "
            f"ideal fidelity={ideal_fidelity:.6f}; "
            f"tomography fidelity={tomography_fidelity:.6f}"
        )

    with (RESULTS_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "varsigma": VARSIGMA,
        "betas": [float(b) for b in betas],
        "shots_per_setting": int(shots),
        "backend": "local statevector sampler" if local_sampling else BACKEND_NAME,
        "ideal_simulation": not local_sampling,
        "qubit_convention": "q0=A1, q1=A2, q2=A3, q3=B1, q4=B2, q5=B3",
        "circuit_directory": str(CIRCUIT_DIR),
    }
    (RESULTS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved summary to {RESULTS_DIR / 'summary.csv'}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--betas", nargs="+", type=float, default=list(DEFAULT_BETAS))
    parser.add_argument("--shots", type=int, default=SHOTS_PER_SETTING)
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--rhobeg", type=float, default=0.5)
    parser.add_argument("--tol", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--local-sampling", action="store_true",
        help="Use local statevector sampling rather than the Quantum Inspire simulator.",
    )
    args = parser.parse_args()
    run_experiment(
        betas=args.betas, shots=args.shots, maxiter=args.maxiter,
        rhobeg=args.rhobeg, tol=args.tol, seed=args.seed,
        local_sampling=args.local_sampling,
    )


if __name__ == "__main__":
    main()

