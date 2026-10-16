"""Validate the 3-spin TFD cost function against Supplementary Note 1.

This script does two things:

1. Runs an independent consistency check of the implemented C_varsigma(beta)
   against the formula in the paper, using independently constructed aggregate
   observables.
2. Reproduces the paper's Xi(varsigma) metric for the 3-spin / 6-qubit
   extension, comparing varsigma=1.00 and varsigma=1.57.

The Xi metric used here is

    Xi(varsigma) = sum_beta sum_O |<TFD(beta)|O|TFD(beta)>
                                  - <Psi(varsigma,beta)|O|Psi(varsigma,beta)>|,

where O contains the 15 aggregate observables from Supplementary Note 1,
extended from two sites to an open three-site chain.

Expected project layout:
    project_root/
      ground_truth.py
      validate_cost_function_3spin.py   # this file
      gibbs_tfd_reproduction/
        __init__.py
        cost_function_3spin.py
        tfd_circuit_3spin.py

Run from the project root in the project's Qiskit environment:
    python validate_cost_function_3spin.py

For a quick smoke test over beta in [0.1, 10]:
    python validate_cost_function_3spin.py --quick

Notes:
- The default beta grid exactly follows the supplied supplement: 10^(x/2),
  for integer x from -8 through 8 (17 beta values, from 1e-4 to 1e4).
- The cost ansatz uses the project's current four-parameters-per-layer
  TFD3SpinCircuit. p is the number of these repeated variational layers.
- The original paper optimized varsigma for its two-spin system. This is a
  three-spin extension of the same diagnostic, not an exact reproduction of
  the published numerical value.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info import DensityMatrix, Pauli, Statevector, partial_trace, state_fidelity

from .cost_function_3spin import CostFunction3Spin
from .tfd_circuit_3spin import TFD3SpinCircuit


PAPER_BETAS = (1.0, 2.5, 5.0, 7.5, 10)
print(f"paper betas {PAPER_BETAS}\n")
SIGMA_VALUES = (1.0, 1.57)
N_QUBITS = 6
N_AGGREGATE_OBSERVABLES = 15


def _kron3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.kron(np.kron(a, b), c)


def three_spin_hamiltonian(g: float = 1.0, periodic: bool = False):
    """Return (eigenvalues, eigenvectors) for H=ZZ+g(X1+X2+X3).

    The default is an open chain: Z1 Z2 + Z2 Z3 + g(X1+X2+X3).
    Matrix tensor-factor order is compatible with the existing project
    convention after accounting for Qiskit's least-significant q0 convention.
    """
    identity = np.array([[1.0, 0.0], [0.0, 1.0]])
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.array([[1.0, 0.0], [0.0, -1.0]])

    h = (
        _kron3(z, z, identity)
        + _kron3(identity, z, z)
        + float(g)
        * (
            _kron3(x, identity, identity)
            + _kron3(identity, x, identity)
            + _kron3(identity, identity, x)
        )
    )
    if periodic:
        h += _kron3(z, identity, z)
    return np.linalg.eigh(h)


def _pauli_string(operators_by_qubit: dict[int, str]) -> str:
    """Build a Qiskit Pauli label; the right-most character acts on q0."""
    return "".join(operators_by_qubit.get(q, "I") for q in reversed(range(N_QUBITS)))


def build_aggregate_observables() -> dict[str, tuple[str, ...]]:
    """Build the paper's 15 aggregate observables for a three-site chain.

    Qubit convention: q0=A1, q1=A2, q2=A3, q3=B1, q4=B2, q5=B3.
    X_A means X_A1+X_A2+X_A3, ZZ_A means Z_A1 Z_A2+Z_A2 Z_A3,
    and XX_BA means X_A1 X_B1+X_A2 X_B2+X_A3 X_B3; similarly for Y and Z.
    """
    observables: dict[str, tuple[str, ...]] = {}

    # Six one-body aggregate operators: X_A, Y_A, Z_A and X_B, Y_B, Z_B.
    for axis in "XYZ":
        observables[f"{axis}_A"] = tuple(
            _pauli_string({q: axis}) for q in (0, 1, 2)
        )
    for axis in "XYZ":
        observables[f"{axis}_B"] = tuple(
            _pauli_string({q: axis}) for q in (3, 4, 5)
        )

    # Two-body correlations on the open chain within systems A and B.
    chain_edges_A = ((0, 1), (1, 2))
    chain_edges_B = ((3, 4), (4, 5))
    cross_edges = ((0, 3), (1, 4), (2, 5))
    for axis in "XYZ":
        label = axis + axis
        observables[f"{label}_A"] = tuple(
            _pauli_string({q1: axis, q2: axis}) for q1, q2 in chain_edges_A
        )
    for axis in "XYZ":
        label = axis + axis
        observables[f"{label}_B"] = tuple(
            _pauli_string({q1: axis, q2: axis}) for q1, q2 in chain_edges_B
        )
    for axis in "XYZ":
        label = axis + axis
        observables[f"{label}_BA"] = tuple(
            _pauli_string({q1: axis, q2: axis}) for q1, q2 in cross_edges
        )

    # Explicitly preserve the order used in the supplementary note.
    order = (
        "X_A", "Y_A", "Z_A", "X_B", "Y_B", "Z_B",
        "XX_A", "YY_A", "ZZ_A", "XX_B", "YY_B", "ZZ_B",
        "XX_BA", "YY_BA", "ZZ_BA",
    )
    return {name: observables[name] for name in order}


OBSERVABLES = build_aggregate_observables()


def aggregate_expectations(state: Statevector) -> dict[str, float]:
    """Expectation of every aggregate observable in the 15-element paper set."""
    result = {}
    for name, pauli_strings in OBSERVABLES.items():
        total = 0.0
        for label in pauli_strings:
            total += float(np.real(state.expectation_value(Pauli(label))))
        result[name] = total
    return result


def exact_tfd_state(beta: float, h_eigendecomposition) -> Statevector:
    """Construct exact |TFD(beta)> = sum_n e^(-beta E_n/2)|n>_B|n>_A/sqrt(Z)."""
    eigenvalues, eigenvectors = h_eigendecomposition
    shifted_energies = np.asarray(eigenvalues, dtype=float) - np.min(eigenvalues)
    weights = np.exp(-float(beta) * shifted_energies / 2.0)

    vector = np.zeros(64, dtype=complex)
    for n in range(len(shifted_energies)):
        eigenstate = eigenvectors[:, n]
        # Ordering matches the project's TFDState convention: np.kron(B, A).
        vector += weights[n] * np.kron(eigenstate, eigenstate)

    norm = np.linalg.norm(vector)
    if norm == 0.0 or not np.isfinite(norm):
        raise ValueError(f"Invalid TFD normalization at beta={beta}: norm={norm}")
    return Statevector(vector / norm)


def exact_gibbs_state(beta: float, h_eigendecomposition) -> DensityMatrix:
    """Construct rho_beta = exp(-beta H)/Z from the exact eigendecomposition."""
    eigenvalues, eigenvectors = h_eigendecomposition
    shifted_energies = np.asarray(eigenvalues, dtype=float) - np.min(eigenvalues)
    weights = np.exp(-float(beta) * shifted_energies)
    probabilities = weights / weights.sum()
    rho = (eigenvectors * probabilities[np.newaxis, :]) @ eigenvectors.conj().T
    return DensityMatrix(rho)


def cost_from_aggregate_expectations(
    values: dict[str, float], beta: float, varsigma: float
) -> float:
    """Evaluate the paper's engineered cost from independently calculated values."""
    if beta == 0:
        return -(values["XX_BA"] + values["ZZ_BA"])
    return float(
        values["X_A"]
        + values["X_B"]
        + varsigma * (values["ZZ_A"] + values["ZZ_B"])
        - beta ** (-varsigma) * (values["XX_BA"] + values["ZZ_BA"])
    )


def check_cost_implementation(p_layers: int = 1, tolerance: float = 1e-9) -> None:
    """Check the cost module against independently built Pauli observables."""
    rng = np.random.default_rng(7321)
    params = rng.uniform(-0.4, 0.4, size=4 * p_layers)
    test_betas = (0.1, 1.0, 10.0)

    print("Cost implementation consistency check:")
    for varsigma in SIGMA_VALUES:
        cost_model = CostFunction3Spin(varsigma=varsigma, p_layers=p_layers, periodic=False)
        for beta in test_betas:
            state = TFD3SpinCircuit(params, p_layers=p_layers, periodic=False).get_state_vector()
            independent_values = aggregate_expectations(state)
            module_values = cost_model.expectation_values(params)
            
            if not np.isclose(
                independent_values["ZZ_A"],
                module_values["ZZ_A"],
                rtol=0.0,
                atol=tolerance,
            ):
                print("\nDEBUG: ZZ_A mismatch")
                print("Parameters:", params)

                print("\nIndependent implementation:")
                for label in OBSERVABLES["ZZ_A"]:
                    value = float(
                        np.real(state.expectation_value(Pauli(label)))
                    )
                    print(f"  {label}: {value:+.12f}")

                print("\nCost module implementation:")
                for label in CostFunction3Spin.OBSERVABLES["ZZ_A"]:
                    module_state = TFD3SpinCircuit(
                        params,
                        p_layers=p_layers,
                        periodic=False,
                    ).get_state_vector()

                    value = float(
                        np.real(module_state.expectation_value(Pauli(label)))
                    )
                    print(f"  {label}: {value:+.12f}")

                print("\nIndependent total:", independent_values["ZZ_A"])
                print("Module total:     ", module_values["ZZ_A"])
                raise AssertionError("ZZ_A mismatch")



            # First compare every expectation used by the engineered cost.
            cost_observables = ("X_A", "X_B", "ZZ_A", "ZZ_B", "XX_BA", "ZZ_BA")
            for name in cost_observables:
                if not np.isclose(
                    independent_values[name], module_values[name],
                    rtol=0.0, atol=tolerance,
                ):
                    raise AssertionError(
                        f"Observable mismatch for {name}, sigma={varsigma}, beta={beta}: "
                        f"independent={independent_values[name]}, module={module_values[name]}"
                    )

            manual_cost = cost_from_aggregate_expectations(independent_values, beta, varsigma)
            implemented_cost = cost_model.cost_function(params, beta)
            if not np.isclose(manual_cost, implemented_cost, rtol=1e-10, atol=tolerance):
                raise AssertionError(
                    f"Cost mismatch for sigma={varsigma}, beta={beta}: "
                    f"independent={manual_cost}, module={implemented_cost}"
                )

        print(f"  PASS: varsigma={varsigma:g}, observables and C_varsigma(beta) agree")
    print()


def evaluate_xi_contribution(
    beta: float,
    optimized_state: Statevector,
    h_eigendecomposition,
):
    """Return Xi's contribution at one beta and the per-observable differences."""
    target_state = exact_tfd_state(beta, h_eigendecomposition)
    target_values = aggregate_expectations(target_state)
    optimized_values = aggregate_expectations(optimized_state)
    errors = {
        name: abs(target_values[name] - optimized_values[name])
        for name in OBSERVABLES
    }
    xi_beta = float(sum(errors.values()))

    rho_A = partial_trace(optimized_state, [3, 4, 5])
    rho_gibbs = exact_gibbs_state(beta, h_eigendecomposition)
    gibbs_fidelity = float(state_fidelity(rho_A, rho_gibbs))
    full_tfd_fidelity = float(state_fidelity(target_state, optimized_state))

    return {
        "xi_beta": xi_beta,
        "mean_abs_observable_error": xi_beta / len(OBSERVABLES),
        "observable_errors": errors,
        "gibbs_fidelity": gibbs_fidelity,
        "full_tfd_fidelity": full_tfd_fidelity,
        "cost": None,
    }


def optimize_for_sigma(
    varsigma: float,
    betas,
    h_eigendecomposition,
    p_layers: int = 1,
    maxiter: int = 200,
    rhobeg: float = 0.5,
    tol: float = 0.02,
):
    """Optimize C_varsigma(beta) at every beta, then calculate Xi contributions."""
    model = CostFunction3Spin(varsigma=varsigma, p_layers=p_layers, periodic=False)
    x0 = np.zeros(4 * p_layers, dtype=float)
    results = {}

    for beta in betas:
        optimizer_result = model.minimize_cost(
            x0=x0,
            beta=float(beta),
            method="COBYLA",
            maxiter=maxiter,
            rhobeg=rhobeg,
            tol=tol,
        )
        params = np.asarray(optimizer_result.x, dtype=float)
        state = TFD3SpinCircuit(params, p_layers=p_layers, periodic=False).get_state_vector()
        score = evaluate_xi_contribution(float(beta), state, h_eigendecomposition)
        score.update(
            {
                "params": params,
                "cost": float(optimizer_result.fun),
                "evaluations": int(getattr(optimizer_result, "nfev", 0)),
                "optimizer_success": bool(optimizer_result.success),
                "optimizer_message": str(optimizer_result.message),
            }
        )
        results[float(beta)] = score

        # Continue the optimization smoothly across beta values, as in the
        # existing extension experiment.
        x0 = params.copy()

        print(
            f"  sigma={varsigma:g}, beta={beta:.6g}: "
            f"Xi_beta={score['xi_beta']:.6g}, "
            f"F_Gibbs={score['gibbs_fidelity']:.8f}, "
            f"F_TFD={score['full_tfd_fidelity']:.8f}, "
            f"nfev={score['evaluations']}"
        )

    return results


def write_outputs(all_results, betas, output_dir: Path, p_layers: int, maxiter: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "xi_by_beta.csv"
    per_observable_path = output_dir / "xi_per_observable.csv"
    summary_path = output_dir / "xi_summary.json"

    summary = {
        "p_layers": int(p_layers),
        "maxiter": int(maxiter),
        "beta_grid": [float(beta) for beta in betas],
        "n_betas": len(betas),
        "n_aggregate_observables": len(OBSERVABLES),
        "sigma_results": {},
    }

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "varsigma", "beta", "xi_beta", "mean_abs_observable_error",
            "gibbs_fidelity", "gibbs_infidelity", "full_tfd_fidelity",
            "full_tfd_infidelity", "cost", "evaluations", "optimizer_success",
        ] + [f"param_{i + 1}" for i in range(4 * p_layers)]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for varsigma, beta_results in all_results.items():
            xi_total = sum(beta_results[float(beta)]["xi_beta"] for beta in betas)
            average_error = xi_total / (len(betas) * len(OBSERVABLES))
            gibbs_fidelities = [beta_results[float(beta)]["gibbs_fidelity"] for beta in betas]
            tfd_fidelities = [beta_results[float(beta)]["full_tfd_fidelity"] for beta in betas]

            summary["sigma_results"][str(varsigma)] = {
                "Xi_total": float(xi_total),
                "mean_abs_observable_error": float(average_error),
                "mean_Gibbs_fidelity": float(np.mean(gibbs_fidelities)),
                "mean_Gibbs_infidelity": float(np.mean(1.0 - np.asarray(gibbs_fidelities))),
                "mean_full_TFD_fidelity": float(np.mean(tfd_fidelities)),
                "mean_full_TFD_infidelity": float(np.mean(1.0 - np.asarray(tfd_fidelities))),
                "Xi_beta": {
                    str(beta): float(beta_results[float(beta)]["xi_beta"])
                    for beta in betas
                },
            }

            for beta in betas:
                record = beta_results[float(beta)]
                row = {
                    "varsigma": float(varsigma),
                    "beta": float(beta),
                    "xi_beta": record["xi_beta"],
                    "mean_abs_observable_error": record["mean_abs_observable_error"],
                    "gibbs_fidelity": record["gibbs_fidelity"],
                    "gibbs_infidelity": 1.0 - record["gibbs_fidelity"],
                    "full_tfd_fidelity": record["full_tfd_fidelity"],
                    "full_tfd_infidelity": 1.0 - record["full_tfd_fidelity"],
                    "cost": record["cost"],
                    "evaluations": record["evaluations"],
                    "optimizer_success": record["optimizer_success"],
                }
                row.update({f"param_{i + 1}": value for i, value in enumerate(record["params"])})
                writer.writerow(row)

    with per_observable_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("varsigma", "beta", "observable", "absolute_expectation_difference"),
        )
        writer.writeheader()
        for varsigma, beta_results in all_results.items():
            for beta in betas:
                errors = beta_results[float(beta)]["observable_errors"]
                for name, error in errors.items():
                    writer.writerow(
                        {
                            "varsigma": float(varsigma),
                            "beta": float(beta),
                            "observable": name,
                            "absolute_expectation_difference": float(error),
                        }
                    )

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Paper's key metric: Xi contribution at each beta, overlaid for both sigma values.
    fig, ax = plt.subplots()
    for varsigma, beta_results in all_results.items():
        ax.plot(
            betas,
            [beta_results[float(beta)]["xi_beta"] for beta in betas],
            marker="o",
            linewidth=1.5,
            label=fr"$\varsigma={varsigma:g}$",
        )
    ax.set_xlabel(r"Inverse temperature $\beta$")
    ax.set_ylabel(r"Per-$\beta$ contribution to $\Xi(\varsigma)$")
    ax.set_title(r"Three-spin cost-function validation: $\Xi$ contributions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "xi_by_beta.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots()
    for varsigma, beta_results in all_results.items():
        ax.plot(
            betas,
            [beta_results[float(beta)]["gibbs_fidelity"] for beta in betas],
            marker="o",
            linewidth=1.5,
            label=fr"$\varsigma={varsigma:g}$",
        )
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel(r"Inverse temperature $\beta$")
    ax.set_ylabel(r"Reduced-state fidelity $F(\rho_A,\rho_\beta)$")
    ax.set_title("Gibbs-state fidelity (additional diagnostic)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "gibbs_fidelity_vs_beta.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots()
    for varsigma, beta_results in all_results.items():
        ax.plot(
            betas,
            [beta_results[float(beta)]["full_tfd_fidelity"] for beta in betas],
            marker="o",
            linewidth=1.5,
            label=fr"$\varsigma={varsigma:g}$",
        )
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel(r"Inverse temperature $\beta$")
    ax.set_ylabel(r"Full-state fidelity $|\langle\mathrm{TFD}|\Psi\rangle|^2$")
    ax.set_title("Full TFD fidelity (additional diagnostic)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "tfd_fidelity_vs_beta.png", dpi=160)
    plt.close(fig)

    return csv_path, per_observable_path, summary_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p", type=int, default=1, help="number of variational layers (default: 1)")
    parser.add_argument("--maxiter", type=int, default=200, help="COBYLA iteration limit per beta")
    parser.add_argument("--rhobeg", type=float, default=0.5, help="initial COBYLA trust-region radius")
    parser.add_argument("--tol", type=float, default=0.02, help="COBYLA final trust-region tolerance")
    parser.add_argument("--outdir", type=Path, default=Path("xi_validation_3spin"))
    args = parser.parse_args()

    if args.p < 1:
        parser.error("--p must be >= 1")
    if args.maxiter < 1:
        parser.error("--maxiter must be >= 1")

    betas = (1.0, 2.5, 5.0, 7.5, 10)
    h_eigendecomposition = three_spin_hamiltonian(g=1.0, periodic=False)

    print("=== Three-spin / six-qubit cost-function validation ===")
    print(f"p = {args.p} variational layers ({4 * args.p} trainable angles in this ansatz)")
    print(f"beta grid ({len(betas)} values): {tuple(float(b) for b in betas)}")
    print(f"aggregate observables in Xi: {len(OBSERVABLES)}")
    print("sigma values:", SIGMA_VALUES)
    print()

    # This validates implementation details independently of optimization.
    check_cost_implementation(p_layers=args.p)

    all_results = {}
    for varsigma in SIGMA_VALUES:
        print(f"Optimizing with varsigma={varsigma:g}:")
        all_results[varsigma] = optimize_for_sigma(
            varsigma=varsigma,
            betas=betas,
            h_eigendecomposition=h_eigendecomposition,
            p_layers=args.p,
            maxiter=args.maxiter,
            rhobeg=args.rhobeg,
            tol=args.tol,
        )
        print()

    paths = write_outputs(all_results, betas, args.outdir, args.p, args.maxiter)

    print("=== Summary ===")
    denominator = len(betas) * len(OBSERVABLES)
    for varsigma, beta_results in all_results.items():
        xi_total = sum(beta_results[float(beta)]["xi_beta"] for beta in betas)
        mean_gibbs_fidelity = np.mean(
            [beta_results[float(beta)]["gibbs_fidelity"] for beta in betas]
        )
        mean_tfd_fidelity = np.mean(
            [beta_results[float(beta)]["full_tfd_fidelity"] for beta in betas]
        )
        print(
            f"varsigma={varsigma:g}: Xi={xi_total:.8g}; "
            f"mean |observable difference|={xi_total / denominator:.8g}; "
            f"mean Gibbs fidelity={mean_gibbs_fidelity:.8f}; "
            f"mean full-TFD fidelity={mean_tfd_fidelity:.8f}"
        )

    print("\nSaved:")
    for path in paths:
        print(f"  {path}")
    print(f"  {args.outdir / 'xi_by_beta.png'}")
    print(f"  {args.outdir / 'gibbs_fidelity_vs_beta.png'}")
    print(f"  {args.outdir / 'tfd_fidelity_vs_beta.png'}")


if __name__ == "__main__":
    main()
