"""First numerical test of the 3-spin / 6-qubit TFD extension.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace, state_fidelity

from cost_function_3spin import CostFunction3Spin
from tfd_circuit_3spin import TFD3SpinCircuit
from ground_truth import GibbsState, TFDState, default_three_spin_subsystem_hamiltonian


DEFAULT_BETAS = (1e-6, 0.5, 1.0, 2.0, 5.0)


def verify_exact_tfd(beta, H_AA):
    """Verify Tr_B |TFD(beta)><TFD(beta)| = rho_Gibbs(beta)."""
    gibbs = GibbsState(H_AA)
    rho_gibbs = DensityMatrix(gibbs.gibbs_density_matrix(beta))

    tfd = TFDState(H_AA, H_AA)
    state = Statevector(tfd.tfd_state_vector(beta))

    # q0,q1,q2 = A; q3,q4,q5 = B.
    rho_A = partial_trace(state, [3, 4, 5])
    fidelity = state_fidelity(rho_A, rho_gibbs)
    return float(fidelity)


def optimize_for_sigma(
    varsigma=1.57,
    betas=DEFAULT_BETAS,
    p_layers=1,
    maxiter=200,
    rhobeg=0.5,
    tol=0.02,
):
    """Optimize at each beta, warm-starting each beta from the previous one."""
    cost_function = CostFunction3Spin(
        varsigma=varsigma,
        p_layers=p_layers,
        periodic=False,
    )

    # The beta=0 TFD has gamma=alpha=0 as the natural ideal starting point.
    x0 = np.zeros(4 * p_layers, dtype=float)
    results = {}

    for beta in betas:
        result = cost_function.minimize_cost(
            x0=x0,
            beta=beta,
            maxiter=maxiter,
            rhobeg=rhobeg,
            tol=tol,
        )

        params = np.asarray(result.x, dtype=float)
        history = np.asarray(cost_function.cost_history, dtype=float).copy()

        results[beta] = {
            "params": params,
            "cost_history": history,
            "final_cost": float(history[-1]),
            "lower_bound": CostFunction3Spin.cost_lower_bound(beta, varsigma),
            "evaluations": len(history),
            "optimizer_success": bool(result.success),
            "optimizer_message": str(result.message),
        }

        # Warm start: this is the same continuation strategy used in the
        # two-spin experiment, where the optimum at the previous beta becomes
        # the initial guess for the next beta.
        x0 = params

    return results


def evaluate_optimized_state(beta, params, H_AA, p_layers=1):
    """Trace B out of the optimized 6-qubit state and calculate Gibbs fidelity."""
    circuit = TFD3SpinCircuit(params, p_layers=p_layers, periodic=False)
    state = circuit.get_state_vector()
    rho_A = partial_trace(state, [3, 4, 5])

    target = DensityMatrix(GibbsState(H_AA).gibbs_density_matrix(beta))
    fidelity = state_fidelity(rho_A, target)
    return rho_A, float(fidelity)


def plot_cost_history(results, varsigma, output_dir="plots_3spin"):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    for beta, data in results.items():
        fig, ax = plt.subplots()
        x = np.arange(1, len(data["cost_history"]) + 1)

        ax.plot(
            x,
            data["cost_history"],
            marker="o",
            markersize=3,
            linewidth=1,
            label="Optimizer",
        )
        ax.axhline(
            data["lower_bound"],
            color="red",
            linestyle="--",
            linewidth=1.3,
            label=f"Exact lower bound = {data['lower_bound']:.3f}",
        )
        ax.set_xlabel("Function evaluation")
        ax.set_ylabel("Cost C(beta)")
        ax.set_title(
            f"3-spin cost convergence, beta={beta:g}, varsigma={varsigma:g}"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            output / f"cost_beta{beta:g}_sigma{varsigma:g}.png",
            dpi=140,
        )
        plt.close(fig)


def run_extension_experiment(
    varsigma=1.57,
    betas=DEFAULT_BETAS,
    p_layers=1,
    maxiter=200,
    rhobeg=0.5,
    tol=0.02,
    output_dir="plots_3spin",
):
    """Run the complete 3-spin numerical extension.

    varsigma is a genuine experiment parameter: change it here without
    changing the cost-function implementation.  This makes it straightforward
    to compare, e.g., varsigma=1.00 and varsigma=1.57 before doing a full
    nested optimization over varsigma.
    """
    H_AA = default_three_spin_subsystem_hamiltonian(g=1.0, periodic=False)

    print("=== 3-spin / 6-qubit TFD extension ===")
    print(f"varsigma = {varsigma}")
    print(f"QAOA layers p = {p_layers}")
    print(f"betas = {tuple(betas)}")
    print()

    # Step 1: exact construction sanity check.
    for beta in betas:
        fidelity = verify_exact_tfd(beta, H_AA)
        print(f"exact TFD -> Gibbs fidelity, beta={beta:g}: {fidelity:.12f}")

    print()

    # Step 2: variational optimization, with beta continuation / warm starts.
    results = optimize_for_sigma(
        varsigma=varsigma,
        betas=betas,
        p_layers=p_layers,
        maxiter=maxiter,
        rhobeg=rhobeg,
        tol=tol,
    )

    # Step 3: physical metric -- fidelity of reduced variational state to Gibbs.
    for beta, data in results.items():
        rho_A, gibbs_fidelity = evaluate_optimized_state(
            beta,
            data["params"],
            H_AA,
            p_layers=p_layers,
        )
        data["rho_A"] = rho_A
        data["gibbs_fidelity"] = gibbs_fidelity

        print(
            f"beta={beta:g}: "
            f"cost={data['final_cost']:.6f}, "
            f"lower_bound={data['lower_bound']:.6f}, "
            f"evaluations={data['evaluations']}, "
            f"Gibbs fidelity={gibbs_fidelity:.8f}"
        )
        print(f"  params={np.round(data['params'], 6)}")
        print(
            f"  optimizer success={data['optimizer_success']}: "
            f"{data['optimizer_message']}"
        )

    plot_cost_history(results, varsigma, output_dir=output_dir)
    return results


def scan_varsigma(
    varsigma_values=(1.0, 1.25, 1.5, 1.57, 1.75, 2.0),
    betas=DEFAULT_BETAS,
    p_layers=1,
    maxiter=200,
):
    """Convenience wrapper for comparing candidate varsigma values.

    This is not yet the nested optimization from Supplementary Note 1; it is
    a first small-system comparison.  Once the 3-spin experiment is working,
    the next step is to define the same aggregate TFD mismatch used there and
    minimize it over varsigma.
    """
    summary = {}
    for varsigma in varsigma_values:
        results = run_extension_experiment(
            varsigma=varsigma,
            betas=betas,
            p_layers=p_layers,
            maxiter=maxiter,
        )
        fidelities = [results[b]["gibbs_fidelity"] for b in betas]
        summary[float(varsigma)] = {
            "mean_infidelity": float(np.mean(1.0 - np.asarray(fidelities))),
            "results": results,
        }
        print(
            f"varsigma={varsigma:g}: "
            f"mean Gibbs infidelity={summary[float(varsigma)]['mean_infidelity']:.8e}"
        )
    return summary


if __name__ == "__main__":
    # First baseline: use the published two-spin value 1.57 unchanged.
    # After this works, compare sigma values with scan_varsigma(...).
    run_extension_experiment(
        varsigma=1.57,
        betas=DEFAULT_BETAS,
        p_layers=1,
        maxiter=200,
        rhobeg=0.5,
        tol=0.02,
    )

