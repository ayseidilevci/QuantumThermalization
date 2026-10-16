"""Three-spin sigma optimization and Gibbs-state infidelity comparison.

Stages
------
1. Coarse scan of Xi(sigma) over sigma in [1, 4].
2. Refined scan around the best coarse sigma.
3. Compare the cost function and Gibbs-state infidelity against beta.

The beta grid is denser around beta = 1 to resolve differences between
sigma values in the region where the variational preparation struggles.

Reference sigma values:
    1.00, 1.57, 2.57

Additional sigma values:
    The best candidates according to the completed Xi scan.

Beta plots use a logarithmic beta axis, matching the supplied reference.

Run from the project root:
    python -m gibbs_tfd_reproduction.xi_validation_3spin

Restart from scratch:
    python -m gibbs_tfd_reproduction.xi_validation_3spin --restart

Outputs:
    outputs/xi_validation_3spin/
"""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .validate_cost_function_3spin import (
    OBSERVABLES,
    optimize_for_sigma,
    three_spin_hamiltonian,
)


# =========================================================
# Configuration
# =========================================================

# Dense sampling around beta = 1, with a few points at the extremes.
BETAS = np.array([
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
    1.1,
    1.2,
    1.3,
    1.5,
    1.75,
    2.0,
    2.5,
    3.0,
    4.0,
    5.0,
    7.5,
    10.0,
], dtype=float)

SIGMA_MIN = 1.0
SIGMA_MAX = 4.0

COARSE_STEP = 0.05
FINE_STEP = 0.01
FINE_HALF_WIDTH = 0.20

P_LAYERS = 1

REFERENCE_SIGMAS = [1.0, 1.57, 2.57]
NUMBER_OF_ADDITIONAL_BEST_SIGMAS = 7

DEFAULT_OUTDIR = Path("outputs/xi_validation_3spin")

RESULT_FIELDS = [
    "sigma",
    "beta",
    "xi_beta",
    "gibbs_fidelity",
    "gibbs_infidelity",
    "full_tfd_fidelity",
    "optimizer_success",
    "evaluations",
]


# =========================================================
# Sigma grids
# =========================================================

def make_coarse_grid():
    count = round((SIGMA_MAX - SIGMA_MIN) / COARSE_STEP)

    return np.round(
        np.linspace(SIGMA_MIN, SIGMA_MAX, count + 1),
        8,
    )


def make_fine_grid(best_coarse_sigma):
    lower = max(
        SIGMA_MIN,
        best_coarse_sigma - FINE_HALF_WIDTH,
    )

    upper = min(
        SIGMA_MAX,
        best_coarse_sigma + FINE_HALF_WIDTH,
    )

    first_index = math.ceil(lower / FINE_STEP - 1e-10)
    last_index = math.floor(upper / FINE_STEP + 1e-10)

    return np.round(
        np.arange(first_index, last_index + 1) * FINE_STEP,
        8,
    )


# =========================================================
# General utilities
# =========================================================

def is_complete(beta_results):
    return all(float(beta) in beta_results for beta in BETAS)


def total_xi(beta_results):
    if not is_complete(beta_results):
        raise ValueError(
            "Cannot calculate total Xi because beta results are incomplete."
        )

    return float(sum(
        beta_results[float(beta)]["xi_beta"]
        for beta in BETAS
    ))


def load_results(output_dir):
    """Load cached results, retaining only beta values in the current grid."""

    csv_path = output_dir / "sigma_sweep_results.csv"

    if not csv_path.exists():
        return {}

    results = {}

    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            sigma = float(row["sigma"])
            beta = float(row["beta"])

            if not any(
                math.isclose(
                    beta,
                    float(target),
                    rel_tol=1e-12,
                    abs_tol=0.0,
                )
                for target in BETAS
            ):
                continue

            if sigma not in results:
                results[sigma] = {}

            fidelity = float(row["gibbs_fidelity"])

            results[sigma][beta] = {
                "xi_beta": float(row["xi_beta"]),
                "gibbs_fidelity": fidelity,
                "gibbs_infidelity": 1.0 - fidelity,
                "full_tfd_fidelity": float(
                    row["full_tfd_fidelity"]
                ),
                "optimizer_success": (
                    row["optimizer_success"].strip().lower()
                    in {"true", "1", "yes"}
                ),
                "evaluations": int(float(row["evaluations"])),
            }

    return results


def save_results(results, output_dir):
    """Save detailed CSV data and an aggregate JSON summary."""

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "sigma_sweep_results.csv"
    json_path = output_dir / "sigma_sweep_summary.json"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=RESULT_FIELDS,
        )

        writer.writeheader()

        for sigma in sorted(results):
            beta_results = results[sigma]

            for beta in BETAS:
                beta = float(beta)

                if beta not in beta_results:
                    continue

                record = beta_results[beta]

                writer.writerow({
                    "sigma": sigma,
                    "beta": beta,
                    "xi_beta": record["xi_beta"],
                    "gibbs_fidelity": record["gibbs_fidelity"],
                    "gibbs_infidelity": (
                        1.0 - record["gibbs_fidelity"]
                    ),
                    "full_tfd_fidelity": record["full_tfd_fidelity"],
                    "optimizer_success": record["optimizer_success"],
                    "evaluations": record["evaluations"],
                })

    summary = {
        "beta_grid": BETAS.tolist(),
        "sigma_range": [SIGMA_MIN, SIGMA_MAX],
        "coarse_step": COARSE_STEP,
        "fine_step": FINE_STEP,
        "fine_half_width": FINE_HALF_WIDTH,
        "p_layers": P_LAYERS,
        "number_of_observables": len(OBSERVABLES),
        "sigma_results": {},
    }

    for sigma, beta_results in sorted(results.items()):
        complete = is_complete(beta_results)

        summary["sigma_results"][str(sigma)] = {
            "Xi_total": (
                total_xi(beta_results)
                if complete
                else None
            ),
            "completed_beta_values": sum(
                float(beta) in beta_results
                for beta in BETAS
            ),
            "mean_gibbs_infidelity": (
                float(np.mean([
                    beta_results[float(beta)]["gibbs_infidelity"]
                    for beta in BETAS
                    if float(beta) in beta_results
                ]))
                if beta_results
                else None
            ),
        }

    json_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


# =========================================================
# Optimization
# =========================================================

def evaluate_sigma(
    sigma,
    results,
    h_eigendecomposition,
    maxiter,
    output_dir,
):
    """Evaluate a sigma or reuse its cached complete results."""

    sigma = float(sigma)
    cached = results.get(sigma, {})

    if is_complete(cached):
        print(f"sigma={sigma:.2f}: complete cached results.")
        return

    print(f"\nOptimizing sigma={sigma:.2f}")

    beta_results = optimize_for_sigma(
        varsigma=sigma,
        betas=BETAS,
        h_eigendecomposition=h_eigendecomposition,
        p_layers=P_LAYERS,
        maxiter=maxiter,
        rhobeg=0.5,
        tol=0.02,
    )

    # Convert fidelity into infidelity for convenient plotting and storage.
    for beta, record in beta_results.items():
        record["gibbs_infidelity"] = (
            1.0 - record["gibbs_fidelity"]
        )

    results[sigma] = beta_results

    # Save immediately after every completed sigma.
    save_results(results, output_dir)

    if is_complete(beta_results):
        print(f"Xi({sigma:.2f}) = {total_xi(beta_results):.10f}")
    else:
        print(
            f"Warning: incomplete beta results for sigma={sigma:.2f}."
        )


def complete_xi_results(results):
    return {
        sigma: total_xi(beta_results)
        for sigma, beta_results in results.items()
        if is_complete(beta_results)
    }


def select_diagnostic_sigmas(results):
    """Include reference sigmas and the best additional Xi candidates."""

    complete = complete_xi_results(results)
    selected = []

    for target in REFERENCE_SIGMAS:
        matching = next(
            (
                sigma
                for sigma in complete
                if math.isclose(
                    sigma,
                    target,
                    rel_tol=0.0,
                    abs_tol=1e-8,
                )
            ),
            None,
        )

        if matching is not None and matching not in selected:
            selected.append(matching)

    additional_count = 0

    for sigma, _ in sorted(
        complete.items(),
        key=lambda item: item[1],
    ):
        if any(
            math.isclose(
                sigma,
                existing,
                rel_tol=0.0,
                abs_tol=1e-8,
            )
            for existing in selected
        ):
            continue

        selected.append(sigma)
        additional_count += 1

        if additional_count >= NUMBER_OF_ADDITIONAL_BEST_SIGMAS:
            break

    return selected


# =========================================================
# Plot 1: coarse Xi scan
# =========================================================

def plot_coarse_scan(results, coarse_sigmas, output_dir):
    xi_results = complete_xi_results(results)

    available = [
        float(sigma)
        for sigma in coarse_sigmas
        if float(sigma) in xi_results
    ]

    if not available:
        print("No completed coarse-scan results available.")
        return

    values = np.array([xi_results[sigma] for sigma in available])

    best_index = int(np.argmin(values))
    best_sigma = available[best_index]
    best_xi = float(values[best_index])

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        available,
        values,
        "k.",
        markersize=5,
        label=r"$\Xi(\varsigma)$",
    )

    ax.scatter(
        [best_sigma],
        [best_xi],
        marker="x",
        s=100,
        linewidths=2,
        color="red",
        label=fr"Best coarse point: $\varsigma={best_sigma:.2f}$",
        zorder=3,
    )

    ax.set_xlim(SIGMA_MIN - 0.05, SIGMA_MAX + 0.05)
    ax.set_xlabel(r"Cost-function parameter $\varsigma$")
    ax.set_ylabel(r"$\Xi(\varsigma)$")
    ax.set_title(r"Three-spin coarse scan of $\Xi(\varsigma)$")
    ax.legend()

    fig.tight_layout()

    path = output_dir / "xi_vs_sigma_coarse.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")
    print(f"Best coarse sigma: {best_sigma:.2f}")
    print(f"Best coarse Xi: {best_xi:.10f}")


# =========================================================
# Plot 2: refined Xi scan
# =========================================================

def plot_refined_scan(results, fine_sigmas, output_dir):
    xi_results = complete_xi_results(results)

    available = [
        float(sigma)
        for sigma in fine_sigmas
        if float(sigma) in xi_results
    ]

    if not available:
        print("No completed refined-scan results available.")
        return

    values = np.array([xi_results[sigma] for sigma in available])

    best_index = int(np.argmin(values))
    best_sigma = available[best_index]
    best_xi = float(values[best_index])

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        available,
        values,
        "k.",
        markersize=6,
        label=r"$\Xi(\varsigma)$",
    )

    if any(
        math.isclose(sigma, 1.57, abs_tol=1e-8)
        for sigma in available
    ):
        ax.scatter(
            [1.57],
            [xi_results[1.57]],
            marker="x",
            s=90,
            linewidths=2,
            color="blue",
            label=r"$\varsigma=1.57$",
        )

    if any(
        math.isclose(sigma, 2.57, abs_tol=1e-8)
        for sigma in available
    ):
        ax.scatter(
            [2.57],
            [xi_results[2.57]],
            marker="x",
            s=90,
            linewidths=2,
            color="green",
            label=r"$\varsigma=2.57$",
        )

    ax.scatter(
        [best_sigma],
        [best_xi],
        marker="x",
        s=100,
        linewidths=2,
        color="red",
        label=fr"Refined minimum: $\varsigma={best_sigma:.2f}$",
        zorder=4,
    )

    ax.set_xlabel(r"Cost-function parameter $\varsigma$")
    ax.set_ylabel(r"$\Xi(\varsigma)$")
    ax.set_title(r"Refined scan of $\Xi(\varsigma)$")
    ax.legend()

    fig.tight_layout()

    path = output_dir / "xi_vs_sigma_refined.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")
    print(f"Best refined sigma: {best_sigma:.2f}")
    print(f"Best refined Xi: {best_xi:.10f}")


# =========================================================
# Plot 3: Gibbs-state infidelity versus beta
# =========================================================

def plot_gibbs_infidelity(results, selected_sigmas, output_dir):
    """Plot 1 - reduced Gibbs-state fidelity against beta."""

    fig, ax = plt.subplots(figsize=(9, 6))

    for sigma in selected_sigmas:
        if sigma not in results:
            continue

        beta_results = results[sigma]

        available_betas = [
            float(beta)
            for beta in BETAS
            if float(beta) in beta_results
        ]

        if not available_betas:
            continue

        infidelities = [
            1.0 - beta_results[beta]["gibbs_fidelity"]
            for beta in available_betas
        ]

        ax.plot(
            available_betas,
            infidelities,
            linewidth=2.0 if sigma in REFERENCE_SIGMAS else 1.3,
            label=fr"$C_{{{sigma:g}}}(\beta)$",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(r"Inverse temperature $\beta$")
    ax.set_ylabel(r"Infidelity to Gibbs state $1-F(\rho_A,\rho_\beta)$")
    ax.set_title("Three-spin Gibbs-state infidelity")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()

    path = output_dir / "gibbs_state_infidelity_vs_beta.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


# =========================================================
# Plot 4: per-beta Xi contribution
# =========================================================

def plot_xi_contribution(results, selected_sigmas, output_dir):
    """Plot per-beta observable discrepancy against beta."""

    fig, ax = plt.subplots(figsize=(9, 6))

    for sigma in selected_sigmas:
        if sigma not in results:
            continue

        beta_results = results[sigma]

        available_betas = [
            float(beta)
            for beta in BETAS
            if float(beta) in beta_results
        ]

        if not available_betas:
            continue

        discrepancies = [
            beta_results[beta]["xi_beta"]
            for beta in available_betas
        ]

        ax.plot(
            available_betas,
            discrepancies,
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=fr"$\varsigma={sigma:g}$",
        )

    ax.set_xscale("log")
    ax.set_xlabel(r"Inverse temperature $\beta$")
    ax.set_ylabel(r"Per-$\beta$ discrepancy $\Xi_\beta(\varsigma)$")
    ax.set_title("Three-spin observable discrepancy versus beta")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()

    path = output_dir / "xi_contribution_vs_beta.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--maxiter",
        type=int,
        default=200,
        help="Maximum COBYLA iterations per beta.",
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Output directory.",
    )

    parser.add_argument(
        "--restart",
        action="store_true",
        help="Ignore cached results and rerun all requested sigmas.",
    )

    args = parser.parse_args()

    if args.maxiter < 1:
        parser.error("--maxiter must be positive.")

    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.restart:
        results = {}
        print("Restart requested: ignoring cached results.")
    else:
        results = load_results(args.outdir)

        if results:
            print(f"Loaded cached results for {len(results)} sigma values.")

    coarse_sigmas = make_coarse_grid()

    print("=" * 65)
    print("THREE-SPIN SIGMA SWEEP AND GIBBS INFIDELITY")
    print("=" * 65)
    print(f"Beta grid: {BETAS.tolist()}")
    print(f"Coarse sigma grid: [{SIGMA_MIN}, {SIGMA_MAX}]")
    print(f"Coarse step: {COARSE_STEP}")
    print(f"Fine step: {FINE_STEP}")
    print(f"Ansatz layers: {P_LAYERS}")
    print(f"Output directory: {args.outdir.resolve()}")
    print("=" * 65)

    h_eigendecomposition = three_spin_hamiltonian(
        g=1.0,
        periodic=False,
    )

    # Stage 1: coarse scan.
    print("\nSTAGE 1: COARSE XI SCAN")

    for index, sigma in enumerate(coarse_sigmas, start=1):
        print(
            f"\nCoarse point {index}/{len(coarse_sigmas)}: "
            f"sigma={sigma:.2f}"
        )

        evaluate_sigma(
            sigma=sigma,
            results=results,
            h_eigendecomposition=h_eigendecomposition,
            maxiter=args.maxiter,
            output_dir=args.outdir,
        )

    save_results(results, args.outdir)
    plot_coarse_scan(results, coarse_sigmas, args.outdir)

    coarse_results = {
        float(sigma): total_xi(results[float(sigma)])
        for sigma in coarse_sigmas
        if float(sigma) in results
        and is_complete(results[float(sigma)])
    }

    if not coarse_results:
        raise RuntimeError(
            "No complete coarse-scan results. Inspect the optimizer output."
        )

    best_coarse_sigma = min(
        coarse_results,
        key=coarse_results.get,
    )

    print(f"\nBest coarse sigma: {best_coarse_sigma:.2f}")
    print(f"Xi: {coarse_results[best_coarse_sigma]:.10f}")

    # Stage 2: refine around the best coarse point.
    print("\nSTAGE 2: REFINED XI SCAN")

    fine_sigmas = make_fine_grid(best_coarse_sigma)

    for index, sigma in enumerate(fine_sigmas, start=1):
        print(
            f"\nFine point {index}/{len(fine_sigmas)}: "
            f"sigma={sigma:.2f}"
        )

        evaluate_sigma(
            sigma=sigma,
            results=results,
            h_eigendecomposition=h_eigendecomposition,
            maxiter=args.maxiter,
            output_dir=args.outdir,
        )

    save_results(results, args.outdir)
    plot_refined_scan(results, fine_sigmas, args.outdir)

    # Stage 3: select reference and best candidates.
    print("\nSTAGE 3: GIBBS-STATE INFIDELITY")

    selected_sigmas = select_diagnostic_sigmas(results)

    print("Sigma values included in the beta plots:")
    for sigma in selected_sigmas:
        print(f"  sigma={sigma:.2f}")

    plot_gibbs_infidelity(
        results,
        selected_sigmas,
        args.outdir,
    )

    plot_xi_contribution(
        results,
        selected_sigmas,
        args.outdir,
    )

    save_results(results, args.outdir)

    # Final ranking.
    complete_results = complete_xi_results(results)

    print("\n" + "=" * 65)
    print("BEST COMPLETED SIGMA VALUES BY TOTAL XI")
    print("=" * 65)

    for rank, (sigma, xi) in enumerate(
        sorted(complete_results.items(), key=lambda item: item[1])[:15],
        start=1,
    ):
        print(f"{rank:3d}. sigma={sigma:.2f}  Xi={xi:.10f}")

    if complete_results:
        best_sigma = min(
            complete_results,
            key=complete_results.get,
        )

        print(f"\nBest tested sigma: {best_sigma:.2f}")
        print(f"Minimum Xi: {complete_results[best_sigma]:.10f}")

    print(f"\nAll results saved to: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
