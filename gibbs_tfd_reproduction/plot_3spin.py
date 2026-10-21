#!/usr/bin/env python3
"""Plot per-beta cost convergence and Manhattan-style density-matrix tomography.

Reads raw data produced by extension_experiment.py and saves figures in the same
folder. No backend jobs are submitted by this script.

Examples:
    python gibbs_tfd_reproduction/plot_3spin.py --indir outputs/three_qubits
"""
from __future__ import annotations
import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import Normalize, LinearSegmentedColormap
import numpy as np

N_A = 3
PAULI_AXES = ("X", "Y", "Z")
PAULI_LABELS = "IXYZ"
TOMOGRAPHY_SETTINGS = tuple(itertools.product(PAULI_AXES, repeat=N_A))


def beta_tag(beta: float) -> str:
    return f"{beta:.6g}".replace("-", "m").replace(".", "p")


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required data file not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_count_key(key: Any) -> int:
    if isinstance(key, (int, np.integer)):
        return int(key)
    text = str(key).replace(" ", "").replace("_", "")
    if text.startswith("0x"):
        return int(text, 16)
    if text.startswith("0b"):
        return int(text, 2)
    return int(text, 2)


def counts_to_probabilities(counts: dict, n_qubits: int, layout: Sequence[int]) -> np.ndarray:
    """Convert QI physical-bit-mask counts into logical-order probabilities."""
    layout = tuple(int(q) for q in layout[:n_qubits])
    if len(layout) != n_qubits or len(set(layout)) != n_qubits or min(layout, default=0) < 0:
        raise ValueError(f"Invalid logical-to-physical layout: {layout}")
    probs = np.zeros(2**n_qubits, dtype=float)
    total = 0
    for key, count in counts.items():
        physical_mask = parse_count_key(key)
        if physical_mask < 0 or physical_mask >> (max(layout) + 1):
            raise ValueError(f"Count key {key!r} has bits outside physical layout {layout}")
        logical_mask = 0
        for logical_q, physical_q in enumerate(layout):
            logical_mask |= ((physical_mask >> physical_q) & 1) << logical_q
        probs[logical_mask] += int(count)
        total += int(count)
    if total <= 0:
        raise ValueError("No counts for tomography setting")
    return probs / total


def hamiltonian_three_spin(g: float = 1.0) -> np.ndarray:
    """H = Z0 Z1 + Z1 Z2 + g (X0 + X1 + X2), with q0 the LSB."""
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return (np.kron(np.kron(Z, Z), I) + np.kron(np.kron(I, Z), Z)
            + float(g) * (np.kron(np.kron(X, I), I) + np.kron(np.kron(I, X), I)
                          + np.kron(np.kron(I, I), X)))


def exact_gibbs_state(beta: float, g: float = 1.0) -> np.ndarray:
    if beta <= 0:
        raise ValueError("beta must be positive")
    vals, vecs = np.linalg.eigh(hamiltonian_three_spin(g))
    weights = np.exp(-beta * (vals - vals.min()))
    weights /= weights.sum()
    return hermitian_trace_one((vecs * weights[np.newaxis, :]) @ vecs.conj().T)


def hermitian_trace_one(rho: np.ndarray) -> np.ndarray:
    rho = 0.5 * (np.asarray(rho, complex) + np.asarray(rho, complex).conj().T)
    tr = float(np.trace(rho).real)
    if abs(tr) < 1e-12:
        raise ValueError("Density matrix has zero trace")
    return rho / tr


def project_physical_density_matrix(rho: np.ndarray) -> np.ndarray:
    """Nearest PSD trace-one projection using eigenvalue simplex projection."""
    rho = 0.5 * (np.asarray(rho, complex) + np.asarray(rho, complex).conj().T)
    vals, vecs = np.linalg.eigh(rho)
    ordered = np.sort(vals)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    candidates = np.where(ordered - cumulative / np.arange(1, len(ordered) + 1) > 0)[0]
    if not candidates.size:
        projected = np.full_like(vals, 1.0 / len(vals))
    else:
        idx = int(candidates[-1])
        threshold = cumulative[idx] / (idx + 1)
        projected = np.maximum(vals - threshold, 0.0)
    projected /= projected.sum()
    result = (vecs * projected) @ vecs.conj().T
    return 0.5 * (result + result.conj().T)


def reconstruct_density_matrix(settings_data: dict, layout: Sequence[int]) -> np.ndarray:
    probs_by_setting = {}
    for setting in TOMOGRAPHY_SETTINGS:
        label = "".join(setting)
        if label not in settings_data:
            raise ValueError(f"Tomography data missing setting {label}")
        probs_by_setting[setting] = counts_to_probabilities(
            settings_data[label].get("counts", {}), N_A, layout[:N_A]
        )
    pauli = {
        "I": np.eye(2, dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }
    rho = np.zeros((2**N_A, 2**N_A), dtype=complex)
    for axes in itertools.product(PAULI_LABELS, repeat=N_A):
        setting = tuple(axis if axis != "I" else "Z" for axis in axes)
        probs = probs_by_setting[setting]
        expectation = 0.0
        for outcome, probability in enumerate(probs):
            sign = 1.0
            for q, axis in enumerate(axes):
                if axis != "I" and ((outcome >> q) & 1):
                    sign *= -1.0
            expectation += float(probability) * sign
        # q0 is LSB, so NumPy tensor order is q2 ⊗ q1 ⊗ q0.
        operator = np.kron(np.kron(pauli[axes[2]], pauli[axes[1]]), pauli[axes[0]])
        rho += expectation * operator / (2**N_A)
    return project_physical_density_matrix(hermitian_trace_one(rho))


def sqrt_psd(matrix: np.ndarray) -> np.ndarray:
    matrix = 0.5 * (matrix + matrix.conj().T)
    vals, vecs = np.linalg.eigh(matrix)
    return (vecs * np.sqrt(np.clip(vals, 0.0, None))) @ vecs.conj().T


def state_fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    rho, sigma = hermitian_trace_one(rho), hermitian_trace_one(sigma)
    middle = sqrt_psd(rho) @ sigma @ sqrt_psd(rho)
    vals = np.clip(np.linalg.eigvalsh(0.5 * (middle + middle.conj().T)), 0.0, None)
    return float(np.clip(np.sum(np.sqrt(vals))**2, 0.0, 1.0))


def pauli_matrix_for_qubits(n_qubits: int, pauli_by_qubit: dict[int, str]) -> np.ndarray:
    mats = {
        "I": np.eye(2, dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }
    # Kronecker product q_(n-1) ⊗ ... ⊗ q0; q0 is the LSB.
    out = np.array([[1.0 + 0j]])
    for q in reversed(range(n_qubits)):
        out = np.kron(out, mats[pauli_by_qubit.get(q, "I")])
    return out


def cost_operator_lower_bound(beta: float, varsigma: float) -> float:
    """Exact minimum eigenvalue of the six-qubit measured cost observable."""
    if beta <= 0:
        raise ValueError("beta must be positive")
    H = np.zeros((64, 64), dtype=complex)
    # X_A + X_B
    for q in range(6):
        H += pauli_matrix_for_qubits(6, {q: "X"})
    # varsigma * (ZZ_A + ZZ_B), open chains in both registers
    for a, b in ((0, 1), (1, 2), (3, 4), (4, 5)):
        H += varsigma * pauli_matrix_for_qubits(6, {a: "Z", b: "Z"})
    # - beta^-varsigma * (XX_BA + ZZ_BA)
    weight = beta ** (-varsigma)
    for a, b in ((0, 3), (1, 4), (2, 5)):
        H -= weight * pauli_matrix_for_qubits(6, {a: "X", b: "X"})
        H -= weight * pauli_matrix_for_qubits(6, {a: "Z", b: "Z"})
    return float(np.linalg.eigvalsh(H).min().real)


def plot_cost_history(indir: Path) -> list[Path]:
    rows = load_csv(indir / "cost_history.csv")
    if not rows:
        raise ValueError(f"No cost history rows in {indir / 'cost_history.csv'}")
    paths = []
    for beta in sorted({float(r["beta"]) for r in rows}):
        data = sorted((r for r in rows if np.isclose(float(r["beta"]), beta)),
                      key=lambda r: int(r["evaluation_within_beta"]))
        x = np.array([int(r["evaluation_within_beta"]) for r in data])
        y = np.array([float(r["cost"]) for r in data])
        varsigma = float(data[0].get("varsigma", 2.05))
        p_layers = data[0].get("p_layers", "?")
        lower = cost_operator_lower_bound(beta, varsigma)
        fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
        ax.plot(x, y, marker="o", linewidth=1.5, markersize=4, label="Measured cost")
        ax.axhline(lower, linestyle="--", linewidth=1.6,
                   label=rf"Exact lower bound $\lambda_{{\min}}={lower:.4f}$")
        ax.set_xlabel("Objective evaluation")
        ax.set_ylabel("Cost function")
        ax.set_title(rf"Cost convergence: $\beta={beta:g}$, $p={p_layers}$, $\varsigma={varsigma:g}$")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(left=1)
        path = indir / f"cost_history_beta_{beta_tag(beta)}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_manhattan_matrix(ax, rho: np.ndarray, title: str, vmax: float | None = None):
    """Manhattan plot: bar height=|rho_ij| and color=arg(rho_ij)."""
    n = rho.shape[0]
    xs, ys = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    xs, ys = xs.ravel(), ys.ravel()
    values = rho.ravel()
    magnitudes = np.abs(values)
    phases = np.angle(values)
    heights = magnitudes
    norm = Normalize(vmin=-np.pi, vmax=np.pi)
    # Muted cyclic phase palette matching the reference figure:
    # red at +/-pi, olive at -pi/2, dark blue at 0, cyan at +pi/2.
    cmap = LinearSegmentedColormap.from_list(
        "density_matrix_phase_muted",
        [
            (0.00, "#d92d27"),  # -pi: muted red
            (0.25, "#a8b83b"),  # -pi/2: olive green
            (0.50, "#344f9b"),  # 0: deep blue
            (0.75, "#55c1d1"),  # +pi/2: muted cyan
            (1.00, "#d92d27"),  # +pi: same phase as -pi
        ],
        N=256,
    )
    bar_width = 0.78
    ax.bar3d(xs, ys, np.zeros_like(heights), bar_width, bar_width, heights,
             color=cmap(norm(phases)), shade=True, edgecolor="black", linewidth=0.12)
    ax.set_xticks(np.arange(n) + bar_width/2, [format(i, "03b") for i in range(n)], rotation=45, ha="right")
    ax.set_yticks(np.arange(n) + bar_width/2, [format(i, "03b") for i in range(n)])
    ax.set_xlabel("Column basis state")
    ax.set_ylabel("Row basis state")
    ax.set_zlabel(r"$|\rho_{ij}|$")
    ax.set_zlim(0, max(vmax or float(magnitudes.max()), 1e-6) * 1.08)
    ax.set_title(title, pad=12)
    ax.view_init(elev=27, azim=-55)
    return cmap, norm


def plot_tomography(indir: Path) -> list[Path]:
    raw_path = indir / "tomography_raw_counts.json"
    if not raw_path.exists():
        print(f"Tomography skipped: {raw_path} not found. Run with --tomography.")
        return []
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    layout = tuple(int(q) for q in payload.get("layout_logical_to_physical", []))
    if len(layout) < N_A:
        raise ValueError(f"Tomography file has no valid layout: {layout}")
    g = float(payload.get("hamiltonian", {}).get("g", 1.0))
    beta_entries = payload.get("betas", {})
    paths = []
    for beta_key, entry in sorted(beta_entries.items(), key=lambda item: float(item[1].get("beta", item[0].replace("beta_", "")))):
        beta = float(entry.get("beta", beta_key.replace("beta_", "")))
        rho_reconstructed = reconstruct_density_matrix(entry.get("settings", {}), layout)
        rho_exact = exact_gibbs_state(beta, g)
        fidelity = state_fidelity(rho_reconstructed, rho_exact)
        vmax = max(float(np.abs(rho_reconstructed).max()), float(np.abs(rho_exact).max()))
        fig = plt.figure(figsize=(14.5, 7.4), constrained_layout=True)
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        cmap, norm = plot_manhattan_matrix(ax1, rho_exact, "Exact Gibbs state $\\rho_\\beta$", vmax)
        plot_manhattan_matrix(ax2, rho_reconstructed, "Reconstructed state $\\hat{\\rho}_A$", vmax)
        scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        scalar_map.set_array([])
        cbar = fig.colorbar(scalar_map, ax=[ax1, ax2], shrink=0.66, pad=0.08)
        cbar.set_label("Phase of density-matrix element")
        cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        cbar.set_ticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
        varsigma = entry.get("varsigma", payload.get("varsigma", float("nan")))
        fig.suptitle(rf"Gibbs-state tomography: $\beta={beta:g}$, $\varsigma={float(varsigma):g}$, $F(\rho_\beta,\hat{{\rho}}_A)={fidelity:.6f}$", fontsize=15)
        path = indir / f"tomography_beta_{beta_tag(beta)}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
        print(f"beta={beta:g}: tomography fidelity = {fidelity:.6f}")
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indir", type=Path, default=Path("outputs/three_qubits"),
                        help="folder containing raw data; plots are saved in this same folder")
    args = parser.parse_args()
    if not args.indir.is_dir():
        parser.error(f"Input data directory does not exist: {args.indir}")
    paths = plot_cost_history(args.indir)
    paths.extend(plot_tomography(args.indir))
    print("Saved figures:")
    for path in paths:
        print(f"  {path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

