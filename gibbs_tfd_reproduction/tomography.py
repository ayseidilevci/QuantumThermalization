from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from qiskit import QuantumCircuit, transpile

from qi_utilities.utility_functions.circuit_modifiers import apply_pre_measurement_rotations
from qi_utilities.utility_functions.raw_data_processing import (
    get_raw_data, get_multi_counts, get_multi_probs, observable_expectation_values_Z_basis,
)
from qi_utilities.error_mitigation.toolkit.readout_error_mitigation import (
    apply_readout_assignment_correction,
)

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from tfd_circuit import TFDCircuit

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

N_QUBITS = 4
PAULIS = ["X", "Y", "Z"]

def extract_ro_matrices(counts_prep0: dict, counts_prep1: dict, num_qubits: int = N_QUBITS):
    shots0, shots1 = sum(counts_prep0.values()), sum(counts_prep1.values())
    ro_matrices = {}
    for q in range(num_qubits):
        d0_0 = sum(n for b, n in counts_prep0.items() if b[-(q + 1)] == '0')
        d0_1 = shots0 - d0_0
        d1_0 = sum(n for b, n in counts_prep1.items() if b[-(q + 1)] == '0')
        d1_1 = shots1 - d1_0
        ro_matrices[f"Qubit {q}"] = np.array([[d0_0 / shots0, d1_0 / shots1], [d0_1 / shots0, d1_1 / shots1]])
    return ro_matrices


def build_tomography_circuit(gamma1, gamma2, alpha1, alpha2, subsystem_qubits=(0, 1)):
    qa, qb = subsystem_qubits
    settings = list(product(PAULIS, PAULIS))

    qc = QuantumCircuit(N_QUBITS, 4 * len(settings) + 8, name="tomography")

    for i, (pa, pb) in enumerate(settings):
        ansatz = TFDCircuit(gamma1, gamma2, alpha1, alpha2)
        ansatz.build_circuit(qc)
        qc.barrier()

        full_obs = ['Z'] * N_QUBITS
        full_obs[N_QUBITS - 1 - qa] = pa
        full_obs[N_QUBITS - 1 - qb] = pb
        bit_register = list(range(4 * i, 4 * i + 4))
        apply_pre_measurement_rotations(qc, "".join(full_obs), bit_register=bit_register)

        qc.barrier()
        for q in range(N_QUBITS):
            qc.reset(q)
        qc.barrier()

    # # readout calibration block
    # offset = 4 * len(settings)
    # for q in range(N_QUBITS):
    #     qc.measure(q, offset + q)
    # qc.barrier()
    # for q in range(N_QUBITS):
    #     qc.reset(q)
    # qc.barrier()
    # qc.x(range(N_QUBITS))
    # qc.barrier()
    # for q in range(N_QUBITS):
    #     qc.measure(q, offset + 4 + q)

    return qc, settings


def run_tomography(backend, initial_layout, gamma1, gamma2, alpha1, alpha2,
                    subsystem_qubits=(0, 1), shots=8192):
    qc, settings = build_tomography_circuit(gamma1, gamma2, alpha1, alpha2, subsystem_qubits)
    qc_t = transpile(qc, backend, initial_layout=initial_layout)

    job = backend.run(qc_t, shots=shots, memory=True)
    result = job.result()

    raw = get_raw_data(qc_t, result)
    counts = get_multi_counts(raw, N_QUBITS)
    probs = get_multi_probs(counts)

    setting_probs = probs[:9]
    calib0_counts, calib1_counts = counts[9], counts[10]
    ro_matrices = extract_ro_matrices(calib0_counts, calib1_counts)

    corrected = [apply_readout_assignment_correction(p, ro_matrices, [0, 1, 2, 3])
                 for p in setting_probs]
    return corrected, settings


def reconstruct_density_matrix(corrected_probs, settings, subsystem_qubits=(0, 1)):
    qa, qb = subsystem_qubits
    terms = {("I", "I"): 1.0}
    marginal_a, marginal_b = {}, {}

    def marg(probs4, oa, ob):
        full = ['I'] * N_QUBITS
        full[N_QUBITS - 1 - qa] = oa
        full[N_QUBITS - 1 - qb] = ob
        return observable_expectation_values_Z_basis([probs4], "".join(full))[0]

    for probs4, (pa, pb) in zip(corrected_probs, settings):
        terms[(pa, pb)] = marg(probs4, "Z", "Z")
        marginal_a.setdefault(pa, []).append(marg(probs4, "Z", "I"))
        marginal_b.setdefault(pb, []).append(marg(probs4, "I", "Z"))

    for pa, vals in marginal_a.items():
        terms[(pa, "I")] = float(np.mean(vals))
    for pb, vals in marginal_b.items():
        terms[("I", pb)] = float(np.mean(vals))

    paulis = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]),
              "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.array([[1, 0], [0, -1]])}

    rho = np.zeros((4, 4), dtype=complex)
    for pa in ["I", "X", "Y", "Z"]:
        for pb in ["I", "X", "Y", "Z"]:
            rho += terms.get((pa, pb), 0.0) * np.kron(paulis[pa], paulis[pb])
    return rho / 4.0


def plot_density_matrix(rho, title="Reconstructed density matrix", save_path=None, rho_target=None):
    labels = ["00", "01", "10", "11"]
    n = 4
    xpos, ypos = np.meshgrid(range(n), range(n), indexing="ij")
    xpos, ypos = xpos.ravel(), ypos.ravel()

    def bar_data(m):
        dz = np.abs(m).ravel()
        phase = np.angle(m).ravel()
        colors = plt.get_cmap("hsv")((phase + np.pi) / (2 * np.pi))
        return dz, colors

    dz_meas, colors_meas = bar_data(rho)
    zmax = max(0.55, dz_meas.max() * 1.1)
    if rho_target is not None:
        dz_target, colors_target = bar_data(rho_target)
        zmax = max(zmax, dz_target.max() * 1.1)

    ncols = 2 if rho_target is not None else 1
    fig = plt.figure(figsize=(6 * ncols, 5.5))

    ax1 = fig.add_subplot(1, ncols, 1, projection="3d")
    ax1.bar3d(xpos, ypos, np.zeros_like(dz_meas), 0.6, 0.6, dz_meas, color=colors_meas, shade=True)
    ax1.set_xticks(np.arange(n) + 0.3); ax1.set_xticklabels(labels)
    ax1.set_yticks(np.arange(n) + 0.3); ax1.set_yticklabels(labels)
    ax1.set_zlim(0, zmax)
    ax1.set_title(f"{title}\n(measured)")

    if rho_target is not None:
        ax2 = fig.add_subplot(1, ncols, 2, projection="3d")
        ax2.bar3d(xpos, ypos, np.zeros_like(dz_target), 0.6, 0.6, dz_target, color=colors_target, shade=True)
        ax2.set_xticks(np.arange(n) + 0.3); ax2.set_xticklabels(labels)
        ax2.set_yticks(np.arange(n) + 0.3); ax2.set_yticklabels(labels)
        ax2.set_zlim(0, zmax)
        ax2.set_title(f"{title}\n(exact target)")

    norm = Normalize(vmin=-np.pi, vmax=np.pi)
    mappable = ScalarMappable(norm=norm, cmap="hsv")
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=fig.get_axes(), shrink=0.6, pad=0.05)
    cbar.set_label("Phase (radians)")
    cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cbar.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])

    if save_path:
        fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return fig

def plot_tomography_grid(rho_A, rho_B, rho_target, beta, save_path=None):
    labels = ["00", "01", "10", "11"]
    n = 4
    xpos, ypos = np.meshgrid(range(n), range(n), indexing="ij")
    xpos, ypos = xpos.ravel(), ypos.ravel()

    def bar_data(m):
        dz = np.abs(m).ravel()
        phase = np.angle(m).ravel()
        colors = plt.get_cmap("hsv")((phase + np.pi) / (2 * np.pi))
        return dz, colors

    dz_A, colors_A = bar_data(rho_A)
    dz_B, colors_B = bar_data(rho_B)
    dz_T, colors_T = bar_data(rho_target)
    zmax = max(0.55, dz_A.max()*1.1, dz_B.max()*1.1, dz_T.max()*1.1)

    fig = plt.figure(figsize=(17, 10))

    def plain_panel(pos, dz, colors, title):
        ax = fig.add_subplot(2, 3, pos, projection="3d")
        ax.bar3d(xpos, ypos, np.zeros_like(dz), 0.6, 0.6, dz, color=colors, shade=True)
        ax.set_xticks(np.arange(n)+0.3); ax.set_xticklabels(labels, fontsize=7)
        ax.set_yticks(np.arange(n)+0.3); ax.set_yticklabels(labels, fontsize=7)
        ax.set_zlim(0, zmax)
        ax.set_title(title, fontsize=10)
        return ax

    def overlay_panel(pos, dz_meas, colors_meas, title):
        ax = fig.add_subplot(2, 3, pos, projection="3d")
        ax.bar3d(xpos, ypos, np.zeros_like(dz_T), 0.6, 0.6, dz_T,
                  color=(0,0,0,0), edgecolor='green', linewidth=0.6,
                  hatch='////', shade=False)
        ax.bar3d(xpos+0.05, ypos+0.05, np.zeros_like(dz_meas), 0.45, 0.45, dz_meas,
                  color=colors_meas, shade=True)
        ax.set_xticks(np.arange(n)+0.3); ax.set_xticklabels(labels, fontsize=7)
        ax.set_yticks(np.arange(n)+0.3); ax.set_yticklabels(labels, fontsize=7)
        ax.set_zlim(0, zmax)
        ax.set_title(title, fontsize=10)
        return ax

    plain_panel(1, dz_A, colors_A, "System A -- measured")
    plain_panel(2, dz_T, colors_T, "Exact target (Gibbs state)")
    overlay_panel(3, dz_A, colors_A, "System A -- overlay\n(solid=measured, hatched=exact)")

    plain_panel(4, dz_B, colors_B, "System B -- measured")
    plain_panel(5, dz_T, colors_T, "Exact target (Gibbs state)")
    overlay_panel(6, dz_B, colors_B, "System B -- overlay\n(solid=measured, hatched=exact)")

    norm = Normalize(vmin=-np.pi, vmax=np.pi)
    mappable = ScalarMappable(norm=norm, cmap="hsv")
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=fig.get_axes(), shrink=0.5, pad=0.02, location='left')
    cbar.set_label("Phase (radians)")
    cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cbar.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])

    fig.suptitle(f"Tomography, beta={beta}", fontsize=14)

    if save_path:
        fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return fig