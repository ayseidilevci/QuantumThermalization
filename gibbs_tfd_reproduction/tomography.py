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

from gibbs_tfd_reproduction.tfd_circuit import TFDCircuit

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
    """
    subsystem_qubits: (0,1) for system A, (2,3) for system B (tfd_circuit.py defines
    q0=A2, q1=A1, q2=B2, q3=B1).
    """
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

    # readout calibration block
    offset = 4 * len(settings)
    for q in range(N_QUBITS):
        qc.measure(q, offset + q)
    qc.barrier()
    for q in range(N_QUBITS):
        qc.reset(q)
    qc.barrier()
    qc.x(range(N_QUBITS))
    qc.barrier()
    for q in range(N_QUBITS):
        qc.measure(q, offset + 4 + q)

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


def plot_density_matrix(rho, title="Reconstructed density matrix", save_path=None):
    labels = ["00", "01", "10", "11"]
    n = 4
    xpos, ypos = np.meshgrid(range(n), range(n), indexing="ij")
    xpos, ypos = xpos.ravel(), ypos.ravel()
    dz = np.abs(rho).ravel()
    phase = np.angle(rho).ravel()
    colors = plt.get_cmap("hsv")((phase + np.pi) / (2 * np.pi))

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.bar3d(xpos, ypos, np.zeros_like(dz), 0.6, 0.6, dz, color=colors, shade=True)
    ax.set_xticks(np.arange(n) + 0.3)
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(n) + 0.3)
    ax.set_yticklabels(labels)
    ax.set_zlim(0, max(0.55, dz.max() * 1.1))
    ax.set_title(title)

    if save_path:
        fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return fig
