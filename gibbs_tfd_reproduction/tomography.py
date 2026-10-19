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

# Seconds to wait for a job's result; None = wait as long as it takes.
# Quantum Inspire's job.result() gives up after 60 s by default, and giving up
# on a queued job only makes the caller resubmit a duplicate behind it -- so
# queued jobs are waited for indefinitely; only real errors trigger a retry.
JOB_TIMEOUT = None


def wait_for_result(job, label="job"):
    """job.result(), but printing every status change (queued -> running ->
    done) with a timestamp, so a long hardware wait isn't a frozen screen."""
    from datetime import datetime
    from qiskit.providers import JobStatus
    import time as _time
    start, last, poll = _time.time(), None, 1.0
    print(f"  [{datetime.now():%H:%M:%S}] {label} submitted, id {_job_id(job)}", flush=True)
    while JOB_TIMEOUT is None or _time.time() - start < JOB_TIMEOUT:
        try:
            status = job.status()
        except Exception as exc:  # a status-query hiccup shouldn't kill the wait
            status = f"status unavailable ({exc})"
        if status != last:
            name = getattr(status, "name", status)
            print(f"  [{datetime.now():%H:%M:%S}] {label}: {name}"
                  f"  (+{_time.time() - start:.0f}s)", flush=True)
            last = status
        if status in (JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED):
            break
        _time.sleep(poll)
        poll = min(poll * 2, 15.0)
    return job.result(timeout=JOB_TIMEOUT)


def _job_id(job):
    # QIJob's job_id() is empty; its id is batch_job_id (set on submit)
    for get in (lambda: job.batch_job_id, lambda: job.job_id(),
                lambda: job.circuits_run_data[0].job_id):
        try:
            value = get()
            if value not in (None, ""):
                return str(value)
        except Exception:
            pass
    return None


def log_raw_result(path, job, circuits, result, meta):
    """Append one JSON line with the job id and, per circuit, the counts of
    every 4-bit measurement block -- enough to redo any analysis later
    (other readout correction, MLE tomography, error bars)."""
    import json
    from datetime import datetime
    blocks = []
    for i, qc in enumerate(circuits):
        raw = get_raw_data(qc, result, circuit_nr=i if len(circuits) > 1 else None)
        blocks.append(get_multi_counts(raw, N_QUBITS))
    record = {"timestamp": datetime.now().isoformat(timespec="seconds"),
              "job_id": _job_id(job), "backend": getattr(result, "backend_name", None),
              **meta, "block_counts": blocks}
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")

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
                    subsystem_qubits=(0, 1), shots=8192, ro_matrices=None,
                    raw_log_path=None, meta=None):
    qc, settings = build_tomography_circuit(gamma1, gamma2, alpha1, alpha2, subsystem_qubits)
    qc_t = transpile(qc, backend, initial_layout=initial_layout)

    job = backend.run(qc_t, shots=shots, memory=True)
    result = wait_for_result(job, f"tomography job{' (' + meta['label'] + ')' if meta and 'label' in meta else ''}")
    if not result.success:
        raise RuntimeError(f"job failed: {getattr(result, 'system_messages', '')}")
    if raw_log_path is not None:
        log_raw_result(raw_log_path, job, [qc_t], result,
                       {"kind": "tomography", "params": [gamma1, gamma2, alpha1, alpha2],
                        "settings": settings, **(meta or {})})

    raw = get_raw_data(qc_t, result)
    counts = get_multi_counts(raw, N_QUBITS)
    probs = get_multi_probs(counts)
    setting_probs = probs[:9]

    if ro_matrices is None:
        corrected = setting_probs
    else:
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

    fig = plt.figure(figsize=(12, 10)) 

    def plain_panel(pos, dz, colors, title):
        ax = fig.add_subplot(2, 2, pos, projection="3d")
        ax.bar3d(xpos, ypos, np.zeros_like(dz), 0.6, 0.6, dz, color=colors, shade=True)
        ax.set_xticks(np.arange(n)+0.3); ax.set_xticklabels(labels, fontsize=7)
        ax.set_yticks(np.arange(n)+0.3); ax.set_yticklabels(labels, fontsize=7)
        ax.set_zlim(0, zmax)
        ax.set_title(title, fontsize=10)
        return ax

    plain_panel(1, dz_A, colors_A, "System A -- measured")
    plain_panel(2, dz_T, colors_T, "Exact target (Gibbs state)")
    plain_panel(3, dz_B, colors_B, "System B -- measured")
    plain_panel(4, dz_T, colors_T, "Exact target (Gibbs state)")

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