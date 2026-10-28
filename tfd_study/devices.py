"""Backends for the study: exact, noiseless shots, noisy Tuna-17 model, real Tuna-17.

Tuna-17's connectivity comes from tuna17.json (fetched from Quantum Inspire;
QI publishes no error rates). The noise model is a uniform per-gate model
fitted to our own Tuna-17 tomography data (see fit_noise.py and
noise_fit.json): depolarizing error plus amplitude damping (T1-like decay
towards |0>) on every CZ, depolarizing error on every rx/ry (rz is virtual,
so error-free), and per-qubit readout error.

coherent_noise_model is the per-coupler alternative (noise_fit_coherent.json):
each CZ also carries a coherent phase error, which the same data show the
uniform model misses (the beta ~ 0 states are partly disentangled on one pair).
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import rustworkx as rx
from qiskit import transpile
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (NoiseModel, ReadoutError, amplitude_damping_error, coherent_unitary_error,
                              depolarizing_error)

from tfd_study.model import B, bonds

HERE = Path(__file__).resolve().parent
NATIVE_GATES = ["rx", "ry", "rz", "cz"]
# Qubits of our own Tuna-17 2+2 runs (results/parallel_20261019_175824).
MEASURED_LAYOUT_2 = [2, 4, 5, 8]


@lru_cache(maxsize=None)
def tuna17():
    return json.loads((HERE / "tuna17.json").read_text())


def coupling_map():
    edges = tuna17()["edges"]
    return CouplingMap([[a, b] for a, b in edges] + [[b, a] for a, b in edges])


@lru_cache(maxsize=None)
def noise_fit():
    path = HERE / "noise_fit.json"
    return json.loads(path.read_text()) if path.exists() else None


# ----------------------------------------------------------------------------- layouts

def ladder_edges(n):
    """Logical couplings the ansatz needs: chain bonds in A and in B, plus rungs A_i-B_i."""
    edges = [(i, j) for i, j in bonds(n)] + [(B(i, n), B(j, n)) for i, j in bonds(n)]
    return edges + [(i, B(i, n)) for i in range(n)]


def ladder_layouts(n):
    """Every placement of the 2 x n ladder on Tuna-17 as {logical qubit: physical qubit}.
    Symmetric copies (same physical qubits and couplers) are listed once."""
    dev = rx.PyGraph()
    dev.add_nodes_from(range(tuna17()["num_qubits"]))
    dev.add_edges_from_no_data([tuple(e) for e in tuna17()["edges"]])
    lad = rx.PyGraph()
    lad.add_nodes_from(range(2 * n))
    lad.add_edges_from_no_data(ladder_edges(n))
    seen, out = set(), []
    for m in rx.vf2_mapping(dev, lad, subgraph=True, induced=False):
        layout = {logical: physical for physical, logical in m.items()}
        key = frozenset(frozenset((layout[a], layout[b])) for a, b in ladder_edges(n))
        if key not in seen:
            seen.add(key)
            out.append([layout[q] for q in range(2 * n)])
    return out


def default_layout(n):
    """n = 2: the qubits of our measured runs. Otherwise the most central ladder."""
    if n == 2:
        cand = [l for l in ladder_layouts(2) if sorted(l) == sorted(MEASURED_LAYOUT_2)]
        if cand:
            return cand[0]
    dist = np.array(rx.distance_matrix(_device_graph()))
    return min(ladder_layouts(n), key=lambda l: dist[np.ix_(l, l)].sum() + dist[l].sum() / 17)


@lru_cache(maxsize=None)
def _device_graph():
    g = rx.PyGraph()
    g.add_nodes_from(range(tuna17()["num_qubits"]))
    g.add_edges_from_no_data([tuple(e) for e in tuna17()["edges"]])
    return g


def to_device(circuits, layout, seed=11):
    """Compile to Tuna-17 native gates on the given physical qubits (no routing needed
    for a ladder layout; the transpiler would add SWAPs otherwise)."""
    return transpile(circuits, basis_gates=NATIVE_GATES, coupling_map=coupling_map(),
                     initial_layout=layout, optimization_level=2, seed_transpiler=seed)


def cz_count(circuit):
    return circuit.count_ops().get("cz", 0)


# ----------------------------------------------------------------------------- simulators

def noise_model(p_cz=None, p_1q=None, gamma_cz=None, readout=None, scale=1.0):
    """Uniform Tuna-17 model. Defaults come from noise_fit.json; `scale` multiplies
    every gate error (for robustness sweeps). gamma_cz is the amplitude-damping
    probability per qubit per CZ. readout = (P(1|0), P(0|1))."""
    fit = noise_fit() or {}
    p_cz = (fit.get("p_cz", 0.0) if p_cz is None else p_cz) * scale
    p_1q = (fit.get("p_1q", 0.0) if p_1q is None else p_1q) * scale
    gamma = (fit.get("gamma_cz", 0.0) if gamma_cz is None else gamma_cz) * scale
    e01, e10 = readout if readout is not None else fit.get("readout", (0.0, 0.0))
    nm = NoiseModel(basis_gates=NATIVE_GATES)
    cz_err = depolarizing_error(min(p_cz, 1.0), 2) if p_cz > 0 else None
    if gamma > 0:
        damp = amplitude_damping_error(min(gamma, 1.0))
        damp2 = damp.tensor(damp)
        cz_err = damp2 if cz_err is None else cz_err.compose(damp2)
    if cz_err is not None:
        nm.add_all_qubit_quantum_error(cz_err, ["cz"])
    if p_1q > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(min(p_1q, 1.0), 1), ["rx", "ry"])
    if e01 or e10:
        nm.add_all_qubit_readout_error(ReadoutError([[1 - e01, e01], [e10, 1 - e10]]))
    return nm


@lru_cache(maxsize=None)
def coherent_fit():
    path = HERE / "noise_fit_coherent.json"
    return json.loads(path.read_text()) if path.exists() else None


def coherent_noise_model(couplers, seed=0, readout=None):
    """Per-coupler Tuna-17 model (fit_noise.py --coherent, noise_fit_coherent.json): every CZ
    is followed by a coherent phase error (conditional phase plus a phase on each qubit) and
    by that coupler's depolarizing error, then amplitude damping on both qubits.

    `couplers` lists the physical couplers the circuits use. Couplers that were never
    measured get the mean fitted depolarizing error and phase errors drawn (with `seed`)
    from a normal distribution with the rms of the fitted phases: one plausible device.
    readout: {physical qubit: (P(1|0), P(0|1))}; default is the fit's uniform readout error."""
    fit = coherent_fit()
    rng = np.random.default_rng(seed)
    nm = NoiseModel(basis_gates=NATIVE_GATES)
    damp = amplitude_damping_error(fit["gamma_cz"])
    for a, b in sorted({tuple(sorted(e)) for e in couplers}):
        c = fit["couplers"].get(f"{a}-{b}")
        if c is None:
            c = {"p_depol": fit["unmeasured"]["p_depol"],
                 "cphase": rng.normal(0, fit["unmeasured"]["phase_rms"]),
                 "phase": {str(a): rng.normal(0, fit["unmeasured"]["phase_rms"]),
                           str(b): rng.normal(0, fit["unmeasured"]["phase_rms"])}}
        for qa, qb in ((a, b), (b, a)):           # instruction qubit 0 (low bit) = qa
            fa, fb = c["phase"][str(qa)], c["phase"][str(qb)]
            U = np.diag(np.exp(1j * np.array([0.0, fa, fb, fa + fb + c["cphase"]])))
            err = coherent_unitary_error(U).compose(depolarizing_error(min(c["p_depol"], 1.0), 2))
            nm.add_quantum_error(err.compose(damp.tensor(damp)), ["cz"], [qa, qb])
    nm.add_all_qubit_quantum_error(depolarizing_error(fit["p_1q"], 1), ["rx", "ry"])
    if readout is None:
        e01, e10 = fit["readout"]
        nm.add_all_qubit_readout_error(ReadoutError([[1 - e01, e01], [e10, 1 - e10]]))
    else:
        for q, (e01, e10) in readout.items():
            nm.add_readout_error(ReadoutError([[1 - e01, e01], [e10, 1 - e10]]), [q])
    return nm


def simulator(kind, threads=1, **noise):
    """'noiseless': ideal shots. 'noisy': density-matrix simulation of the Tuna-17 model.

    One Aer thread by default: these circuits are small, and the studies run
    many simulations side by side in worker processes. Aer's default (one
    thread per core in every process) oversubscribes the machine until worker
    processes die."""
    if kind == "noiseless":
        return AerSimulator(method="statevector", max_parallel_threads=threads)
    if kind == "noisy":
        return AerSimulator(method="density_matrix", noise_model=noise_model(**noise), max_parallel_threads=threads)
    raise ValueError(kind)


def real_backend(name="Tuna-17"):
    from qiskit_quantuminspire.qi_provider import QIProvider
    return QIProvider().get_backend(name)
