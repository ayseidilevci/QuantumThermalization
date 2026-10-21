"""Compare a hardware run with the noisy and the ideal simulator, on identical circuits.

For every tomography in a run (results/tfd_*), the exact circuits that were
submitted are in compiled.jsonl (recorded by run_hardware.py). Each is
re-run on:
  * ideal      noiseless density matrix of the compiled circuit
  * noisy      the Tuna-17 noise model (devices.noise_model, or a model refitted
               with fit_run below) plus the run's own measured readout error,
               sampled with the run's shots, readout-corrected and
               reconstructed exactly like the hardware data
The hardware states are reconstructed again from the raw counts with the
same code, so all three go through one analysis path.

fit_run refits the noise model on a run's reference tomographies, either
uniform (one CZ error for all couplers) or per coupler, and keeps the end-of-run
references as a validation set when they exist.

    python -m tfd_study.analyze_run results/tfd_n3p1_<tag>
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, qasm3
from qiskit.quantum_info import DensityMatrix, partial_trace, state_fidelity
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, amplitude_damping_error, depolarizing_error
from scipy.optimize import least_squares

from tfd_study import devices, model

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------------- loading

def load(run_dir):
    run_dir = Path(run_dir)
    state = json.loads((run_dir / "state.json").read_text())
    compiled = {}
    for line in (run_dir / "compiled.jsonl").read_text().splitlines():
        row = json.loads(line)
        compiled[row["name"]] = row["qasm"]          # a resubmission overwrites: last one wins
    return state["cfg"], state["exp"], compiled


def tomo_circuits(compiled, label, n):
    return [qasm3.loads(compiled[f"tomo|{label}|{s}"]) for s in model.tomography_settings(n)]


def readout_error_per_qubit(exp, cfg):
    """(P(1|0), P(0|1)) per physical qubit, from the run's latest readout calibration."""
    mats = exp["readout"]["mats"]
    return {cfg["layout"][k]: (m[1][0], m[0][1]) for k, m in enumerate(mats)}


# ----------------------------------------------------------------------------- noise models

def build_noise(n_layout_edges, p_cz, gamma, p_1q, readout=None):
    """p_cz: a number (uniform) or {frozenset(edge): p}. readout: {physical qubit: (e01, e10)}."""
    nm = NoiseModel(basis_gates=devices.NATIVE_GATES)

    def cz_error(p):
        err = depolarizing_error(min(max(p, 0.0), 1.0), 2)
        if gamma > 0:
            d = amplitude_damping_error(min(gamma, 1.0))
            err = err.compose(d.tensor(d))
        return err

    if isinstance(p_cz, dict):
        for edge, p in p_cz.items():
            a, b = tuple(edge)
            nm.add_quantum_error(cz_error(p), ["cz"], [a, b])
            nm.add_quantum_error(cz_error(p), ["cz"], [b, a])
    else:
        nm.add_all_qubit_quantum_error(cz_error(p_cz), ["cz"])
    if p_1q > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(min(p_1q, 1.0), 1), ["rx", "ry"])
    for q, (e01, e10) in (readout or {}).items():
        nm.add_readout_error(ReadoutError([[1 - e01, e01], [e10, 1 - e10]]), [q])
    return nm


def default_noise(cfg, exp):
    fit = devices.noise_fit()
    return {"p_cz": fit["p_cz"], "gamma": fit["gamma_cz"], "p_1q": fit["p_1q"]}


# ----------------------------------------------------------------------------- reconstruction

def reconstruct(probs_list, n, mats=None):
    if mats is not None:
        probs_list = [model.correct_readout(p, mats) for p in probs_list]
    by = dict(zip(model.tomography_settings(n), probs_list))
    return {s: model.tomography_rho(by, n, s) for s in ("A", "B")}


def hardware_states(run_dir, cfg, exp):
    """rho_A, rho_B per tomography label, from the run's own reconstruction."""
    out = {}
    for label, t in exp["tomo"].items():
        out[label] = {s: np.array(t[s]["rho"][0]) + 1j * np.array(t[s]["rho"][1]) for s in ("A", "B")}
    return out


def measured_qubits(qc):
    """Physical qubit measured into each clbit (clbit k = logical qubit k), read from the
    compiled circuit itself rather than assumed from the planned layout."""
    out = {}
    for inst in qc.data:
        if inst.operation.name == "measure":
            out[qc.find_bit(inst.clbits[0]).index] = qc.find_bit(inst.qubits[0]).index
    return [out[k] for k in sorted(out)]


def ansatz_state(compiled, label, n, layout=None, noise=None):
    """rho_A and rho_B that the run's compiled tomography circuits measure, without shot noise:
    exact outcome probabilities of all 3^n compiled circuits, reconstructed like the hardware data.

    (The compiled all-Z circuit is not the ansatz itself: before a Z measurement the compiler
    may drop diagonal gates such as rz and cz, which cannot change Z-basis outcomes. Only the
    full set of tomography circuits pins the state down.)"""
    circs = []
    for full in tomo_circuits(compiled, label, n):
        qubits = measured_qubits(full)                 # clbit order = logical order
        qc = full.remove_final_measurements(inplace=False)
        qc.save_probabilities(qubits=qubits)
        circs.append(qc)
    sim = AerSimulator(method="density_matrix", noise_model=noise, max_parallel_threads=1)
    res = sim.run(circs).result()
    probs = [np.asarray(res.data(i)["probabilities"]) for i in range(len(circs))]
    return reconstruct(probs, n)


def simulated_tomography(compiled, label, n, shots, noise, mats, seed=0):
    """The run's tomography circuits sampled on a simulator, then the same analysis as the hardware."""
    sim = AerSimulator(method="density_matrix", noise_model=noise, max_parallel_threads=1)
    res = sim.run(tomo_circuits(compiled, label, n), shots=shots, seed_simulator=seed).result()
    probs = [model.counts_to_probs(res.get_counts(i), 2 * n) for i in range(len(res.results))]
    return reconstruct(probs, n, mats)


# ----------------------------------------------------------------------------- comparison

def compare(run_dir, noise_params=None):
    """One row per (tomography, system): fidelities to Gibbs of hardware / noisy / ideal,
    hardware-vs-simulator fidelities, purities."""
    cfg, exp, compiled = load(run_dir)
    n, layout = cfg["n"], cfg["layout"]
    hw = hardware_states(run_dir, cfg, exp)
    mats = [np.array(m) for m in exp["readout"]["mats"]]
    ro = readout_error_per_qubit(exp, cfg)
    params = noise_params or default_noise(cfg, exp)
    p_cz = params["p_cz"]
    if isinstance(p_cz, dict):
        p_cz = {frozenset(map(int, k.split("-"))): v for k, v in p_cz.items()}
    gate_noise = build_noise(None, p_cz, params["gamma"], params["p_1q"])
    full_noise = build_noise(None, p_cz, params["gamma"], params["p_1q"], readout=ro)
    rows = []
    for label, t in exp["tomo"].items():
        beta = t["beta"]
        ideal = ansatz_state(compiled, label, n, layout)
        noisy = ansatz_state(compiled, label, n, layout, gate_noise)
        noisy_tomo = simulated_tomography(compiled, label, n, cfg["shots_tomo"], full_noise, mats)
        G = DensityMatrix(model.gibbs_state(n, beta))
        for s in ("A", "B"):
            f = lambda r: float(state_fidelity(DensityMatrix(r), G, validate=False))
            pur = lambda r: float(np.real(np.trace(r @ r)))
            rows.append({
                "label": label, "beta": beta, "system": s,
                "F_hw": f(hw[label][s]), "F_noisy_tomo": f(noisy_tomo[s]), "F_noisy": f(noisy[s]), "F_ideal": f(ideal[s]),
                "F_hw_vs_noisy": float(state_fidelity(DensityMatrix(hw[label][s]), DensityMatrix(noisy[s]), validate=False)),
                "F_hw_vs_ideal": float(state_fidelity(DensityMatrix(hw[label][s]), DensityMatrix(ideal[s]), validate=False)),
                "purity_hw": pur(hw[label][s]), "purity_noisy": pur(noisy[s]), "purity_ideal": pur(ideal[s]),
            })
    return rows


# ----------------------------------------------------------------------------- refit

def _sim_states(compiled, labels, n, layout, p_cz, gamma, p_1q):
    nm = build_noise(None, p_cz, gamma, p_1q)
    return {lab: ansatz_state(compiled, lab, n, layout, nm) for lab in labels}


def fit_run(run_dir, per_coupler=True, fit_on="reference_start", start=None):
    """Refit the gate noise on a run's reference tomographies by matching the full density
    matrices (Frobenius distance over rho_A and rho_B). Returns the fitted parameters and the
    residuals on the fitting set and, if present, on the end-of-run references (validation).
    `start` ({"p_cz", "gamma", "p_1q"}) overrides the starting point (default: the 2+2 fit)."""
    cfg, exp, compiled = load(run_dir)
    n, layout = cfg["n"], cfg["layout"]
    hw = hardware_states(run_dir, cfg, exp)
    fit_labels = [l for l in exp["tomo"] if l.startswith(fit_on)]
    val_labels = [l for l in exp["tomo"] if l.startswith("reference_end")]
    edges = [frozenset((layout[a], layout[b])) for a, b in devices.ladder_edges(n)]
    start = start or default_noise(cfg, exp)

    def unpack(x):
        gamma, p_1q = x[0], x[1]
        p = dict(zip(edges, x[2:])) if per_coupler else x[2]
        return p, gamma, p_1q

    def residual(x, labels):
        p, gamma, p_1q = unpack(x)
        sim = _sim_states(compiled, labels, n, layout, p, gamma, p_1q)
        return np.concatenate([np.abs(sim[l][s] - hw[l][s]).ravel() for l in labels for s in ("A", "B")])

    x0 = [start["gamma"], start["p_1q"]] + ([start["p_cz"]] * len(edges) if per_coupler else [start["p_cz"]])
    res = least_squares(residual, x0, args=(fit_labels,), bounds=(0.0, 0.3), diff_step=0.05, max_nfev=200)
    p, gamma, p_1q = unpack(res.x)
    rms = lambda labels, x: float(np.sqrt(np.mean(residual(x, labels) ** 2))) if labels else None
    out = {"per_coupler": per_coupler, "gamma": float(gamma), "p_1q": float(p_1q),
           "p_cz": ({"-".join(map(str, sorted(e))): float(v) for e, v in p.items()} if per_coupler else float(p)),
           "rms_fit": rms(fit_labels, res.x), "rms_validation": rms(val_labels, res.x),
           "rms_fit_before": rms(fit_labels, np.array(x0)), "rms_validation_before": rms(val_labels, np.array(x0)),
           "fit_on": fit_labels, "validated_on": val_labels}
    return out


def main():
    run_dir = Path(sys.argv[1])
    rows = compare(run_dir)
    print(f"{'tomography':<30} sys   F_hw  F_noisy(tomo) F_noisy F_ideal | F(hw,noisy) F(hw,ideal) | purity hw/noisy")
    for r in rows:
        print(f"{r['label']:<30} {r['system']}   {r['F_hw']:.3f}   {r['F_noisy_tomo']:.3f}       {r['F_noisy']:.3f}   "
              f"{r['F_ideal']:.3f}  |   {r['F_hw_vs_noisy']:.3f}       {r['F_hw_vs_ideal']:.3f}   |  "
              f"{r['purity_hw']:.3f}/{r['purity_noisy']:.3f}")


if __name__ == "__main__":
    main()
