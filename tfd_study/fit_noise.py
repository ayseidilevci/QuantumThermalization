"""Fit the uniform Tuna-17 noise model to our own hardware tomographies.

Data: results/parallel_20261019_175824 (Tuna-17, qubits 2-4-5-8, 2+2 ansatz,
36-setting readout-corrected tomography of systems A and B for the reference
angles and every SPSA replica). For each recorded tomography we simulate the
exact circuit that ran (repo TFDCircuit, same layout, compiled to Tuna's native
gates) with depolarizing error p_cz plus amplitude damping gamma_cz per CZ and
depolarizing error p_1q per rx/ry, and compare
fidelity-to-Gibbs and purity of rho_A and rho_B with what tomography measured.
Both quantities are independent of the qubit-ordering convention.

Readout error is taken directly from the run's 16-state calibration matrix,
averaged over the four qubits. The measured states were readout-corrected, so
readout does not enter the gate-error fit.

    python -m tfd_study.fit_noise        # writes tfd_study/noise_fit.json

The uniform model misses a coherent error: at beta ~ 0 one measured pair (qubits
2-5) is partly disentangled, with a single-qubit Bloch component whose sign
changes with the angles. `--coherent` fits the per-coupler alternative
(devices.coherent_noise_model) to the full density matrices instead, and
reports its residual on a held-out half of the tomographies.

    python -m tfd_study.fit_noise --coherent    # writes tfd_study/noise_fit_coherent.json
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, partial_trace, state_fidelity
from qiskit_aer import AerSimulator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gibbs_tfd_reproduction"))
from tfd_circuit import TFDCircuit  # noqa: E402  (the circuit that actually ran)

from tfd_study import devices, model  # noqa: E402

warnings.filterwarnings("ignore")
RUN = ROOT / "results" / "parallel_20261019_175824" / "state.json"
OUT = Path(__file__).resolve().parent / "noise_fit.json"
OUT_COHERENT = Path(__file__).resolve().parent / "noise_fit_coherent.json"
SYSTEMS = {"A": [0, 1], "B": [2, 3]}            # run_hardware_parallel convention


def load_tomographies():
    state = json.loads(RUN.read_text())
    rows = []
    for label, t in state["exp"]["tomo"].items():
        for system in ("A", "B"):
            if system not in t or "rho" not in t[system]:
                continue
            re, im = np.array(t[system]["rho"])
            rho = re + 1j * im
            rows.append({"label": f"{label}/{system}", "beta": float(t["beta"]), "params": t["params"],
                         "system": system, "fid": float(t[system]["fid"]), "rho": rho,
                         "purity": float(np.real(np.trace(rho @ rho)))})
    return state, rows


def readout_from_run(state):
    """Average per-qubit P(1|0), P(0|1) from the full 16x16 calibration matrix M[measured, prepared]."""
    M = np.array(state["exp"]["readout"]["M"])
    nq = int(np.log2(len(M)))
    e01, e10 = [], []
    for q in range(nq):
        bit = (np.arange(2 ** nq) >> q) & 1
        e01 += [M[bit == 1, j].sum() for j in range(2 ** nq) if not (j >> q) & 1]
        e10 += [M[bit == 0, j].sum() for j in range(2 ** nq) if (j >> q) & 1]
    return float(np.mean(e01)), float(np.mean(e10))


def simulate(rows, p_cz, p_1q, gamma_cz, layout):
    """Noisy rho of each row's system, as fidelity and purity."""
    sim = devices.simulator("noisy", p_cz=p_cz, p_1q=p_1q, gamma_cz=gamma_cz, readout=(0.0, 0.0))
    circs = []
    for r in rows:
        qc = QuantumCircuit(4)
        TFDCircuit(*r["params"]).build_circuit(qc)
        qc = devices.to_device(qc, layout)
        qc.save_density_matrix(qubits=[qc.layout.final_index_layout()[q] for q in range(4)])
        circs.append(qc)
    res = sim.run(circs).result()
    out = []
    for i, r in enumerate(rows):
        rho4 = DensityMatrix(res.data(i)["density_matrix"])
        keep = SYSTEMS[r["system"]]
        rho = partial_trace(rho4, [q for q in range(4) if q not in keep])
        gibbs = DensityMatrix(model.gibbs_state(2, r["beta"]))
        out.append((float(state_fidelity(rho, gibbs, validate=False)), float(np.real(rho.purity()))))
    return np.array(out)


def compiled_circuits(rows, layout):
    circs = []
    for r in rows:
        qc = QuantumCircuit(4)
        TFDCircuit(*r["params"]).build_circuit(qc)
        qc = devices.to_device(qc, layout)
        qc.save_density_matrix(qubits=[qc.layout.final_index_layout()[q] for q in range(4)])
        circs.append(qc)
    return circs


def coherent_fit(rows, circs, couplers, train, start_seeds=6):
    """Least-squares fit of devices.coherent_noise_model to the full measured rho_A/rho_B of
    the `train` rows. x = [gamma_cz, p_1q, p_depol per coupler, (cphase, phase_a, phase_b) per
    coupler]. theta = 0 phases is a stationary point of the residual, so random starts."""
    from scipy.optimize import least_squares
    K = len(couplers)

    def fit_dict(x):
        return {"gamma_cz": float(x[0]), "p_1q": float(x[1]), "readout": [0.0, 0.0],
                "couplers": {f"{a}-{b}": {"p_depol": float(x[2 + k]), "cphase": float(x[2 + K + 3 * k]),
                                          "phase": {str(a): float(x[3 + K + 3 * k]), str(b): float(x[4 + K + 3 * k])}}
                             for k, (a, b) in enumerate(couplers)}}

    def states(x, idx):
        nm = _coherent_model(fit_dict(x), couplers)
        sim = AerSimulator(method="density_matrix", noise_model=nm, max_parallel_threads=1)
        res = sim.run([circs[i] for i in idx]).result()
        return [partial_trace(DensityMatrix(res.data(j)["density_matrix"]),
                              [q for q in range(4) if q not in SYSTEMS[rows[i]["system"]]]).data
                for j, i in enumerate(idx)]

    def residual(x, idx):
        d = [s - rows[i]["rho"] for s, i in zip(states(x, idx), idx)]
        return np.concatenate([np.concatenate([e.real.ravel(), e.imag.ravel()]) for e in d])

    lb = [0, 0] + [0] * K + [-np.pi] * 3 * K
    ub = [0.3, 0.05] + [0.3] * K + [np.pi] * 3 * K
    best = None
    for s in range(start_seeds):
        x0 = [0.01, 0.003] + [0.03] * K + list(np.random.default_rng(s).uniform(-0.4, 0.4, 3 * K))
        r = least_squares(residual, x0, args=(train,), bounds=(lb, ub), diff_step=1e-3, max_nfev=150)
        print(f"  start {s}: train cost {r.cost:.5f}", flush=True)
        best = r if best is None or r.cost < best.cost else best
    rms = lambda x, idx: float(np.sqrt(np.mean(residual(x, idx) ** 2)))
    return fit_dict(best.x), rms, best.x


def _coherent_model(fit, couplers):
    """devices.coherent_noise_model for a fit dict that is not (yet) on disk."""
    from unittest import mock
    with mock.patch.object(devices, "coherent_fit", lambda: fit):
        return devices.coherent_noise_model(couplers)


def main_coherent():
    state, rows = load_tomographies()
    layout = state["cfg"]["layout"]
    circs = compiled_circuits(rows, layout)
    couplers = sorted({tuple(sorted((qc.find_bit(i.qubits[0]).index, qc.find_bit(i.qubits[1]).index)))
                       for qc in circs for i in qc.data if i.operation.name == "cz"})
    print(f"{len(rows)} measured states, couplers {couplers}")

    # held-out check: fit on half the tomographies (A and B of one tomography together)
    labels = sorted({r["label"].rsplit("/", 1)[0] for r in rows})
    np.random.default_rng(1).shuffle(labels)
    held = set(labels[: len(labels) // 2])
    test = [i for i, r in enumerate(rows) if r["label"].rsplit("/", 1)[0] in held]
    train = [i for i in range(len(rows)) if i not in test]
    print("fit on half, test on the other half:")
    _, rms, x_half = coherent_fit(rows, circs, couplers, train)
    uni = devices.noise_fit()

    def rms_uniform(idx):
        nm = devices.noise_model(readout=(0.0, 0.0))
        res = AerSimulator(method="density_matrix", noise_model=nm, max_parallel_threads=1).run(
            [circs[i] for i in idx]).result()
        d = [partial_trace(DensityMatrix(res.data(j)["density_matrix"]),
                           [q for q in range(4) if q not in SYSTEMS[rows[i]["system"]]]).data - rows[i]["rho"]
             for j, i in enumerate(idx)]
        return float(np.sqrt(np.mean(np.concatenate([np.abs(e).ravel() ** 2 for e in d]) / 2)))

    held_out = {"coherent": rms(x_half, test), "uniform": rms_uniform(test)}
    print(f"  held-out rms |rho_model - rho_hw|: coherent {held_out['coherent']:.4f}, uniform {held_out['uniform']:.4f}")

    print("final fit on all tomographies:")
    fit, rms, x = coherent_fit(rows, circs, couplers, list(range(len(rows))))
    phases = [v for c in fit["couplers"].values() for v in [c["cphase"], *c["phase"].values()]]
    fit.update({
        "readout": uni["readout"],
        "unmeasured": {"p_depol": float(np.mean([c["p_depol"] for c in fit["couplers"].values()])),
                       "phase_rms": float(np.sqrt(np.mean(np.square(phases))))},
        "rms_rho_all": rms(x, list(range(len(rows)))), "rms_rho_held_out": held_out,
        "source": f"{RUN.parent.name} ({len(rows)} tomographies, Tuna-17 qubits {layout})",
        "model": "per CZ: coherent phase error (conditional + single-qubit) then per-coupler depolarizing, "
                 "then amplitude damping on both qubits; per rx/ry: depolarizing; uniform readout; "
                 "unmeasured couplers: mean depolarizing, random phases with the fitted rms",
    })
    OUT_COHERENT.write_text(json.dumps(fit, indent=1))
    print(f"gamma_cz {fit['gamma_cz']:.4f}, p_1q {fit['p_1q']:.4f}, phase rms {np.degrees(fit['unmeasured']['phase_rms']):.1f} deg")
    for e, c in fit["couplers"].items():
        print(f"  {e}: p_depol {c['p_depol']:.4f}, cphase {np.degrees(c['cphase']):+.1f} deg, "
              + ", ".join(f"q{q} {np.degrees(v):+.1f} deg" for q, v in c["phase"].items()))


def main():
    if "--coherent" in sys.argv:
        return main_coherent()
    state, rows = load_tomographies()
    layout = state["cfg"]["layout"]
    meas = np.array([[r["fid"], r["purity"]] for r in rows])
    print(f"{len(rows)} measured states from {RUN.parent.name}, layout {layout}")

    # Depolarizing alone can only mix towards 1/4; the hardware states at beta ~ 0
    # are purer than that, which needs a non-unital (T1-like) part, hence gamma_cz.
    ratio = 0.05                                  # p_1q / p_cz; 1q errors barely matter here
    best = (np.inf, 0.0, 0.0)
    for p_cz in np.arange(0.0, 0.0601, 0.005):
        for gamma in np.arange(0.0, 0.0401, 0.005):
            sim = simulate(rows, p_cz, ratio * p_cz, gamma, layout)
            rms = float(np.sqrt(np.mean((sim - meas) ** 2)))
            best = min(best, (rms, p_cz, gamma))
        print(f"  p_cz={p_cz:.3f}: best so far rms={best[0]:.4f} at p_cz={best[1]:.3f}, gamma={best[2]:.3f}")
    rms, p_cz, gamma = best
    for p in np.arange(max(0.0, p_cz - 0.004), p_cz + 0.0041, 0.002):     # refine
        for g in np.arange(max(0.0, gamma - 0.004), gamma + 0.0041, 0.002):
            sim = simulate(rows, p, ratio * p, g, layout)
            r_ = float(np.sqrt(np.mean((sim - meas) ** 2)))
            if r_ < rms:
                rms, p_cz, gamma = r_, float(p), float(g)

    sim = simulate(rows, p_cz, ratio * p_cz, gamma, layout)
    ideal = simulate(rows, 0.0, 0.0, 0.0, layout)
    depol_only = min((float(np.sqrt(np.mean((simulate(rows, p, ratio * p, 0.0, layout) - meas) ** 2))), p)
                     for p in np.arange(0.0, 0.0601, 0.005))
    fit = {
        "p_cz": round(p_cz, 4), "p_1q": round(ratio * p_cz, 5), "gamma_cz": round(gamma, 4),
        "readout": readout_from_run(state), "rms_fid_purity": round(rms, 4),
        "rms_depolarizing_only": round(depol_only[0], 4),
        "source": f"{RUN.parent.name} ({len(rows)} tomographies, Tuna-17 qubits {layout})",
        "model": "per CZ: depolarizing + amplitude damping on both qubits; per rx/ry: depolarizing; uniform over qubits; readout from the run's calibration",
        "per_state": [{"label": r["label"], "beta": r["beta"], "measured_fid": r["fid"],
                       "model_fid": float(s[0]), "ideal_fid": float(i[0]),
                       "measured_purity": r["purity"], "model_purity": float(s[1])}
                      for r, s, i in zip(rows, sim, ideal)],
    }
    OUT.write_text(json.dumps(fit, indent=1))
    print(f"\nbest: p_cz={fit['p_cz']}, p_1q={fit['p_1q']}, gamma_cz={fit['gamma_cz']}, "
          f"readout P(1|0),P(0|1)={np.round(fit['readout'], 4)}, rms={rms:.4f} "
          f"(depolarizing only: {depol_only[0]:.4f})")
    for s in fit["per_state"]:
        print(f"  {s['label']:<28} beta={s['beta']:<6g} F meas {s['measured_fid']:.3f}  model {s['model_fid']:.3f}  "
              f"(no noise {s['ideal_fid']:.3f})   purity meas {s['measured_purity']:.3f}  model {s['model_purity']:.3f}")


if __name__ == "__main__":
    main()
