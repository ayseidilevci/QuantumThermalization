"""Error-mitigated state preparation and tomography of the n+n TFD on Tuna-17 (no optimization).

The earlier 3+3 and 4+4 runs showed that the device, not the optimizer, limits the fidelity: the
noiseless-optimal angles measured on the device were as good as SPSA's. This runner therefore prepares
the noiseless-optimal state at each beta (reference_angles, as in run_hardware.py) and spends the
hardware time on tomography with error mitigation, so that qubit choice and mitigation can be compared.

Circuit modes (per tomography):
  platform  logical circuits compiled by the default transpile against the backend (as in the earlier
            runs; Quantum Inspire decomposes the CNOT/controlled-phase gates itself)
  native    compiled here to rx, ry, rz, cz on the chosen qubits, submitted without further optimization
  twirl     native, plus Pauli twirling of every CZ and a random X before each measurement (undone in
            the analysis): cfg["twirls"] random variants per setting, the shots split evenly

Readout: every tomography is preceded by its own calibration of the 2^n basis states, prepared on A
and B at the same time (2^n circuits). The analysis applies three corrections to the same data:
  none      raw marginal distribution of each system
  perqubit  tensor product of per-qubit matrices from the all-0 and all-1 preparations (as before)
  full      the full 2^n x 2^n assignment matrix of each system (captures correlated readout errors)
Each state is reconstructed by linear inversion over the 3^n settings and projected onto the closest
physical state (Smolin, Gambetta, Smith 2012). Fidelities are stored in both conventions.

A plan lists stages; each stage expands to, for every repeat, layout and beta: one calibration, then
one tomography per mode. Example: tfd_study/mitigated_plan_3_scan.json.

    python -m tfd_study.run_mitigated --config <plan.json> --simulate {ideal,coherent}
    python -m tfd_study.run_mitigated --config <plan.json>                # hardware
    python -m tfd_study.run_mitigated --resume results/mitig-<tag>
"""

import argparse
import json
import sys
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, qasm3, transpile
from qiskit.circuit import pauli_twirl_2q_gates
from qiskit.quantum_info import DensityMatrix, Statevector, state_fidelity
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGatesDecomposition

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gibbs_tfd_reproduction"))
import run_hardware_parallel as rhp  # noqa: E402  (scheduler, unchanged apart from the compile hook)

from tfd_study import devices, model  # noqa: E402
from tfd_study.run_hardware import reference_angles  # noqa: E402

log = rhp.log
MODES = ("platform", "native", "twirl")
CORRECTIONS = ("none", "perqubit", "full")
MERGE_1Q = PassManager([Optimize1qGatesDecomposition(basis=["rz", "ry", "rx"])])


# ----------------------------------------------------------------------------- circuits

def logical_circuit(n, p, theta, setting, flip_mask=0):
    """Ansatz, basis change for `setting` (n letters, applied to A and B alike), an X before the
    measurement of every qubit in flip_mask (bit k = logical qubit k), measurement qubit k -> clbit k."""
    qc = QuantumCircuit(2 * n, 2 * n)
    qc.compose(model.bound_ansatz(n, p, theta), inplace=True)
    for q, basis in enumerate(setting * 2):      # same basis changes as model.measured
        if basis == "X":
            qc.ry(-np.pi / 2, q)
        elif basis == "Y":
            qc.rx(np.pi / 2, q)
    for k in range(2 * n):
        if (flip_mask >> k) & 1:
            qc.x(k)
    qc.measure(range(2 * n), range(2 * n))
    return qc


def calibration_circuit(n, s):
    """Basis state s (n bits) prepared on A and the same on B, all 2n qubits measured."""
    qc = QuantumCircuit(2 * n, 2 * n, name=f"cal{s}")
    for k in range(n):
        if (s >> k) & 1:
            qc.x(model.A(k))
            qc.x(model.B(k, n))
    qc.measure(range(2 * n), range(2 * n))
    return qc


def trimmed(circuit):
    """The same circuit on qubits 0..(highest used); physical qubit numbers are kept, so a per-coupler
    noise model still applies. Only for the simulators: Aer's density-matrix method takes at most 14
    qubits, while compiled circuits span all 17 qubits of Tuna-17."""
    used = {circuit.find_bit(q).index for inst in circuit.data for q in inst.qubits}
    out = QuantumCircuit(max(used) + 1, circuit.num_clbits, name=circuit.name)
    for inst in circuit.data:
        out.append(inst.operation, [circuit.find_bit(q).index for q in inst.qubits],
                   [circuit.find_bit(c).index for c in inst.clbits])
    return out


def twirled(native_circuit, seed):
    """One Pauli-twirled copy of a native circuit, with the Paulis merged into the 1-qubit rotations.
    The CZ gates are untouched, so the number of CZ is the same as in the native circuit."""
    tw = pauli_twirl_2q_gates(native_circuit, twirling_gate=["cz"], seed=int(seed))
    out = MERGE_1Q.run(tw)
    out.name = native_circuit.name
    return out


# ----------------------------------------------------------------------------- analysis helpers

def marginals(p, n):
    """Full 2n-bit distribution -> (A marginal, B marginal); index bit k = logical qubit k."""
    t = p.reshape(2 ** n, 2 ** n)              # [B bits, A bits]
    return t.sum(axis=0), t.sum(axis=1)


def unflip(q, mask):
    return q[np.arange(len(q)) ^ mask]


def rho_from_marginals(margs, n):
    """Linear inversion from {setting (n letters): distribution over n bits}; returns (raw, physical)."""
    paulis = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]),
              "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1.0, -1.0])}
    rho = np.zeros((2 ** n, 2 ** n), complex)
    for label in product("IXYZ", repeat=n):
        support = [k for k in range(n) if label[k] != "I"]
        vals = [model.parity(q, support) for s, q in margs.items() if all(s[k] == label[k] for k in support)]
        op = np.array([[1.0]])
        for k in reversed(range(n)):
            op = np.kron(op, paulis[label[k]])
        rho += np.mean(vals) * op
    rho /= 2 ** n
    return rho, model.project_physical(rho)


# ----------------------------------------------------------------------------- experiment

class Experiment:
    """Task interface of run_hardware_parallel.Experiment, plus compile() for pre-compiled circuits."""

    def __init__(self, cfg, state=None):
        self.cfg = cfg
        self.n, self.p = cfg["n"], cfg["p"]
        self._compiled = {}
        self.compiled_log = None
        if state is not None:
            self.st = state
            return
        self.st = {"tasks": {}, "pending": [], "next_tid": 0, "cals": {}, "results": {}, "deferred": [],
                   "submissions_done": 0, "n_tomo": 0}
        seed = 0
        for si, stage in enumerate(cfg["stages"]):
            for rep in range(stage.get("repeats", 1)):
                for layout in stage["layouts"]:
                    for beta in stage["betas"]:
                        key = f"s{si}/{'-'.join(map(str, layout))}/b={beta:g}/r{rep}"
                        self.add({"kind": "cal", "cal_key": key, "layout": layout})
                        for mode in stage["modes"]:
                            seed += 1
                            self.add(self._tomo_task(key, layout, beta, mode, seed))
                            self.st["n_tomo"] += 1

    def _tomo_task(self, cal_key, layout, beta, mode, seed):
        k = self.cfg["twirls"] if mode == "twirl" else 1
        rng = np.random.default_rng([self.cfg["seed"], seed])
        n_set = 3 ** self.n
        masks = rng.integers(0, 2 ** (2 * self.n), size=n_set * k).tolist() if mode == "twirl" else [0] * n_set
        return {"kind": "tomo", "label": f"{cal_key}/{mode}", "cal_key": cal_key, "layout": layout,
                "beta": beta, "mode": mode, "variants": k, "masks": masks, "seed": seed,
                "params": [float(x) for x in self.cfg["reference_angles"][str(float(beta))]]}

    # -- bookkeeping (as in run_hardware_parallel)
    def add(self, task):
        tid = str(self.st["next_tid"])
        self.st["next_tid"] += 1
        n_circ = 2 ** self.n if task["kind"] == "cal" else len(task["masks"])
        task.update({"n_chunks": -(-n_circ // self.cfg["max_circuits"]), "counts": {}})
        self.st["tasks"][tid] = task
        self.st["pending"] += [{"tid": tid, "chunk": c, "attempt": 0} for c in range(task["n_chunks"])]

    def calibration_due(self):
        return False                           # every tomography has its own calibration

    def describe(self, tid):
        t = self.st["tasks"][tid]
        return f"calibration {t['cal_key']}" if t["kind"] == "cal" else f"tomography {t['label']}"

    def shots(self, task):
        if task["kind"] == "cal":
            return self.cfg["shots_cal"]
        return self.cfg["shots_tomo"] // task["variants"]

    def finished(self):
        return len(self.st["results"]) >= self.st["n_tomo"]

    # -- circuits
    def logical_circuits(self, task):
        n, p = self.n, self.p
        if task["kind"] == "cal":
            return [calibration_circuit(n, s) for s in range(2 ** n)]
        k = task["variants"]
        out = []
        for i, s in enumerate(model.tomography_settings(n)):
            for v in range(k):
                c = logical_circuit(n, p, task["params"], s[:n], task["masks"][i * k + v])
                c.name = f"{task['label']}|{s[:n]}|v{v}"
                out.append(c)
        return out

    def circuits(self, task, chunk=None):
        circs = self.logical_circuits(task)
        if chunk is None:
            return circs
        m = self.cfg["max_circuits"]
        return circs[chunk * m:(chunk + 1) * m]

    def compile(self, task, chunk, backend):
        """All circuits of a task compiled once (cached); returns the chunk's slice."""
        key = task.get("label") or task["cal_key"]
        if key not in self._compiled:
            logical = self.logical_circuits(task)
            if task["kind"] == "tomo" and task["mode"] == "platform":
                out = transpile(logical, backend, initial_layout=task["layout"])
            else:
                native = devices.to_device(logical, task["layout"])
                native = native if isinstance(native, list) else [native]
                if task["kind"] == "tomo" and task["mode"] == "twirl":
                    native = [twirled(c, 1000 * task["seed"] + i) for i, c in enumerate(native)]
                if self.cfg.get("sim"):
                    native = [trimmed(c) for c in native]
                # already native and on physical qubits: only map gate names onto the backend
                out = transpile(native, backend, optimization_level=0)
            out = out if isinstance(out, list) else [out]
            self._compiled[key] = out
            if self.compiled_log is not None:
                with self.compiled_log.open("a") as fh:
                    for c in out:
                        fh.write(json.dumps({"name": c.name, "cz": c.count_ops().get("cz", 0),
                                             "qasm": qasm3.dumps(c)}) + "\n")
        m = self.cfg["max_circuits"]
        return self._compiled[key][chunk * m:(chunk + 1) * m]

    # -- results
    def chunk_done(self, tid, chunk, counts):
        task = self.st["tasks"][tid]
        task["counts"][str(chunk)] = counts
        self.st["submissions_done"] += 1
        if len(task["counts"]) < task["n_chunks"]:
            return
        if task["kind"] == "cal":
            self._calibration(tid)
            for d in [d for d in self.st["deferred"] if self.st["tasks"][d]["cal_key"] == task["cal_key"]]:
                self.st["deferred"].remove(d)
                self._analyse(d)
        elif task["cal_key"] in self.st["cals"]:
            self._analyse(tid)
        else:
            self.st["deferred"].append(tid)

    def _counts(self, task):
        return [c for k in range(task["n_chunks"]) for c in task["counts"][str(k)]]

    def _calibration(self, tid):
        task = self.st["tasks"].pop(tid)
        n = self.n
        probs = [model.counts_to_probs(c, 2 * n) for c in self._counts(task)]
        MA = np.array([marginals(p, n)[0] for p in probs]).T          # M[measured, prepared]
        MB = np.array([marginals(p, n)[1] for p in probs]).T
        mats = model.readout_matrices(probs[0], probs[-1], 2 * n)
        self.st["cals"][task["cal_key"]] = {"MA": MA.tolist(), "MB": MB.tolist(),
                                           "perqubit": [m.tolist() for m in mats],
                                           "time": datetime.now().isoformat(timespec="seconds")}
        log(f"calibration {task['cal_key']}: P(correct) A min {MA.diagonal().min():.3f} mean "
            f"{MA.diagonal().mean():.3f}, B min {MB.diagonal().min():.3f} mean {MB.diagonal().mean():.3f}")

    def _analyse(self, tid):
        task = self.st["tasks"].pop(tid)
        n, beta = self.n, task["beta"]
        cal = self.st["cals"][task["cal_key"]]
        MA, MB = np.array(cal["MA"]), np.array(cal["MB"])
        mats = [np.array(m) for m in cal["perqubit"]]
        k, mask_low = task["variants"], 2 ** n - 1
        settings = [s[:n] for s in model.tomography_settings(n)]
        probs = [model.counts_to_probs(c, 2 * n) for c in self._counts(task)]
        exact = {"A": model.exact_rho_A(n, self.p, task["params"]), "B": model.exact_rho_B(n, self.p, task["params"])}
        entry = {"label": task["label"], "layout": task["layout"], "beta": beta, "mode": task["mode"],
                 "variants": k, "params": task["params"], "calibration": cal["time"],
                 "finished": datetime.now().isoformat(timespec="seconds"),
                 "fid_paper_noiseless": float(np.sqrt(model.exact_fidelity(n, self.p, task["params"], beta))),
                 "corrections": {}}
        for corr in CORRECTIONS:
            margs = {"A": {}, "B": {}}
            for i, s in enumerate(settings):
                acc = {"A": 0.0, "B": 0.0}
                for v in range(k):
                    p = probs[i * k + v]
                    if corr == "perqubit":
                        p = model.correct_readout(p, mats)
                    qa, qb = marginals(p, n)
                    if corr == "full":
                        qa, qb = np.linalg.solve(MA, qa), np.linalg.solve(MB, qb)
                    mask = task["masks"][i * k + v]
                    acc["A"] = acc["A"] + unflip(qa, mask & mask_low)
                    acc["B"] = acc["B"] + unflip(qb, mask >> n)
                margs["A"][s], margs["B"][s] = acc["A"] / k, acc["B"] / k
            res = {}
            for system in ("A", "B"):
                raw, rho = rho_from_marginals(margs[system], n)
                f = model.fidelity(rho, beta, n)
                f_circ = float(np.sqrt(max(state_fidelity(DensityMatrix(rho), DensityMatrix(exact[system]),
                                                          validate=False), 0)))
                res[system] = {"fid": f, "fid_paper": float(np.sqrt(max(f, 0))), "fid_paper_to_circuit": f_circ,
                               "purity": float(np.real(np.trace(rho @ rho))),
                               "min_eig_raw": float(np.linalg.eigvalsh((raw + raw.conj().T) / 2).min()),
                               "rho": [np.real(rho).tolist(), np.imag(rho).tolist()]}
            entry["corrections"][corr] = res
        self.st["results"][task["label"]] = entry
        r = entry["corrections"]
        log(f"tomography {task['label']}: F_A (paper) none {r['none']['A']['fid_paper']:.3f} / perqubit "
            f"{r['perqubit']['A']['fid_paper']:.3f} / full {r['full']['A']['fid_paper']:.3f}; "
            f"F_B full {r['full']['B']['fid_paper']:.3f}  (noiseless {entry['fid_paper_noiseless']:.3f})")


# ----------------------------------------------------------------------------- backends, report, main

def make_backend(cfg, sim):
    if sim is None:
        return devices.real_backend(cfg["backend"])
    from qiskit_aer import AerSimulator
    noise = None
    if sim == "coherent":
        couplers = {(lay[a], lay[b]) for st in cfg["stages"] for lay in st["layouts"]
                    for a, b in devices.ladder_edges(cfg["n"])}
        noise = devices.coherent_noise_model(sorted(couplers), seed=cfg["seed"])
    return AerSimulator(method="density_matrix" if noise else "automatic", noise_model=noise,
                        coupling_map=devices.coupling_map(), basis_gates=devices.NATIVE_GATES)


def report(exp, run_dir):
    lines = ["label                                             F_A paper: none  perqubit  full   F_B full  "
             "noiseless  F_A to circuit (full)"]
    for label, e in exp.st["results"].items():
        c = e["corrections"]
        lines.append(f"{label:48s}  {c['none']['A']['fid_paper']:.3f}    {c['perqubit']['A']['fid_paper']:.3f}   "
                     f"{c['full']['A']['fid_paper']:.3f}   {c['full']['B']['fid_paper']:.3f}     "
                     f"{e['fid_paper_noiseless']:.3f}      {c['full']['A']['fid_paper_to_circuit']:.3f}")
    text = "\n".join(lines)
    (run_dir / "analysis.txt").write_text(text + "\n")
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--simulate", choices=["ideal", "coherent"])
    ap.add_argument("--resume", type=Path)
    args = ap.parse_args()

    if args.resume:
        run_dir = args.resume
        state = json.loads((run_dir / "state.json").read_text())
        cfg = state["cfg"]
        exp = Experiment(cfg, state["exp"])
        if state.get("inflight"):
            log(f"resuming; {len(state['inflight'])} submission(s) that were in the queue are resubmitted")
            exp.st["pending"] = [x["sub"] for x in state["inflight"]] + exp.st["pending"]
            state["inflight"] = []
        sim = cfg.get("sim")
    else:
        cfg = json.loads(args.config.read_text())["run"]
        cfg.setdefault("max_circuits", 5)
        cfg["sim"] = sim = args.simulate
        if sim is None:
            bt = devices.real_backend(cfg["backend"]).get_backend_type()
            cfg["max_circuits"] = bt.max_jobs_per_batch_job
            cfg["max_inflight"] = min(cfg["max_inflight"], bt.batchjobs_per_queue_limit)
        betas = sorted({float(b) for st in cfg["stages"] for b in st["betas"]})
        log("computing the noiseless-optimal angles ...")
        cfg["reference_angles"] = reference_angles(cfg["n"], cfg["p"], cfg["varsigma"], betas)
        tag = f"sim-{sim}" if sim else "hw"
        run_dir = ROOT / "results" / f"mitig-{tag}_n{cfg['n']}_{datetime.now():%Y%m%d_%H%M%S}"
        run_dir.mkdir(parents=True, exist_ok=False)
        state = {"cfg": cfg}
        exp = Experiment(cfg)
    exp.compiled_log = run_dir / "compiled.jsonl"
    n_sub = sum(t["n_chunks"] for t in exp.st["tasks"].values())
    log(f"run directory {run_dir}\n  {cfg['n']}+{cfg['n']}, {exp.st['n_tomo']} tomographies, "
        f"{n_sub} submissions to go, backend {'simulator ' + sim if sim else cfg['backend']}")
    backend = make_backend(cfg, sim)
    rhp.run_scheduler(exp, backend, None, run_dir, state, cfg["max_inflight"], poll=1 if sim else 15)
    report(exp, run_dir)
    log(f"done; everything is in {run_dir}")


if __name__ == "__main__":
    main()
