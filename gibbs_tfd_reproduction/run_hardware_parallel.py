"""Parallel, paper-style hardware run: several SPSA replicas through a job scheduler.

Self-contained: only TFDCircuit and ground_truth are imported; cost_function.py,
tomography.py, optimizers.py and run_hardware.py are not used or changed.

Why (measured on Tuna-17, 2026-10-18/19): a submission waits minutes to hours
in the queue, but submissions queued together run back to back (0-3 s apart).
So the scarce resource is queue waits. This script therefore

  * measures each observable in its own circuit (no mid-circuit
    measurement/reset), as in the paper:
      - cost: one all-X-basis and one all-Z-basis circuit per parameter point
        (an SPSA step = 4 circuits = one submission)
      - readout calibration as a shared task, refreshed every --cal-interval
        minutes: all 16 basis states (paper, full 16x16 inversion) or
        all-0/all-1 (per-qubit) -- --calibration full|perqubit
      - tomography: 36 settings {+-X,+-Y,+-Z}^2 with sign flips (paper) or 9
        -- --tomography 36|9; one set of circuits gives systems A AND B
  * respects Quantum Inspire's limits (read from the backend): at most 5
    circuits per submission (tasks are split) and 5 submissions in the queue.
    Execution time grows with circuits x shots (~5 s per circuit at 8192
    shots on Tuna-17), so packing saves queue waits, not device time.
  * keeps up to --max-inflight submissions in the queue at once, fed by
    --replicas independent SPSA runs (default 4) of the chain protocol
    (all angles start at --x0, warm-start through beta, as in experiment.py and
    the paper). SPSA is sequential, so the parallelism comes from replicas plus
    tomography jobs that run alongside the next beta's optimization.

Every job's counts and id go to raw_jobs.jsonl; state.json is rewritten after
every finished job; --resume continues (jobs in flight at the time of the stop
are resubmitted; their old ids are printed so they can be checked/cancelled).

Fidelities are reported both as qiskit's state_fidelity, F = (Tr sqrt(..))^2,
and as the paper's convention, sqrt(F).

Usage (from the repo root):
  python gibbs_tfd_reproduction/run_hardware_parallel.py --self-test         # ideal sim, checks the analysis
  python gibbs_tfd_reproduction/run_hardware_parallel.py --dry-run           # local noisy sim, tiny budget
  python gibbs_tfd_reproduction/run_hardware_parallel.py --capability-test   # 5 jobs on Tuna-17
  python gibbs_tfd_reproduction/run_hardware_parallel.py                     # the real run
  python gibbs_tfd_reproduction/run_hardware_parallel.py --resume results/parallel_<tag>
"""

import argparse
import json
import sys
import time
import warnings
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.providers import JobStatus
from qiskit.quantum_info import DensityMatrix, Pauli, partial_trace, state_fidelity
from scipy.optimize import minimize

from ground_truth import GibbsState, default_subsystem_hamiltonian
from tfd_circuit import TFDCircuit

warnings.filterwarnings("ignore", message="Providing `coupling_map`")

VARSIGMA = 1.57
N = 4                       # logical qubits: q0=A2, q1=A1, q2=B2, q3=B1
SYSTEMS = {"A": (0, 1), "B": (2, 3)}
PAULI = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]),
         "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1, -1])}
FINAL = (JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED)


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# =========================================================== ground truth

def gibbs_target(beta):
    return GibbsState(default_subsystem_hamiltonian()).gibbs_density_matrix(beta)


def fidelities(rho, beta):
    """(qiskit F = (Tr sqrt)^2, paper F = Tr sqrt) against the Gibbs state."""
    f = float(state_fidelity(DensityMatrix(rho), DensityMatrix(gibbs_target(beta)), validate=False))
    return f, float(np.sqrt(max(f, 0.0)))


def exact_rho(params, system="A"):
    traced = [2, 3] if system == "A" else [0, 1]
    return partial_trace(TFDCircuit(*params).get_state_vector(), traced).data


def exact_cost(params, beta):
    sv = TFDCircuit(*params).get_state_vector()
    e = lambda p: sv.expectation_value(Pauli(p)).real
    return (e("IIXI") + e("IIIX") + e("XIII") + e("IXII")
            + VARSIGMA * (e("IIZZ") + e("ZZII"))
            - beta ** (-VARSIGMA) * (e("XIXI") + e("IXIX") + e("ZIZI") + e("IZIZ")))


def simulator_optimum(beta, n_starts=20, seed=0):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_starts):
        r = minimize(exact_cost, rng.uniform(-np.pi, np.pi, 4), args=(beta,),
                     method="COBYLA", options={"maxiter": 1000, "rhobeg": 0.5})
        if best is None or r.fun < best.fun:
            best = r
    return best.x


# =========================================================== circuits
# Every circuit measures all 4 logical qubits into clbits 0..3 (clbit q <- qubit q),
# so a counts key "b3b2b1b0" read as an int has bit q = qubit q.

def measured_circuit(params, bases, flips=(0, 0, 0, 0), name="c"):
    """Ansatz, then rotate qubit q into bases[q] (X/Y/Z); flips[q]=1 adds an X
    before measuring, i.e. measures -P (the analysis flips the sign back)."""
    qc = QuantumCircuit(N, N, name=name)
    TFDCircuit(*params).build_circuit(qc)
    qc.barrier()
    for q, (b, f) in enumerate(zip(bases, flips)):
        if b == "X":
            qc.h(q)
        elif b == "Y":
            qc.sdg(q)
            qc.h(q)
        if f:
            qc.x(q)
    qc.barrier()
    qc.measure(range(N), range(N))
    return qc


def calibration_circuits(mode):
    states = range(2 ** N) if mode == "full" else (0, 2 ** N - 1)
    out = []
    for s in states:
        qc = QuantumCircuit(N, N, name=f"cal{s}")
        for q in range(N):
            if (s >> q) & 1:
                qc.x(q)
        qc.barrier()
        qc.measure(range(N), range(N))
        out.append(qc)
    return out


def cost_circuits(params):
    return [measured_circuit(params, "XXXX", name="costX"),
            measured_circuit(params, "ZZZZ", name="costZ")]


def tomography_settings(mode):
    """(bases, flips) per circuit. The same Pauli pair is applied to A=(q0,q1)
    and B=(q2,q3) at once, so one set of circuits reconstructs both systems."""
    signs = [(0, 0), (0, 1), (1, 0), (1, 1)] if mode == "36" else [(0, 0)]
    out = []
    for (p0, p1), (s0, s1) in product(product("XYZ", repeat=2), signs):
        out.append(((p0, p1, p0, p1), (s0, s1, s0, s1)))
    return out


def tomography_circuits(params, mode):
    return [measured_circuit(params, "".join(b), f, name="tomo")
            for b, f in tomography_settings(mode)]


# =========================================================== analysis

def counts_to_probs(counts):
    p = np.zeros(2 ** N)
    for key, n in counts.items():
        k = int(key, 16) if str(key).startswith("0x") else int(str(key).replace(" ", ""), 2)
        p[k] += n
    return p / p.sum()


def readout_matrix(cal_probs, mode):
    """M[m, s] = P(measure m | prepared s)."""
    if mode == "full":
        return np.array(cal_probs).T
    p0, p1 = cal_probs  # all-0 and all-1 preparations
    M = np.array([[1.0]])
    for q in range(N):  # kron order: highest qubit first -> index bit q = qubit q
        r00 = sum(p0[m] for m in range(2 ** N) if not (m >> q) & 1)
        r11 = sum(p1[m] for m in range(2 ** N) if (m >> q) & 1)
        Mq = np.array([[r00, 1 - r11], [1 - r00, r11]])
        M = np.kron(Mq, M)
    return M


def correct(p, M):
    return np.linalg.solve(M, p)


def z_expect(p, qubits):
    signs = np.array([(-1) ** sum((m >> q) & 1 for q in qubits) for m in range(2 ** N)])
    return float(np.dot(signs, p))


def cost_from_probs(px, pz, beta):
    E = z_expect
    return (E(px, [0]) + E(px, [1]) + E(px, [2]) + E(px, [3])
            + VARSIGMA * (E(pz, [0, 1]) + E(pz, [2, 3]))
            - beta ** (-VARSIGMA) * (E(px, [1, 3]) + E(px, [0, 2]) + E(pz, [1, 3]) + E(pz, [0, 2])))


def reconstruct(probs_list, settings, system):
    """Linear-inversion 2-qubit tomography of `system`, averaging every Pauli
    expectation over all settings (and sign flips) that measure it."""
    qa, qb = SYSTEMS[system]
    acc = {}
    for p, (bases, flips) in zip(probs_list, settings):
        for use_a, use_b in ((1, 0), (0, 1), (1, 1)):
            qs = [q for q, use in ((qa, use_a), (qb, use_b)) if use]
            sign = (-1) ** sum(flips[q] for q in qs)
            label = (bases[qa] if use_a else "I", bases[qb] if use_b else "I")
            acc.setdefault(label, []).append(sign * z_expect(p, qs))
    rho = np.kron(PAULI["I"], PAULI["I"]).astype(complex)
    for (la, lb), vals in acc.items():
        rho += np.mean(vals) * np.kron(PAULI[lb], PAULI[la])  # qubit qb is the high bit
    return rho / 4


# =========================================================== SPSA (ask / tell)

class SPSA:
    """Spall's SPSA with a capped step, driven from outside: ask() returns the
    points to evaluate, tell(costs) consumes their costs. The calibration of
    the gain `a` is one ask/tell round with all its pairs at x0."""

    def __init__(self, x0, n_iter, c=0.15, first_step=0.3, alpha=0.602, gamma=0.101,
                 n_calibrate=3, seed=0, state=None):
        self.n_iter, self.c, self.alpha, self.gamma = n_iter, c, alpha, gamma
        self.first_step, self.n_calibrate, self.seed = first_step, n_calibrate, seed
        self.A = 0.1 * n_iter
        self.s = state or {"k": -1, "x": list(map(float, x0)), "a": None,
                           "iterates": [], "costs": []}

    def _delta(self, k, i=0):
        return np.random.default_rng([self.seed, k + 1, i]).choice([-1.0, 1.0], size=N)

    @property
    def done(self):
        return self.s["k"] >= self.n_iter

    def ask(self):
        x = np.array(self.s["x"])
        if self.s["k"] < 0:  # gain calibration
            pts = []
            for i in range(self.n_calibrate):
                d = self._delta(-1, i)
                pts += [x + self.c * d, x - self.c * d]
            return pts
        k = self.s["k"]
        ck = self.c / (k + 1) ** self.gamma
        d = self._delta(k)
        return [x + ck * d, x - ck * d]

    def tell(self, costs):
        x = np.array(self.s["x"])
        if self.s["k"] < 0:
            g = [abs(costs[2 * i] - costs[2 * i + 1]) / (2 * self.c) for i in range(self.n_calibrate)]
            g_mag = float(np.mean(g)) or 1.0
            self.s["a"] = self.first_step * (self.A + 1) ** self.alpha / g_mag
            self.s["k"] = 0
            return
        k = self.s["k"]
        ak = self.s["a"] / (k + 1 + self.A) ** self.alpha
        ck = self.c / (k + 1) ** self.gamma
        step = ak * (costs[0] - costs[1]) / (2 * ck) * self._delta(k)
        x = x - np.clip(step, -self.first_step, self.first_step)
        self.s["x"] = x.tolist()
        self.s["iterates"].append(x.tolist())
        self.s["costs"].append([float(costs[0]), float(costs[1])])
        self.s["k"] = k + 1

    def result(self):
        tail = max(1, self.n_iter // 4)
        return np.mean(self.s["iterates"][-tail:], axis=0)


# =========================================================== experiment logic

class Experiment:
    """Replicas of the chain protocol, reference tomography and a shared,
    periodically refreshed readout calibration, expressed as tasks.

    A task is split into submissions of at most `max_circuits` circuits (the
    backend's limit); it is analysed when all its submissions are back. Its
    circuits are rebuilt from the task dict, so the whole run is
    JSON-serializable. Cost and tomography results are corrected with the
    latest calibration; a task finishing before the first calibration waits
    in `deferred` and is analysed as soon as the calibration arrives."""

    def __init__(self, cfg, state=None):
        self.cfg = cfg
        if state is not None:
            self.st = state
            return
        self.st = {"replicas": [], "tomo": {}, "tasks": {}, "pending": [], "deferred": [],
                   "next_tid": 0, "x_sim": {}, "readout": None, "calibrations": 0,
                   "submissions_done": 0}
        for beta in cfg["betas"]:
            self.st["x_sim"][str(beta)] = simulator_optimum(beta).tolist()
        x_start = cfg.get("x_start") or [[cfg["x0"]] * N] * cfg["replicas"]
        states = cfg.get("spsa_states") or [None] * cfg["replicas"]
        for r in range(cfg["replicas"]):
            self.st["replicas"].append({"id": r, "beta_idx": 0, "x_prev": list(x_start[r]),
                                        "spsa": states[r], "per_beta": {}})
        # calibration first, then the replicas' first steps, then the references
        self.add({"kind": "cal"}, front=True)
        for r in range(cfg["replicas"]):
            self.add(self._next_cost_task(r))
        for beta in cfg["betas"] if cfg.get("references", True) else []:
            self.add(self._tomo_task(f"reference_start/beta={beta}", beta, self.st["x_sim"][str(beta)]))

    # -- task bookkeeping
    def add(self, task, front=False):
        tid = str(self.st["next_tid"])
        self.st["next_tid"] += 1
        n = len(self.circuits(task))
        task.update({"n_chunks": -(-n // self.cfg["max_circuits"]), "counts": {}})
        self.st["tasks"][tid] = task
        subs = [{"tid": tid, "chunk": c, "attempt": 0} for c in range(task["n_chunks"])]
        self.st["pending"] = subs + self.st["pending"] if front else self.st["pending"] + subs

    def calibration_due(self):
        if any(t["kind"] == "cal" for t in self.st["tasks"].values()):
            return False
        ro = self.st["readout"]
        return ro is not None and time.time() - ro["time"] > self.cfg["cal_interval"] * 60

    def describe(self, tid):
        t = self.st["tasks"][tid]
        if t["kind"] == "cost":
            return f"cost replica {t['replica']} beta={t['beta']} ({len(t['points'])} points)"
        if t["kind"] == "tomo":
            return f"tomography {t['label']}"
        return "readout calibration"

    # -- task factories
    def _spsa(self, rep):
        # A continued run (--continue-from) keeps the source run's random-direction seed and
        # gain offset A, so steps k+1.. are exactly those of one longer run.
        seed_idx = self.cfg.get("seed_beta_idx", rep["beta_idx"])
        spsa = SPSA(rep["x_prev"], self.cfg["iters"], first_step=self.cfg["first_step"],
                    n_calibrate=self.cfg["calibrate"],
                    seed=1000 * (rep["id"] + self.cfg.get("replica_offset", 0)) + seed_idx,
                    state=rep["spsa"])
        if self.cfg.get("spsa_A") is not None:
            spsa.A = self.cfg["spsa_A"]
        return spsa

    def _next_cost_task(self, r):
        rep = self.st["replicas"][r]
        spsa = self._spsa(rep)
        rep["spsa"] = spsa.s
        return {"kind": "cost", "replica": r, "beta": self.cfg["betas"][rep["beta_idx"]],
                "points": [list(map(float, p)) for p in spsa.ask()]}

    @staticmethod
    def _tomo_task(label, beta, params):
        return {"kind": "tomo", "label": label, "beta": beta, "params": list(map(float, params))}

    # -- circuits
    def circuits(self, task, chunk=None):
        if task["kind"] == "cost":
            circs = [c for p in task["points"] for c in cost_circuits(p)]
        elif task["kind"] == "tomo":
            circs = tomography_circuits(task["params"], self.cfg["tomography"])
        else:
            circs = calibration_circuits(self.cfg["calibration"])
        if chunk is None:
            return circs
        m = self.cfg["max_circuits"]
        return circs[chunk * m:(chunk + 1) * m]

    def shots(self, task):
        return self.cfg["shots_tomo"] if task["kind"] == "tomo" else self.cfg["shots_cost"]

    # -- results
    def chunk_done(self, tid, chunk, counts):
        task = self.st["tasks"][tid]
        task["counts"][str(chunk)] = counts
        self.st["submissions_done"] += 1
        if len(task["counts"]) < task["n_chunks"]:
            return
        counts_list = [c for k in range(task["n_chunks"]) for c in task["counts"][str(k)]]
        if task["kind"] == "cal":
            M = readout_matrix([counts_to_probs(c) for c in counts_list], self.cfg["calibration"])
            self.st["readout"] = {"M": M.tolist(), "time": time.time(),
                                  "assignment_fidelity": [float(M[s, s]) for s in range(2 ** N)]}
            self.st["calibrations"] += 1
            del self.st["tasks"][tid]
            log(f"readout calibration #{self.st['calibrations']} done: P(correct) per basis state "
                f"min {min(np.diag(M)):.3f}, mean {np.mean(np.diag(M)):.3f}")
            deferred, self.st["deferred"] = self.st["deferred"], []
            for d in deferred:
                self._analyse(d)
            return
        if self.st["readout"] is None:
            self.st["deferred"].append(tid)
        else:
            self._analyse(tid)

    def _analyse(self, tid):
        task = self.st["tasks"].pop(tid)
        counts_list = [c for k in range(task["n_chunks"]) for c in task["counts"][str(k)]]
        M = np.array(self.st["readout"]["M"])
        probs = [correct(counts_to_probs(c), M) for c in counts_list]

        if task["kind"] == "tomo":
            settings = tomography_settings(self.cfg["tomography"])
            entry = {"beta": task["beta"], "params": task["params"],
                     "finished": datetime.now().isoformat(timespec="seconds"),
                     "calibration_age_min": round((time.time() - self.st["readout"]["time"]) / 60, 1)}
            for system in SYSTEMS:
                rho = reconstruct(probs, settings, system)
                f, f_paper = fidelities(rho, task["beta"])
                entry[system] = {"fid": f, "fid_paper": f_paper,
                                 "rho": [np.real(rho).tolist(), np.imag(rho).tolist()]}
            entry["fid_exact_angles"] = fidelities(exact_rho(task["params"]), task["beta"])[0]
            self.st["tomo"][task["label"]] = entry
            log(f"tomography {task['label']}: F(A)={entry['A']['fid']:.4f} F(B)={entry['B']['fid']:.4f} "
                f"(paper convention {entry['A']['fid_paper']:.4f}/{entry['B']['fid_paper']:.4f})")
            if (self._all_replicas_done() and self.cfg.get("references", True)
                    and not self.st.get("reference_end_queued")):
                # measured again at the end, to see how much the device drifted
                self.st["reference_end_queued"] = True
                for b in self.cfg["betas"]:
                    self.add(self._tomo_task(f"reference_end/beta={b}", b, self.st["x_sim"][str(b)]))
            return

        # cost task
        r, beta = task["replica"], task["beta"]
        rep = self.st["replicas"][r]
        costs = [cost_from_probs(probs[2 * i], probs[2 * i + 1], beta) for i in range(len(task["points"]))]
        spsa = self._spsa(rep)
        spsa.tell(costs)
        rep["spsa"] = spsa.s
        phase = ("gain calibration" if spsa.s["k"] == 0 and not spsa.s["iterates"]
                 else f"step {spsa.s['k']}/{self.cfg['iters']}")
        log(f"replica {r} beta={beta}: {phase}  costs {np.round(costs, 3).tolist()}")
        if spsa.done:
            x_final = spsa.result()
            rep["per_beta"][str(beta)] = {"x_start": rep["x_prev"], "x_final": x_final.tolist(),
                                          "spsa": spsa.s,
                                          "fid_exact_angles": fidelities(exact_rho(x_final), beta)[0]}
            self.add(self._tomo_task(f"replica{r}/beta={beta}", beta, x_final))
            rep["x_prev"] = x_final.tolist()
            rep["spsa"] = None
            rep["beta_idx"] += 1
            log(f"replica {r} finished beta={beta}; angles' exact fidelity "
                f"{rep['per_beta'][str(beta)]['fid_exact_angles']:.4f}")
        if rep["beta_idx"] < len(self.cfg["betas"]):
            self.add(self._next_cost_task(r))

    def _all_replicas_done(self):
        return all(rep["beta_idx"] >= len(self.cfg["betas"]) for rep in self.st["replicas"])

    def finished(self):
        n_tomo = len(self.cfg["betas"]) * (self.cfg["replicas"] + (2 if self.cfg.get("references", True) else 0))
        return self._all_replicas_done() and len(self.st["tomo"]) >= n_tomo


# =========================================================== scheduler

def job_id(job):
    for get in (lambda: job.batch_job_id, lambda: job.job_id()):
        try:
            v = get()
            if v not in (None, ""):
                return str(v)
        except Exception:
            pass
    return None


def run_scheduler(exp, backend, layout, run_dir, state, max_inflight, poll=15):
    inflight = []  # (submission dict, job, submitted_at)
    raw_path = run_dir / "raw_jobs.jsonl"

    def checkpoint():
        state["exp"] = exp.st
        state["inflight"] = [{"sub": s, "job_id": job_id(j)} for s, j, _ in inflight]
        tmp = run_dir / "state.json.tmp"
        tmp.write_text(json.dumps(state))
        tmp.replace(run_dir / "state.json")

    while not exp.finished():
        if exp.calibration_due():
            log("readout calibration is older than "
                f"{exp.cfg['cal_interval']} min -> refreshing")
            exp.add({"kind": "cal"}, front=True)

        # fill the free queue slots
        while len(inflight) < max_inflight and exp.st["pending"]:
            sub = exp.st["pending"][0]
            if sub.get("not_before", 0) > time.time():
                break
            task = exp.st["tasks"][sub["tid"]]
            try:
                # An experiment that compiles its own circuits (fixed native gates, twirling)
                # provides compile(); the default transpile could re-merge such circuits.
                circuits = (exp.compile(task, sub["chunk"], backend) if hasattr(exp, "compile")
                            else transpile(exp.circuits(task, sub["chunk"]), backend, initial_layout=layout))
                job = backend.run(circuits, shots=exp.shots(task))
            except Exception as exc:
                log(f"! submit failed ({exc}); retrying in 60 s")
                sub["not_before"] = time.time() + 60
                break
            exp.st["pending"].pop(0)
            inflight.append((sub, job, time.time()))
            log(f"submitted {job_id(job)}: {exp.describe(sub['tid'])} part {sub['chunk'] + 1}/"
                f"{task['n_chunks']}, {len(circuits)} circuits x {exp.shots(task)} shots   "
                f"[{len(inflight)}/{max_inflight} in queue]")
            checkpoint()

        # collect finished submissions
        still = []
        for sub, job, t0 in inflight:
            try:
                status = job.status()
            except Exception as exc:
                log(f"! status query failed for {job_id(job)}: {exc}")
                still.append((sub, job, t0))
                continue
            if status not in FINAL:
                still.append((sub, job, t0))
                continue
            try:
                result = job.result(timeout=None)
                if not result.success:
                    raise RuntimeError(getattr(result, "system_messages", "job failed"))
                counts = [result.get_counts(i) for i in range(len(result.results))]
            except Exception as exc:
                sub["attempt"] += 1
                if sub["attempt"] > 5:
                    raise RuntimeError(f"submission failed 6 times, giving up: {sub}") from exc
                log(f"! {job_id(job)} failed ({exc}); resubmitting (attempt {sub['attempt']})")
                sub["not_before"] = time.time() + 60 * sub["attempt"]
                exp.st["pending"].insert(0, sub)
                continue
            task = exp.st["tasks"][sub["tid"]]
            with open(raw_path, "a") as f:
                f.write(json.dumps({"timestamp": datetime.now().isoformat(timespec="seconds"),
                                    "job_id": job_id(job), "chunk": sub["chunk"],
                                    "task": {k: v for k, v in task.items() if k != "counts"},
                                    "seconds": round(time.time() - t0, 1),
                                    "counts": counts}) + "\n")
            exp.chunk_done(sub["tid"], sub["chunk"], counts)
        inflight = still
        checkpoint()
        if not inflight and not exp.st["pending"] and not exp.finished():
            raise RuntimeError("scheduler has nothing to run but the experiment is not finished")
        if inflight or exp.st["pending"]:
            time.sleep(poll if inflight else 1)
    checkpoint()


# =========================================================== reporting

def report(exp, run_dir):
    cfg, st = exp.cfg, exp.st
    lines = ["beta     reference(start)  reference(end)   replicas mean +- std (n)        "
             "replicas' angles exact",
             "         F_A     F_B       F_A     F_B      F_A                F_B"]
    means, stds, ref_s, ref_e = [], [], [], []
    for beta in cfg["betas"]:
        rs = st["tomo"].get(f"reference_start/beta={beta}", {})
        re = st["tomo"].get(f"reference_end/beta={beta}", {})
        fa = [st["tomo"][f"replica{r}/beta={beta}"]["A"]["fid"] for r in range(cfg["replicas"])
              if f"replica{r}/beta={beta}" in st["tomo"]]
        fb = [st["tomo"][f"replica{r}/beta={beta}"]["B"]["fid"] for r in range(cfg["replicas"])
              if f"replica{r}/beta={beta}" in st["tomo"]]
        fe = [rep["per_beta"][str(beta)]["fid_exact_angles"] for rep in st["replicas"]
              if str(beta) in rep["per_beta"]]
        g = lambda d, s: d.get(s, {}).get("fid", float("nan"))
        lines.append(f"{beta:<8g} {g(rs,'A'):.4f}  {g(rs,'B'):.4f}    {g(re,'A'):.4f}  {g(re,'B'):.4f}   "
                     f"{np.mean(fa):.4f} +- {np.std(fa):.4f} ({len(fa)})  {np.mean(fb):.4f} +- {np.std(fb):.4f}   "
                     f"{np.mean(fe):.4f}")
        means.append(np.mean(fa)); stds.append(np.std(fa)); ref_s.append(g(rs, "A")); ref_e.append(g(re, "A"))
    lines.append("(qiskit fidelity F = (Tr sqrt)^2; the paper reports sqrt(F))")
    text = "\n".join(lines)
    (run_dir / "summary.txt").write_text(text + "\n")
    print("\n" + text)

    x = np.arange(len(cfg["betas"]))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(x, means, yerr=stds, fmt="o-", capsize=4, label=f"SPSA replicas (mean +- std, n={cfg['replicas']})")
    ax.plot(x - 0.08, ref_s, "s", label="reference: simulator angles (start)")
    ax.plot(x + 0.08, ref_e, "D", label="reference: simulator angles (end)")
    ax.set_xticks(x, [f"{b:g}" for b in cfg["betas"]])
    ax.set_xlabel("beta")
    ax.set_ylabel("fidelity of system A (qiskit F)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "fidelity_vs_beta.png", dpi=130)
    plt.close(fig)


# =========================================================== modes

def make_backend(args):
    if args.dry_run or args.self_test:
        from qi_utilities.device_simulation.simulators import NoisySimulator
        return NoisySimulator("Starmon-7", ideal_simulation=args.self_test), [0, 2, 3, 5]
    from qiskit_quantuminspire.qi_provider import QIProvider
    return QIProvider().get_backend(args.backend), args.layout


def self_test(backend, layout):
    """Ideal simulator: the measured cost and both reduced states must match
    the exact ones within shot noise (checks bit order, bases, sign flips,
    readout inversion and reconstruction)."""
    rng = np.random.default_rng(7)
    ok = True
    for calibration, tomography in (("full", "36"), ("perqubit", "9")):
        params = rng.uniform(-np.pi, np.pi, 4)
        beta = 1.0
        circs = cost_circuits(params) + tomography_circuits(params, tomography) + calibration_circuits(calibration)
        res = backend.run(transpile(circs, backend, initial_layout=layout), shots=50000).result()
        counts = [res.get_counts(i) for i in range(len(circs))]
        n_cal = 16 if calibration == "full" else 2
        M = readout_matrix([counts_to_probs(c) for c in counts[-n_cal:]], calibration)
        probs = [correct(counts_to_probs(c), M) for c in counts[:-n_cal]]
        c_meas, c_exact = cost_from_probs(probs[0], probs[1], beta), exact_cost(params, beta)
        print(f"[{calibration}/{tomography}] cost measured {c_meas:+.4f}  exact {c_exact:+.4f}")
        ok &= abs(c_meas - c_exact) < 0.1
        for system in SYSTEMS:
            rho = reconstruct(probs[2:], tomography_settings(tomography), system)
            err = np.abs(rho - exact_rho(params, system)).max()
            print(f"[{calibration}/{tomography}] system {system}: max |rho - exact| = {err:.4f}")
            ok &= err < 0.03
    print("SELF-TEST", "PASSED" if ok else "FAILED")
    return ok


def capability_test(backend, layout, args):
    """Five jobs submitted at once with 2..56 circuits, to measure how
    execution time scales with circuits per job and whether 5 jobs are
    accepted and run back to back."""
    x = simulator_optimum(1.0)
    cal = calibration_circuits("full")
    # Tuna backends accept at most 5 circuits per submission and 5 queued
    # submissions per user (backend type: max_jobs_per_batch_job,
    # batchjobs_per_queue_limit) -- so: 1 / 2 / 4 / 5 / 5 circuits, all at once.
    plan = [("1 cost circuit", cost_circuits(x)[:1], args.shots_cost),
            ("2 cost circuits (one parameter point)", cost_circuits(x), args.shots_cost),
            ("4 cost circuits (one SPSA step)", cost_circuits(x) + cost_circuits(x + 0.1), args.shots_cost),
            ("5 calibration circuits", cal[:5], args.shots_cost),
            ("5 tomography circuits, tomography shots", tomography_circuits(x, "36")[:5], args.shots_tomo)]
    if args.also_time:  # an already-queued 2-circuit job takes that slot in the queue
        plan = [item for item in plan if not item[0].startswith("2 cost")]
    jobs = []
    for label, circs, shots in plan:
        job = backend.run(transpile(circs, backend, initial_layout=layout), shots=shots)
        jobs.append((label, len(circs), shots, job))
        log(f"submitted {job_id(job)}: {label}, {len(circs)} circuits x {shots} shots")
    for label, n, shots, job in jobs:
        while job.status() not in FINAL:
            time.sleep(10)
        log(f"done {job_id(job)}: {label}, success={job.result(timeout=None).success}")
    if args.dry_run:
        return
    from qi2_shared.utils import run_async
    from qi2_shared.client import config
    from qi2_shared.pagination import PageReader
    from compute_api_client import ApiClient, BatchJobsApi, PageBatchJob, BatchJob

    async def timing(bid):
        async with ApiClient(config()) as c:
            return await PageReader[PageBatchJob, BatchJob]().get_single(
                BatchJobsApi(c).read_batch_jobs_batch_jobs_get, id=int(bid))

    print("\ncircuits  shots   queued->started  started->finished   label")
    rows = []
    for bid in args.also_time or []:
        jobs.append((f"already queued job {bid}", 2, args.shots_cost, bid))
    for label, n, shots, job in jobs:
        b = run_async(timing(job if isinstance(job, str) else job_id(job)))
        while b.finished_at is None:  # an --also-time job may still be running
            time.sleep(15)
            b = run_async(timing(job if isinstance(job, str) else job_id(job)))
        rows.append((b.reserved_at, b.finished_at))
        print(f"{n:>8}  {shots:>6}  {(b.reserved_at - b.queued_at).total_seconds():>12.0f} s  "
              f"{(b.finished_at - b.reserved_at).total_seconds():>14.1f} s   {label}")
    rows.sort()
    gaps = [(s2 - f1).total_seconds() for (s1, f1), (s2, f2) in zip(rows, rows[1:])]
    print(f"gaps between consecutive executions: {[round(g) for g in gaps]} s")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", default="Tuna-17")
    p.add_argument("--layout", type=int, nargs=4, default=[2, 4, 5, 8])
    p.add_argument("--betas", type=float, nargs="+", default=[1e-8, 0.5, 1, 5])
    p.add_argument("--x0", type=float, default=0.1)
    p.add_argument("--replicas", type=int, default=4,
                   help="independent SPSA runs; one fewer than the queue limit leaves a slot "
                        "for tomography and calibration")
    p.add_argument("--max-inflight", type=int, default=5,
                   help="submissions kept in the queue (capped by the backend's limit)")
    p.add_argument("--iters", type=int, default=20, help="SPSA steps per beta per replica")
    p.add_argument("--calibrate", type=int, default=3, help="SPSA gain-calibration pairs (one job)")
    p.add_argument("--first-step", type=float, default=0.3)
    p.add_argument("--calibration", choices=["full", "perqubit"], default="full")
    p.add_argument("--tomography", choices=["36", "9"], default="36")
    p.add_argument("--shots-cost", type=int, default=8192)
    p.add_argument("--shots-tomo", type=int, default=4096,
                   help="per tomography circuit; 36 x 4096 costs the same as 9 x 16384")
    p.add_argument("--cal-interval", type=float, default=30,
                   help="minutes between readout-calibration refreshes")
    p.add_argument("--dry-run", action="store_true", help="local noisy Starmon-7, tiny budget")
    p.add_argument("--self-test", action="store_true", help="ideal simulator correctness check")
    p.add_argument("--capability-test", action="store_true", help="5 timing jobs (hardware unless --dry-run)")
    p.add_argument("--also-time", nargs="*", help="capability test: ids of already-queued "
                   "2-circuit batch jobs to include in the timing table")
    p.add_argument("--resume", type=Path)
    p.add_argument("--start-from", type=Path,
                   help="continue a finished run: replica r starts from that run's replica r angles at --start-beta")
    p.add_argument("--start-beta", default="1", help="key of the beta whose final angles --start-from uses")
    p.add_argument("--continue-from", type=Path,
                   help="continue SPSA of a finished run at one beta (--continue-beta) for --extra-iters "
                        "more steps, from each replica's saved SPSA state: same qubits, same schedule")
    p.add_argument("--continue-beta", default="5.0", help="key of the beta to continue (as in per_beta)")
    p.add_argument("--extra-iters", type=int, default=20)
    p.add_argument("--replica-offset", type=int, default=0,
                   help="number replicas from this offset for their SPSA seeds, so new replicas are "
                        "independent of an earlier run's replicas 0..n-1")
    p.add_argument("--spsa-A", type=float, default=None, dest="spsa_A",
                   help="SPSA gain offset A (default 0.1 x iters); set it to match another run's schedule")
    p.add_argument("--no-references", dest="references", action="store_false",
                   help="skip the best-angle reference tomographies at the start and end of the run")
    args = p.parse_args()

    backend, layout = make_backend(args)
    # Quantum Inspire limits: circuits per submission and submissions in the queue
    try:
        bt = backend.get_backend_type()
        max_circuits, queue_limit = bt.max_jobs_per_batch_job, bt.batchjobs_per_queue_limit
    except Exception:  # local simulator: mirror the Tuna limits
        max_circuits, queue_limit = 5, 5
    if args.self_test:
        sys.exit(0 if self_test(backend, layout) else 1)
    if args.capability_test:
        capability_test(backend, layout, args)
        return

    if args.resume:
        run_dir = args.resume
        state = json.loads((run_dir / "state.json").read_text())
        cfg = state["cfg"]
        exp = Experiment(cfg, state["exp"])
        if state.get("inflight"):
            ids = [x["job_id"] for x in state["inflight"]]
            log(f"resuming; {len(ids)} submission(s) were in the queue and will be resubmitted "
                f"(old ids, may still run: {ids})")
            exp.st["pending"] = [x["sub"] for x in state["inflight"]] + exp.st["pending"]
            state["inflight"] = []
        backend, layout = make_backend(argparse.Namespace(**{**vars(args), "dry_run": cfg["dry_run"]}))
        layout = cfg["layout"]
    else:
        if args.dry_run:
            args.iters, args.replicas, args.max_inflight = 3, 2, 3
            args.shots_cost, args.shots_tomo, args.calibrate = 2048, 2048, 1
            args.cal_interval = 0.5  # exercise the refresh path
        cfg = {k: v for k, v in vars(args).items() if k not in ("resume", "start_from", "continue_from")}
        cfg["layout"] = layout
        if args.continue_from:
            prev = json.loads((args.continue_from / "state.json").read_text())
            pc, reps, key = prev["cfg"], prev["exp"]["replicas"], args.continue_beta
            if list(pc["layout"]) != list(layout) and not args.dry_run:
                sys.exit(f"--continue-from run used layout {pc['layout']}; pass --layout {' '.join(map(str, pc['layout']))}")
            src = [r["per_beta"][key] for r in reps]
            cfg.update(betas=[float(key)], replicas=len(reps), iters=pc["iters"] + args.extra_iters,
                       first_step=pc["first_step"], calibrate=pc["calibrate"], shots_cost=pc["shots_cost"],
                       shots_tomo=pc["shots_tomo"], calibration=pc["calibration"], tomography=pc["tomography"],
                       spsa_A=0.1 * pc["iters"], seed_beta_idx=[float(b) for b in pc["betas"]].index(float(key)),
                       continue_from=str(args.continue_from), x_start=[s["x_start"] for s in src],
                       spsa_states=[s["spsa"] for s in src])
            log(f"continuing {args.continue_from} at beta={key}: {len(reps)} replicas from step "
                f"{src[0]['spsa']['k']} to {cfg['iters']}")
        if args.start_from:
            prev = json.loads((args.start_from / "state.json").read_text())
            reps = prev["exp"]["replicas"]
            if args.replicas > len(reps):
                sys.exit(f"--start-from run has only {len(reps)} replicas")
            if list(prev["cfg"]["layout"]) != list(layout):
                log(f"warning: --start-from run used layout {prev['cfg']['layout']}, this run uses {layout}")
            cfg["start_from"] = str(args.start_from)
            cfg["x_start"] = [reps[r]["per_beta"][args.start_beta]["x_final"] for r in range(args.replicas)]
            log(f"starting replica angles from {args.start_from} at beta={args.start_beta}")
        cfg["max_circuits"] = max_circuits
        cfg["max_inflight"] = min(args.max_inflight, queue_limit)
        run_dir = Path("results") / f"{'parallel-dryrun' if args.dry_run else 'parallel'}_{datetime.now():%Y%m%d_%H%M%S}"
        run_dir.mkdir(parents=True, exist_ok=True)
        state = {"cfg": cfg}
        log("computing simulator optima for the reference tomography ...")
        exp = Experiment(cfg)

    m = cfg["max_circuits"]
    n_cost = cfg["replicas"] * len(cfg["betas"]) * (cfg["iters"] + -(-2 * cfg["calibrate"] * 2 // m))
    n_tomo = len(cfg["betas"]) * (cfg["replicas"] + 2) * -(-len(tomography_settings(cfg["tomography"])) // m)
    log(f"run directory {run_dir}\n  backend {cfg['backend'] if not cfg['dry_run'] else 'local noisy sim'}, "
        f"layout {layout}, betas {cfg['betas']}, {cfg['replicas']} replicas x {cfg['iters']} SPSA steps\n"
        f"  at most {m} circuits per submission, {cfg['max_inflight']} submissions in the queue\n"
        f"  readout calibration {cfg['calibration']} every {cfg['cal_interval']} min, tomography "
        f"{cfg['tomography']} settings, shots {cfg['shots_cost']} (cost) / {cfg['shots_tomo']} (tomography)\n"
        f"  submissions: ~{n_cost} cost + {n_tomo} tomography + calibrations")
    run_scheduler(exp, backend, layout, run_dir, state, cfg["max_inflight"],
                  poll=1 if cfg["dry_run"] else 15)
    report(exp, run_dir)
    log(f"done; everything is in {run_dir}")


if __name__ == "__main__":
    main()
