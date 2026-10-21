"""n+n-qubit, p-layer TFD run on Tuna-17, on top of the parallel hardware runner.

The scheduler, the Quantum Inspire limits (5 circuits per submission, 5
submissions queued), checkpointing, retries and raw-count logging are those of
gibbs_tfd_reproduction/run_hardware_parallel.py, imported unchanged. This file
only supplies the experiment for general n and p:

  * loss: the paper's engineered cost C_s(beta) (tfd_study/model.py); SPSA's
    gain calibration makes its steps independent of the cost's scale.
    2 circuits per parameter point (all-Z,
    all-X), so an SPSA step is 4 circuits = 1 submission
  * replicas of the chain protocol from theta = 0 (Bell pairs). theta = 0 is a
    stationary point, so the first beta starts from the best of n_kicks random
    small kicks (one task)
  * readout calibration per qubit (all-0 / all-1, 2 circuits; a full 2^(2n)
    calibration is impractical beyond 2+2), refreshed every --cal-interval min
  * tomography: 3^n settings, A_i and B_i in the same basis, so one set of
    circuits gives rho_A and rho_B
  * reference tomographies of the noiseless-optimal angles at the start (they
    double as a characterization of the exact qubits used: known circuits with
    known ideal states) and, optionally, again at the end to measure drift

Usage (repo root):
  python -m tfd_study.run_hardware --config tfd_study/hardware_plan.json --self-test
  python -m tfd_study.run_hardware --config tfd_study/hardware_plan.json --dry-run
  python -m tfd_study.run_hardware --config tfd_study/hardware_plan.json --simulate coherent
  python -m tfd_study.run_hardware --config tfd_study/hardware_plan.json
  python -m tfd_study.run_hardware --resume results/tfd_<tag>
"""

import argparse
import json
import math
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gibbs_tfd_reproduction"))
import run_hardware_parallel as rhp  # noqa: E402  (scheduler and SPSA, unchanged)

from tfd_study import devices, model  # noqa: E402

warnings.filterwarnings("ignore", message="Providing `coupling_map`")
log = rhp.log
SUBMISSIONS_PER_HOUR = 33   # measured: 603 submissions in 18.3 h on Tuna-17 (parallel_20261019_175824)


class SPSA(rhp.SPSA):
    """The runner's SPSA; perturbation directions sized to the number of angles (4p)."""

    def _delta(self, k, i=0):
        return np.random.default_rng([self.seed, k + 1, i]).choice([-1.0, 1.0], size=len(self.s["x"]))


def reference_angles(n, p, s, betas, restarts=4, seed=0):
    """Noiseless optimum of the loss at each beta (warm start + random restarts)."""
    from scipy.optimize import minimize
    rng = np.random.default_rng([seed, n, p])
    x, out = np.zeros(4 * p), {}
    for b in betas:
        starts = [x] + [rng.uniform(-np.pi, np.pi, 4 * p) for _ in range(restarts)]
        x = min((minimize(model.exact_loss, x0, args=(n, p, b, s), method="L-BFGS-B") for x0 in starts),
                key=lambda r: r.fun).x
        out[str(b)] = x.tolist()
    return out


class Experiment:
    """Same task interface as run_hardware_parallel.Experiment, for any n and p."""

    def __init__(self, cfg, state=None):
        self.cfg = cfg
        self.n, self.p, self.s = cfg["n"], cfg["p"], cfg["varsigma"]
        if state is not None:
            self.st = state
            return
        self.st = {"replicas": [], "tomo": {}, "tasks": {}, "pending": [], "deferred": [],
                   "next_tid": 0, "x_ref": cfg["reference_angles"], "readout": None,
                   "calibrations": 0, "submissions_done": 0}
        for r in range(cfg["replicas"]):
            self.st["replicas"].append({"id": r, "beta_idx": 0, "x_prev": [0.0] * 4 * self.p,
                                        "kicked": False, "spsa": None, "per_beta": {}})
        self.add({"kind": "cal"}, front=True)
        for r in range(cfg["replicas"]):
            self.add(self._next_task(r))
        for b in cfg["betas"]:
            self.add(self._tomo_task(f"reference_start/beta={b}", b, self.st["x_ref"][str(b)]))

    # -- bookkeeping (as in run_hardware_parallel)
    def add(self, task, front=False):
        tid = str(self.st["next_tid"])
        self.st["next_tid"] += 1
        task.update({"n_chunks": -(-len(self.circuits(task)) // self.cfg["max_circuits"]), "counts": {}})
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
        if t["kind"] in ("cost", "kick"):
            return f"{t['kind']} replica {t['replica']} beta={t['beta']} ({len(t['points'])} points)"
        if t["kind"] == "tomo":
            return f"tomography {t['label']}"
        return "readout calibration"

    # -- task factories
    def _spsa(self, rep):
        c = self.cfg
        return SPSA(rep["x_prev"], c["iters"], first_step=c["first_step"], n_calibrate=c["calibrate"],
                    seed=1000 * rep["id"] + rep["beta_idx"], state=rep["spsa"])

    def _next_task(self, r):
        rep = self.st["replicas"][r]
        beta = self.cfg["betas"][rep["beta_idx"]]
        if not rep["kicked"] and self.cfg["n_kicks"] and np.allclose(rep["x_prev"], 0):
            rng = np.random.default_rng([self.cfg["seed"], r, 17])
            pts = [rng.normal(0, self.cfg["kick"], 4 * self.p).tolist() for _ in range(self.cfg["n_kicks"])]
            return {"kind": "kick", "replica": r, "beta": beta, "points": pts}
        spsa = self._spsa(rep)
        rep["spsa"] = spsa.s
        return {"kind": "cost", "replica": r, "beta": beta, "points": [list(map(float, x)) for x in spsa.ask()]}

    @staticmethod
    def _tomo_task(label, beta, params):
        return {"kind": "tomo", "label": label, "beta": beta, "params": list(map(float, params))}

    # -- circuits (logical qubits 0..2n-1 = A1..An, B1..Bn; clbit k <- qubit k)
    def circuits(self, task, chunk=None):
        n, p = self.n, self.p
        if task["kind"] in ("cost", "kick"):
            circs = []
            for i, x in enumerate(task["points"]):
                for b in ("Z", "X"):
                    c = model.measured(n, p, x, b)
                    c.name = f"{task['kind']}|r{task['replica']}|b{task['beta']}|pt{i}|{b}"
                    circs.append(c)
        elif task["kind"] == "tomo":
            circs = []
            for s in model.tomography_settings(n):
                c = model.measured(n, p, task["params"], s)
                c.name = f"tomo|{task['label']}|{s}"      # names survive compilation
                circs.append(c)
        else:
            circs = model.calibration_circuits(n)
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
        if task["kind"] == "cal":
            c0, c1 = (model.counts_to_probs(c, 2 * self.n) for c in self._counts(task))
            mats = model.readout_matrices(c0, c1, 2 * self.n)
            self.st["readout"] = {"mats": [m.tolist() for m in mats], "time": time.time(),
                                  "assignment": [[float(m[0, 0]), float(m[1, 1])] for m in mats]}
            self.st["calibrations"] += 1
            del self.st["tasks"][tid]
            worst = min(min(a) for a in self.st["readout"]["assignment"])
            log(f"readout calibration #{self.st['calibrations']} done: worst per-qubit P(correct) {worst:.3f}")
            deferred, self.st["deferred"] = self.st["deferred"], []
            for d in deferred:
                self._analyse(d)
            return
        if self.st["readout"] is None:
            self.st["deferred"].append(tid)
        else:
            self._analyse(tid)

    def _counts(self, task):
        return [c for k in range(task["n_chunks"]) for c in task["counts"][str(k)]]

    def _probs(self, task):
        mats = [np.array(m) for m in self.st["readout"]["mats"]]
        return [model.correct_readout(model.counts_to_probs(c, 2 * self.n), mats) for c in self._counts(task)]

    def _losses(self, probs, beta):
        return [model.loss(model.loss_terms(probs[2 * i], probs[2 * i + 1], self.n), beta, self.s)
                for i in range(len(probs) // 2)]

    def _analyse(self, tid):
        task = self.st["tasks"].pop(tid)
        probs = self._probs(task)
        n, p = self.n, self.p

        if task["kind"] == "tomo":
            by_setting = dict(zip(model.tomography_settings(n), probs))
            entry = {"beta": task["beta"], "params": task["params"],
                     "finished": datetime.now().isoformat(timespec="seconds"),
                     "calibration_age_min": round((time.time() - self.st["readout"]["time"]) / 60, 1),
                     "fid_exact_angles": model.exact_fidelity(n, p, task["params"], task["beta"])}
            for system in ("A", "B"):
                rho = model.tomography_rho(by_setting, n, system)
                f = model.fidelity(rho, task["beta"], n)
                entry[system] = {"fid": f, "fid_paper": float(np.sqrt(max(f, 0))),
                                 "purity": float(np.real(np.trace(rho @ rho))),
                                 "rho": [np.real(rho).tolist(), np.imag(rho).tolist()]}
            self.st["tomo"][task["label"]] = entry
            log(f"tomography {task['label']}: F(A)={entry['A']['fid']:.4f} F(B)={entry['B']['fid']:.4f} "
                f"(angles without noise {entry['fid_exact_angles']:.4f})")
            if self._all_replicas_done() and self.cfg["reference_end"] and not self.st.get("reference_end_queued"):
                self.st["reference_end_queued"] = True
                for b in self.cfg["betas"]:
                    self.add(self._tomo_task(f"reference_end/beta={b}", b, self.st["x_ref"][str(b)]))
            return

        r, beta = task["replica"], task["beta"]
        rep = self.st["replicas"][r]
        losses = self._losses(probs, beta)
        if task["kind"] == "kick":
            best = int(np.argmin(losses))
            rep["x_prev"], rep["kicked"] = task["points"][best], True
            rep["kick"] = {"points": task["points"], "losses": losses, "chosen": best}
            log(f"replica {r} beta={beta}: kick start, best of {len(losses)} (loss {losses[best]:.4f})")
            self.add(self._next_task(r))
            return

        spsa = self._spsa(rep)
        spsa.tell(losses)
        rep["spsa"] = spsa.s
        phase = ("gain calibration" if spsa.s["k"] == 0 and not spsa.s["iterates"]
                 else f"step {spsa.s['k']}/{self.cfg['iters']}")
        log(f"replica {r} beta={beta}: {phase}  losses {np.round(losses, 4).tolist()}")
        if spsa.done:
            x = spsa.result()
            rep["per_beta"][str(beta)] = {"x_start": rep["x_prev"], "x_final": x.tolist(), "spsa": spsa.s,
                                          "fid_exact_angles": model.exact_fidelity(n, p, x, beta)}
            self.add(self._tomo_task(f"replica{r}/beta={beta}", beta, x))
            rep["x_prev"], rep["spsa"] = x.tolist(), None
            rep["beta_idx"] += 1
            log(f"replica {r} finished beta={beta}; angles without noise "
                f"{rep['per_beta'][str(beta)]['fid_exact_angles']:.4f}")
        if rep["beta_idx"] < len(self.cfg["betas"]):
            self.add(self._next_task(r))

    def _all_replicas_done(self):
        return all(rep["beta_idx"] >= len(self.cfg["betas"]) for rep in self.st["replicas"])

    def finished(self):
        refs = 2 if self.cfg["reference_end"] else 1
        return self._all_replicas_done() and len(self.st["tomo"]) >= len(self.cfg["betas"]) * (self.cfg["replicas"] + refs)


def record_compiled_circuits(run_dir):
    """Save every circuit exactly as the scheduler compiles it for submission
    (run_dir/compiled.jsonl: name + OpenQASM 3), so the analysis can simulate the
    identical circuits. Wraps the scheduler's transpile call; run_hardware_parallel.py
    itself is not modified."""
    from qiskit import qasm3
    original = rhp.transpile
    path = run_dir / "compiled.jsonl"

    def transpile_and_record(circuits, *args, **kwargs):
        out = original(circuits, *args, **kwargs)
        with path.open("a") as fh:
            for c in (out if isinstance(out, list) else [out]):
                fh.write(json.dumps({"name": c.name, "qasm": qasm3.dumps(c)}) + "\n")
        return out

    rhp.transpile = transpile_and_record


# =========================================================== budget, backends, report

def budget(cfg):
    m = cfg["max_circuits"]
    cal_pairs = -(-4 * cfg["calibrate"] // m)                   # gain calibration: 2*calibrate points x 2 circuits
    cost = cfg["replicas"] * len(cfg["betas"]) * (cfg["iters"] + cal_pairs)
    kicks = cfg["replicas"] * -(-2 * cfg["n_kicks"] // m) if cfg["n_kicks"] else 0
    tomo_each = -(-3 ** cfg["n"] // m)
    tomos = len(cfg["betas"]) * (cfg["replicas"] + (2 if cfg["reference_end"] else 1))
    total = cost + kicks + tomo_each * tomos
    hours = total / SUBMISSIONS_PER_HOUR
    cals = math.ceil(hours * 60 / cfg["cal_interval"]) + 1
    return {"cost": cost, "kicks": kicks, "tomography": tomo_each * tomos, "calibrations": cals,
            "total": total + cals, "hours": round((total + cals) / SUBMISSIONS_PER_HOUR, 1)}


def make_backend(cfg, mode):
    """mode: 'hardware', 'dry-run' (fitted Tuna-17 noise model), 'self-test' (ideal), or
    'simulate' (the full plan on the simulator cfg['sim']: 'ideal', 'uniform' = the dry-run
    model, 'coherent' = devices.coherent_noise_model on the plan's couplers)."""
    if mode == "hardware":
        return devices.real_backend(cfg["backend"])
    from qiskit_aer import AerSimulator
    noise = devices.noise_model() if mode == "dry-run" else None
    if mode == "simulate" and cfg["sim"] == "uniform":
        noise = devices.noise_model()
    elif mode == "simulate" and cfg["sim"] == "coherent":
        lay = cfg["layout"]
        noise = devices.coherent_noise_model([(lay[a], lay[b]) for a, b in devices.ladder_edges(cfg["n"])],
                                             seed=cfg["seed"])
    return AerSimulator(method="density_matrix" if noise else "automatic", noise_model=noise,
                        coupling_map=devices.coupling_map(), basis_gates=devices.NATIVE_GATES)


def self_test(cfg):
    """Ideal simulator: measured loss and both reconstructed states must match the exact ones."""
    from qiskit import transpile
    backend = make_backend(cfg, "self-test")
    exp = Experiment(cfg, state={})          # circuits only, no tasks
    n, p = cfg["n"], cfg["p"]
    th = np.random.default_rng(7).uniform(-1, 1, 4 * p)
    tasks = [{"kind": "cost", "replica": 0, "beta": 1.0, "points": [th.tolist()]},
             {"kind": "tomo", "label": "self-test", "params": th.tolist()}, {"kind": "cal"}]
    circs = [c for t in tasks for c in exp.circuits(t)]
    res = backend.run(transpile(circs, backend, initial_layout=cfg["layout"]), shots=100000).result()
    probs = [model.counts_to_probs(res.get_counts(i), 2 * n) for i in range(len(circs))]
    mats = model.readout_matrices(probs[-2], probs[-1], 2 * n)
    probs = [model.correct_readout(q, mats) for q in probs[:-2]]
    beta = 1.0
    meas = model.loss(model.loss_terms(probs[0], probs[1], n), beta, cfg["varsigma"])
    exact = model.exact_loss(th, n, p, beta, cfg["varsigma"])
    # Shot-noise bound: every Pauli term has std <= 1/sqrt(shots); sum of |coefficients| of C_s
    s = cfg["varsigma"]
    ok = abs(meas - exact) < (2 * n + 2 * (n - 1) * s + 2 * n * beta ** -s) / np.sqrt(100000)
    print(f"loss measured {meas:+.4f}  exact {exact:+.4f}")
    by = dict(zip(model.tomography_settings(n), probs[2:]))
    for system, ex in (("A", model.exact_rho_A(n, p, th)), ("B", model.exact_rho_B(n, p, th))):
        err = np.abs(model.tomography_rho(by, n, system) - ex).max()
        print(f"system {system}: max |rho - exact| = {err:.4f}")
        ok &= err < 0.03
    print("SELF-TEST", "PASSED" if ok else "FAILED")
    return ok


def report(exp, run_dir):
    st, cfg = exp.st, exp.cfg
    lines = [f"n={cfg['n']} p={cfg['p']} varsigma={cfg['varsigma']} layout={cfg['layout']}",
             "beta    reference F_A/F_B    replicas F_A mean +- std (k)   F_B          angles without noise"]
    for b in cfg["betas"]:
        ref = st["tomo"].get(f"reference_start/beta={b}")
        reps = [st["tomo"][k] for k in st["tomo"] if k.startswith("replica") and k.endswith(f"beta={b}")]
        fa = [t["A"]["fid"] for t in reps]
        fb = [t["B"]["fid"] for t in reps]
        ang = [t["fid_exact_angles"] for t in reps]
        lines.append(f"{b:<7g} " + (f"{ref['A']['fid']:.3f}/{ref['B']['fid']:.3f}" if ref else "   -   ") + "        "
                     + (f"{np.mean(fa):.3f} +- {np.std(fa):.3f} ({len(fa)})       {np.mean(fb):.3f}        "
                        f"{np.mean(ang):.3f}" if fa else "-"))
    text = "\n".join(lines)
    (run_dir / "analysis.txt").write_text(text + "\n")
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, help="hardware plan (JSON), e.g. tfd_study/hardware_plan.json")
    ap.add_argument("--dry-run", action="store_true", help="run locally on the fitted Tuna-17 noise model")
    ap.add_argument("--self-test", action="store_true", help="ideal-simulator correctness check")
    ap.add_argument("--simulate", choices=["ideal", "uniform", "coherent"],
                    help="run the full plan (no dry-run shortcuts) on a local simulator")
    ap.add_argument("--resume", type=Path)
    args = ap.parse_args()

    if args.resume:
        run_dir = args.resume
        state = json.loads((run_dir / "state.json").read_text())
        cfg = state["cfg"]
        exp = Experiment(cfg, state["exp"])
        if state.get("inflight"):
            ids = [x["job_id"] for x in state["inflight"]]
            log(f"resuming; {len(ids)} submission(s) were in the queue and are resubmitted (old ids: {ids})")
            exp.st["pending"] = [x["sub"] for x in state["inflight"]] + exp.st["pending"]
            state["inflight"] = []
        mode = "simulate" if cfg.get("sim") else "dry-run" if cfg["dry_run"] else "hardware"
    else:
        cfg = json.loads(args.config.read_text())["run"]
        cfg.setdefault("layout", devices.default_layout(cfg["n"]))
        cfg.setdefault("max_circuits", 5)
        cfg["dry_run"] = args.dry_run
        cfg["sim"] = args.simulate
        if args.self_test:
            sys.exit(0 if self_test(cfg) else 1)
        mode = "simulate" if args.simulate else "dry-run" if args.dry_run else "hardware"
        if mode == "hardware":
            bt = devices.real_backend(cfg["backend"]).get_backend_type()
            cfg["max_circuits"] = bt.max_jobs_per_batch_job
            cfg["max_inflight"] = min(cfg["max_inflight"], bt.batchjobs_per_queue_limit)
        if args.dry_run:
            cfg.update(iters=4, replicas=min(cfg["replicas"], 2), max_inflight=3, calibrate=1,
                       cal_interval=0.5, shots_cost=2048, shots_tomo=2048)
        log("computing the noiseless-optimal reference angles ...")
        cfg["reference_angles"] = reference_angles(cfg["n"], cfg["p"], cfg["varsigma"], cfg["betas"])
        tag = f"tfd-sim-{args.simulate}" if args.simulate else "tfd-dryrun" if args.dry_run else "tfd"
        if mode == "hardware" and cfg["backend"] != "Tuna-17":      # e.g. QX emulator -> tfd-qx-emulator
            tag += "-" + cfg["backend"].lower().replace(" ", "-")
        run_dir = ROOT / "results" / f"{tag}_n{cfg['n']}p{cfg['p']}_{datetime.now():%Y%m%d_%H%M%S}"
        run_dir.mkdir(parents=True, exist_ok=True)
        state = {"cfg": cfg}
        exp = Experiment(cfg)

    b = budget(cfg)
    log(f"run directory {run_dir}\n  {cfg['n']}+{cfg['n']} qubits, p={cfg['p']}, varsigma={cfg['varsigma']}, "
        f"layout {cfg['layout']}, betas {cfg['betas']}\n  {cfg['replicas']} replicas x {cfg['iters']} SPSA steps, "
        f"{cfg['n_kicks']} kicks, shots {cfg['shots_cost']}/{cfg['shots_tomo']}\n  submissions ~{b['total']} "
        f"({b['cost']} cost, {b['kicks']} kicks, {b['tomography']} tomography, ~{b['calibrations']} calibrations)"
        f" ~= {b['hours']} h at {SUBMISSIONS_PER_HOUR}/h")
    record_compiled_circuits(run_dir)
    rhp.run_scheduler(exp, make_backend(cfg, mode), cfg["layout"], run_dir, state, cfg["max_inflight"],
                      poll=15 if mode == "hardware" else 1)
    report(exp, run_dir)
    log(f"done; everything is in {run_dir}")


if __name__ == "__main__":
    main()
