"""Study 2: SPSA warm-start chains with shots and with the fitted Tuna-17 noise.

Parts (each chain = protocol.run_chain over beta = 0.5, 1, 2, 5):
  main        n in {2,3,4} x p in {1,2,3} x {noiseless shots, noisy} x SPSA steps
              per beta in {20, 40, 80} x 4 seeds
  robustness  noisy, p in {1,2}, 40 steps: gate noise x0.5 and x1.5, and the
              depolarizing-only model (fits our Tuna-17 data equally well but
              extrapolates differently to deeper circuits)
  varsigma    noisy, p = 2, 40 steps: the re-fitted varsigma vs the paper's 1.57

varsigma per (n, p) is the best one from study 1 (results/noiseless_scan.json).
Every chain is appended to results/spsa_study.jsonl as soon as it finishes, and
finished chains are skipped on a re-run, so the study can be stopped and resumed.

    python -m tfd_study.run_spsa_study [main|robustness|varsigma ...]
"""

import faulthandler
import json
import os
import sys
import tempfile
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np

from tfd_study.protocol import run_chain

warnings.filterwarnings("ignore")
RESULTS = Path(__file__).resolve().parent / "results"
OUT = RESULTS / "spsa_study.jsonl"
SEEDS = range(4)
DEPOL_ONLY = {"p_cz": 0.046, "p_1q": 0.0023, "gamma_cz": 0.0}   # fit_noise.py, depolarizing-only optimum


def best_varsigma():
    """(n, p) -> varsigma with the lowest mean noiseless infidelity in study 1."""
    scan = json.loads((RESULTS / "noiseless_scan.json").read_text())
    best = {}
    for r in scan["results"]:
        if r["loss"] != "C":
            continue
        infid = float(np.mean([1 - row["fid"] for row in r["rows"]]))
        key = (r["n"], r["p"])
        if key not in best or infid < best[key][1]:
            best[key] = (r["s"], infid)
    return {k: v[0] for k, v in best.items()}


def jobs(parts, sig):
    out = []
    if "main" in parts:
        for n in (2, 3, 4):
            for p in (1, 2, 3):
                for kind in ("noiseless", "noisy"):
                    for iters in (20, 40, 80):
                        for seed in SEEDS:
                            out.append(dict(part="main", n=n, p=p, s=sig[(n, p)], kind=kind, iters=iters,
                                            seed=seed, noise=None, noise_label="fit"))
    if "robustness" in parts:
        for n in (2, 3, 4):
            for p in (1, 2):
                for label, noise in (("x0.5", {"scale": 0.5}), ("x1.5", {"scale": 1.5}),
                                     ("depolarizing-only", DEPOL_ONLY)):
                    for seed in SEEDS:
                        out.append(dict(part="robustness", n=n, p=p, s=sig[(n, p)], kind="noisy", iters=40,
                                        seed=seed, noise=noise, noise_label=label))
    if "varsigma" in parts:
        for n in (2, 3, 4):
            for s in sorted({sig[(n, 2)], 1.57}):
                for seed in SEEDS:
                    out.append(dict(part="varsigma", n=n, p=2, s=s, kind="noisy", iters=40,
                                    seed=seed, noise=None, noise_label="fit"))
    return out


def key(j):
    return (j["part"], j["n"], j["p"], j["s"], j["kind"], j["iters"], j["seed"], j["noise_label"])


def _worker_init():
    """Native crashes (Aer) would otherwise kill a worker without a trace."""
    crash_dir = Path(tempfile.gettempdir()) / "tfd_study_worker_crashes"
    crash_dir.mkdir(exist_ok=True)
    faulthandler.enable(file=open(crash_dir / f"worker_{os.getpid()}.log", "w"), all_threads=True)


def run_job(j):
    r = run_chain(j["n"], j["p"], j["s"], j["kind"], j["iters"], seed=j["seed"], noise=j["noise"])
    r.update(part=j["part"], noise_label=j["noise_label"])
    return j, r


def main():
    parts = sys.argv[1:] or ["main", "robustness", "varsigma"]
    sig = best_varsigma()
    print("varsigma per (n, p):", {f"{n},{p}": s for (n, p), s in sorted(sig.items())})
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            r = json.loads(line)
            done.add((r["part"], r["n"], r["p"], r["s"], r["kind"], r["iters"], r["seed"], r["noise_label"]))
    todo = [j for j in jobs(parts, sig) if key(j) not in done]
    # longest chains first so the pool finishes evenly
    todo.sort(key=lambda j: -(j["iters"] * 2 ** j["n"] * j["p"]))
    print(f"{len(todo)} chains to run ({len(done)} already in {OUT.name})", flush=True)
    t0 = time.time()
    workers = min(14, max(1, (os.cpu_count() or 2) - 2))
    for attempt in range(5):
        if not todo:
            break
        finished = set()
        try:
            # one fresh process per chain: a crashed or bloated worker cannot take others down
            with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1, initializer=_worker_init) as pool, \
                    OUT.open("a") as fh:
                futs = {pool.submit(run_job, j): key(j) for j in todo}
                for fut in as_completed(futs):
                    j, r = fut.result()
                    fh.write(json.dumps(r) + "\n")
                    fh.flush()
                    finished.add(key(j))
                    f = " ".join(f"{st['fid_device']:.3f}" for st in r["steps"])
                    print(f"[{time.time() - t0:6.0f}s {len(done) + len(finished)}/{len(done) + len(todo)}] "
                          f"{j['part']:<10} n={j['n']} p={j['p']} s={j['s']} {j['kind']:<9} {j['iters']:>2} it "
                          f"seed {j['seed']} {j['noise_label']:<17} F_device: {f}", flush=True)
        except BrokenProcessPool as exc:
            print(f"! worker pool broke ({exc}); {len(finished)} chains saved this round, restarting the rest", flush=True)
        done |= finished
        todo = [j for j in todo if key(j) not in done]
    print(f"finished: {len(done)} chains in {OUT.name}, {len(todo)} not run", flush=True)


if __name__ == "__main__":
    main()
