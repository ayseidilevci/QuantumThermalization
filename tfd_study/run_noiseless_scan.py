"""Study 1: which loss exponent varsigma, and how many layers, without noise.

For every system size n and depth p, minimise each loss over the 4p angles at
9 log-spaced beta in [0.01, 100] with exact (statevector) expectation values:
annealed in beta (warm start) plus 3 random restarts per beta, L-BFGS-B. This
is the procedure of the tfd_qi note, redone with this package's qiskit
circuits. Losses: the exact free energy (reference: the best the circuit can
be steered to), C0 (varsigma = 1, Premaratne & Matsuura) and the engineered
loss C_s over a grid of varsigma.

    python -m tfd_study.run_noiseless_scan             # ~45 min on 20 cores
    python -m tfd_study.run_noiseless_scan --redo-F 10 # re-optimise the exact free
                                                       # energy with 10 restarts
Writes tfd_study/results/noiseless_scan.json. With 2+ layers the landscape is
rugged; the free-energy reference is meant to be a floor, so it gets more
random restarts than the scanned losses.
"""

import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from tfd_study import model

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent / "results" / "noiseless_scan.json"
BETAS = np.logspace(-2, 2, 9)
SIZES = [(n, p) for n in (2, 3, 4) for p in (1, 2, 3)]
VARSIGMAS = [round(1.0 + 0.1 * k, 1) for k in range(11)]       # 1.0 ... 2.0 (1.0 is C0)
EXTRA_VARSIGMAS = [1.57]                                        # the published value


def objective(kind, n, p, s):
    if kind == "F":
        return lambda th, b: model.free_energy(model.exact_rho_A(n, p, th), b, n)
    return lambda th, b: model.exact_loss(th, n, p, b, s)


def scan_one(n, p, kind, s, restarts=3, seed=0):
    rng = np.random.default_rng([seed, n, p])
    f = objective(kind, n, p, s)
    x = np.zeros(4 * p)
    rows = []
    for b in BETAS:
        starts = [x] + [rng.uniform(-np.pi, np.pi, 4 * p) for _ in range(restarts)]
        best = min((minimize(f, x0, args=(b,), method="L-BFGS-B") for x0 in starts), key=lambda r: r.fun)
        x = best.x
        rows.append({"beta": float(b), "theta": x.tolist(), "fid": model.exact_fidelity(n, p, x, b)})
    return {"n": n, "p": p, "loss": kind, "s": s, "rows": rows}


def run(jobs):
    results, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 2)) as pool:
        futs = {pool.submit(scan_one, *j): j for j in jobs}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            f = np.array([row["fid"] for row in r["rows"]])
            print(f"[{time.time() - t0:5.0f}s] n={r['n']} p={r['p']} {r['loss']}{'' if r['s'] is None else r['s']}: "
                  f"mean infidelity {np.mean(1 - f):.4f}, worst F {f.min():.3f}", flush=True)
    return results


def save(results):
    results.sort(key=lambda r: (r["n"], r["p"], r["loss"], r["s"] or 0))
    OUT.write_text(json.dumps({"betas": BETAS.tolist(), "fidelity": "qiskit (squared)",
                               "results": results}, indent=1))
    print(f"wrote {OUT}")


def main():
    OUT.parent.mkdir(exist_ok=True)
    if "--redo-F" in sys.argv:
        restarts = int(sys.argv[sys.argv.index("--redo-F") + 1])
        old = json.loads(OUT.read_text())["results"]
        new = run([(n, p, "F", None, restarts) for n, p in SIZES])
        # per beta keep the angles with the lower free energy (selected by the
        # objective, not by fidelity, so the reference is not biased upwards)
        merged = []
        for r in new:
            prev = next(o for o in old if o["loss"] == "F" and (o["n"], o["p"]) == (r["n"], r["p"]))
            n, p = r["n"], r["p"]
            fe = lambda row: model.free_energy(model.exact_rho_A(n, p, row["theta"]), row["beta"], n)
            merged.append({**r, "rows": [a if fe(a) <= fe(b) else b for a, b in zip(r["rows"], prev["rows"])]})
        save([o for o in old if o["loss"] != "F"] + merged)
        return
    jobs = [(n, p, "F", None) for n, p in SIZES]
    jobs += [(n, p, "C", s) for n, p in SIZES for s in VARSIGMAS + EXTRA_VARSIGMAS]
    save(run(jobs))


if __name__ == "__main__":
    main()
