"""Reference ceilings at the hardware betas, without and with noise.

For every (n, p) and beta in protocol.DEFAULT_BETAS, find the best angles with
exact expectation values (warm start + 4 random restarts, L-BFGS-B) for
  - the exact free energy (the most the circuit can do), and
  - the engineered loss at the re-fitted varsigma (what a perfect optimizer of
    our loss would reach),
then evaluate the state those angles produce on each noise-model variant.
The gap between these and an SPSA run separates optimizer error from noise.

    python -m tfd_study.run_ceilings      # needs results/noiseless_scan.json
Writes tfd_study/results/ceilings.json.
"""

import json
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from tfd_study import model
from tfd_study.protocol import DEFAULT_BETAS, Evaluator
from tfd_study.run_spsa_study import DEPOL_ONLY, best_varsigma

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent / "results" / "ceilings.json"
NOISE_VARIANTS = {"fit": {}, "x0.5": {"scale": 0.5}, "x1.5": {"scale": 1.5}, "depolarizing-only": DEPOL_ONLY}


def optimum(n, p, f, restarts=4, seed=0):
    rng = np.random.default_rng([seed, n, p])
    x, out = np.zeros(4 * p), {}
    for b in DEFAULT_BETAS:
        starts = [x] + [rng.uniform(-np.pi, np.pi, 4 * p) for _ in range(restarts)]
        x = min((minimize(f, x0, args=(b,), method="L-BFGS-B") for x0 in starts), key=lambda r: r.fun).x
        out[b] = x
    return out


def one(n, p, s):
    angles = {
        "exact F": optimum(n, p, lambda th, b: model.free_energy(model.exact_rho_A(n, p, th), b, n)),
        f"loss (s={s})": optimum(n, p, lambda th, b: model.exact_loss(th, n, p, b, s)),
    }
    rows = []
    for target, per_beta in angles.items():
        for b, th in per_beta.items():
            row = {"target": target, "beta": b, "theta": th.tolist(),
                   "fid_noiseless": model.exact_fidelity(n, p, th, b)}
            for label, noise in NOISE_VARIANTS.items():
                rho = Evaluator(n, p, "noisy", noise=noise).noisy_rho_A(th)
                row[f"fid_{label}"] = model.fidelity(rho, b, n)
            rows.append(row)
    return {"n": n, "p": p, "s": s, "rows": rows}


def main():
    sig = best_varsigma()
    results, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 2)) as pool:
        futs = [pool.submit(one, n, p, sig[(n, p)]) for n in (2, 3, 4) for p in (1, 2, 3)]
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            ex = [row for row in r["rows"] if row["target"] == "exact F"]
            print(f"[{time.time() - t0:5.0f}s] n={r['n']} p={r['p']}: noiseless "
                  + " ".join(f"{row['fid_noiseless']:.3f}" for row in ex) + " | noisy(fit) "
                  + " ".join(f"{row['fid_fit']:.3f}" for row in ex), flush=True)
    results.sort(key=lambda r: (r["n"], r["p"]))
    OUT.write_text(json.dumps({"betas": list(DEFAULT_BETAS), "noise_variants": NOISE_VARIANTS,
                               "results": results}, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
