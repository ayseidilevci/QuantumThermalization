"""SPSA optimizer with a scipy-like interface, for the GUI's optimization run.

scipy.optimize.minimize has no SPSA, so this adapts the implementation in
gibbs_tfd_reproduction/experiments/noisy_optimizer_benchmark.py (where SPSA
was compared against COBYLA and others under shot noise), adding the
per-iteration `callback(xk)` the GUI uses for its Stop button, and handing
each round of cost evaluations to the caller as one batch so that on real
hardware a round can share a single queued job.
"""

import numpy as np
from scipy.optimize import OptimizeResult


def spsa_minimize(fun_batch, x0, maxiter, callback=None, rng=None, c=0.15, alpha=0.602,
                  gamma=0.101, first_step=0.3, n_calibrate=4):
    """Simultaneous-perturbation stochastic approximation (Spall's standard
    gains). Each iteration estimates the gradient from just two evaluations,
    whatever the dimension, which makes it robust to noisy cost estimates.

    `fun_batch(points, current)` must return the costs of `points` (a list
    of parameter vectors). `current` is the iterate those points probe
    around — None while calibrating — so the caller can measure it in the
    same batch for live monitoring; SPSA itself never needs its cost.

    The step size `a` is calibrated so the first update moves about
    `first_step` rad: the raw cost spans ~1e13 at beta -> 0 and ~10 at
    beta = 5, so a fixed `a` can't work for both. Calibration costs
    2 * n_calibrate extra evaluations; the run is then exactly `maxiter`
    iterations (SPSA has no convergence test).
    """
    rng = rng if rng is not None else np.random.default_rng()
    x = np.array(x0, dtype=float)
    dim = len(x)

    deltas = [rng.choice([-1.0, 1.0], size=dim) for _ in range(n_calibrate)]
    probes = [p for d in deltas for p in (x + c * d, x - c * d)]
    costs = fun_batch(probes, None)
    grads = [abs(costs[2 * i] - costs[2 * i + 1]) / (2 * c) for i in range(n_calibrate)]
    g_mag = float(np.mean(grads)) or 1.0

    A = 0.1 * maxiter
    a = first_step * (A + 1) ** alpha / g_mag

    for k in range(maxiter):
        ak = a / (k + 1 + A) ** alpha
        ck = c / (k + 1) ** gamma
        delta = rng.choice([-1.0, 1.0], size=dim)
        cost_plus, cost_minus = fun_batch([x + ck * delta, x - ck * delta], x.copy())
        g = (cost_plus - cost_minus) / (2 * ck) * delta
        x = x - ak * g
        if callback is not None:
            callback(x.copy())

    return OptimizeResult(x=x, nit=maxiter, success=True)
