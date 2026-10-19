"""SPSA for hardware-in-the-loop optimization.

Ported from experiments/noisy_optimizer_benchmark.py's spsa(), with two
changes for real hardware:

- `fun_pair(x_plus, x_minus)` evaluates both perturbed points together, so
  a caller can submit them as ONE batched job (half the queue waits).
- The state after every iteration is handed to `on_iter`, and a run can be
  restarted from such a state -- a multi-hour hardware run survives a crash.

Gains follow Spall's standard schedule. The step size `a` is calibrated so
the first update moves about `first_step` rad, which makes it independent
of the cost's scale (~1e13 at beta -> 0, ~10 at beta = 5).
"""

import numpy as np


def spsa(fun_pair, x0, n_iter, c=0.15, alpha=0.602, gamma=0.101,
         first_step=0.15, n_calibrate=3, seed=0, state=None, on_iter=None,
         max_step=None):
    """Minimize with SPSA. Returns (x_final, state).

    Each update moves every angle by at most `max_step` rad (default:
    `first_step`).

    x_final is the mean of the last quarter of the iterates, which strips most
    of the leftover shot-noise jitter. `state` is a JSON-serializable dict;
    pass a previous one back in to resume where it stopped.
    """
    A = 0.1 * n_iter
    if max_step is None:
        max_step = first_step

    if state is None:
        x = np.array(x0, dtype=float)
        rng = np.random.default_rng([seed, 0])
        grads = []
        for _ in range(n_calibrate):
            delta = rng.choice([-1.0, 1.0], size=x.size)
            f_plus, f_minus = fun_pair(x + c * delta, x - c * delta)
            grads.append(abs(f_plus - f_minus) / (2 * c))
        g_mag = float(np.mean(grads)) or 1.0
        state = {"k": 0, "x": x.tolist(), "a": first_step * (A + 1) ** alpha / g_mag,
                 "iterates": [], "costs": []}
        if on_iter is not None:
            on_iter(state)

    x = np.array(state["x"], dtype=float)
    a = state["a"]
    for k in range(state["k"], n_iter):
        ak = a / (k + 1 + A) ** alpha
        ck = c / (k + 1) ** gamma
        # seeded per iteration, so a resumed run draws the same directions
        delta = np.random.default_rng([seed, k + 1]).choice([-1.0, 1.0], size=x.size)
        f_plus, f_minus = fun_pair(x + ck * delta, x - ck * delta)
        # Cap the step. Started near the optimum, the calibration sees only
        # shot noise (true gradient ~0), so `a` comes out huge and one noisy
        # gradient estimate would throw the angles radians away.
        step = np.clip(ak * (f_plus - f_minus) / (2 * ck) * delta, -max_step, max_step)
        x = x - step

        state["k"] = k + 1
        state["x"] = x.tolist()
        state["iterates"].append(x.tolist())
        state["costs"].append([float(f_plus), float(f_minus)])
        if on_iter is not None:
            on_iter(state)

    tail = max(1, n_iter // 4)
    x_final = np.mean(state["iterates"][-tail:], axis=0) if state["iterates"] else x
    return x_final, state
