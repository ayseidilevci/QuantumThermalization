"""SPSA warm-start protocol for n+n qubits on any backend, with hardware bookkeeping.

One chain = the canonical experiment: theta = 0 is the exact beta = 0 optimum
(Bell pairs), then SPSA at each beta in turn, each step warm-started from the
previous one. theta = 0 is a stationary point of the loss (every gradient
vanishes there), so before the first beta > 0 a batch of random small kicks is
evaluated and SPSA starts from the best one.

Every loss evaluation = all-Z and all-X circuit for each parameter point; SPSA
evaluates its +/- pair in one batch (4 circuits = 1 submission, since Quantum
Inspire takes at most 5 circuits per submission). The simulation re-measures
readout calibration with every batch, but the cost bookkeeping follows the
hardware runner, which refreshes calibration about every 30 minutes: one
calibration submission per beta step. So each simulated setup comes with the
circuits and submissions it would cost on Tuna-17.

Evaluator kinds:
  'exact'      statevector expectation values, no shots (a perfect device)
  'noiseless'  Aer, shot noise only
  'noisy'      Aer density matrix with the fitted Tuna-17 model, readout-corrected
"""

import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gibbs_tfd_reproduction"))
from optimizers import spsa  # noqa: E402  (the SPSA used on Tuna-17)

from qiskit.quantum_info import DensityMatrix, partial_trace  # noqa: E402

from tfd_study import devices, model  # noqa: E402

CIRCUITS_PER_SUBMISSION = 5
DEFAULT_BETAS = (0.5, 1.0, 2.0, 5.0)


class Evaluator:
    """Loss for a batch of angle vectors on one backend, with circuit counting."""

    def __init__(self, n, p, kind, shots=4096, layout=None, seed=0, noise=None):
        self.n, self.p, self.kind, self.shots = n, p, kind, shots
        self.layout = layout or devices.default_layout(n)
        self.noise = noise or {}
        self.sim = None if kind == "exact" else devices.simulator(kind, **self.noise)
        self.rng_seed = seed
        self.n_circuits = 0
        self.n_submissions = 0
        self._calls = 0

    def charge(self, circuits):
        """Hardware cost of one batch of `circuits` circuits."""
        self.n_circuits += circuits
        self.n_submissions += math.ceil(circuits / CIRCUITS_PER_SUBMISSION)

    def _run(self, circuits):
        self._calls += 1
        device = devices.to_device(circuits, self.layout)
        res = self.sim.run(device, shots=self.shots, seed_simulator=self.rng_seed * 100003 + self._calls).result()
        return [model.counts_to_probs(res.get_counts(i), 2 * self.n) for i in range(len(circuits))]

    def terms(self, thetas):
        self.charge(2 * len(thetas))
        if self.kind == "exact":
            return [model.exact_terms(self.n, self.p, th) for th in thetas]
        circs = []
        for th in thetas:
            circs += [model.measured(self.n, self.p, th, "Z"), model.measured(self.n, self.p, th, "X")]
        probs = self._run(circs + model.calibration_circuits(self.n))
        mats = model.readout_matrices(probs[-2], probs[-1], 2 * self.n)
        out = []
        for k in range(len(thetas)):
            pz, px = (model.correct_readout(probs[2 * k + j], mats) for j in (0, 1))
            out.append(model.loss_terms(pz, px, self.n))
        return out

    def losses(self, thetas, beta, s):
        return [model.loss(t, beta, s) for t in self.terms(thetas)]

    def noisy_rho_A(self, theta):
        """rho_A the device model would produce (exact density matrix, no shot or tomography noise)."""
        if self.kind != "noisy":
            return model.exact_rho_A(self.n, self.p, theta)
        qc = devices.to_device(model.bound_ansatz(self.n, self.p, theta), self.layout)
        qc.save_density_matrix(qubits=[qc.layout.final_index_layout()[q] for q in range(2 * self.n)])
        sim = devices.simulator("noisy", **{**self.noise, "readout": (0.0, 0.0)})
        rho = DensityMatrix(sim.run(qc).result().data(0)["density_matrix"])
        return partial_trace(rho, list(range(self.n, 2 * self.n))).data


def run_chain(n, p, s, kind, iters, betas=DEFAULT_BETAS, shots=4096, seed=0, first_step=0.3,
              n_kicks=8, kick=0.4, layout=None, noise=None):
    """One warm-start chain. Returns per-beta records with angle quality, device-state
    fidelity and the cumulative hardware cost."""
    ev = Evaluator(n, p, kind, shots=shots, layout=layout, seed=seed, noise=noise)
    x = np.zeros(4 * p)
    steps = []
    t0 = time.time()
    for b in betas:
        ev.charge(2)                                   # readout calibration for this beta step
        if np.allclose(x, 0) and n_kicks:
            rng = np.random.default_rng([seed, 17])
            cands = [rng.normal(0, kick, 4 * p) for _ in range(n_kicks)]
            vals = ev.losses(cands, b, s)
            x = cands[int(np.argmin(vals))]
        history = []

        def fun_pair(xp, xm, b=b):
            fp, fm = ev.losses([xp, xm], b, s)
            history.append(0.5 * (fp + fm))
            return fp, fm

        x, _ = spsa(fun_pair, x, iters, first_step=first_step, seed=seed)
        x = np.asarray(x, float)
        rho = ev.noisy_rho_A(x)
        steps.append({
            "beta": b,
            "theta": x.tolist(),
            "fid_angles": model.exact_fidelity(n, p, x, b),            # optimizer quality
            "fid_device": model.fidelity(rho, b, n),                   # what tomography would see
            "purity_device": float(np.real(np.trace(rho @ rho))),
            "exact_loss": model.exact_loss(x, n, p, b, s),
            "loss_history": history,
            "circuits": ev.n_circuits,
            "submissions": ev.n_submissions,
        })
    cz = devices.cz_count(devices.to_device(model.bound_ansatz(n, p, np.full(4 * p, 0.3)), ev.layout))
    return {"n": n, "p": p, "s": s, "kind": kind, "iters": iters, "shots": shots, "seed": seed,
            "layout": ev.layout, "cz": cz, "noise": noise or {}, "steps": steps,
            "seconds": time.time() - t0}


def tomography_cost(n):
    """Circuits for one tomography of system A (3^n settings + 2 calibrations)."""
    return 3 ** n + 2
