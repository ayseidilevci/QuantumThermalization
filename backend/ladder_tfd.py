"""The n+n-qubit TFD experiment (n = 3, 4) on a real device, for the GUI.

Physics, circuits and analysis are tfd_study/model.py's, as run on Tuna-17
by tfd_study/run_hardware.py on the tfd-study branch:

  * loss: the engineered cost C_s(beta), from 2 circuits per parameter point
    (all-Z, all-X); an SPSA step (+/- points) is 4 circuits = 1 job
  * readout: per-qubit calibration (all-0 / all-1, 2 circuits), refreshed
    every CALIBRATION_INTERVAL seconds, applied to every distribution
  * tomography: 3^n settings, A_i and B_i in the same basis; its all-Z and
    all-X settings double as the loss circuits, so the final loss is free

Every job goes through `wait(job, purpose)`, the GUI's poller (see
frontend/experiment.py), so jobs show in the Hardware jobs tab and Stop works.
"""

import time

import numpy as np
from qiskit import transpile
from qiskit.quantum_info import SparsePauliOp

from tfd_study import model

# As tfd_study/run_hardware.py's hardware plans.
SHOTS = 4096
CALIBRATION_INTERVAL = 30 * 60


def loss_lower_bound(n: int, beta: float, varsigma: float = 1.57) -> float:
    """Smallest possible C_s(beta): the lowest eigenvalue of its operator on
    2n qubits (model.py's qubit order: A_i = q(i-1), B_i = q(n+i-1))."""
    terms = [("X", [q], model.G_FIELD) for q in range(2 * n)]
    for i, j in model.bonds(n):
        terms += [("ZZ", [model.A(i), model.A(j)], varsigma),
                  ("ZZ", [model.B(i, n), model.B(j, n)], varsigma)]
    for i in range(n):
        terms += [(pauli, [model.A(i), model.B(i, n)], -beta ** (-varsigma)) for pauli in ("XX", "ZZ")]
    op = SparsePauliOp.from_sparse_list(terms, num_qubits=2 * n)
    return float(np.linalg.eigvalsh(op.to_matrix()).min())


class LadderTFD:
    """`layout`: physical qubits of A1..An, B1..Bn. `max_circuits`: the
    device's limit per job (5 on Quantum Inspire's Tuna devices)."""

    def __init__(self, n, backend, layout, wait, max_circuits, varsigma=1.57, p=1, shots=SHOTS):
        self.n, self.p, self.varsigma = n, p, varsigma
        self.shots = shots
        self.backend, self.layout = backend, list(layout)
        self.wait = wait
        self.max_circuits = max_circuits or 5
        self._mats = None
        self._calibrated_at = None

    # ---- jobs ------------------------------------------------------------
    def _run(self, circuits, purpose):
        """Probability vectors (readout uncorrected) for `circuits`, in as few
        jobs as the device allows."""
        chunks = [circuits[k:k + self.max_circuits]
                  for k in range(0, len(circuits), self.max_circuits)]
        probs = []
        for k, chunk in enumerate(chunks):
            label = purpose + (f" ({k + 1}/{len(chunks)})" if len(chunks) > 1 else "")
            compiled = transpile(chunk, self.backend, initial_layout=self.layout)
            result = self.wait(self.backend.run(compiled, shots=self.shots), label)
            failed = [f"circuit {i}: {exp.status}" for i, exp in enumerate(result.results)
                      if not exp.success]
            if failed:
                raise RuntimeError("Backend job failed — " + "; ".join(failed))
            for i in range(len(chunk)):
                counts = result.get_counts(i)
                if any(int(key.replace(" ", ""), 2) >= 4 ** self.n for key in counts):
                    raise RuntimeError(
                        f"{getattr(self.backend, 'name', 'the backend')} returned results with more "
                        f"than {2 * self.n} bits (e.g. {max(counts, key=len)!r}): they're indexed "
                        f"by physical qubit, not by the circuit's bits, for layout {self.layout}.")
                probs.append(model.counts_to_probs(counts, 2 * self.n))
        return probs

    def _corrected(self, circuits, purpose):
        if self._mats is None or time.time() - self._calibrated_at > CALIBRATION_INTERVAL:
            cal0, cal1 = self._run(model.calibration_circuits(self.n), "Readout calibration")
            self._mats = model.readout_matrices(cal0, cal1, 2 * self.n)
            self._calibrated_at = time.time()
        return [model.correct_readout(p, self._mats) for p in self._run(circuits, purpose)]

    # ---- experiment ------------------------------------------------------
    def losses(self, points, beta, purpose):
        """C_s(beta) at every point, measured on the device."""
        circuits = [model.measured(self.n, self.p, x, basis) for x in points for basis in "ZX"]
        probs = self._corrected(circuits, purpose)
        return [model.loss(model.loss_terms(probs[2 * i], probs[2 * i + 1], self.n), beta, self.varsigma)
                for i in range(len(points))]

    def tomography(self, theta, beta, purpose):
        """(rho_A, C_s(beta)) at `theta`, from the 3^n tomography settings."""
        settings = model.tomography_settings(self.n)
        circuits = [model.measured(self.n, self.p, theta, s) for s in settings]
        by_setting = dict(zip(settings, self._corrected(circuits, purpose)))
        rho_a = model.tomography_rho(by_setting, self.n, "A")
        terms = model.loss_terms(by_setting["Z" * 2 * self.n], by_setting["X" * 2 * self.n], self.n)
        return rho_a, model.loss(terms, beta, self.varsigma)

    def counts_circuit(self, theta):
        """The prepared state measured in the computational basis, for the
        final counts histogram."""
        return model.measured(self.n, self.p, theta, "Z")
