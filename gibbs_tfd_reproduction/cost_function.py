import json
import os
from datetime import datetime

import numpy as np
from qi_utilities.error_mitigation.toolkit.readout_error_mitigation import apply_readout_assignment_correction
from qi_utilities.utility_functions.circuit_modifiers import apply_pre_measurement_rotations
from qi_utilities.utility_functions.raw_data_processing import (
    get_multi_counts, get_multi_probs, get_raw_data, observable_expectation_values_Z_basis,
)
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize

from tfd_circuit import TFDCircuit
from tomography import build_tomography_circuit, plot_density_matrix, reconstruct_density_matrix

# q0 = A2
# q1 = A1
# q2 = B2
# q3 = B1

class CostFunction:
    def __init__(self, varsigma=1.57, backend=None, initial_layout=(0,2,3,5), subsystem_qubits=(0,1),
                 tomography=True, results_dir="results", run_tag=None, shots=8192,
                 wait_for_result=None):
        # tomography=False skips the per-evaluation 9-setting tomography, which
        # costs several times more than the cost circuit itself.
        # wait_for_result(job) -> Result replaces job.result(): on real
        # hardware QIJob.result() gives up after 60 s, less than a queue wait.
        self.varsigma = varsigma
        self.tomography = tomography
        self.backend = backend
        self.initial_layout = list(initial_layout)
        self.shots = shots
        self.wait_for_result = wait_for_result or (lambda job: job.result())
        self.ro_matrices = None
        self.iteration = 0
        self.cost_history = []
        self.tomography_history = []
        self.subsystem_qubits = subsystem_qubits

        # Progress log: one line per cost evaluation, so a crash or Ctrl-C
        # loses nothing. Instances sharing a run_tag share one file.
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        self.run_tag = run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(self.results_dir, f"progress_{self.run_tag}.jsonl")

    def cost_function(self, params, beta):
        tomography_for = (0,) if self.tomography else ()
        return self.evaluate_batch([params], beta, tomography_for=tomography_for)[0][0]

    def evaluate_batch(self, points, beta, tomography_for=()):
        """Evaluate the cost at every point in `points` in ONE backend job.
        Points whose index is in `tomography_for` also get a tomography
        circuit. Returns [(cost, rho or None)] in the order of `points`.

        On real hardware every job waits in the device's queue, so batching
        the points an optimizer step needs is much faster than one job each.
        A job may hold a limited number of circuits (5 on QI's Tuna devices):
        every point costs one circuit, plus one per tomography."""
        circuits, slots = [], []
        for i, params in enumerate(points):
            circuits.append(self._cost_circuit(params))
            slots.append(("cost", i, None))
        for i in tomography_for:
            qc_tomo_t, settings = self._tomography_circuit(points[i])
            circuits.append(qc_tomo_t)
            slots.append(("tomo", i, settings))

        job = self.backend.run(circuits, shots=self.shots, memory=True)
        result = self.wait_for_result(job)
        failed = [f"circuit {n}: {exp.status}" for n, exp in enumerate(result.results)
                  if not exp.success]
        if failed:
            raise RuntimeError("Backend job failed — " + "; ".join(failed))

        costs, rhos = [None] * len(points), [None] * len(points)
        # Costs first: the first one may refresh the readout calibration
        # that the tomography below is corrected with.
        for n, (kind, i, _) in enumerate(slots):
            if kind == "cost":
                costs[i] = self._process_cost(circuits[n], result, n, points[i], beta)
        for n, (kind, i, settings) in enumerate(slots):
            if kind == "tomo":
                rhos[i] = self._process_tomography(circuits[n], result, n, settings, beta)
        return list(zip(costs, rhos))

    def _cost_circuit(self, params):
        gamma1, gamma2, alpha1, alpha2 = params

        qc_cost = QuantumCircuit(4, 16)
        ansatz = TFDCircuit(gamma1, gamma2, alpha1, alpha2)
        ansatz.build_circuit(qc_cost)
        qc_cost.barrier()
        apply_pre_measurement_rotations(qc_cost, "XXXX", bit_register=[0,1,2,3])
        qc_cost.barrier()
        for q in range(4): qc_cost.reset(q)
        qc_cost.barrier()
        ansatz.build_circuit(qc_cost)
        qc_cost.barrier()
        apply_pre_measurement_rotations(qc_cost, "ZZZZ", bit_register=[4,5,6,7])
        qc_cost.barrier()
        for q in range(4): qc_cost.reset(q)
        qc_cost.barrier()
        for q in range(4): qc_cost.measure(q, 8+q)
        qc_cost.barrier()
        for q in range(4): qc_cost.reset(q)
        qc_cost.barrier()
        qc_cost.x(range(4))
        qc_cost.barrier()
        for q in range(4): qc_cost.measure(q, 12+q)

        return transpile(qc_cost, self.backend, initial_layout=self.initial_layout)

    def _tomography_circuit(self, params):
        qc_tomo, settings = build_tomography_circuit(*params, subsystem_qubits=self.subsystem_qubits)
        return transpile(qc_tomo, self.backend, initial_layout=self.initial_layout), settings

    def _process_cost(self, qc_cost_t, result, circuit_nr, params, beta):
        gamma1, gamma2, alpha1, alpha2 = params
        raw_cost = get_raw_data(qc_cost_t, result, circuit_nr=circuit_nr)
        counts_cost = get_multi_counts(raw_cost, 4)
        probs_cost = get_multi_probs(counts_cost)
        x_probs, z_probs, calib0, calib1 = probs_cost[0], probs_cost[1], counts_cost[2], counts_cost[3]

        if self.iteration % 10 == 0 or self.ro_matrices is None:
            self.ro_matrices = self._extract_ro_matrices(calib0, calib1)
        self.iteration += 1

        x_probs = apply_readout_assignment_correction(x_probs, self.ro_matrices, [0,1,2,3])
        z_probs = apply_readout_assignment_correction(z_probs, self.ro_matrices, [0,1,2,3])

        exp_X_A = observable_expectation_values_Z_basis([x_probs], "IIZI")[0] + observable_expectation_values_Z_basis([x_probs], "IIIZ")[0]
        exp_X_B = observable_expectation_values_Z_basis([x_probs], "ZIII")[0] + observable_expectation_values_Z_basis([x_probs], "IZII")[0]
        exp_ZZ_A = observable_expectation_values_Z_basis([z_probs], "IIZZ")[0]
        exp_ZZ_B = observable_expectation_values_Z_basis([z_probs], "ZZII")[0]
        exp_XX_BA = observable_expectation_values_Z_basis([x_probs], "ZIZI")[0] + observable_expectation_values_Z_basis([x_probs], "IZIZ")[0]
        exp_ZZ_BA = observable_expectation_values_Z_basis([z_probs], "ZIZI")[0] + observable_expectation_values_Z_basis([z_probs], "IZIZ")[0]

        cost_B = exp_X_A + exp_X_B + self.varsigma*(exp_ZZ_A+exp_ZZ_B) - beta**(-self.varsigma)*(exp_XX_BA+exp_ZZ_BA)

        # Print and log every evaluation as soon as it's known.
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "beta": beta,
            "iteration": self.iteration,
            "gamma1": float(gamma1), "gamma2": float(gamma2),
            "alpha1": float(alpha1), "alpha2": float(alpha2),
            "cost": float(cost_B),
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(
            f"[{record['timestamp']}] beta={beta:<10} iter={self.iteration:<4} "
            f"g1={gamma1:+.4f} g2={gamma2:+.4f} a1={alpha1:+.4f} a2={alpha2:+.4f} "
            f"cost={cost_B:+.5f}",
            flush=True,
        )

        self.cost_history.append(cost_B)
        return cost_B

    def _process_tomography(self, qc_tomo_t, result, circuit_nr, settings, beta):
        raw_tomo = get_raw_data(qc_tomo_t, result, circuit_nr=circuit_nr)
        counts_tomo = get_multi_counts(raw_tomo, 4)
        probs_tomo = get_multi_probs(counts_tomo)
        tomography_probs = [apply_readout_assignment_correction(p, self.ro_matrices, [0,1,2,3])
                             for p in probs_tomo[:9]]
        rho = reconstruct_density_matrix(tomography_probs, settings, subsystem_qubits=self.subsystem_qubits)
        self.tomography_history.append(rho)

        os.makedirs("plots", exist_ok=True)
        plot_density_matrix(rho, title=f"beta={beta}, iteration={self.iteration}",
                     save_path=f"plots/tomography_live_beta{beta}.png")
        return rho

    def minimize_cost(self, x0, beta):
        optimized_params = minimize(self.cost_function, x0=x0, args=(beta,), method='COBYLA', options={'maxiter': 100, 'rhobeg': 0.5})
        return optimized_params.x

    def cost_lower_bound(beta, varsigma=1.57):
        paulis = {
            "IIXI": 1, "IIIX": 1,
            "XIII": 1, "IXII": 1,
            "IIZZ": varsigma,
            "ZZII": varsigma,
            "XIXI": -beta**(-varsigma), "IXIX": -beta**(-varsigma),
            "ZIZI": -beta**(-varsigma), "IZIZ": -beta**(-varsigma),
        }
        op = SparsePauliOp(list(paulis.keys()), list(paulis.values()))
        mat = op.to_matrix()
        return np.linalg.eigvalsh(mat).min()

    def _extract_ro_matrices(self, counts_prep0, counts_prep1):
        shots0, shots1 = sum(counts_prep0.values()), sum(counts_prep1.values())
        ro_matrices = {}
        for q in range(4):
            d0_0 = sum(n for b, n in counts_prep0.items() if b[-(q+1)] == '0')
            d0_1 = shots0 - d0_0
            d1_0 = sum(n for b, n in counts_prep1.items() if b[-(q+1)] == '0')
            d1_1 = shots1 - d1_0
            ro_matrices[f"Qubit {q}"] = np.array([[d0_0/shots0, d1_0/shots1], [d0_1/shots0, d1_1/shots1]])
        return ro_matrices