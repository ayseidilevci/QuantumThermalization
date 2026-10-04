from qiskit import QuantumCircuit, transpile
from qi_utilities.utility_functions.circuit_modifiers import apply_pre_measurement_rotations
from qi_utilities.utility_functions.raw_data_processing import (
    get_raw_data, get_multi_counts, get_multi_probs, observable_expectation_values_Z_basis,
)
from qi_utilities.error_mitigation.toolkit.readout_error_mitigation import apply_readout_assignment_correction
from qiskit.quantum_info import Pauli
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt

from tfd_circuit import TFDCircuit
from qiskit.quantum_info import SparsePauliOp
import numpy as np
from tomography import plot_density_matrix


from tomography import build_tomography_circuit, reconstruct_density_matrix
# q0 = A2
# q1 = A1
# q2 = B2
# q3 = B1

class CostFunction:
    def __init__(self, varsigma=1.57, backend=None, initial_layout=(0,2,3,5), subsystem_qubits=(0,1)):
        self.varsigma = varsigma
        self.backend = backend
        self.initial_layout = list(initial_layout)
        self.ro_matrices = None
        self.iteration = 0
        self.cost_history = []
        self.tomography_history = []
        self.subsystem_qubits = subsystem_qubits

    def cost_function(self, params, beta):
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

        qc_tomo, settings = build_tomography_circuit(gamma1, gamma2, alpha1, alpha2,
                                                       subsystem_qubits=self.subsystem_qubits)

        qc_cost_t = transpile(qc_cost, self.backend, initial_layout=self.initial_layout)
        qc_tomo_t = transpile(qc_tomo, self.backend, initial_layout=self.initial_layout)

        job = self.backend.run([qc_cost_t, qc_tomo_t], shots=8192, memory=True)
        result = job.result()

        raw_cost = get_raw_data(qc_cost_t, result, circuit_nr=0)
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

        raw_tomo = get_raw_data(qc_tomo_t, result, circuit_nr=1)
        counts_tomo = get_multi_counts(raw_tomo, 4)
        probs_tomo = get_multi_probs(counts_tomo)
        tomography_probs = [apply_readout_assignment_correction(p, self.ro_matrices, [0,1,2,3])
                             for p in probs_tomo[:9]]
        rho = reconstruct_density_matrix(tomography_probs, settings, subsystem_qubits=self.subsystem_qubits)
        self.tomography_history.append(rho)

        plot_density_matrix(rho, title=f"beta={beta}, iteration={self.iteration}",
                     save_path=f"plots/tomography_live_beta{beta}.png")
        
        self.cost_history.append(cost_B)
        return cost_B
    
    # def minimize_cost(self, x0, beta, method='COBYLA', maxiter=None, callback=None):
    #     """Run the classical optimizer. `method` is any scipy.optimize.minimize
    #     method that supports a plain `callback(xk)` (COBYLA, Nelder-Mead and
    #     Powell are verified to; see frontend/setup.py's OPTIMIZER_METHODS).
    #     `callback`, if given, is invoked with the current parameter vector
    #     after every iteration — used by the GUI to stream progress live.
    #     """
    #     options = {'maxiter': maxiter} if maxiter is not None else None
    #     optimized_params = minimize(
    #         self.cost_function, x0=x0, args=(beta,), method=method,
    #         options=options, callback=callback,
    #     )
    def minimize_cost(self, x0, beta):
        optimized_params = minimize(self.cost_function, x0=x0, args=(beta,), method='COBYLA', options={'maxiter': 100, 'rhobeg': 0.5})
        # optimized_params = minimize(self.cost_function, x0=x0, args=(beta,), method='BFGS', options={'maxiter': 100, 'eps': 0.1})
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