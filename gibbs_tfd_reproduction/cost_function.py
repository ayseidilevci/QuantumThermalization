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

from gibbs_tfd_reproduction.tfd_circuit import TFDCircuit

# q0 = A2
# q1 = A1
# q2 = B2
# q3 = B1

class CostFunction:
    def __init__(self, varsigma=1.57, backend=None, initial_layout=(0, 2, 3, 5)):
        self.varsigma = varsigma
        self.backend = backend
        self.initial_layout = list(initial_layout)
        self.ro_matrices = None
        self.iteration = 0
        self.cost_history = []
        self.circuit_pdf = None

    def cost_function(self, params, beta):
        gamma1, gamma2, alpha1, alpha2 = params
        qc = QuantumCircuit(4, 16)
        ansatz = TFDCircuit(gamma1, gamma2, alpha1, alpha2)

        ansatz.build_circuit(qc)
        qc.barrier()
        apply_pre_measurement_rotations(qc, "XXXX", bit_register=[0, 1, 2, 3])

        qc.barrier()
        for q in range(4):
            qc.reset(q)
        qc.barrier()
        ansatz.build_circuit(qc)
        qc.barrier()
        apply_pre_measurement_rotations(qc, "ZZZZ", bit_register=[4, 5, 6, 7])

        qc.barrier()
        for q in range(4):
            qc.reset(q)
        qc.barrier()
        for q in range(4):
            qc.measure(q, 8 + q)
        qc.barrier()
        for q in range(4):
            qc.reset(q)
        qc.barrier()
        qc.x(range(4))
        qc.barrier()
        for q in range(4):
            qc.measure(q, 12 + q)

        if self.circuit_pdf is not None:
            fig = qc.draw(output='mpl')
            self.circuit_pdf.savefig(fig)
            plt.close(fig)

        qc_t = transpile(qc, self.backend, initial_layout=self.initial_layout)
        job = self.backend.run(qc_t, shots=10000, memory=True)
        result = job.result()

        raw = get_raw_data(qc_t, result)
        counts = get_multi_counts(raw, 4)
        probs = get_multi_probs(counts)
        x_probs, z_probs, calib0, calib1 = probs[0], probs[1], counts[2], counts[3]

        # if self.iteration % 10 == 0 or self.ro_matrices is None:
        #     self.ro_matrices = self._extract_ro_matrices(calib0, calib1)
        # self.iteration += 1

        # # M_q = [[prepared0read0, prepared0read1],
        # #       [preapred1read0, prepared1prepared1]]
        # # kronecker product --> invert the matrix --> apply to raw measurement probability
        # x_probs = apply_readout_assignment_correction(x_probs, self.ro_matrices, [0, 1, 2, 3])
        # z_probs = apply_readout_assignment_correction(z_probs, self.ro_matrices, [0, 1, 2, 3])

        exp_X_A = observable_expectation_values_Z_basis([x_probs], "IIZI")[0] + observable_expectation_values_Z_basis([x_probs], "IIIZ")[0]
        exp_X_B = observable_expectation_values_Z_basis([x_probs], "ZIII")[0] + observable_expectation_values_Z_basis([x_probs], "IZII")[0]
        exp_ZZ_A = observable_expectation_values_Z_basis([z_probs], "IIZZ")[0]
        exp_ZZ_B = observable_expectation_values_Z_basis([z_probs], "ZZII")[0]
        exp_XX_BA = observable_expectation_values_Z_basis([x_probs], "ZIZI")[0] + observable_expectation_values_Z_basis([x_probs], "IZIZ")[0]
        exp_ZZ_BA = observable_expectation_values_Z_basis([z_probs], "ZIZI")[0] + observable_expectation_values_Z_basis([z_probs], "IZIZ")[0]

        # Intermediate terms for granular debugging
        term_ZZ_sum = exp_ZZ_A + exp_ZZ_B
        term_scaled_ZZ = self.varsigma * term_ZZ_sum
        
        term_BA_sum = exp_XX_BA + exp_ZZ_BA
        term_power_beta = np.power(beta, -self.varsigma)
        term_scaled_BA = term_power_beta * term_BA_sum

        # Final calculation
        cost_B = exp_X_A + exp_X_B + term_scaled_ZZ - term_scaled_BA
            
        # # Detailed printouts for every value and term
        # print("--- Cost B Evaluation Debug ---")
        # print(f"exp_X_A       : {exp_X_A}")
        # print(f"exp_X_B       : {exp_X_B}")
        # print(f"exp_ZZ_A      : {exp_ZZ_A}")
        # print(f"exp_ZZ_B      : {exp_ZZ_B}")
        # print(f"exp_XX_BA     : {exp_XX_BA}")
        # print(f"exp_ZZ_BA     : {exp_ZZ_BA}")
        # print(f"self.varsigma : {self.varsigma}")
        # print(f"beta          : {beta}")
        # print(f"beta^(-varsig): {term_power_beta}")
        # print("-------------------------------")
        # print(f"Summation Term 1 (X_A + X_B)           : {exp_X_A + exp_X_B}")
        # print(f"Summation Term 2 (varsigma * ZZ_sum)   : {term_scaled_ZZ}")
        # print(f"Summation Term 3 (beta^-. * BA_sum)    : {term_scaled_BA}")
        # print(f"Calculated cost_B                      : {cost_B}")
        # print("-------------------------------")

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
    def minimize_cost(self, x0, beta, rhobeg=0.5, maxiter=200, callback=None):
        optimized_params = minimize(
            self.cost_function, x0=x0, args=(beta,), method='COBYLA',
            options={'maxiter': maxiter, 'rhobeg': rhobeg, 'tol': 0.02},
            callback=callback
        )
        # optimized_params = minimize(self.cost_function, x0=x0, args=(beta,), method='BFGS', options={'maxiter': 100, 'eps': 0.1})
        return optimized_params.x
    
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
