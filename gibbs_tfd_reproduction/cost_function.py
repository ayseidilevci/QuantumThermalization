from qiskit.quantum_info import Pauli
from scipy.optimize import minimize

from .tfd_circuit import TFDCircuit

# q0 = A2
# q1 = A1
# q2 = B2
# q3 = B1

class CostFunction:
    def __init__(self, varsigma=1.57):
        self.varsigma = varsigma

    def cost_function(self, params, beta):
        gamma1, gamma2, alpha1, alpha2 = params
        tfd_circuit = TFDCircuit(gamma1, gamma2, alpha1, alpha2)
        state_vector = tfd_circuit.get_state_vector()

        # Pauli string --> right most character is qubit 0
        exp_X_A = state_vector.expectation_value(Pauli("IIXI")).real + state_vector.expectation_value(Pauli("IIIX")).real
        exp_X_B = state_vector.expectation_value(Pauli("XIII")).real + state_vector.expectation_value(Pauli("IXII")).real
        exp_ZZ_A = state_vector.expectation_value(Pauli("IIZZ")).real
        exp_ZZ_B = state_vector.expectation_value(Pauli("ZZII")).real
        exp_XX_BA = state_vector.expectation_value(Pauli("XIXI")).real + state_vector.expectation_value(Pauli("IXIX")).real
        exp_ZZ_BA = state_vector.expectation_value(Pauli("ZIZI")).real + state_vector.expectation_value(Pauli("IZIZ")).real


        cost_B = exp_X_A + exp_X_B + self.varsigma * (exp_ZZ_A + exp_ZZ_B) - beta**(-self.varsigma) * (exp_XX_BA + exp_ZZ_BA)

        return cost_B
    
    def minimize_cost(self, x0, beta, method='COBYLA', maxiter=None, callback=None):
        """Run the classical optimizer. `method` is any scipy.optimize.minimize
        method that supports a plain `callback(xk)` (COBYLA, Nelder-Mead and
        Powell are verified to; see frontend/setup.py's OPTIMIZER_METHODS).
        `callback`, if given, is invoked with the current parameter vector
        after every iteration — used by the GUI to stream progress live.
        """
        options = {'maxiter': maxiter} if maxiter is not None else None
        optimized_params = minimize(
            self.cost_function, x0=x0, args=(beta,), method=method,
            options=options, callback=callback,
        )
        return optimized_params.x
    

