import numpy as np
from qiskit.quantum_info import Pauli, SparsePauliOp
from scipy.optimize import minimize

from .tfd_circuit_3spin import TFD3SpinCircuit

class CostFunction3Spin:
    """Engineered C_varsigma(beta) cost for the 3-spin / 6-qubit ansatz.

    Qubit convention:
        q0=A1, q1=A2, q2=A3, q3=B1, q4=B2, q5=B3

    For an open three-spin chain:
        X_A   = X_A1 + X_A2 + X_A3
        X_B   = X_B1 + X_B2 + X_B3
        ZZ_A  = Z_A1 Z_A2 + Z_A2 Z_A3
        ZZ_B  = Z_B1 Z_B2 + Z_B2 Z_B3
        XX_BA = X_A1 X_B1 + X_A2 X_B2 + X_A3 X_B3
        ZZ_BA = Z_A1 Z_B1 + Z_A2 Z_B2 + Z_A3 Z_B3

    The engineered cost is

        C_varsigma(beta) = <X_A> + <X_B>
            + varsigma (<ZZ_A> + <ZZ_B>)
            - beta^(-varsigma) (<XX_BA> + <ZZ_BA>).

    At beta=0 we use the limiting form
        C(0) = -<XX_BA + ZZ_BA>.
    """

    N_QUBITS = 6

    # Qiskit Pauli strings: right-most character acts on q0.
    OBSERVABLES = {
        "X_A": ("IIIXII", "IIIIXI", "IIIIIX"),      # X1 + X2 + X3  (A)
        "X_B": ("IIXIII", "IXIIII", "XIIIII"),      # X1 + X2 + X3  (B)
        "ZZ_A": ("IIIZZI", "IIIIZZ"),               # Z3Z2 + Z2Z1   (A)
        "ZZ_B": ("ZZIIII", "IZZIII"),               # Z3Z2 + Z2Z1   (B)
        "XX_BA": ("XIIXII", "IXIIXI", "IIXIIX"),    # X1aX1b + X2aX2b + X3aX3b
        "ZZ_BA": ("ZIIZII", "IZIIZI", "IIZIIZ"),    # Z1aZ1b + Z2aZ2b + Z3aZ3b
    }

    def __init__(self, varsigma=2.05, p_layers=1, periodic=False):
        self.varsigma = float(varsigma)
        self.p_layers = int(p_layers)
        self.periodic = bool(periodic)
        self.cost_history = []
        self.parameter_history = []

    def expectation_values(self, params):
        state = TFD3SpinCircuit(
            params,
            p_layers=self.p_layers,
            periodic=self.periodic,
        ).get_state_vector()

        return {
            name: sum(self._expectation(state, s) for s in strings)
            for name, strings in self.OBSERVABLES.items()
        }

    @staticmethod
    def _expectation(state, pauli_string):
        return float(np.real(state.expectation_value(Pauli(pauli_string))))

    def minimize_cost(
        self,
        x0,
        beta,
        method="COBYLA",
        maxiter=200,
        rhobeg=0.5,
        tol=0.02,
        callback=None,
    ):
        """Minimize the cost using COBYLA or Nelder-Mead."""

        self.cost_history = []
        self.parameter_history = []

        method = method.strip().lower().replace("_", "-")

        methods = {
            "cobyla": "COBYLA",
            "nelder-mead": "Nelder-Mead",
            "neldermead": "Nelder-Mead",
        }

        if method not in methods:
            raise ValueError(
                "Unsupported optimizer. Choose 'COBYLA' or 'Nelder-Mead'."
            )

        method = methods[method]
        x0 = np.asarray(x0, dtype=float)

        expected_parameters = 4 * self.p_layers
        if x0.shape != (expected_parameters,):
            raise ValueError(
                f"Expected {expected_parameters} initial parameters for "
                f"p_layers={self.p_layers}, but received {x0.size}."
            )

        if method == "COBYLA":
            options = {
                "maxiter": int(maxiter),
                "rhobeg": float(rhobeg),
                "tol": float(tol),
            }
        else:
            options = {
                "maxiter": int(maxiter),
                "xatol": float(tol),
                "fatol": float(tol),
            }

        result = minimize(
            self.cost_function,
            x0=x0,
            args=(beta,),
            method=method,
            options=options,
            callback=callback,
        )

        return result


    def cost_function(self, params, beta):
        values = self.expectation_values(params)

        if beta == 0:
            cost = -(values["XX_BA"] + values["ZZ_BA"])
        else:
            beta_weight = beta ** (-self.varsigma)
            cost = (
                values["X_A"]
                + values["X_B"]
                + self.varsigma * (values["ZZ_A"] + values["ZZ_B"])
                - beta_weight * (values["XX_BA"] + values["ZZ_BA"])
            )

        cost = float(cost)
        self.cost_history.append(cost)
        self.parameter_history.append(np.asarray(params, dtype=float).copy())
        return cost

    @staticmethod
    def cost_lower_bound(beta, varsigma=1.57, periodic=False):
        """Exact minimum eigenvalue of the 6-qubit engineered cost operator."""
        varsigma = float(varsigma)

        if beta == 0:
            terms = [
                ("XIIXII", -1.0),
                ("IXIIXI", -1.0),
                ("IIXIIX", -1.0),
                ("ZIIZII", -1.0),
                ("IZIIZI", -1.0),
                ("IIZIIZ", -1.0),
            ]
        else:
            weight = beta ** (-varsigma)
            terms = CostFunction3Spin._cost_terms(varsigma)
            terms += [
                ("XIIXII", -weight),
                ("IXIIXI", -weight),
                ("IIXIIX", -weight),
                ("ZIIZII", -weight),
                ("IZIIZI", -weight),
                ("IIZIIZ", -weight),
            ]

            if periodic:
                terms += [
                    ("IZZIII", varsigma),
                    ("ZZI III".replace(" ", ""), varsigma),
                ]
                # The helper above is intentionally not used for periodic
                # cost construction yet; see the note in extension_experiment.
                raise NotImplementedError(
                    "Periodic lower_bound needs the extra A3-A1 and B3-B1 terms."
                )

        op = SparsePauliOp.from_list(terms)
        return float(np.linalg.eigvalsh(op.to_matrix()).min())
