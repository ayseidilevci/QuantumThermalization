import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


class TFD3SpinCircuit:
    """QAOA-motivated TFD ansatz for three spins in A and three in B.

    Qubit convention: q0=A1, q1=A2, q2=A3, q3=B1, q4=B2, q5=B3.
    Each layer has four shared angles: gamma1, gamma2, alpha1, alpha2.
    Barriers separate the initial TFD preparation and each variational block;
    they are visual/scheduling markers and do not change the ideal state.
    """

    N_QUBITS = 6

    def __init__(self, params, p_layers=None, periodic=False):
        params = np.asarray(params, dtype=float)
        if params.ndim != 1 or params.size == 0 or params.size % 4 != 0:
            raise ValueError("params must contain 4 parameters per QAOA layer")
        inferred_p = params.size // 4
        self.p_layers = inferred_p if p_layers is None else int(p_layers)
        if self.p_layers < 1:
            raise ValueError("p_layers must be >= 1")
        if params.size != 4 * self.p_layers:
            raise ValueError("len(params) must equal 4 * p_layers")
        self.params = params.copy()
        self.periodic = bool(periodic)
        self.circuit = QuantumCircuit(self.N_QUBITS)
        self.build_circuit(self.circuit)

    def build_circuit(self, circuit):
        # |TFD(0)> = |Phi+>_{A1,B1}|Phi+>_{A2,B2}|Phi+>_{A3,B3}.
        for a, b in zip(range(3), range(3, 6)):
            circuit.h(b)
            circuit.cx(b, a)
        circuit.barrier(*range(6))  # End of initial entanglement preparation.

        intra_edges = [(0, 1), (1, 2), (3, 4), (4, 5)]
        if self.periodic:
            intra_edges += [(2, 0), (5, 3)]

        for layer in range(self.p_layers):
            gamma1, gamma2, alpha1, alpha2 = self.params[4 * layer:4 * (layer + 1)]

            # U_gamma1 = exp(-i gamma1 X_total / 2).
            circuit.rx(gamma1, range(6))
            circuit.barrier(*range(6))

            # U_gamma2 = exp(-i gamma2 ZZ_intra / 2).
            for q1, q2 in intra_edges:
                circuit.rzz(gamma2, q1, q2)
            circuit.barrier(*range(6))

            # U_alpha1 = exp(-i alpha1 XX_BA / 2).
            for a, b in zip(range(3), range(3, 6)):
                circuit.rxx(alpha1, a, b)
            circuit.barrier(*range(6))

            # U_alpha2 = exp(-i alpha2 ZZ_BA / 2).
            for a, b in zip(range(3), range(3, 6)):
                circuit.rzz(alpha2, a, b)
            circuit.barrier(*range(6))

        # Explicit end marker for readable circuit diagrams.
        circuit.barrier(*range(6))
        return circuit

    def get_state_vector(self):
        return Statevector(self.circuit)

    def as_qiskit_circuit(self) -> QuantumCircuit:
        return self.circuit

