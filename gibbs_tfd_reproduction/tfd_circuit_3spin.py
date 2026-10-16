"""Fixed six-qubit TFD ansatz for the three-spin extension.

Qubit convention: q0=A1, q1=A2, q2=A3, q3=B1, q4=B2, q5=B3.
The fixed variational block uses gamma1, gamma2, alpha1, alpha2.
"""
from __future__ import annotations
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


class TFD3SpinCircuit:
    """Build the fixed three-spin-per-register TFD variational circuit."""
    N_QUBITS = 6
    N_PARAMETERS = 4

    def __init__(self, params, periodic: bool = False, **legacy_kwargs):
        # Compatibility with the existing CostFunction3Spin call site.
        legacy_kwargs.pop("p_layers", None)
        if legacy_kwargs:
            raise TypeError(f"Unexpected keyword argument(s): {', '.join(legacy_kwargs)}")
        params = np.asarray(params, dtype=float)
        if params.shape != (self.N_PARAMETERS,):
            raise ValueError(
                "params must contain exactly four angles: gamma1, gamma2, alpha1, alpha2"
            )
        self.params = params.copy()
        self.periodic = bool(periodic)
        self.circuit = QuantumCircuit(self.N_QUBITS)
        self.build_circuit(self.circuit)

    def build_circuit(self, circuit: QuantumCircuit | None = None) -> QuantumCircuit:
        """Prepare the three-spin TFD ansatz."""
        if circuit is None:
            circuit = self.circuit

        # Prepare three Bell pairs.
        for a, b in zip(range(3), range(3, 6)):
            circuit.h(b)
            circuit.cx(b, a)

        circuit.barrier(*range(6))

        gamma1, gamma2, alpha1, alpha2 = self.params

        # Single-qubit rotations.
        circuit.rx(gamma1, range(6))
        circuit.barrier(*range(6))

        # Open-chain interactions within each subsystem.
        for q0, q1 in [(0, 1), (1, 2), (3, 4), (4, 5)]:
            circuit.rzz(gamma2, q0, q1)

        circuit.barrier(*range(6))

        # Cross-subsystem XX interactions.
        for a, b in zip(range(3), range(3, 6)):
            circuit.rxx(alpha1, a, b)

        circuit.barrier(*range(6))

        # Cross-subsystem ZZ interactions.
        for a, b in zip(range(3), range(3, 6)):
            circuit.rzz(alpha2, a, b)

        circuit.barrier(*range(6))

        return circuit

    def get_state_vector(self) -> Statevector:
        return Statevector(self.circuit)

    def as_qiskit_circuit(self) -> QuantumCircuit:
        return self.circuit

