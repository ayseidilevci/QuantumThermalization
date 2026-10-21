"""Parameterized six-qubit TFD ansatz for the three-spin extension.

Qubit convention:
    q0=A1, q1=A2, q2=A3, q3=B1, q4=B2, q5=B3.

Each variational layer has four independent parameters:
    gamma1, gamma2, alpha1, alpha2.

The initial Bell pairs are prepared once, before all variational layers.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


class TFD3SpinCircuit:
    """Build a generalized p-layer TFD variational circuit."""

    N_QUBITS = 6
    PARAMETERS_PER_LAYER = 4

    def __init__(self, params, p_layers=1, periodic=False):
        self.p_layers = int(p_layers)
        self.periodic = bool(periodic)

        if self.p_layers < 1:
            raise ValueError("p_layers must be at least 1.")

        params = np.asarray(params, dtype=float)
        expected_parameters = self.PARAMETERS_PER_LAYER * self.p_layers

        if params.shape != (expected_parameters,):
            raise ValueError(
                f"Expected {expected_parameters} parameters for "
                f"p_layers={self.p_layers}, but received {params.size}."
            )

        self.params = params.copy()
        self.circuit = QuantumCircuit(self.N_QUBITS)
        self.build_circuit(self.circuit)

    def build_circuit(
        self,
        circuit: QuantumCircuit | None = None,
    ) -> QuantumCircuit:
        """Prepare Bell pairs and apply p independent variational layers."""

        if circuit is None:
            circuit = self.circuit

        # Prepare the three Bell pairs once.
        for a, b in zip(range(3), range(3, 6)):
            circuit.h(b)
            circuit.cx(b, a)

        circuit.barrier(*range(6))

        # Nearest-neighbour interactions within registers A and B.
        edges = [(0, 1), (1, 2), (3, 4), (4, 5)]

        if self.periodic:
            edges += [(2, 0), (5, 3)]

        # Apply p layers, each with four independent angles.
        for layer in range(self.p_layers):
            offset = self.PARAMETERS_PER_LAYER * layer

            gamma1, gamma2, alpha1, alpha2 = self.params[
                offset : offset + self.PARAMETERS_PER_LAYER
            ]

            # Single-qubit rotations.
            circuit.rx(gamma1, range(6))
            circuit.barrier(*range(6))

            # Interactions within A and B.
            for q1, q2 in edges:
                circuit.rzz(gamma2, q1, q2)

            circuit.barrier(*range(6))

            # Interactions between corresponding A and B spins.
            for a, b in zip(range(3), range(3, 6)):
                circuit.rxx(alpha1, a, b)

            circuit.barrier(*range(6))

            for a, b in zip(range(3), range(3, 6)):
                circuit.rzz(alpha2, a, b)

            circuit.barrier(*range(6))

        return circuit

    def get_state_vector(self) -> Statevector:
        return Statevector(self.circuit)

    def as_qiskit_circuit(self) -> QuantumCircuit:
        return self.circuit
