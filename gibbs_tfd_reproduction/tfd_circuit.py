from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

class TFDCircuit:
    def __init__(self, gamma1, gamma2, alpha1, alpha2, qubits=4):
        # q0 = A2
        # q1 = A1
        # q2 = B2
        # q3 = B1
        qc = QuantumCircuit(qubits)
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.circuit = self.build_circuit(qc)


    def build_circuit(self, circuit):
        circuit.h([2, 3])
        circuit.cx(3, 1)
        circuit.cx(2, 0)
        circuit.rx(self.gamma1, [0, 1, 2, 3])
        circuit.rzz(self.gamma2, 2, 3)
        circuit.rzz(self.gamma2, 0, 1)
        circuit.rxx(self.alpha1, 1, 3)
        circuit.rxx(self.alpha1, 0, 2)
        circuit.rzz(self.alpha2, 1, 3)
        circuit.rzz(self.alpha2, 0, 2)
        return circuit

    def get_state_vector(self):
        return Statevector(self.circuit)
    
    def as_qiskit_circuit(self) -> QuantumCircuit:
        return self.circuit