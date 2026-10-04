from qiskit.quantum_info import Pauli
from tfd_circuit import TFDCircuit
from qiskit.quantum_info import SparsePauliOp
import numpy as np

def exact_lower_bound(beta, varsigma=1.57):
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
