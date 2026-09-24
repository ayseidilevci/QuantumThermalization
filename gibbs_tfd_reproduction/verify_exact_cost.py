from qiskit.quantum_info import Pauli
from tfd_circuit import TFDCircuit

def exact_cost(g1, g2, a1, a2, beta, varsigma=1.57):
    tfd = TFDCircuit(g1, g2, a1, a2)
    sv = tfd.get_state_vector()
    exp_X_A = sv.expectation_value(Pauli("IIXI")).real + sv.expectation_value(Pauli("IIIX")).real
    exp_X_B = sv.expectation_value(Pauli("XIII")).real + sv.expectation_value(Pauli("IXII")).real
    exp_ZZ_A = sv.expectation_value(Pauli("IIZZ")).real
    exp_ZZ_B = sv.expectation_value(Pauli("ZZII")).real
    exp_XX_BA = sv.expectation_value(Pauli("XIXI")).real + sv.expectation_value(Pauli("IXIX")).real
    exp_ZZ_BA = sv.expectation_value(Pauli("ZIZI")).real + sv.expectation_value(Pauli("IZIZ")).real
    return exp_X_A + exp_X_B + varsigma * (exp_ZZ_A + exp_ZZ_B) - beta**(-varsigma) * (exp_XX_BA + exp_ZZ_BA)

print("cost at a nearby point:", exact_cost(0.1,0.1,0.1,0.1, beta=5))