from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace, state_fidelity
import numpy as np

from ground_truth import GibbsState, TFDState 
from tfd_circuit import TFDCircuit
from cost_function import CostFunction

X = [[0, 1], [1, 0]]
Z = [[1, 0], [0, -1]]
I = [[1, 0], [0, 1]]
ZZ_A = np.kron(Z, Z)
XI_A = np.kron(X, I)
IX_A = np.kron(I, X)
H = ZZ_A + XI_A + IX_A
H_AA = np.linalg.eigh(H)

betas = [0.02, 0.2, 1, 5, 50] 

for beta in betas:
    gibbs = GibbsState(H_AA)
    gibbs_state = gibbs.gibbs_density_matrix(beta=beta)

    tfd = TFDState(H_AA, H_AA)
    tfd_state = tfd.tfd_state_vector(beta=beta)
    sv_tfd = Statevector(tfd_state)

    tfd_traceout = partial_trace(sv_tfd, [2, 3])
    fidelity = state_fidelity(DensityMatrix(gibbs_state), tfd_traceout)
    print(f"beta={beta}: fidelity = {fidelity}")
    print("Gibbs state:", gibbs_state)
    print("TFD state:", tfd_traceout)


cost_function = CostFunction(varsigma=1.57)
x0 = np.array([0.0, 0.0, 0.0, 0.0])
results = {0: x0.copy()}

for beta in betas:
    angles = cost_function.minimize_cost(x0, beta)
    x0 = angles
    results[beta] = angles
    print(f"beta={beta}: angles={angles}")

for beta, angles in results.items():
    g1, g2, a1, a2 = angles
    qc = TFDCircuit(g1, g2, a1, a2)
    sv = qc.get_state_vector()
    rho_A = partial_trace(sv, [2, 3])

    target = DensityMatrix(gibbs.gibbs_density_matrix(beta))
    fid = state_fidelity(rho_A, target)
    print(f"beta={beta}: fidelity to true Gibbs state = {fid}")