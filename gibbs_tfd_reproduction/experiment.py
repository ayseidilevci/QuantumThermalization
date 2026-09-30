import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector, DensityMatrix, partial_trace, state_fidelity
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator
from ground_truth import GibbsState, TFDState, default_subsystem_hamiltonian
from tfd_circuit import TFDCircuit
from cost_function import CostFunction
from tomography import run_tomography, reconstruct_density_matrix, plot_density_matrix
from qi_utilities.device_simulation.simulators import NoisySimulator

H_AA = default_subsystem_hamiltonian()

betas = [0.01, 0.1, 0.25, 0.5, 1, 2, 3, 4, 5]

def naive_lower_bound(beta, varsigma=1.57):
    w = beta ** (-varsigma)
    # <X_A> = <X_B> = -2   (each is a sum of two single-qubit X terms)
    # <ZZ_A> = <ZZ_B> = -1
    # <XX_BA> = <ZZ_BA> = +2   (each is a sum of two terms)
    return (-2) + (-2) + varsigma * (-1 + -1) - w * (2 + 2)

def cost_lower_bound(beta, varsigma=1.57):
    w = beta ** (-varsigma)
    terms = (
        [("X", [q], 1.0) for q in range(4)]                      # X_A + X_B
        + [("ZZ", [0, 1], varsigma), ("ZZ", [2, 3], varsigma)]   # ZZ_A, ZZ_B
        + [("XX", [0, 2], -w), ("XX", [1, 3], -w)]               # XX_BA
        + [("ZZ", [0, 2], -w), ("ZZ", [1, 3], -w)]               # ZZ_BA
    )
    op = SparsePauliOp.from_sparse_list(terms, num_qubits=4)
    return np.linalg.eigvalsh(op.to_matrix())[0]


for beta in betas:
    gibbs = GibbsState(H_AA)
    gibbs_state = gibbs.gibbs_density_matrix(beta=beta)

    tfd = TFDState(H_AA, H_AA)
    tfd_state = tfd.tfd_state_vector(beta=beta)
    sv_tfd = Statevector(tfd_state)

    tfd_traceout = partial_trace(sv_tfd, [2, 3])
    fidelity = state_fidelity(DensityMatrix(gibbs_state), tfd_traceout)


backend = NoisySimulator("Starmon-7", ideal_simulation=True)
initial_layout = [0, 2, 3, 5]
cost_function = CostFunction(varsigma=1.57, backend=backend)
x0 = np.array([0.1, 0.1, 0.1, 0.1])
results = {0: x0.copy()}

for beta in betas:
    cost_function.cost_history = []
    cost_function.circuit_pdf = PdfPages(f"plots/circuits_beta{beta}.pdf")

    cost_function.circuit_pdf = PdfPages(f"plots/circuits_beta{beta}.pdf")
    angles = cost_function.minimize_cost(x0, beta)
    cost_function.circuit_pdf.close()

    results[beta] = angles
    x0 = angles
    print(f"beta={beta}: angles={angles}")

    c_min = cost_lower_bound(beta, varsigma=1.57)
    c_naive = naive_lower_bound(beta, varsigma=1.57)

    fig, ax = plt.subplots()
    ax.plot(range(1, len(cost_function.cost_history) + 1), cost_function.cost_history, marker='o', label="Optimizer")
    ax.axhline(c_min, color='r', ls='--', label=f"Exact lower bound (eigenvalue) = {c_min:.3f}")
    #ax.axhline(c_naive, color='k', ls=':', label=f"Naive bound (each term at its extreme) = {c_naive:.3f}")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Function evaluation")
    ax.set_ylabel("Cost C(beta)")
    ax.set_title(f"Cost convergence, beta={beta}")
    ax.legend()
    fig.savefig(f"plots/cost_beta{beta}.png", dpi=120)
    plt.close(fig)

    g1, g2, a1, a2 = angles
    corrected_probs_A, settings_A = run_tomography(backend, initial_layout, g1, g2, a1, a2, subsystem_qubits=(0, 1), shots=8192)
    rho_A = reconstruct_density_matrix(corrected_probs_A, settings_A, subsystem_qubits=(0, 1))
    plot_density_matrix(rho_A, title=f"System A, beta={beta}", save_path=f"plots/tomography_beta{beta}_A.png")

    corrected_probs_B, settings_B = run_tomography(backend, initial_layout, g1, g2, a1, a2, subsystem_qubits=(2, 3), shots=8192)
    rho_B = reconstruct_density_matrix(corrected_probs_B, settings_B, subsystem_qubits=(2, 3))
    plot_density_matrix(rho_B, title=f"System B, beta={beta}", save_path=f"plots/tomography_beta{beta}_B.png")    

for beta, angles in results.items():
    g1, g2, a1, a2 = angles
    qc = TFDCircuit(g1, g2, a1, a2)
    sv = qc.get_state_vector()
    rho_A = partial_trace(sv, [2, 3])

    target = DensityMatrix(gibbs.gibbs_density_matrix(beta))
    fid = state_fidelity(rho_A, target)
    print(f"beta={beta}: fidelity to true Gibbs state = {fid}")
