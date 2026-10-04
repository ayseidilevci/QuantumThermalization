import numpy as np
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace, state_fidelity
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from ground_truth import GibbsState, TFDState, default_subsystem_hamiltonian
from tfd_circuit import TFDCircuit
from cost_function import CostFunction
from tomography import run_tomography, reconstruct_density_matrix, plot_tomography_grid
from qi_utilities.device_simulation.simulators import NoisySimulator


H_AA = default_subsystem_hamiltonian()

betas = [0.00000001, 1, 5] 

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

    lower_bound = CostFunction.exact_lower_bound(beta, cost_function.varsigma)

    
    fig, ax = plt.subplots()
    ax.plot(cost_function.cost_history, marker='o', label="Cost C(beta)")
    ax.axhline(lower_bound, color='red', linestyle='--', label=f"theoretical lower bound = {lower_bound:.2f}")
    ax.set_xlabel("Function evaluation")
    ax.set_ylabel("Cost C(beta)")
    ax.set_title(f"Cost convergence, beta={beta}")
    ax.legend()
    fig.savefig(f"plots/cost_beta{beta}.png", dpi=120)
    plt.close(fig)

    g1, g2, a1, a2 = angles
    classical_target = gibbs.gibbs_density_matrix(beta=beta)

    corrected_probs_A, settings_A = run_tomography(backend, initial_layout, g1, g2, a1, a2, subsystem_qubits=(0, 1), shots=8192)
    rho_A = reconstruct_density_matrix(corrected_probs_A, settings_A, subsystem_qubits=(0, 1))

    corrected_probs_B, settings_B = run_tomography(backend, initial_layout, g1, g2, a1, a2, subsystem_qubits=(2, 3), shots=8192)
    rho_B = reconstruct_density_matrix(corrected_probs_B, settings_B, subsystem_qubits=(2, 3))

    plot_tomography_grid(rho_A, rho_B, classical_target, beta,
                          save_path=f"plots/tomography_beta{beta}_full.png")
    
    print("rho_A:\n", np.round(rho_A, 4))
    print("rho_B:\n", np.round(rho_B, 4))
    print("max abs difference:", np.max(np.abs(rho_A - rho_B)))
    

for beta, angles in results.items():
    g1, g2, a1, a2 = angles
    qc = TFDCircuit(g1, g2, a1, a2)
    sv = qc.get_state_vector()
    rho_A = partial_trace(sv, [2, 3])

    target = DensityMatrix(gibbs.gibbs_density_matrix(beta))
    fid = state_fidelity(rho_A, target)
    print(f"beta={beta}: fidelity to true Gibbs state = {fid}")