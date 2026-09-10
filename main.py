import sys
import numpy as np

from PySide6.QtWidgets import QApplication, QStackedWidget

from backend.qi_manager import QuantumInspireManager
from frontend.login import LoginWindow
from frontend.home import HomeWindow

from qiskit import transpile
from qiskit.quantum_info import DensityMatrix, partial_trace, state_fidelity
import numpy as np

from gibbs_tfd_reproduction.tfd_circuit import TFDCircuit
from gibbs_tfd_reproduction.cost_function import CostFunction
from gibbs_tfd_reproduction.ground_truth import GibbsState 

from io import open_code
from qi_utilities.utility_functions.data_handling import StoreProjectRecord, RetrieveProjectRecord

X = [[0, 1], [1, 0]]
Z = [[1, 0], [0, -1]]
I = [[1, 0], [0, 1]]
ZZ_A = np.kron(Z, Z)
XI_A = np.kron(X, I)
IX_A = np.kron(I, X)
H = ZZ_A + XI_A + IX_A
H_AA = np.linalg.eigh(H)

def run():

    app = QApplication(sys.argv)

    manager = QuantumInspireManager()

    window = QStackedWidget()

    login = LoginWindow(manager)
    home = HomeWindow()

    window.addWidget(login)
    window.addWidget(home)

    login.login_successful.connect(
        lambda: window.setCurrentWidget(home)
    )

    if manager.check_login():
        window.setCurrentWidget(home)
    else:
        window.setCurrentWidget(login)

    window.resize(1100, 700)
    window.show()

    # Center window
    screen = window.screen().availableGeometry()

    window.move(
        screen.center() - window.rect().center()
    )

    send_circuit(manager=manager)
    sys.exit(app.exec())


def send_circuit(manager):
    betas = [0.02, 0.2, 1, 5, 50] 
    cost_function = CostFunction(varsigma=1.57)
    x0 = np.array([0.0, 0.0, 0.0, 0.0])
    results = {0: x0.copy()}

    for beta in betas:
        angles = cost_function.minimize_cost(x0, beta)
        x0 = angles
        results[beta] = angles
        print(f"beta={beta}: angles={angles}")

    gibbs = GibbsState(H_AA)

    backend = manager.provider.get_backend("QX emulator")

    for beta, angles in results.items():
        g1, g2, a1, a2 = angles
        qc = TFDCircuit(g1, g2, a1, a2)

        sv = qc.get_state_vector()
        rho_A = partial_trace(sv, [2, 3])

        target = DensityMatrix(gibbs.gibbs_density_matrix(beta))
        fid = state_fidelity(rho_A, target)
        print(f"beta={beta}: fidelity to true Gibbs state = {fid}")

        job = manager.submit(transpile(qc.as_qiskit_circuit(), backend), backend)
        job_id = str(job.circuits_run_data[0].job_id)

        job_result = job.result()
        counts = job_result.data()["counts"]
        print(counts)

        StoreProjectRecord(job)

        loaded_job = RetrieveProjectRecord(job_id)
        loaded_results = loaded_job.get_counts()
        print(loaded_results)


if __name__ == "__main__":
    run()
