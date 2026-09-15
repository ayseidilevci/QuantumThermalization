import sys
import numpy as np

from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QStackedWidget,
)

from backend.qi_manager import QuantumInspireManager

from frontend.colors import Colors
from frontend.topbar import TopBar
from frontend.login import LoginWindow
from frontend.home import HomeWindow
from frontend.experiment import ExperimentWindow

def run():

    app = QApplication(sys.argv)

    # Global application style
    app.setStyleSheet(f"""
        QWidget {{
            background-color: {Colors.BG_1};
            color: {Colors.TEXT};
        }}
    """)

    # Backend
    manager = QuantumInspireManager()

    # Main window
    window = QWidget()

    window.resize(1100, 700)

    layout = QVBoxLayout(window)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)

    topbar = TopBar()

    layout.addWidget(topbar)

    stack = QStackedWidget()

    login = LoginWindow(manager)
    home = HomeWindow(manager)
    experiment = ExperimentWindow(manager)

    stack.addWidget(login)
    stack.addWidget(home)
    stack.addWidget(experiment)

    layout.addWidget(stack)

    login.login_successful.connect(
        lambda: stack.setCurrentWidget(home)
    )

    # Home -> Experiment
    home.experiment_requested.connect(
        lambda: stack.setCurrentWidget(experiment)
    )

    # Experiment -> Home
    experiment.go_home.connect(
        lambda: stack.setCurrentWidget(home)
    )


    if manager.check_login():
        stack.setCurrentWidget(home)
    else:
        stack.setCurrentWidget(login)

    window.show()

    screen = window.screen().availableGeometry()

    window.move(
        screen.center() - window.rect().center()
    )

    # send_circuit(manager=manager)
    sys.exit(app.exec())


# def send_circuit(manager):
#     betas = [0.02, 0.2, 1, 5, 50] 
#     cost_function = CostFunction(varsigma=1.57)
#     x0 = np.array([0.0, 0.0, 0.0, 0.0])
#     results = {0: x0.copy()}
#
#     for beta in betas:
#         angles = cost_function.minimize_cost(x0, beta)
#         x0 = angles
#         results[beta] = angles
#         print(f"beta={beta}: angles={angles}")
#
#     gibbs = GibbsState(H_AA)
#
#     backend = manager.provider.get_backend("QX emulator")
#
#     for beta, angles in results.items():
#         g1, g2, a1, a2 = angles
#         qc = TFDCircuit(g1, g2, a1, a2)
#
#         sv = qc.get_state_vector()
#         rho_A = partial_trace(sv, [2, 3])
#
#         target = DensityMatrix(gibbs.gibbs_density_matrix(beta))
#         fid = state_fidelity(rho_A, target)
#         print(f"beta={beta}: fidelity to true Gibbs state = {fid}")
#
#         job = manager.submit(transpile(qc.as_qiskit_circuit(), backend), backend)
#         job_id = str(job.circuits_run_data[0].job_id)
#
#         job_result = job.result()
#         counts = job_result.data()["counts"]
#         print(counts)
#
#         StoreProjectRecord(job)
#
#         loaded_job = RetrieveProjectRecord(job_id)
#         loaded_results = loaded_job.get_counts()
#         print(loaded_results)


if __name__ == "__main__":
    run()
