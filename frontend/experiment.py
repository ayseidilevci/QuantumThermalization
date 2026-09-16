import threading

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
)

from gibbs_tfd_reproduction.tfd_circuit import TFDCircuit
from qiskit import transpile

class ExperimentWindow(QWidget):

    go_home = Signal()

    # Worker -> GUI signals
    experiment_finished = Signal(bool)
    experiment_error = Signal(str)

    def __init__(self, manager):
        super().__init__()

        self.manager = manager

        self.setWindowTitle("Run Experiment")

        layout = QVBoxLayout(self)

        # -------------------------------------------------
        # Top bar
        # -------------------------------------------------

        top_bar = QHBoxLayout()

        title = QLabel("Run Experiment")

        self.start_button = QPushButton(
            "▶  START EXPERIMENT"
        )

        self.start_button.setFixedSize(200, 50)

        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #87CEEB;
                color: #06101C;
                border: none;
                border-radius: 10px;
                font-size: 15px;
                font-weight: bold;
            }

            QPushButton:hover {
                background-color: #A8DDF2;
            }

            QPushButton:pressed {
                background-color: #6BBBDD;
            }
        """)

        top_bar.addWidget(title)
        top_bar.addStretch()
        top_bar.addWidget(self.start_button)

        layout.addLayout(top_bar)

        # -------------------------------------------------
        # Circuit area
        # -------------------------------------------------

        self.circuit_label = QLabel(
            "Circuit will appear here"
        )

        self.circuit_label.setAlignment(
            Qt.AlignCenter
        )

        layout.addWidget(self.circuit_label)

        layout.addStretch()

        # -------------------------------------------------
        # Bottom bar
        # -------------------------------------------------

        bottom_bar = QHBoxLayout()

        bottom_bar.addStretch()

        self.home_button = QPushButton(
            "←  GO BACK HOME"
        )

        self.home_button.setFixedSize(180, 45)

        self.home_button.setStyleSheet("""
            QPushButton {
                background-color: #1F2A38;
                color: #7F8B95;
                border: none;
                border-radius: 10px;
                font-size: 14px;
                font-weight: bold;
            }

            QPushButton:hover {
                background-color: #2A3746;
                color: #FFFFFF;
            }

            QPushButton:pressed {
                background-color: #17212C;
            }
        """)

        bottom_bar.addWidget(self.home_button)

        layout.addLayout(bottom_bar)

        # -------------------------------------------------
        # Connections
        # -------------------------------------------------

        self.start_button.clicked.connect(
            self.start_experiment
        )

        self.home_button.clicked.connect(
            self.go_home.emit
        )

        self.experiment_finished.connect(
            self.handle_experiment_finished
        )

        self.experiment_error.connect(
            self.handle_experiment_error
        )

    def start_experiment(self):
        """
        Create the circuit and start the experiment worker.
        """

        print("\n========================================")
        print("STARTING EXPERIMENT")
        print("========================================")

        # Prevent multiple experiments being started
        # while one is already running.
        self.start_button.setEnabled(False)

        self.circuit_label.setText(
            "Preparing experiment..."
        )

        gamma1 = 0.5
        gamma2 = 0.5
        alpha1 = 0.5
        alpha2 = 0.5

        # Circuit construction is very fast, so this can
        # safely happen before starting the worker.
        try:
            print("\nCreating TFD circuit...")

            tfd = TFDCircuit(
                gamma1=gamma1,
                gamma2=gamma2,
                alpha1=alpha1,
                alpha2=alpha2,
                qubits=4,
            )

            print("\nTFD circuit created.")

        except Exception as exc:
            self.start_button.setEnabled(True)
            self.circuit_label.setText(
                "Failed to create circuit."
            )
            print("\nERROR creating circuit:")
            print(exc)
            return

        # Start the potentially blocking work in a thread.
        threading.Thread(
            target=self.run_experiment,
            args=(tfd,),
            daemon=True,
        ).start()

    def run_experiment(self, tfd):
        """
        Perform transpilation, backend selection, submission,
        and result waiting in a background thread.
        """

        try:
            print("\nSelecting backend...")

            backend = self.manager.get_backend()

            # TODO: MAKE THE BACKEND SELECTION
            # BEFORE OR IN ANOTHER WAY

            print("\nSelected backend:")
            print(backend)

            print("\nTranspiling circuit...")

            circuit = transpile(
                tfd.as_qiskit_circuit(),
                backend,
            )

            print("\nCircuit transpiled.")

            print("\nSubmitting circuit...")

            job, results = self.manager.submit_circuit(
                circuit,
                backend,
            ) 

            if job is not None:
                self.experiment_finished.emit(True)
                self.manager.save_job(job)
            else:
                self.experiment_finished.emit(False)

        except Exception as exc:
            print("\nERROR running experiment:")
            print(exc)

            self.experiment_error.emit(str(exc))

    def handle_experiment_finished(self, success):
        """
        Handle the experiment result on the GUI thread.
        """

        if success:
            self.circuit_label.setText(
                "Experiment completed successfully."
            )
        else:
            self.circuit_label.setText(
                "Experiment submission failed."
            )

        self.start_button.setEnabled(True)

        print("Experiment ended!")

    def handle_experiment_error(self, error):
        """
        Handle worker errors on the GUI thread.
        """

        self.circuit_label.setText(
            f"Experiment failed: {error}"
        )

        self.start_button.setEnabled(True)
