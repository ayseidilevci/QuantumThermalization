from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QPushButton,
    QComboBox, QDoubleSpinBox, QSpinBox,
)

from frontend.colors import Colors
from frontend.session import SessionConfig
from frontend.tutorial import TutorialCoach, TutorialStep
from frontend.device_panel import DevicePanel

SIDEBAR_WIDTH = 300

# scipy.optimize.minimize methods that (a) support gradient-free
# optimization of the 4 ansatz angles and (b) accept a per-iteration
# `callback` — verified against the scipy version this project pins.
# "SPSA" (used as a placeholder option in the `gui` branch's tfd_gui.py) is
# NOT a scipy.optimize.minimize method and has been left out rather than
# faked; COBYLA matches gibbs_tfd_reproduction/cost_function.py's previous
# hardcoded default.
OPTIMIZER_METHODS = ("COBYLA", "Nelder-Mead", "Powell")


def _field_style():
    return f"""
        QComboBox, QDoubleSpinBox, QSpinBox {{
            background: {Colors.BG_3};
            border: 1px solid {Colors.BORDER};
            border-radius: 8px;
            padding: 5px 8px;
            color: {Colors.TEXT};
        }}
        QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus {{
            border: 1px solid {Colors.ACCENT};
        }}
        QLabel {{ color: {Colors.TEXT_DIM}; }}
    """


class SetupWindow(QWidget):

    run_requested = Signal(object)   # SessionConfig
    go_home = Signal()

    def __init__(self, manager):
        super().__init__()
        self.manager = manager

        self._backend_infos = {}     # name -> BackendInfo
        self._backends_loaded = False
        self._pending_backend_name = None
        self._pending_qubit_mapping = None
        self._qubit_mapping = {}

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # -------------------------------------------------
        # Left sidebar — parameters
        # -------------------------------------------------

        sidebar = QWidget()
        sidebar.setFixedWidth(SIDEBAR_WIDTH)
        sidebar.setStyleSheet(f"background-color: {Colors.BG_2};")
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(20, 28, 20, 24)
        side.setSpacing(10)

        title = QLabel("Setup")
        title_font = QFont()
        title_font.setPointSize(15)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet(f"color: {Colors.TEXT};")
        side.addWidget(title)
        side.addSpacing(6)

        form = QFormLayout()
        form.setVerticalSpacing(10)

        self.backend = QComboBox()
        self.backend.addItem(self.manager.STANDARD_BACKEND)
        self.backend.currentTextChanged.connect(self._on_backend_changed)

        self.backend_status_lbl = QLabel("")
        self.backend_status_lbl.setWordWrap(True)
        self.backend_status_lbl.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 8.5pt;")

        self.beta = QDoubleSpinBox()
        self.beta.setRange(0.01, 50.0)
        self.beta.setSingleStep(0.1)
        self.beta.setDecimals(2)
        self.beta.setValue(1.0)

        self.optimizer = QComboBox()
        self.optimizer.addItems(OPTIMIZER_METHODS)

        self.iterations = QSpinBox()
        self.iterations.setRange(5, 2000)
        self.iterations.setValue(100)

        form.addRow("Backend:", self.backend)
        form.addRow(self.backend_status_lbl)
        form.addRow("Inverse temperature β:", self.beta)
        form.addRow("Optimizer:", self.optimizer)
        form.addRow("Iterations:", self.iterations)
        side.addLayout(form)

        side.addSpacing(4)
        angles_lbl = QLabel("Optimizer starting guess (γ, α)")
        angles_lbl.setStyleSheet(f"color: {Colors.TEXT_DIM}; font-weight: 600;")
        side.addWidget(angles_lbl)

        self.angles_group = QWidget()
        angle_form = QFormLayout(self.angles_group)
        angle_form.setContentsMargins(0, 0, 0, 0)
        angle_form.setVerticalSpacing(8)
        self.gamma1 = self._angle_spinbox()
        self.gamma2 = self._angle_spinbox()
        self.alpha1 = self._angle_spinbox()
        self.alpha2 = self._angle_spinbox()
        angle_form.addRow("γ₁:", self.gamma1)
        angle_form.addRow("γ₂:", self.gamma2)
        angle_form.addRow("α₁:", self.alpha1)
        angle_form.addRow("α₂:", self.alpha2)
        side.addWidget(self.angles_group)

        sidebar.setStyleSheet(sidebar.styleSheet() + _field_style())

        side.addStretch()

        home_btn = QPushButton("←  Home")
        home_btn.setCursor(Qt.PointingHandCursor)
        home_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; color: {Colors.TEXT_DIM};
                border: 1px solid {Colors.BORDER}; border-radius: 10px;
                padding: 8px;
            }}
            QPushButton:hover {{ color: {Colors.TEXT}; border-color: {Colors.ACCENT}; }}
        """)
        home_btn.clicked.connect(self.go_home.emit)
        side.addWidget(home_btn)

        self.start_experiment_btn = QPushButton("▶  Start Experiment")
        self.start_experiment_btn.setMinimumHeight(46)
        self.start_experiment_btn.setEnabled(False)
        self.start_experiment_btn.setCursor(Qt.PointingHandCursor)
        self.start_experiment_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.ACCENT}; color: #06101C; border: none;
                border-radius: 10px; font-weight: bold;
            }}
            QPushButton:hover {{ background: {Colors.ACCENT_HOVER}; }}
            QPushButton:pressed {{ background: {Colors.ACCENT_PRESSED}; }}
            QPushButton:disabled {{ background: {Colors.BG_3}; color: {Colors.TEXT_MUTED}; }}
        """)
        self.start_experiment_btn.clicked.connect(self._on_start_experiment)
        side.addWidget(self.start_experiment_btn)

        root.addWidget(sidebar)

        # -------------------------------------------------
        # Right main area — device panel
        # -------------------------------------------------

        main = QWidget()
        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(28, 24, 28, 24)
        main_layout.setSpacing(12)

        device_title = QLabel("Device")
        device_title.setStyleSheet(f"color: {Colors.TEXT}; font-weight: bold; font-size: 12pt;")
        main_layout.addWidget(device_title)

        self.device_panel = DevicePanel()
        self.device_panel.mapping_changed.connect(self._on_mapping_changed)
        main_layout.addWidget(self.device_panel, stretch=1)

        root.addWidget(main, stretch=1)

        # -------------------------------------------------
        # Guided tutorial — a step-by-step coach mark over the controls
        # above, instead of one static wall of text (see frontend/tutorial.py)
        # -------------------------------------------------

        self.tutorial = TutorialCoach(self, steps=[
            TutorialStep(
                self.backend, "Backend",
                "Which Quantum Inspire device (or the QX emulator) runs the circuit."),
            TutorialStep(
                self.beta, "Inverse temperature β",
                "The target inverse temperature of the Gibbs state you're preparing."),
            TutorialStep(
                self.optimizer, "Optimizer",
                "The classical method that adjusts γ/α so the prepared state's cost "
                "function converges for this β."),
            TutorialStep(
                self.iterations, "Iterations",
                "The optimizer's step budget before it stops."),
            TutorialStep(
                self.angles_group, "Starting guess (γ, α)",
                "The ansatz's variational angles. These are only the optimizer's "
                "starting guess — it adjusts them itself during the run."),
            TutorialStep(
                self.device_panel, "Device panel",
                "Assign physical qubits to the roles A1/A2/B1/B2 in the "
                "thermofield-double ansatz. They must be wired the way the protocol "
                "needs: A1–A2, B1–B2, A1–B1 and A2–B2 all adjacent."),
            TutorialStep(
                self.start_experiment_btn, "Start Experiment",
                "Once your qubit assignment is valid, click here to run."),
        ])

    def _angle_spinbox(self) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(-6.283, 6.283)
        box.setSingleStep(0.05)
        box.setDecimals(3)
        box.setValue(0.0)
        return box

    # -----------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------

    def showEvent(self, event):
        super().showEvent(event)
        if not self._backends_loaded:
            self._load_backends()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.tutorial.reposition()

    def _load_backends(self):
        self._backends_loaded = True
        try:
            self.manager.set_provider()
            infos = self.manager.backends
            self.backend_status_lbl.setText("")
        except Exception as exc:
            infos = []
            self.backend_status_lbl.setText(
                f"Couldn't fetch the live backend list ({exc}); "
                f"defaulting to {self.manager.STANDARD_BACKEND}.")

        self._backend_infos = {info.name: info for info in infos}
        names = list(self._backend_infos.keys()) or [self.manager.STANDARD_BACKEND]

        self.backend.blockSignals(True)
        self.backend.clear()
        self.backend.addItems(names)
        self.backend.blockSignals(False)

        self._apply_pending_backend()

    def _apply_pending_backend(self):
        name = self._pending_backend_name
        mapping = self._pending_qubit_mapping
        self._pending_backend_name = None
        self._pending_qubit_mapping = None

        if name:
            idx = self.backend.findText(name)
            if idx >= 0 and idx != self.backend.currentIndex():
                self.backend.setCurrentIndex(idx)
                # setCurrentIndex already triggered _on_backend_changed via
                # currentTextChanged; fall through to also restore qubits.
            else:
                self._on_backend_changed(self.backend.currentText())
        else:
            self._on_backend_changed(self.backend.currentText())

        if mapping:
            self.device_panel.set_assignment(mapping)

    # -----------------------------------------------------
    # Backend / device panel wiring
    # -----------------------------------------------------

    def _on_backend_changed(self, name: str):
        info = self._backend_infos.get(name)
        qi_backend = info.backend if info else None
        self.device_panel.set_backend(name, qi_backend)

    def _on_mapping_changed(self, mapping: dict):
        self._qubit_mapping = mapping
        self.start_experiment_btn.setEnabled(bool(mapping))

    # -----------------------------------------------------
    # Config in / out
    # -----------------------------------------------------

    def load_config(self, config: SessionConfig = None):
        config = config or SessionConfig()

        self.beta.setValue(config.beta)
        idx = self.optimizer.findText(config.optimizer)
        self.optimizer.setCurrentIndex(idx if idx >= 0 else 0)
        self.iterations.setValue(config.iterations)
        self.gamma1.setValue(config.gamma1)
        self.gamma2.setValue(config.gamma2)
        self.alpha1.setValue(config.alpha1)
        self.alpha2.setValue(config.alpha2)

        self._pending_backend_name = config.backend_name
        self._pending_qubit_mapping = config.qubit_mapping

        if self._backends_loaded:
            self._apply_pending_backend()
        # else: showEvent -> _load_backends() will apply it once the
        # backend list (and therefore the device panel's topology) exists.

    def set_tutorial_visible(self, visible: bool):
        if visible:
            self.tutorial.start()
        else:
            self.tutorial.stop()

    def current_config(self) -> SessionConfig:
        return SessionConfig(
            backend_name=self.backend.currentText(),
            beta=self.beta.value(),
            optimizer=self.optimizer.currentText(),
            iterations=self.iterations.value(),
            gamma1=self.gamma1.value(),
            gamma2=self.gamma2.value(),
            alpha1=self.alpha1.value(),
            alpha2=self.alpha2.value(),
            qubit_mapping=dict(self._qubit_mapping) if self._qubit_mapping else None,
        )

    def _on_start_experiment(self):
        if not self._qubit_mapping:
            return
        self.run_requested.emit(self.current_config())
