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

    # Backend is chosen on the prior screen (frontend/backend_select.py);
    # emitted when the user wants to change it without starting over.
    change_backend = Signal()

    def __init__(self, manager):
        super().__init__()
        self.manager = manager

        self._backend_name = None
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
        side.setSpacing(16)

        title = QLabel("Setup")
        title_font = QFont()
        title_font.setPointSize(15)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet(f"color: {Colors.TEXT};")
        side.addWidget(title)

        self._backend_name_lbl = QLabel(self.manager.STANDARD_BACKEND)
        self._backend_name_lbl.setStyleSheet(f"color: {Colors.TEXT}; font-weight: 600;")

        form = QFormLayout()
        form.setVerticalSpacing(16)

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

        form.addRow("Backend:", self._backend_name_lbl)
        form.addRow("Inverse temperature β:", self.beta)
        form.addRow("Optimizer:", self.optimizer)
        form.addRow("Iterations:", self.iterations)
        side.addLayout(form)

        angles_lbl = QLabel("Optimizer starting guess (γ, α)")
        angles_lbl.setStyleSheet(f"color: {Colors.TEXT_DIM}; font-weight: 600;")
        side.addWidget(angles_lbl)

        self.angles_group = QWidget()
        angle_form = QFormLayout(self.angles_group)
        angle_form.setContentsMargins(0, 0, 0, 0)
        angle_form.setVerticalSpacing(12)
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

        # No separate "Home" button here — the persistent top bar already
        # has one, visible on this screen. Saving a session lives on the
        # Run screen instead, next to the converged result — see
        # frontend/experiment.py.

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

        self.change_backend_btn = QPushButton("⇄  Change backend")
        self.change_backend_btn.setCursor(Qt.PointingHandCursor)
        self.change_backend_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; color: {Colors.TEXT_DIM};
                border: 1px solid {Colors.BORDER}; border-radius: 10px;
                padding: 8px;
            }}
            QPushButton:hover {{ color: {Colors.TEXT}; border-color: {Colors.ACCENT}; }}
        """)
        self.change_backend_btn.clicked.connect(self.change_backend.emit)
        side.addWidget(self.change_backend_btn)

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
                self._backend_name_lbl, "Backend",
                "The device you picked on the previous screen. Use "
                "the “Change backend” button below to go back and pick a "
                "different one."),
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
                "Pick subsystem A of the thermofield-double ansatz: drag a line "
                "from one qubit (A1) to a coupled neighbour (A2), or click them "
                "one by one. The mirrored subsystem B (B1 next to A1, B2 next to "
                "A2, B1–B2 coupled) is chosen automatically — or you're told "
                "it's impossible on this device."),
            TutorialStep(
                self.start_experiment_btn, "Start Experiment",
                "Once your qubit assignment is valid, click here to run."),
            TutorialStep(
                self.change_backend_btn, "Change backend",
                "Not the right device? Go back and pick a "
                "different one without losing these settings."),
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

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.tutorial.reposition()

    # -----------------------------------------------------
    # Backend / device panel wiring
    # -----------------------------------------------------

    def set_backend(self, name: str, qi_backend=None):
        """Called by main.py's navigation once a backend is picked on the
        previous screen (frontend/backend_select.py) — or when restoring a
        saved session, with qi_backend=None since that skips re-fetching
        the live object. Redraws the device panel's topology for it, which
        clears any qubit assignment; call before load_config() so a
        restored qubit_mapping isn't wiped out afterwards.
        """
        self._backend_name = name
        self._backend_name_lbl.setText(name)
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

        # Assumes set_backend() (for this config's backend_name) already
        # ran — see that method's docstring — so the device panel's
        # topology matches this qubit_mapping before it's restored.
        if config.qubit_mapping:
            self.device_panel.set_assignment(config.qubit_mapping)

    def set_tutorial_visible(self, visible: bool):
        if visible:
            self.tutorial.start()
        else:
            self.tutorial.stop()

    def current_config(self) -> SessionConfig:
        return SessionConfig(
            backend_name=self._backend_name or self.manager.STANDARD_BACKEND,
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
