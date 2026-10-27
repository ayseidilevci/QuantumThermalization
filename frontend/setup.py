from PySide6.QtCore import Qt, Signal, QSignalBlocker
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QPushButton,
    QComboBox, QDoubleSpinBox, QSpinBox, QMessageBox,
)

from backend import starmon7

from frontend.colors import Colors
from frontend.session import (
    SessionConfig, DEFAULT_START_ANGLE, BETA_CHOICES, BETA_ZERO, KB_UNITS,
    DEFAULT_KB_UNIT, HARDWARE_OPTIMIZER, format_beta, format_temperature,
    nearest_beta_choice,
)
from frontend.tutorial import TutorialCoach, TutorialStep
from frontend.device_panel import DevicePanel

SIDEBAR_WIDTH = 300

# Gradient-free optimizers with a per-iteration callback: COBYLA runs via
# scipy.optimize.minimize, SPSA via backend/spsa.py (scipy has none). Real
# hardware runs are restricted to HARDWARE_OPTIMIZER.
OPTIMIZER_METHODS = ("COBYLA", "SPSA")
_ANGLES_TITLE = "Optimizer starting guess (γ, α)"


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
        self._real_hardware = False

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

        # Shown only while a real device is selected (see set_backend).
        self.hardware_banner = QLabel()
        self.hardware_banner.setWordWrap(True)
        self.hardware_banner.setStyleSheet(f"""
            color: {Colors.TEXT}; background: rgba(254, 138, 39, 0.12);
            border: 1px solid {Colors.WARN}; border-radius: 8px; padding: 8px 10px;
        """)
        self.hardware_banner.hide()
        side.addWidget(self.hardware_banner)

        self._backend_name_lbl = QLabel(self.manager.STANDARD_BACKEND)
        self._backend_name_lbl.setStyleSheet(f"color: {Colors.TEXT}; font-weight: 600;")

        form = QFormLayout()
        form.setVerticalSpacing(16)

        self.beta = QComboBox()
        for choice in BETA_CHOICES:
            self.beta.addItem(format_beta(choice), choice)
        self.beta.setCurrentIndex(BETA_CHOICES.index(1.0))

        self.temperature_lbl = QLabel()
        self.temperature_lbl.setStyleSheet(f"color: {Colors.ACCENT}; font-weight: 600;")

        self.beta_row = QWidget()
        beta_layout = QVBoxLayout(self.beta_row)
        beta_layout.setContentsMargins(0, 0, 0, 0)
        beta_layout.setSpacing(6)
        beta_layout.addWidget(self.beta)
        beta_layout.addWidget(self.temperature_lbl)

        self.kb_unit = QComboBox()
        for key, (label, _, _, _) in KB_UNITS.items():
            self.kb_unit.addItem(label, key)

        self.kb_note = QLabel()
        self.kb_note.setWordWrap(True)
        self.kb_note.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 10pt;")

        self.beta.currentIndexChanged.connect(self._update_temperature)
        self.kb_unit.currentIndexChanged.connect(self._update_temperature)
        self._update_temperature()

        self.optimizer = QComboBox()
        self.optimizer.addItems(OPTIMIZER_METHODS)
        self.optimizer.currentTextChanged.connect(self._on_optimizer_changed)

        self.iterations = QSpinBox()
        self.iterations.setRange(5, 2000)
        self.iterations.setValue(100)

        form.addRow("Backend:", self._backend_name_lbl)
        form.addRow("Inverse temperature β:", self.beta_row)
        form.addRow("Boltzmann constant:", self.kb_unit)
        form.addRow(self.kb_note)
        form.addRow("Optimizer:", self.optimizer)
        form.addRow("Max iterations:", self.iterations)
        side.addLayout(form)

        self.angles_lbl = QLabel(_ANGLES_TITLE)
        self.angles_lbl.setWordWrap(True)
        self.angles_lbl.setStyleSheet(f"color: {Colors.TEXT_DIM}; font-weight: 600;")
        side.addWidget(self.angles_lbl)

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
                self.beta_row, "Inverse temperature β",
                "The target inverse temperature of the Gibbs state you're preparing, "
                "in units of 1/g. The matching temperature T = 1/(k_B β) is shown "
                "next to it; β = 0 is infinite temperature."),
            TutorialStep(
                self.kb_unit, "Boltzmann constant",
                "Which units T is shown in. Natural units (k_B = 1) give T in "
                "units of g/k_B; the others give kelvin, taking the coupling g "
                "as 1 of that energy unit. Display only — it doesn't change the run."),
            TutorialStep(
                self.optimizer, "Optimizer",
                "The classical method that adjusts γ/α so the prepared state's cost "
                "function converges for this β. Real hardware always uses SPSA, "
                "which copes best with noisy measurements."),
            TutorialStep(
                self.iterations, "Max iterations",
                "The most optimizer steps allowed per β step. It may stop earlier "
                "if it converges."),
            TutorialStep(
                self.angles_group, "Starting guess (γ, α)",
                "The ansatz's variational angles. These are only the optimizer's "
                "starting guess — it adjusts them itself during the run. On real "
                "hardware the run starts from random angles instead (cold start)."),
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

    def _beta_value(self) -> float:
        choice = self.beta.currentData()
        return BETA_ZERO if choice == 0 else choice

    def _update_temperature(self):
        unit = self.kb_unit.currentData()
        self.temperature_lbl.setText(format_temperature(self._beta_value(), unit))
        _, k_b_text, _, energy_unit = KB_UNITS[unit]
        self.kb_note.setText(
            f"{k_b_text}.  T = 1/(k_B β), in units of the coupling g." if energy_unit is None
            else f"{k_b_text}.  T = g/(k_B β), taking g = 1 {energy_unit}.")

    def _angle_spinbox(self) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(-6.283, 6.283)
        box.setSingleStep(0.05)
        box.setDecimals(3)
        box.setValue(DEFAULT_START_ANGLE)
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
        self._real_hardware = not starmon7.is_starmon7(name)
        self._apply_hardware_mode()

    def _apply_hardware_mode(self):
        """Real devices run a cold start from random angles with SPSA (see
        OptimizationWorker.run); lock the controls that don't apply."""
        real = self._real_hardware
        self.hardware_banner.setText(
            f"<b>Real quantum computer.</b> This experiment will be sent to "
            f"{self._backend_name}. It runs a cold start at the chosen β from "
            f"random angles, using {HARDWARE_OPTIMIZER}.")
        self.hardware_banner.setVisible(real)
        self.angles_group.setEnabled(not real)
        self.angles_lbl.setText(
            f"{_ANGLES_TITLE} — random on real hardware" if real else _ANGLES_TITLE)
        if real and self.optimizer.currentText() != HARDWARE_OPTIMIZER:
            with QSignalBlocker(self.optimizer):
                self.optimizer.setCurrentText(HARDWARE_OPTIMIZER)

    def _on_optimizer_changed(self, name: str):
        if not self._real_hardware or name == HARDWARE_OPTIMIZER:
            return
        QMessageBox.information(
            self, f"{HARDWARE_OPTIMIZER} is used on real hardware",
            f"{HARDWARE_OPTIMIZER} is suggested for noisy setups like a real quantum "
            f"computer: it estimates the gradient from just two noisy measurements "
            f"per step, where {name} is easily misled by shot noise.\n\n"
            f"Real hardware runs therefore always use {HARDWARE_OPTIMIZER}. Pick "
            f"the Starmon-7 emulator to use {name}.")
        with QSignalBlocker(self.optimizer):
            self.optimizer.setCurrentText(HARDWARE_OPTIMIZER)

    def _on_mapping_changed(self, mapping: dict):
        self._qubit_mapping = mapping
        self.start_experiment_btn.setEnabled(bool(mapping))

    # -----------------------------------------------------
    # Config in / out
    # -----------------------------------------------------

    def load_config(self, config: SessionConfig = None):
        config = config or SessionConfig()

        self.beta.setCurrentIndex(BETA_CHOICES.index(nearest_beta_choice(config.beta)))
        unit_idx = self.kb_unit.findData(config.kb_unit)
        self.kb_unit.setCurrentIndex(unit_idx if unit_idx >= 0 else
                                     self.kb_unit.findData(DEFAULT_KB_UNIT))
        idx = self.optimizer.findText(config.optimizer)
        with QSignalBlocker(self.optimizer):
            self.optimizer.setCurrentIndex(idx if idx >= 0 else 0)
        self.iterations.setValue(config.iterations)
        self.gamma1.setValue(config.gamma1)
        self.gamma2.setValue(config.gamma2)
        self.alpha1.setValue(config.alpha1)
        self.alpha2.setValue(config.alpha2)
        self._apply_hardware_mode()

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
            beta=self._beta_value(),
            kb_unit=self.kb_unit.currentData(),
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
