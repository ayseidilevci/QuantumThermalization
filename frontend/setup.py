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
MIN_SUGGESTED_ITERATIONS = 100


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
        self.sidebar = sidebar
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
        self.iterations.setValue(SessionConfig.iterations)

        self.iterations_note = QLabel()
        self.iterations_note.setWordWrap(True)
        self.iterations.valueChanged.connect(self._update_iterations_note)
        self._update_iterations_note()

        form.addRow("Backend:", self._backend_name_lbl)
        form.addRow("Inverse temperature β:", self.beta_row)
        form.addRow("Boltzmann constant:", self.kb_unit)
        form.addRow(self.kb_note)
        form.addRow("Optimizer:", self.optimizer)
        form.addRow("Max iterations:", self.iterations)
        form.addRow(self.iterations_note)
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
                self.sidebar, "Welcome",
                "This tour walks through setting up a run. The app prepares a "
                "<b>thermofield-double state</b> whose half A is the Gibbs state at "
                "your chosen temperature, tuning the circuit's angles γ and α with a "
                "classical optimizer.<br><br>"
                "On the <b>Starmon-7 emulator</b> it runs a <b>chain experiment</b> "
                "(β ≈ 0 → 0.5 → 1 → your β, each step starting from the last). On a "
                "<b>real quantum computer</b> it runs a <b>cold start</b>: one "
                "optimization straight at your β from random angles, measured on the "
                "device itself."),
            TutorialStep(
                self.hardware_banner, "Real quantum computer",
                "This banner appears whenever a real device is selected. Every cost "
                "evaluation then runs on that device: one job per optimizer iteration, "
                "each waiting in the device's queue, so runs can take much longer than "
                "on the emulator."),
            TutorialStep(
                self._backend_name_lbl, "Backend",
                "The device you picked on the previous screen: the Starmon-7 emulator "
                "or a real Tuna device. Use “Change backend” below to pick another."),
            TutorialStep(
                self.beta_row, "Inverse temperature β",
                "The temperature of the Gibbs state you're preparing, as β = 1/(k<sub>B</sub>T) "
                "in units of 1/g (g is the Hamiltonian's coupling). Choose 0, 0.5, 1, 2, "
                "3, 4 or 5; the matching temperature is shown underneath. β = 0 is "
                "infinite temperature (a maximally mixed state), β = 5 is close to the "
                "ground state."),
            TutorialStep(
                self.kb_unit, "Boltzmann constant",
                "The units the temperature is shown in. <b>Natural</b> (k<sub>B</sub> = 1) "
                "gives T in units of g/k<sub>B</sub>. <b>eV/K</b>, <b>GHz/K</b> and "
                "<b>J/K</b> give kelvin, taking g as 1 of that energy unit; GHz/K is "
                "the natural scale for superconducting qubits. Display only, it doesn't "
                "change the run."),
            TutorialStep(
                self.optimizer, "Optimizer",
                "The classical method that adjusts γ and α to minimise the cost.<br>"
                "<b>COBYLA</b> builds a local model of the cost; fast and accurate "
                "on the noiseless emulator.<br>"
                "<b>SPSA</b> estimates the gradient from two measurements per step and "
                "copes far better with noise. Real hardware always uses SPSA; picking "
                "COBYLA there switches back with a note."),
            TutorialStep(
                self.iterations, "Max iterations",
                "The most optimizer steps per β step (default 200). It's an upper "
                "limit: COBYLA stops earlier once it converges, which is fine. SPSA "
                "has no convergence test, so it always uses the full budget. At "
                "least 100 is suggested; fewer may stop before converging.<br><br>"
                "On real hardware each iteration is one job, plus 4 extra jobs "
                "(2 for SPSA's calibration, 1 for the final state, 1 for the final "
                "counts), each waiting in the device's queue."),
            TutorialStep(
                self.angles_group, "Starting guess (γ, α)",
                "Where the optimizer starts on the emulator; it adjusts the angles "
                "itself during the run. Avoid all zeros, a saddle point of the cost.<br><br>"
                "On real hardware these are disabled: the cold start draws random "
                "angles (each between 0.1 and π in size), shown on the Run screen."),
            TutorialStep(
                self.device_panel, "Device panel",
                "Pick subsystem A of the thermofield-double ansatz: drag a line "
                "from one qubit (A1) to a coupled neighbour (A2), or click them "
                "one by one. The mirrored subsystem B (B1 next to A1, B2 next to "
                "A2, B1–B2 coupled) is chosen automatically — or you're told "
                "it's impossible on this device."),
            TutorialStep(
                self.start_experiment_btn, "Start Experiment",
                "Once your qubit assignment is valid, click here to run. The Run "
                "screen shows the fidelity to the target Gibbs state and the cost "
                "live, plus the measured density matrix.<br><br>"
                "For a real device it also shows the device's <b>status</b> (idle, "
                "busy, calibrating, benchmarking, offline) and a <b>Hardware jobs</b> "
                "tab following every job from Queued to Results received. "
                "<b>Stop</b> tries to delete the waiting job from the queue and tells "
                "you whether Quantum Inspire accepted."),
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

    def _update_iterations_note(self):
        low = self.iterations.value() < MIN_SUGGESTED_ITERATIONS
        self.iterations_note.setText(
            f"An upper limit: the run may stop earlier once it converges. "
            f"At least {MIN_SUGGESTED_ITERATIONS} is suggested"
            + (" — this may stop before converging." if low else "."))
        self.iterations_note.setStyleSheet(
            f"color: {Colors.WARN if low else Colors.TEXT_MUTED}; font-size: 10pt;")

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
