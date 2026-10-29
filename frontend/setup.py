import numpy as np
from PySide6.QtCore import QRegularExpression, QSignalBlocker, Qt, Signal
from PySide6.QtGui import QFont, QRegularExpressionValidator
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFormLayout, QFrame, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
    QScrollArea, QSpinBox, QVBoxLayout, QWidget,
)

from backend import backends
from frontend.colors import Colors
from frontend.device_panel import DevicePanel
from frontend.help_button import HELP_SIZE, HelpButton
from frontend.latex import set_tex, tex_label
from frontend.session import (
    BETA_MAX, BETA_ZERO, DEFAULT_START_ANGLE, HARDWARE_OPTIMIZER, MAX_ITERATIONS, MIN_ITERATIONS, SPIN_CHOICES,
    SessionConfig, clamp_beta,
    format_beta, format_temperature_tex, random_start_angles, spins_allowed,
)
from frontend.tutorial import TutorialCoach, TutorialStep

SIDEBAR_WIDTH = 300

# Gradient-free optimizers with a per-iteration callback: COBYLA runs via
# scipy.optimize.minimize, SPSA via backend/spsa.py (scipy has none). Real
# hardware runs are restricted to HARDWARE_OPTIMIZER.
OPTIMIZER_METHODS = ("COBYLA", "SPSA")
_OPTIMIZER_HELP = (
    "<b>Which optimizer?</b><br>"
    "<b>COBYLA</b>: no gradients; fits a local model to individual cost values and "
    "stops once it converges. Fast and precise when the cost is (nearly) noise-free, "
    "but noise easily misleads it.<br><br>"
    "<b>SPSA</b>: estimates the gradient from 2 cost measurements per step, however "
    "many angles there are, and its shrinking steps average noise away. The choice "
    "for noisy costs; always runs every iteration. Cloud runs always use it.")
# Plain decimal or scientific notation, "." or "," as the decimal mark.
BETA_PATTERN = r"[-+]?\d*([.,]\d*)?([eE][-+]?\d*)?"
_ANGLES_TITLE = "Starting angles:"
MIN_SUGGESTED_ITERATIONS = 100


def _field_style():
    return f"""
        QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit {{
            background: {Colors.BG_3};
            border: 1px solid {Colors.BORDER};
            border-radius: 8px;
            padding: 5px 8px;
            color: {Colors.TEXT};
        }}
        QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus, QLineEdit:focus {{
            border: 1px solid {Colors.ACCENT};
        }}
        QLabel {{ color: {Colors.TEXT_DIM}; }}
    """


def _group(title: str) -> tuple:
    """A titled settings group: (the group widget, the form to add rows to)."""
    box = QFrame()
    box.setObjectName("settingsBox")
    box.setStyleSheet(f"#settingsBox {{ background: transparent; border: 1px solid {Colors.BORDER}; "
                      f"border-radius: 10px; }}")
    layout = QVBoxLayout(box)
    layout.setContentsMargins(12, 10, 12, 12)
    layout.setSpacing(10)
    heading = QLabel(title.upper())
    heading.setStyleSheet(f"color: {Colors.ACCENT}; font-size: 9pt; font-weight: 700; "
                          f"letter-spacing: 1px; background: transparent;")
    layout.addWidget(heading)
    form = QFormLayout()
    form.setVerticalSpacing(12)
    layout.addLayout(form)
    return box, form


def _note_label() -> QLabel:
    label = QLabel()
    label.setWordWrap(True)
    label.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 9pt;")
    return label


def _with_help(widget: QWidget, help_button: HelpButton = None) -> QWidget:
    """`widget` with its "?" on the right. Rows without help get an empty slot
    of the same width, so every field in the form ends at the same edge."""
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(6)
    layout.addWidget(widget, stretch=1)
    if help_button is None:
        help_button = QWidget()
        help_button.setFixedSize(HELP_SIZE, HELP_SIZE)
    layout.addWidget(help_button, alignment=Qt.AlignTop)
    return row


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
        self._remote = False   # QX emulator or real hardware: jobs in the cloud

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

        # Shown while a cloud backend (QX emulator or real hardware) is
        # selected; see _apply_hardware_mode.
        self.hardware_banner = QLabel()
        self.hardware_banner.setWordWrap(True)
        self.hardware_banner.hide()
        side.addWidget(self.hardware_banner)

        self._backend_name_lbl = QLabel(self.manager.STANDARD_BACKEND)
        self._backend_name_lbl.setStyleSheet(f"color: {Colors.TEXT}; font-weight: 600;")

        # ---- Physical system: what state is prepared -----------------------
        self.spins = QComboBox()
        for n in SPIN_CHOICES:
            self.spins.addItem(str(n), n)
        self.spins.currentIndexChanged.connect(self._on_spins_changed)
        # Why this size, and what it means on this backend (see _update_spins_note).
        self.spins_help = HelpButton(width=320)

        # Any β in [BETA_ZERO, BETA_MAX], typed (see _on_beta_edited).
        self.beta = QLineEdit(format_beta(1.0))
        # Not QDoubleValidator: it follows the system locale, so with a comma decimal
        # mark (e.g. Dutch) "0.0000001" counts as unfinished and is never applied.
        self.beta.setValidator(QRegularExpressionValidator(QRegularExpression(BETA_PATTERN)))
        self.beta.setPlaceholderText(f"{format_beta(BETA_ZERO)} to {format_beta(BETA_MAX)}")
        self.beta.editingFinished.connect(self._on_beta_edited)
        self.temperature_lbl = QLabel()
        self.temperature_lbl.setStyleSheet(f"color: {Colors.ACCENT}; font-weight: 600;")
        self.beta_note = _note_label()
        self.beta_help = HelpButton()
        self.beta_help.set_help(
            r"$\beta = 1/(k_B T)$ is <b>unitless</b>: energies are measured in units of the "
            r"Hamiltonian's own scale and $k_B = 1$, so $\beta$ and $T = 1/\beta$ are plain "
            r"numbers.<br><br>"
            r"Type any value from $10^{{-8}}$ to {top}: values above {top} are set to {top}, "
            r"values below $10^{{-8}}$ to $10^{{-8}}$. Small $\beta$ is high temperature; "
            r"exactly 0 (infinite temperature) isn't allowed because the cost's "
            r"$\beta^{{-1.57}}$ term diverges there."
            .format(top=format_beta(BETA_MAX)))

        system_box, system = _group("Physical system")
        system.addRow("Backend:", self._backend_name_lbl)
        system.addRow("Spins per system:", _with_help(self.spins, self.spins_help))
        self.beta_row = _with_help(self.beta, self.beta_help)
        system.addRow(tex_label(r"Inverse temp. $\beta$:", Colors.TEXT_DIM), self.beta_row)
        system.addRow(self.temperature_lbl)
        system.addRow(self.beta_note)

        # ---- Optimization: how the angles are found -------------------------
        self.optimizer = QComboBox()
        self.optimizer.addItems(OPTIMIZER_METHODS)
        self.optimizer.currentTextChanged.connect(self._on_optimizer_changed)
        self.optimizer_help = HelpButton(width=340)
        self.optimizer_help.set_help(_OPTIMIZER_HELP)

        self.iterations = QSpinBox()
        self.iterations.setRange(MIN_ITERATIONS, MAX_ITERATIONS)
        # A typed value out of range (e.g. 2) snaps to the nearest limit (5)
        # instead of reverting to the previous value.
        self.iterations.setCorrectionMode(QSpinBox.CorrectToNearestValue)
        self.iterations.setValue(SessionConfig.iterations)
        self.iterations_help = HelpButton()
        self.iterations.valueChanged.connect(self._update_iterations_help)
        self._update_iterations_help()

        optimization_box, optimization = _group("Optimization")
        optimization.addRow("Optimizer:", _with_help(self.optimizer, self.optimizer_help))
        optimization.addRow("Max iterations:", _with_help(self.iterations, self.iterations_help))

        self.angles_lbl = QLabel()
        self.angles_lbl.setWordWrap(True)
        set_tex(self.angles_lbl, _ANGLES_TITLE, Colors.TEXT_DIM)
        self.randomize_btn = QPushButton("🎲 Randomize")
        self.randomize_btn.setCursor(Qt.PointingHandCursor)
        self.randomize_btn.setToolTip("Draw new starting angles near θ = 0 (normal, σ = 0.4)")
        self.randomize_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; color: {Colors.ACCENT}; border: 1px solid {Colors.ACCENT};
                border-radius: 7px; padding: 3px 8px; font-weight: 600;
            }}
            QPushButton:hover {{ background: rgba(135,206,235,0.12); }}
        """)
        self.randomize_btn.clicked.connect(self._randomize_angles)
        angles_head = QHBoxLayout()
        angles_head.setSpacing(6)
        angles_head.addWidget(self.angles_lbl, stretch=1)
        angles_head.addWidget(self.randomize_btn)
        optimization.addRow(angles_head)

        self.angles_group = QWidget()
        angle_form = QFormLayout(self.angles_group)
        angle_form.setContentsMargins(0, 0, 0, 0)
        angle_form.setVerticalSpacing(10)
        self.gamma1 = self._angle_spinbox()
        self.gamma2 = self._angle_spinbox()
        self.alpha1 = self._angle_spinbox()
        self.alpha2 = self._angle_spinbox()
        for tex, box in ((r"\gamma_1", self.gamma1), (r"\gamma_2", self.gamma2),
                         (r"\alpha_1", self.alpha1), (r"\alpha_2", self.alpha2)):
            angle_form.addRow(tex_label(f"${tex}$:", Colors.TEXT_DIM), box)
        optimization.addRow(self.angles_group)

        # The settings scroll if the window is short; the buttons stay put.
        settings = QWidget()
        settings_layout = QVBoxLayout(settings)
        settings_layout.setContentsMargins(0, 0, 4, 0)
        settings_layout.setSpacing(18)
        for box in (system_box, optimization_box):
            settings_layout.addWidget(box)
        settings_layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setStyleSheet("QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; }")
        scroll.setWidget(settings)
        side.addWidget(scroll, stretch=1)
        self._update_temperature()

        sidebar.setStyleSheet(sidebar.styleSheet() + _field_style())

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
        # Guided tutorial (see frontend/tutorial.py)
        # -------------------------------------------------

        self.tutorial = TutorialCoach(self, steps=[
            TutorialStep(
                self.sidebar, "Welcome",
                "This tour walks through setting up a run. The app prepares a "
                "<b>thermofield-double state</b> whose half A is the Gibbs state at "
                r"your chosen temperature, tuning the circuit's angles $\gamma$ and $\alpha$ with a "
                "classical optimizer.<br><br>"
                r"Every run is <b>one optimization straight at your $\beta$</b>, from random "
                "starting angles (on Starmon-7 you can also set them yourself). The settings "
                "come in two groups: the <b>physical system</b> (size and temperature) and the "
                "<b>optimization</b> (method, iterations, starting angles)."),
            TutorialStep(
                self.hardware_banner, "Cloud backend",
                "This banner appears when the QX emulator (blue) or a real quantum "
                "computer (orange) is selected. Every cost evaluation then runs there: "
                "one job per optimizer iteration, each waiting in a queue, so runs can "
                "take much longer than on Starmon-7."),
            TutorialStep(
                self._backend_name_lbl, "Backend",
                "The backend you picked on the previous screen: the Starmon-7 noisy "
                "simulator (on this computer), the noiseless QX emulator (in the cloud) "
                "or a real Tuna device. Use “Change backend” below to pick another."),
            TutorialStep(
                self.spins, "System size",
                "How many spins each system (A and its copy B) has: 2, 3 or 4, so 4, 6 or 8 "
                "qubits. The “?” next to it explains what the chosen size means on this "
                "backend.<br><br>"
                "<b>2</b> (4 qubits) is the original experiment and runs everywhere. "
                "<b>3</b> and <b>4</b> need a 2 × n ladder of coupled qubits, so they run "
                "on <b>Tuna-17</b> and the <b>QX emulator</b>; their fidelity is measured once, "
                "by tomography at the end ($3^n$ = 27 or 81 circuits)."),
            TutorialStep(
                self.beta_row, r"Inverse temperature $\beta$",
                r"The temperature of the Gibbs state you're preparing, $\beta = 1/(k_B T)$. It's "
                r"unitless: the Hamiltonian's energy scale and $k_B$ are 1. Type any value from $10^{-8}$ to 5: anything above 5 is set to 5, "
                r"anything below $10^{-8}$ to $10^{-8}$. Small $\beta$ is close to infinite "
                r"temperature (a maximally mixed state), $\beta = 5$ is close to the ground state."),
            TutorialStep(
                self.optimizer, "Optimizer",
                r"The classical method that adjusts $\gamma$ and $\alpha$ to minimise the cost. "
                "The “?” next to it says when to use which.<br>"
                "<b>COBYLA</b>: gradient-free, precise on a (nearly) noise-free cost.<br>"
                "<b>SPSA</b>: two measurements per step, copes best with noise; cloud runs "
                "always use it."),
            TutorialStep(
                self.iterations, "Max iterations",
                r"The most optimizer steps (default 200). It's an upper "
                "limit: COBYLA stops earlier once it converges, which is fine. SPSA "
                "has no convergence test, so it always uses the full budget. At "
                "least 100 is suggested; fewer may stop before converging.<br><br>"
                "In the cloud each iteration is one job, plus 4 extra jobs "
                "(2 for SPSA's calibration, 1 for the final state, 1 for the final "
                "counts), each waiting in the device's queue."),
            TutorialStep(
                self.angles_group, r"Starting angles ($\gamma$, $\alpha$)",
                r"Where the optimizer starts. They're drawn at random near $\theta = 0$ (the "
                r"$\beta = 0$ solution, so the optimizer starts in the right valley; exactly 0 "
                "is a saddle point). <b>🎲 Randomize</b> draws new ones.<br><br>"
                "On Starmon-7 you can also type them; in the cloud they stay random."),
            TutorialStep(
                self.device_panel, "Device panel",
                "Pick subsystem A of the thermofield-double ansatz: a chain of "
                "coupled qubits A1, A2, … (as many as the system size). Drag a "
                "path through them, or click them one by one; clicking the last "
                "one again undoes it. The mirrored subsystem B (each B<sub>i</sub> "
                "next to its A<sub>i</sub>, and the B qubits coupled in a chain) is "
                "chosen automatically, or you're told it's impossible on this device."),
            TutorialStep(
                self.start_experiment_btn, "Start Experiment",
                "Once your qubit assignment is valid, click here to run. The Run "
                "screen shows the fidelity to the target Gibbs state and the cost "
                "live, plus the measured density matrix. Runs that measure the fidelity "
                "only at the end (QX emulator, 3 or 4 spins per system) show the cost live and the "
                "fidelity in a Final fidelity card when they finish.<br><br>"
                "For a cloud backend it also shows the backend's <b>status</b> (idle, "
                "busy, calibrating, benchmarking, offline) and a <b>Hardware jobs</b> "
                "tab following every job from Queued to Results received. "
                "<b>Stop</b> tries to delete the waiting job from the queue and tells "
                "you whether Quantum Inspire accepted."),
            TutorialStep(
                self.change_backend_btn, "Change backend",
                "Not the right device? Go back and pick a "
                "different one without losing these settings."),
        ])

    def _typed_beta(self):
        """The β field's number, or None if it isn't one."""
        try:
            return float(self.beta.text().strip().replace(",", "."))
        except ValueError:
            return None

    def _beta_value(self) -> float:
        typed = self._typed_beta()
        return clamp_beta(1.0 if typed is None else typed)

    def _on_beta_edited(self):
        """Clamp a typed β into [BETA_ZERO, BETA_MAX] and say so."""
        typed = self._typed_beta()
        beta = self._beta_value()
        if typed is None:
            note = "Not a number: using 1."
        elif typed > BETA_MAX:
            note = f"{typed:.12g} is above the maximum: set to {format_beta(BETA_MAX)}."
        elif typed < BETA_ZERO:
            note = f"{typed:.12g} is below the minimum: set to {format_beta(BETA_ZERO)}."
        else:
            note = ""
        self.beta.setText(format_beta(beta))
        self._update_temperature(note)

    def _update_temperature(self, note: str = ""):
        beta = self._beta_value()
        set_tex(self.temperature_lbl, f"${format_temperature_tex(beta)}$", Colors.ACCENT)
        set_tex(self.beta_note, note, Colors.TEXT_MUTED)
        self.beta_note.setVisible(bool(note))

    def _randomize_angles(self):
        for box, value in zip((self.gamma1, self.gamma2, self.alpha1, self.alpha2),
                              random_start_angles(np.random.default_rng())):
            box.setValue(float(value))

    def _update_iterations_help(self):
        low = self.iterations.value() < MIN_SUGGESTED_ITERATIONS
        self.iterations_help.set_help(
            f"An upper limit: the run may stop earlier once it converges. "
            f"At least {MIN_SUGGESTED_ITERATIONS} is suggested"
            + (" — this may stop before converging." if low else "."), warn=low)

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
        self._remote = not backends.is_local(name)
        self._update_spin_choices()
        self._apply_hardware_mode()

    def _n_spins(self) -> int:
        return self.spins.currentData()

    def _update_spin_choices(self):
        """3+3 and 4+4 need Tuna-17 or QX (see session.spins_allowed): grey the
        sizes this backend can't run out, and fall back to 2 spins if needed."""
        model = self.spins.model()
        for row, n in enumerate(SPIN_CHOICES):
            model.item(row).setEnabled(spins_allowed(self._backend_name, n))
        if not spins_allowed(self._backend_name, self._n_spins()):
            self.spins.setCurrentIndex(SPIN_CHOICES.index(2))
        self._update_spins_note()
        self.device_panel.set_spins(self._n_spins())

    def _on_spins_changed(self):
        self.device_panel.set_spins(self._n_spins())
        self._update_spins_note()
        self._apply_hardware_mode()

    def _update_spins_note(self):
        """Why the chosen size works the way it does on this backend."""
        n, name = self._n_spins(), self._backend_name or ""
        qx = backends.is_qx(name)
        note = f"<b>{n} spins per system</b>: two copies, A and B, so "
        if n == 2:
            note += ("4 qubits: the original experiment (Sagastizabal et al.). Fits on every "
                    "backend; " + ("on QX the fidelity is measured once, at the end (9 circuits)."
                                   if qx else "the fidelity is measured live, every iteration."))
        else:
            note += (f"{2 * n} qubits: the ansatz needs a 2 × {n} ladder of coupled qubits, which "
                    f"fits on Tuna-17 and the all-to-all QX emulator. The fidelity needs "
                    f"{3 ** n} tomography circuits, so it's measured once, at the end.")
        if not spins_allowed(name, max(SPIN_CHOICES)):
            note += f" 3 and 4 spins don't fit on {name}."
        self.spins_help.set_help(note)

    def _apply_hardware_mode(self):
        """Cloud backends (QX emulator, real hardware) start from small random
        angles with SPSA (see OptimizationWorker.run); lock the
        controls that don't apply."""
        remote = self._remote
        n = self._n_spins()
        if backends.is_qx(self._backend_name):
            color, rgb = Colors.ACCENT, "135, 206, 235"
            headline = (f"<b>Cloud simulator.</b> This {2 * n}-qubit experiment runs on "
                        f"Quantum Inspire's noiseless QX emulator; its jobs wait in a queue.")
        else:
            color, rgb = Colors.WARN, "254, 138, 39"
            headline = (f"<b>Real quantum computer.</b> This {2 * n}-qubit experiment will "
                        f"be sent to {self._backend_name}.")
        self.hardware_banner.setStyleSheet(f"""
            color: {Colors.TEXT}; background: rgba({rgb}, 0.12);
            border: 1px solid {color}; border-radius: 8px; padding: 8px 10px;
        """)
        set_tex(self.hardware_banner, 
            f"{headline} It optimizes at the chosen $\\beta$ from random starting angles, "
            f"using {HARDWARE_OPTIMIZER}.", Colors.TEXT)
        self.hardware_banner.setVisible(remote)
        # In the cloud the start is always random: the drawn angles are shown,
        # and only Randomize changes them.
        self.angles_group.setEnabled(not remote)
        self.angles_lbl.setToolTip("Random in the cloud: only Randomize changes them" if remote else "")
        if remote and self.optimizer.currentText() != HARDWARE_OPTIMIZER:
            with QSignalBlocker(self.optimizer):
                self.optimizer.setCurrentText(HARDWARE_OPTIMIZER)

    def _on_optimizer_changed(self, name: str):
        if not self._remote or name == HARDWARE_OPTIMIZER:
            return
        QMessageBox.information(
            self, f"{HARDWARE_OPTIMIZER} is used in the cloud",
            f"{HARDWARE_OPTIMIZER} is suggested for noisy setups: it estimates the "
            f"gradient from just two noisy measurements per step, where {name} is easily "
            f"misled by noise. Real hardware has gate noise and shot noise, and the QX "
            f"emulator has shot noise.\n\n"
            f"Runs on {self._backend_name} therefore always use {HARDWARE_OPTIMIZER}. Pick "
            f"the Starmon-7 simulator to use {name}.")
        with QSignalBlocker(self.optimizer):
            self.optimizer.setCurrentText(HARDWARE_OPTIMIZER)

    def _on_mapping_changed(self, mapping: dict):
        self._qubit_mapping = mapping
        self.start_experiment_btn.setEnabled(bool(mapping))

    # -----------------------------------------------------
    # Config in / out
    # -----------------------------------------------------

    def load_config(self, config: SessionConfig = None):
        """A saved session's settings, or (config None) the defaults with
        fresh random starting angles."""
        fresh = config is None
        config = config or SessionConfig()

        # Before the qubit mapping below: a new size clears the selection.
        n = config.n_spins if spins_allowed(self._backend_name, config.n_spins) else 2
        self.spins.setCurrentIndex(SPIN_CHOICES.index(n) if n in SPIN_CHOICES else 0)
        self.beta.setText(format_beta(clamp_beta(config.beta)))
        self._update_temperature()
        idx = self.optimizer.findText(config.optimizer)
        with QSignalBlocker(self.optimizer):
            self.optimizer.setCurrentIndex(idx if idx >= 0 else 0)
        self.iterations.setValue(config.iterations)
        self.gamma1.setValue(config.gamma1)
        self.gamma2.setValue(config.gamma2)
        self.alpha1.setValue(config.alpha1)
        self.alpha2.setValue(config.alpha2)
        if fresh:
            self._randomize_angles()
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
            n_spins=self._n_spins(),
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
