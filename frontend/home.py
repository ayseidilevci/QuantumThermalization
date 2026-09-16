from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QSizePolicy,
)

from frontend.colors import Colors
from frontend.upload import load_session_dialog

SIDEBAR_WIDTH = 280


class HomeWindow(QWidget):

    # (SessionConfig or None for defaults, show_tutorial)
    open_setup = Signal(object, bool)

    def __init__(self, manager):
        super().__init__()

        self.manager = manager

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # -------------------------------------------------
        # Left sidebar
        # -------------------------------------------------

        sidebar = QWidget()
        sidebar.setFixedWidth(SIDEBAR_WIDTH)
        sidebar.setStyleSheet(f"background-color: {Colors.BG_2};")

        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(24, 40, 24, 40)
        side_layout.setSpacing(16)

        heading = QLabel("Quantum Inspire\nExperiment")
        heading_font = QFont()
        heading_font.setPointSize(16)
        heading_font.setBold(True)
        heading.setFont(heading_font)
        heading.setStyleSheet(f"color: {Colors.TEXT};")
        side_layout.addWidget(heading)
        side_layout.addSpacing(20)

        self.start_button = self._sidebar_button("▶  Start")
        self.load_button = self._sidebar_button("📂  Load")
        self.tutorial_button = self._sidebar_button("❓  Tutorial")

        side_layout.addWidget(self.start_button)
        side_layout.addWidget(self.load_button)
        side_layout.addWidget(self.tutorial_button)
        side_layout.addStretch()

        self.start_button.clicked.connect(self._on_start)
        self.load_button.clicked.connect(self._on_load)
        self.tutorial_button.clicked.connect(self._on_tutorial)

        layout.addWidget(sidebar)

        # -------------------------------------------------
        # Right main area — static explanatory content
        # -------------------------------------------------

        main = QWidget()
        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(60, 60, 60, 60)
        main_layout.setAlignment(Qt.AlignTop)

        title = QLabel("Thermofield-double state preparation")
        title_font = QFont()
        title_font.setPointSize(22)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet(f"color: {Colors.TEXT};")
        main_layout.addWidget(title)
        main_layout.addSpacing(18)

        # ===== CONTENT STUB — replace with final copy =====
        body = QLabel(
            "This tool prepares an approximation to a thermofield-double (TFD) "
            "state on a Quantum Inspire backend, following the variational "
            "protocol of Sagastizabal et al. (npj Quantum Information 7:130, 2021).\n\n"
            "A TFD state is a purification of the thermal (Gibbs) state of a "
            "subsystem: tracing out one half of the TFD state leaves the other "
            "half in the Gibbs state ρ ∝ exp(−βH) at inverse temperature β. "
            "The circuit prepares this using two mirrored pairs of qubits "
            "(A1/A2 and B1/B2) connected by Bell pairs, then applies two "
            "variational layers — γ (intra-system coupling) and α (inter-system "
            "coupling) — whose angles are tuned by a classical optimizer to "
            "minimize an engineered cost function for the chosen β.\n\n"
            "Use Start to configure a new run, Load to resume a previously "
            "saved parameter set, or Tutorial for an annotated walkthrough of "
            "each control."
        )
        body.setWordWrap(True)
        body.setStyleSheet(f"color: {Colors.TEXT_DIM}; font-size: 11pt;")
        main_layout.addWidget(body)

        layout.addWidget(main, stretch=1)

    def _sidebar_button(self, text: str) -> QPushButton:
        button = QPushButton(text)
        button.setMinimumHeight(52)
        button.setCursor(Qt.PointingHandCursor)
        button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        button.setFont(font)
        button.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.BG_3};
                color: {Colors.TEXT};
                border: 1px solid {Colors.BORDER};
                border-radius: 12px;
                text-align: left;
                padding-left: 16px;
            }}
            QPushButton:hover {{
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 {Colors.GRADIENT_1_START},
                    stop: 1 {Colors.GRADIENT_1_END}
                );
            }}
            QPushButton:pressed {{ background-color: #162330; }}
        """)
        return button

    # -----------------------------------------------------
    # Handlers
    # -----------------------------------------------------

    def _on_start(self):
        self.open_setup.emit(None, False)

    def _on_tutorial(self):
        self.open_setup.emit(None, True)

    def _on_load(self):
        config = load_session_dialog(self)
        if config is not None:
            self.open_setup.emit(config, False)
