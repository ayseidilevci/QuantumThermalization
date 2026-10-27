from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QSizePolicy,
    QScrollArea,
)

from frontend.colors import Colors
from frontend.flashcards import FlashCardGrid
from frontend.latex import font_px, tex_html

SIDEBAR_WIDTH = 300

# Longer lines than this get hard to read, so the text column stops growing
# on wide windows.
CONTENT_MAX_WIDTH = 1200

# (section heading, paragraphs). Qt rich text, with $...$ math (see
# frontend/latex.py).
MOTIVATION = (
    ("Why Gibbs states?", (
        "Superconductivity, phase transitions and the early universe are all "
        "physics at <b>finite temperature</b>, described by the Gibbs state "
        "$\\rho = e^{-\\beta H}/Z$. Simulating them on a quantum computer starts "
        "with preparing that state.",
    )),
    ("The catch", (
        "At any finite temperature $\\rho$ is <b>mixed</b>, and unitary gates "
        "acting on a pure input only ever produce pure states.",
    )),
    ("The trick", (
        "Prepare two copies A and B in a pure, entangled <b>thermofield-double</b> "
        "state. On its own, A is then exactly in the Gibbs state. The angles "
        "$\\gamma$ (within each copy) and $\\alpha$ (between copies) set the "
        "temperature.",
    )),
    ("Variational", (
        "A classical optimizer tunes $\\gamma$ and $\\alpha$ from measurements "
        "alone, with no precomputed answers. On hardware, the paper reached "
        "fidelities of 99%, 92% and 75% at $\\beta = 0, 1, 5$.",
    )),
)

# The paragraph text size; math is rendered to match it.
_BODY_PT = 15


class HomeWindow(QWidget):

    # (SessionConfig or None for defaults, show_tutorial)
    open_setup = Signal(object, bool)
    open_upload = Signal()

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
        heading_font.setPointSize(20)
        heading_font.setBold(True)
        heading.setFont(heading_font)
        heading.setStyleSheet(f"color: {Colors.TEXT};")
        side_layout.addWidget(heading)
        side_layout.addSpacing(20)

        self.start_button = self._sidebar_button("▶  Start")
        self.load_button = self._sidebar_button("📂  Past Experiments")
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
        # Right main area — motivation text, then concept flash cards,
        # all in one scrollable column
        # -------------------------------------------------

        content = QWidget()
        content.setMaximumWidth(CONTENT_MAX_WIDTH)
        main_layout = QVBoxLayout(content)
        main_layout.setContentsMargins(60, 50, 60, 60)
        main_layout.setSpacing(0)

        title = QLabel("Thermofield-double state preparation")
        title_font = QFont()
        title_font.setPointSize(30)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet(f"color: {Colors.TEXT};")
        title.setWordWrap(True)
        main_layout.addWidget(title)
        main_layout.addSpacing(8)

        subtitle = QLabel(
            "Following Sagastizabal et al., “Variational preparation of "
            "finite-temperature states on a quantum computer”, "
            "npj Quantum Information 7:130 (2021).")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(f"color: {Colors.TEXT_DIM}; font-size: 13pt; font-style: italic;")
        main_layout.addWidget(subtitle)
        main_layout.addSpacing(28)

        for heading_text, paragraphs in MOTIVATION:
            main_layout.addWidget(self._section_heading(heading_text))
            main_layout.addSpacing(10)
            for text in paragraphs:
                main_layout.addWidget(self._paragraph(text))
                main_layout.addSpacing(12)
            main_layout.addSpacing(16)

        # Flash cards
        cards_header = QHBoxLayout()
        cards_header.addWidget(self._section_heading("Key concepts"))
        cards_header.addStretch()
        self.reveal_all_button = QPushButton("Reveal all answers")
        self.reveal_all_button.setCheckable(True)
        self.reveal_all_button.setCursor(Qt.PointingHandCursor)
        self.reveal_all_button.setStyleSheet(f"""
            QPushButton {{
                background: transparent; color: {Colors.TEXT_DIM};
                border: 1px solid {Colors.BORDER}; border-radius: 10px;
                padding: 8px 14px; font-size: 12pt;
            }}
            QPushButton:hover {{ color: {Colors.TEXT}; border-color: {Colors.ACCENT}; }}
        """)
        cards_header.addWidget(self.reveal_all_button)
        main_layout.addLayout(cards_header)
        main_layout.addSpacing(6)

        cards_hint = QLabel("Click a card to flip it.")
        cards_hint.setStyleSheet(f"color: {Colors.TEXT_DIM}; font-size: 13pt;")
        main_layout.addWidget(cards_hint)
        main_layout.addSpacing(18)

        self.flashcards = FlashCardGrid()
        main_layout.addWidget(self.flashcards)
        main_layout.addStretch()

        self.reveal_all_button.toggled.connect(self._on_reveal_all)

        # Centre the readable-width column inside the scroll area.
        holder = QWidget()
        holder_layout = QHBoxLayout(holder)
        holder_layout.setContentsMargins(0, 0, 0, 0)
        holder_layout.addStretch()
        holder_layout.addWidget(content, stretch=100)
        holder_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"""
            QScrollArea {{ border: none; background: transparent; }}
            QScrollArea > QWidget > QWidget {{ background: transparent; }}
            QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
            QScrollBar::handle:vertical {{
                background: {Colors.BORDER}; border-radius: 5px; min-height: 30px;
            }}
            QScrollBar::handle:vertical:hover {{ background: {Colors.TEXT_MUTED}; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
        """)
        scroll.setWidget(holder)

        layout.addWidget(scroll, stretch=1)

    def _section_heading(self, text: str) -> QLabel:
        label = QLabel(text)
        font = QFont()
        font.setPointSize(20)
        font.setBold(True)
        label.setFont(font)
        label.setStyleSheet(f"color: {Colors.TEXT};")
        label.setWordWrap(True)
        return label

    def _paragraph(self, text: str) -> QLabel:
        label = QLabel(tex_html(text, font_px(_BODY_PT)))
        label.setTextFormat(Qt.RichText)
        label.setWordWrap(True)
        label.setStyleSheet(f"color: {Colors.TEXT}; font-size: {_BODY_PT}pt;")
        return label

    def _on_reveal_all(self, checked: bool):
        self.flashcards.set_all_flipped(checked)
        self.reveal_all_button.setText("Hide all answers" if checked else "Reveal all answers")

    def _sidebar_button(self, text: str) -> QPushButton:
        button = QPushButton(text)
        button.setMinimumHeight(58)
        button.setCursor(Qt.PointingHandCursor)
        button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        font = QFont()
        font.setPointSize(15)
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
        self.open_upload.emit()
