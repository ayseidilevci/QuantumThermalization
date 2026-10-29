"""Click-to-flip concept flash cards for the Home screen.

Content is drawn from Sagastizabal et al., "Variational preparation of
finite-temperature states on a quantum computer", npj Quantum Information
7:130 (2021) — equation numbers below refer to that paper.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QVBoxLayout, QWidget

from frontend.colors import Colors
from frontend.latex import font_px, tex_html

# (topic, question, answer). Qt rich text, with $...$ for inline and
# $$...$$ for display math (see frontend/latex.py).
FLASHCARDS = (
    ("Basics", "What is a Gibbs state?",
     "The equilibrium state at inverse temperature $\\beta$ (Eq. 2):"
     "$$\\rho_{\\mathrm{Gibbs}}(\\beta) = \\dfrac{1}{Z}\\sum_j e^{-\\beta E_j}\\,|j\\rangle\\langle j|$$"
     "Each eigenstate is weighted by its Boltzmann factor."),
    ("Basics", "What does $\\beta$ mean?",
     "$$\\beta = \\dfrac{1}{k_B T}$$"
     "$\\beta = 0$: infinite temperature, $\\rho = I/4$.<br>"
     "$\\beta \\to \\infty$: zero temperature, ground state only."),
    ("Basics", "What is the partition function $Z$?",
     "The normalisation that makes $\\mathrm{Tr}\\,\\rho = 1$ (Eq. 3):"
     "$$Z = \\sum_j e^{-\\beta E_j}$$"),
    ("Basics", "Why can't gates prepare a Gibbs state directly?",
     "For finite $\\beta$ it is <b>mixed</b>, but unitaries map pure states to pure "
     "states:"
     "$$\\mathrm{Tr}\\,\\rho^2 = 1 \\;\\Rightarrow\\; \\mathrm{Tr}\\,(U\\rho U^\\dagger)^2 = 1$$"),
    ("TFD", "What is a thermofield-double state?",
     "A <b>pure</b> state of two copies A and B (Eq. 4):"
     "$$|\\mathrm{TFD}(\\beta)\\rangle = \\dfrac{1}{\\sqrt{Z}}\\sum_j e^{-\\beta E_j/2}\\,|j\\rangle_B|j\\rangle_A$$"
     "Tracing out either copy leaves the Gibbs state."),
    ("TFD", "How does the TFD avoid the mixed-state problem?",
     "It is a <b>purification</b>: pure as a whole, mixed in each half."
     "$$\\mathrm{Tr}_B\\,|\\mathrm{TFD}\\rangle\\langle\\mathrm{TFD}| = \\rho_{\\mathrm{Gibbs}}$$"
     "Copy B acts as the heat bath."),
    ("TFD", "What is the TFD state at $\\beta = 0$?",
     "A Bell pair per qubit, from $H$ + CNOT:"
     "$$|\\Phi^+\\rangle = \\dfrac{|00\\rangle + |11\\rangle}{\\sqrt{2}}$$"
     "Each half is maximally mixed, $\\rho_A = I/4$."),
    ("TFD", "Where else do TFD states appear?",
     "In high-energy physics: eternal black holes and traversable wormholes, "
     "with A and B as the two sides."),
    ("Model", "Which Hamiltonian is simulated?",
     "A 2-spin transverse-field Ising chain (Eq. 5):"
     "$$H_A = Z_{A2}Z_{A1} + g\\,(X_{A2} + X_{A1})$$"
     "with $g = 1$, the critical point for long chains."),
    ("Model", "What couples the two copies?",
     "The inter-system Hamiltonian (Eq. 6):"
     "$$H_{BA} = XX_{BA} + ZZ_{BA}$$"
     "The $\\beta = 0$ TFD is a $+2$ eigenstate of both terms."),
    ("Ansatz", "What does the circuit apply?",
     "Bell pairs, then one QAOA-style layer (Eq. 7):"
     "$$U(\\alpha, \\gamma) = U_{\\mathrm{inter}}(\\alpha)\\,U_{\\mathrm{intra}}(\\gamma)$$"),
    ("Ansatz", "What do $\\gamma$ and $\\alpha$ control?",
     "$\\gamma_1, \\gamma_2$: rotations <b>within</b> each copy, from $X$ and $ZZ$.<br>"
     "$\\alpha_1, \\alpha_2$: rotations <b>between</b> copies, from $XX_{BA}$ and $ZZ_{BA}$.<br>"
     "Together they set the temperature."),
    ("Ansatz", "Why a QAOA-style ansatz?",
     "It is shallow. Quantum Metropolis needs deep phase estimation, and "
     "imaginary-time evolution has no clear ansatz."),
    ("Cost", "What cost does the optimizer minimise?",
     "Eq. 10:"
     "$$C = \\langle X_A\\rangle + \\langle X_B\\rangle$$"
     "$$+ 1.57\\,(\\langle ZZ_A\\rangle + \\langle ZZ_B\\rangle)$$"
     "$$- \\beta^{-1.57}\\,(\\langle XX_{BA}\\rangle + \\langle ZZ_{BA}\\rangle)$$"
     "Energy pulls towards low $T$, the $\\beta^{-1.57}$ term towards entanglement."),
    ("Cost", "Why the exponent 1.57?",
     "It is engineered: compared with exponent 1, infidelity drops by 54% on "
     "average and by up to 98% near $\\beta \\approx 1$."),
    ("Cost", "What is the landscape at $\\beta = 0$?",
     "$C \\to -\\langle H_{BA}\\rangle$. Over $\\gamma$ it has period $\\pi$ and runs "
     "from $-4$ to $+4$. Over $\\alpha$ it is flat: the $\\beta = 0$ TFD is an eigenstate of "
     "$H_{BA}$."),
    ("Algorithm", "How does QAOA work?",
     "The Quantum Approximate Optimization Algorithm alternates evolution under "
     "two Hamiltonians for $p$ layers:"
     "$$U = \\prod_{k=1}^{p} e^{-i\\alpha_k H_2}\\, e^{-i\\gamma_k H_1}$$"
     "A classical optimizer tunes the angles from measurements. Here "
     "$H_1 = H_A + H_B$, $H_2 = H_{BA}$ and $p = 1$."),
    ("Algorithm", "Where does the optimizer start?",
     "Every run is one optimization straight at $\\beta_{\\mathrm{target}}$. In the cloud it "
     "starts from small random angles near the $\\beta = 0$ solution:"
     "$$\\gamma_i,\\ \\alpha_i \\sim \\mathcal{N}(0,\\, 0.4)$$"
     "Fully random angles would often get stuck in a poor local minimum with 3 or 4 spins per system. "
     "On Starmon-7 it starts from the angles set in Setup."),
    ("Results", "How is success measured?",
     "Fidelity (Eq. 11) and purity (Eq. 12):"
     "$$F = \\mathrm{Tr}\\sqrt{\\sqrt{\\rho_G}\\,\\rho_{\\mathrm{Exp}}\\sqrt{\\rho_G}}$$"
     "$$P = \\mathrm{Tr}\\,\\rho_{\\mathrm{Exp}}^2$$"),
    ("Results", "How does $\\rho_{\\mathrm{Gibbs}}$ change with $\\beta$?",
     "From $I/4$ it builds coherences with phase $0$ or $\\pi$ by parity. At "
     "$\\beta = 5$ it is nearly pure:"
     "$$|\\Upsilon\\rangle \\approx 0.6\\,(|01\\rangle + |10\\rangle)$$$$-\\, 0.37\\,(|00\\rangle + |11\\rangle)$$"),
)

_FLIP_HINT_FRONT = "Click to reveal answer"
_FLIP_HINT_BACK = "Click to see question"
_QUESTION_PT = 16
_MIN_HEIGHT = 230
# Card text width at FlashCardGrid.MIN_CARD_WIDTH, minus the side margins.
_MATH_MAX_WIDTH = 290
_ANSWER_PT = 14


class FlashCard(QFrame):
    """One card: shows the question; a click flips it to show the answer."""

    def __init__(self, topic: str, question: str, answer: str):
        super().__init__()
        self._question = question
        self._answer = answer
        self._flipped = False

        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(_MIN_HEIGHT)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 16)
        layout.setSpacing(10)

        self._topic = QLabel(topic.upper())
        self._body = QLabel()
        self._body.setTextFormat(Qt.RichText)
        self._body.setWordWrap(True)
        self._body.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self._hint = QLabel()

        layout.addWidget(self._topic)
        layout.addWidget(self._body, stretch=1)
        layout.addWidget(self._hint)

        self._render()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.set_flipped(not self._flipped)
        super().mousePressEvent(event)

    def set_flipped(self, flipped: bool):
        self._flipped = flipped
        self._render()
        self._fit_height()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if event.size().width() != event.oldSize().width():
            self._fit_height()

    def _fit_height(self):
        # The grid sizes rows from sizeHint, not height-for-width, so
        # content that wraps more at the actual column width would be
        # clipped; claim the height it really needs as the minimum.
        needed = max(_MIN_HEIGHT, self.heightForWidth(self.width()))
        if needed != self.minimumHeight():
            self.setMinimumHeight(needed)

    def _render(self):
        border = Colors.ACCENT if self._flipped else Colors.BORDER
        bg = Colors.BG_3 if self._flipped else Colors.BG_2
        self.setStyleSheet(f"""
            FlashCard {{
                background: {bg};
                border: 1px solid {border};
                border-radius: 14px;
            }}
            FlashCard:hover {{ border: 1px solid {Colors.ACCENT}; }}
        """)
        self._topic.setStyleSheet(
            f"color: {Colors.ACCENT}; font-size: 10pt; font-weight: 700; "
            f"letter-spacing: 1px; background: transparent; border: none;")
        if self._flipped:
            self._body.setText(tex_html(self._answer, font_px(_ANSWER_PT), max_width=_MATH_MAX_WIDTH))
            self._body.setStyleSheet(
                f"color: {Colors.TEXT}; font-size: {_ANSWER_PT}pt; "
                f"background: transparent; border: none;")
            self._hint.setText(_FLIP_HINT_BACK)
        else:
            self._body.setText(tex_html(self._question, font_px(_QUESTION_PT), max_width=_MATH_MAX_WIDTH))
            self._body.setStyleSheet(
                f"color: {Colors.TEXT}; font-size: {_QUESTION_PT}pt; font-weight: 600; "
                f"background: transparent; border: none;")
            self._hint.setText(_FLIP_HINT_FRONT)
        self._hint.setStyleSheet(
            f"color: {Colors.TEXT_MUTED}; font-size: 10pt; background: transparent; border: none;")


class FlashCardGrid(QWidget):
    """Lays the cards out in as many columns as the width allows, reflowing
    on resize."""

    MIN_CARD_WIDTH = 340

    def __init__(self, cards=FLASHCARDS):
        super().__init__()
        self.cards = [FlashCard(*c) for c in cards]
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(16)
        self._columns = 0
        self._relayout(2)

    def set_all_flipped(self, flipped: bool):
        for card in self.cards:
            card.set_flipped(flipped)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        columns = max(1, min(3, event.size().width() // self.MIN_CARD_WIDTH))
        if columns != self._columns:
            self._relayout(columns)

    def _relayout(self, columns: int):
        for card in self.cards:
            self._grid.removeWidget(card)
        for old in range(self._columns):
            self._grid.setColumnStretch(old, 0)
        for i, card in enumerate(self.cards):
            self._grid.addWidget(card, i // columns, i % columns)
        for col in range(columns):
            self._grid.setColumnStretch(col, 1)
        self._columns = columns
        self.updateGeometry()
