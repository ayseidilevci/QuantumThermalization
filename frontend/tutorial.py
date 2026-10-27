"""Step-by-step guided tour shown on the Setup screen when the user enters
via Home's "Tutorial" card (see frontend/home.py, frontend/setup.py).

Rather than one static wall of text, this highlights each control in turn
with a small floating card (title + short explanation + Back/Next/Skip),
the way a product-tour "coach mark" works. TutorialCoach drives the
sequence; TutorialCard is the floating text bubble; TutorialHighlight is the
border drawn around whichever control the current step is about.
"""

from dataclasses import dataclass

from PySide6.QtCore import QObject, QPoint, QRect, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget

from frontend.colors import Colors


@dataclass
class TutorialStep:
    """One stop on the tour. Steps whose target isn't shown (e.g. the
    real-hardware banner while the emulator is selected) are skipped."""
    target: QWidget
    title: str
    text: str


# Longest the body text may grow before it scrolls, and the card's margin
# from the window edge.
_MAX_BODY_HEIGHT = 260
_EDGE = 12


class TutorialHighlight(QWidget):
    """A borrowed-outline rectangle drawn around the widget the current step
    is talking about. Purely decorative — never intercepts clicks, so the
    controls underneath stay usable during the tour."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        # A plain QWidget doesn't paint a QSS background/border on its own —
        # it needs this attribute set, or the style below is silently ignored.
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(f"""
            background: transparent;
            border: 2px solid {Colors.ACCENT};
            border-radius: 10px;
        """)
        self.hide()

    def point_at(self, target: QWidget):
        top_left = target.mapTo(self.parentWidget(), QPoint(0, 0))
        rect = QRect(top_left, target.size()).adjusted(-6, -6, 6, 6)
        self.setGeometry(rect)
        self.show()
        self.raise_()


class TutorialCard(QWidget):
    """The floating talking-balloon: step counter, title, body text, and
    Back / Next (or Finish) / Skip controls."""

    next_clicked = Signal()
    back_clicked = Signal()
    skip_clicked = Signal()

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setFixedWidth(300)
        # A plain QWidget doesn't paint a QSS background/border on its own —
        # it needs this attribute set, or the style below is silently ignored.
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(f"""
            TutorialCard {{
                background: {Colors.BG_2};
                border: 1px solid {Colors.ACCENT};
                border-radius: 12px;
            }}
            QLabel {{ background: transparent; }}
        """)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 14, 16, 14)
        outer.setSpacing(8)

        self.step_lbl = QLabel("")
        self.step_lbl.setStyleSheet(f"color: {Colors.ACCENT}; font-weight: 600; font-size: 9pt;")
        outer.addWidget(self.step_lbl)

        self.title_lbl = QLabel("")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        self.title_lbl.setFont(title_font)
        self.title_lbl.setStyleSheet(f"color: {Colors.TEXT};")
        outer.addWidget(self.title_lbl)

        # The body scrolls when it's longer than the room around the target
        # (see fit_body), so a long explanation never runs off the card.
        self.body_lbl = QLabel("")
        self.body_lbl.setWordWrap(True)
        self.body_lbl.setTextFormat(Qt.RichText)
        self.body_lbl.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.body_lbl.setStyleSheet(f"color: {Colors.TEXT_DIM};")
        self.body_scroll = QScrollArea()
        self.body_scroll.setWidget(self.body_lbl)
        self.body_scroll.setWidgetResizable(True)
        self.body_scroll.setFrameShape(QFrame.NoFrame)
        self.body_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.body_scroll.setStyleSheet(f"""
            QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; }}
            QScrollBar:vertical {{ background: transparent; width: 8px; margin: 0; }}
            QScrollBar::handle:vertical {{
                background: {Colors.BORDER}; border-radius: 4px; min-height: 24px;
            }}
            QScrollBar::handle:vertical:hover {{ background: {Colors.TEXT_MUTED}; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
        """)
        outer.addWidget(self.body_scroll)

        btn_row = QHBoxLayout()
        self.skip_btn = QPushButton("Skip")
        self.back_btn = QPushButton("Back")
        self.next_btn = QPushButton("Next")
        for btn in (self.skip_btn, self.back_btn, self.next_btn):
            btn.setCursor(Qt.PointingHandCursor)
        self.skip_btn.setStyleSheet(f"""
            QPushButton {{ background: transparent; color: {Colors.TEXT_MUTED}; border: none; }}
            QPushButton:hover {{ color: {Colors.TEXT_DIM}; }}
        """)
        self.back_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; color: {Colors.TEXT_DIM};
                border: 1px solid {Colors.BORDER}; border-radius: 8px; padding: 4px 12px;
            }}
            QPushButton:hover {{ color: {Colors.TEXT}; border-color: {Colors.ACCENT}; }}
            QPushButton:disabled {{ color: {Colors.BORDER}; }}
        """)
        self.next_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.ACCENT}; color: #06101C;
                border: none; border-radius: 8px; padding: 4px 14px; font-weight: 600;
            }}
            QPushButton:hover {{ background: {Colors.ACCENT_HOVER}; }}
        """)
        btn_row.addWidget(self.skip_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.back_btn)
        btn_row.addWidget(self.next_btn)
        outer.addLayout(btn_row)

        self.skip_btn.clicked.connect(self.skip_clicked.emit)
        self.back_btn.clicked.connect(self.back_clicked.emit)
        self.next_btn.clicked.connect(self.next_clicked.emit)

    def set_content(self, title: str, text: str, step_no: int, total: int, can_back: bool, is_last: bool):
        self.step_lbl.setText(f"STEP {step_no} OF {total}")
        self.title_lbl.setText(title)
        self.body_lbl.setText(text)
        self.body_scroll.verticalScrollBar().setValue(0)
        self.back_btn.setEnabled(can_back)
        self.next_btn.setText("Finish" if is_last else "Next")

    def fit_body(self, max_height: int):
        """Size the body to its text, up to `max_height`; beyond that it
        scrolls."""
        margins = self.layout().contentsMargins()
        # Leave room for the scrollbar so text never hides under it.
        width = self.width() - margins.left() - margins.right() - 10
        text_height = self.body_lbl.heightForWidth(width)
        self.body_scroll.setFixedHeight(max(40, min(text_height + 4, max_height)))

    def position_near(self, target: QWidget):
        """Prefer floating to the right of `target`; fall back to below it
        if that would run off the parent's edge, then clamp fully on-screen
        either way — no per-step tuning needed for the handful of controls
        this tour actually points at."""
        parent = self.parentWidget()
        top_left = target.mapTo(parent, QPoint(0, 0))
        target_rect = QRect(top_left, target.size())

        # Room for the body: the window height minus the card's fixed parts
        # (step counter, title, buttons), capped at _MAX_BODY_HEIGHT.
        self.body_scroll.setFixedHeight(0)
        self.adjustSize()
        chrome = self.sizeHint().height()
        self.fit_body(max(40, min(_MAX_BODY_HEIGHT, parent.height() - 2 * _EDGE - chrome)))
        self.adjustSize()          # this is a floating, non-layout-managed
        size = self.size()         # child — nothing else sizes it to its content

        x = target_rect.right() + 16
        y = target_rect.center().y() - size.height() // 2
        if x + size.width() > parent.width() - 12:
            x = target_rect.left()
            y = target_rect.bottom() + 12

        x = max(_EDGE, min(x, max(_EDGE, parent.width() - size.width() - _EDGE)))
        y = max(_EDGE, min(y, max(_EDGE, parent.height() - size.height() - _EDGE)))
        self.move(x, y)


class TutorialCoach(QObject):
    """Drives a TutorialCard + TutorialHighlight through a fixed sequence of
    TutorialStep entries, one control at a time."""

    finished = Signal()

    def __init__(self, host: QWidget, steps: list):
        super().__init__(host)
        self.host = host
        self.steps = steps
        self.active = []
        self.index = 0

        self.card = TutorialCard(host)
        self.highlight = TutorialHighlight(host)
        self.card.hide()

        self.card.next_clicked.connect(self._next)
        self.card.back_clicked.connect(self._back)
        self.card.skip_clicked.connect(self.stop)

    def start(self):
        self.active = [step for step in self.steps if step.target.isVisibleTo(self.host)]
        if not self.active:
            return
        self.index = 0
        self._show_step()

    def stop(self):
        self.card.hide()
        self.highlight.hide()
        self.finished.emit()

    def is_active(self) -> bool:
        return self.card.isVisible()

    def reposition(self):
        """Call from the host window's resizeEvent while the tour is up so
        the card and highlight keep tracking their target."""
        if self.is_active():
            step = self.active[self.index]
            self.highlight.point_at(step.target)
            self.card.position_near(step.target)

    def _next(self):
        if self.index + 1 < len(self.active):
            self.index += 1
            self._show_step()
        else:
            self.stop()

    def _back(self):
        if self.index > 0:
            self.index -= 1
            self._show_step()

    def _show_step(self):
        step = self.active[self.index]
        self.card.set_content(
            step.title, step.text, self.index + 1, len(self.active),
            can_back=self.index > 0, is_last=self.index + 1 == len(self.active),
        )
        self.highlight.point_at(step.target)
        self.card.position_near(step.target)
        self.card.show()
        self.card.raise_()
        self.highlight.raise_()
