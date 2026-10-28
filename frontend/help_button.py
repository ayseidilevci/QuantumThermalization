"""A small round "?" that shows an explanation beside itself while hovered,
for the Setup form and the Run screen's plot cards."""

from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QLabel, QToolButton

from frontend.colors import Colors
from frontend.latex import set_tex

HELP_SIZE = 22


class _HelpPopup(QLabel):
    """The explanation box a HelpButton shows beside itself."""

    def __init__(self, button, width):
        # A floating window, owned by its button so it's deleted with it.
        super().__init__(button, Qt.ToolTip | Qt.FramelessWindowHint)
        self.setWordWrap(True)
        self.setFixedWidth(width)
        self.setStyleSheet(f"""
            QLabel {{
                background: {Colors.BG_2}; color: {Colors.TEXT};
                border: 1px solid {Colors.ACCENT}; border-radius: 6px; padding: 6px 8px;
            }}
        """)


class HelpButton(QToolButton):
    """A small round "?" that shows its explanation right beside it while
    hovered; orange when the explanation is a warning."""

    def __init__(self, width: int = 260):
        super().__init__()
        self.setText("?")
        self.setFixedSize(HELP_SIZE, HELP_SIZE)
        self.setCursor(Qt.WhatsThisCursor)
        self._popup = _HelpPopup(self, width)
        self.set_help("")

    def set_help(self, text: str, warn: bool = False):
        set_tex(self._popup, text)
        color = Colors.WARN if warn else Colors.ACCENT
        self.setStyleSheet(f"""
            QToolButton {{
                background: transparent; color: {color}; border: 1px solid {color};
                border-radius: {HELP_SIZE // 2}px; font-weight: bold;
            }}
            QToolButton:hover {{ background: {color}; color: #06101C; }}
        """)

    def enterEvent(self, event):
        popup = self._popup
        popup.adjustSize()
        # Right of the button, vertically centred on it; to its left if the
        # screen edge is in the way.
        pos = self.mapToGlobal(QPoint(self.width() + 8, (self.height() - popup.height()) // 2))
        screen = self.screen().availableGeometry()
        if pos.x() + popup.width() > screen.right():
            pos.setX(self.mapToGlobal(QPoint(0, 0)).x() - 8 - popup.width())
        pos.setY(max(screen.top(), min(pos.y(), screen.bottom() - popup.height())))
        popup.move(pos)
        popup.show()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._popup.hide()
        super().leaveEvent(event)

    def hideEvent(self, event):
        self._popup.hide()
        super().hideEvent(event)
