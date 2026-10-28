"""The Run screen's Final fidelity card, in place of the live fidelity plot
for runs that measure the fidelity once, by tomography at the end."""

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from frontend.colors import Colors
from frontend.latex import set_tex


class _Gauge(QWidget):
    """A horizontal bar filled to `value` (0..1)."""

    def __init__(self):
        super().__init__()
        self.value = None
        self.setFixedHeight(14)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        rect = QRectF(0, 3, self.width(), 8)
        painter.setBrush(QColor(Colors.BORDER))
        painter.drawRoundedRect(rect, 4, 4)
        if self.value is not None:
            painter.setBrush(QColor(Colors.GOOD))
            painter.drawRoundedRect(QRectF(0, 3, self.width() * max(0.0, min(1.0, self.value)), 8), 4, 4)
        painter.end()


class FinalFidelityView(QWidget):
    """reset() before a run; fidelity() and purity() as the final tomography
    comes in (live, or replayed from a saved run)."""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(340)
        # The app's global QLabel style paints a background: keep the card's.
        self.setObjectName("finalFidelity")
        self.setStyleSheet("#finalFidelity, #finalFidelity QLabel { background: transparent; border: none; }")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 10, 24, 10)
        layout.setSpacing(10)
        layout.addStretch()

        self._value = QLabel()
        self._value.setAlignment(Qt.AlignCenter)
        self._caption = QLabel("Fidelity to the target Gibbs state")
        self._caption.setAlignment(Qt.AlignCenter)
        self._caption.setStyleSheet(f"color: {Colors.TEXT_DIM}; font-size: 11pt;")
        self._gauge = _Gauge()
        self._purity = QLabel()
        self._purity.setAlignment(Qt.AlignCenter)
        self._purity.setStyleSheet(f"color: {Colors.TEXT}; font-size: 11pt;")
        self._note = QLabel()
        self._note.setAlignment(Qt.AlignCenter)
        self._note.setWordWrap(True)
        self._note.setStyleSheet(f"color: {Colors.TEXT_DIM}; font-size: 10pt;")
        for widget in (self._value, self._caption, self._gauge, self._purity, self._note):
            layout.addWidget(widget)
        layout.addStretch()
        self.reset()

    def reset(self, settings: int = None, target_purity: float = None):
        """`settings`: tomography settings (3^n); `target_purity`: Tr rho^2 of
        the exact Gibbs state."""
        self._settings, self._target_purity = settings, target_purity
        self._set_value(None)
        self._purity.setText("")
        self._note.setText("Measured by tomography"
                           + (f" ({settings} measurement settings)" if settings else "")
                           + " after the last iteration. Follow the cost plot until then.")

    def fidelity(self, value: float):
        self._set_value(value)
        self._note.setText("Measured by tomography at the final angles"
                           + (f" ({self._settings} measurement settings)." if self._settings else "."))

    def purity(self, value: float):
        target = (f"  ·  target {self._target_purity:.1%}" if self._target_purity is not None else "")
        set_tex(self._purity, rf"Purity $\mathrm{{Tr}}\,\rho^2$ <b>{value:.1%}</b>{target}")

    def _set_value(self, value):
        self._gauge.value = value
        self._gauge.update()
        if value is None:
            self._value.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 24pt; font-weight: 600;")
            self._value.setText("Pending")
        else:
            self._value.setStyleSheet(f"color: {Colors.TEXT}; font-size: 40pt; font-weight: 700;")
            self._value.setText(f"{value:.1%}")
