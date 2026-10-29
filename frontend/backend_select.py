"""Backend-selection screen, shown between Home's Start/Tutorial and Setup.

One vertical (portrait) card per backend, all in a single row that fills
the full width and height below the header — N backends divide the row into
N equal-width columns (see BackendSelectWindow's QGridLayout), a thin
vertical rule separating real hardware from the simulators (the local noisy
Starmon-7 and the cloud's noiseless QX emulator; see backend/backends.py).
Each card's chip schematic (or a placeholder
for a device we ship none for) fills it as a background image, colored if
the backend is online/busy and grayscale if it's offline
(backend.qi_manager.classify_backend_status is the "which devices are
available" function this reads). The name sits near the bottom and slides
up to vertical-center on hover; a small status line with a colored dot
stays fixed at the bottom.
"""

from pathlib import Path

from PySide6.QtCore import QEasingCurve, QPoint, QPropertyAnimation, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QLinearGradient, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QLabel, QMessageBox, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout,
    QWidget,
)

from backend import backends
from backend.qi_manager import STATE_OFFLINE, STATE_ONLINE, classify_backend_status
from frontend.colors import Colors
from frontend.device_panel import BACKEND_IMAGES, _lookup_by_name

MIN_CARD_SIZE = QSize(110, 180)
_NAME_HEIGHT = 40
_STATUS_HEIGHT = 22
_STATUS_MARGIN = 14

_STATE_DOT = {
    STATE_OFFLINE: ("🔴", "Offline"),
    "busy": ("🟡", "Busy"),
    STATE_ONLINE: ("🟢", "Online"),
}


def _render_card_background(name: str, size: QSize) -> QPixmap:
    """A full-bleed, opaque background image for the card: the real chip
    schematic, or a placeholder gradient for a device with no entry in
    BACKEND_IMAGES."""
    path = _lookup_by_name(name, BACKEND_IMAGES)

    image = QImage(size, QImage.Format_ARGB32_Premultiplied)
    image.fill(QColor(Colors.BG_3))
    painter = QPainter(image)

    if path is not None and Path(path).exists():
        if str(path).lower().endswith((".svg", ".svgz")):
            renderer = QSvgRenderer(str(path))
            if renderer.isValid():
                base = renderer.defaultSize()
                if base.width() > 0 and base.height() > 0:
                    scale = max(size.width() / base.width(), size.height() / base.height())
                    w, h = base.width() * scale, base.height() * scale
                    x, y = (size.width() - w) / 2, (size.height() - h) / 2
                    renderer.render(painter, QRectF(x, y, w, h))
        else:
            raster = QPixmap(str(path))
            if not raster.isNull():
                scaled = raster.scaled(size, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
                x = (scaled.width() - size.width()) // 2
                y = (scaled.height() - size.height()) // 2
                painter.drawPixmap(0, 0, scaled, x, y, size.width(), size.height())
    else:
        # Placeholder: a soft accent gradient with a simple glyph, standing
        # in for any device missing a schematic.
        grad = QLinearGradient(0, 0, 0, size.height())
        grad.setColorAt(0.0, QColor(Colors.GRADIENT_1_START))
        grad.setColorAt(1.0, QColor(Colors.GRADIENT_1_END))
        painter.fillRect(0, 0, size.width(), size.height(), grad)
        painter.setPen(QColor(255, 255, 255, 60))
        glyph_font = QFont()
        glyph_font.setPointSize(48)
        painter.setFont(glyph_font)
        painter.drawText(image.rect(), Qt.AlignCenter, "◆")

    painter.end()
    return QPixmap.fromImage(image)


def _grayscale(pixmap: QPixmap) -> QPixmap:
    image = pixmap.toImage().convertToFormat(QImage.Format_Grayscale8)
    return QPixmap.fromImage(image.convertToFormat(QImage.Format_ARGB32_Premultiplied))


class BackendCard(QFrame):
    """One backend, filling whatever space its layout column gives it (see
    BackendSelectWindow — every card gets an equal-stretch column reaching
    the full height of the row). Emits `clicked(name, qi_backend)` when a
    non-offline card is pressed."""

    clicked = Signal(str, object)

    def __init__(self, info):
        super().__init__()
        self.info = info
        self.state = classify_backend_status(info.status)
        self._online = self.state != STATE_OFFLINE
        self._hovered = False
        self._bg_size = None  # last size the background was rendered at

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(MIN_CARD_SIZE)
        self.setCursor(Qt.PointingHandCursor if self._online else Qt.ForbiddenCursor)
        self._set_border(Colors.BORDER)

        self._bg_label = QLabel(self)

        # Dark scrim behind the bottom text so it reads over any image.
        self._scrim = QWidget(self)
        self._scrim.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 rgba(6,16,28,0), stop:1 rgba(6,16,28,200));
        """)
        self._scrim.setAttribute(Qt.WA_TransparentForMouseEvents)

        self._name_label = QLabel(
            f"{info.name}<br><span style='font-size:9.5pt; font-weight:400; color:{Colors.TEXT_DIM};'>"
            f"{backends.description(info.name)}</span>", self)
        self._name_label.setTextFormat(Qt.RichText)
        self._name_label.setAlignment(Qt.AlignCenter)
        name_font = QFont()
        name_font.setPointSize(13)
        name_font.setBold(True)
        self._name_label.setFont(name_font)
        self._name_label.setStyleSheet(f"color: {Colors.TEXT}; background: transparent;")
        self._name_label.setWordWrap(True)
        self._name_label.setAttribute(Qt.WA_TransparentForMouseEvents)

        dot, text = _STATE_DOT[self.state]
        self._status_label = QLabel(f"{dot}  {text}", self)
        self._status_label.setAlignment(Qt.AlignCenter)
        self._status_label.setStyleSheet(
            f"color: {Colors.TEXT_DIM}; background: transparent; font-size: 9.5pt;")
        self._status_label.setAttribute(Qt.WA_TransparentForMouseEvents)

        self._overlay = None
        if not self._online:
            self._overlay = QWidget(self)
            self._overlay.setStyleSheet("background: rgba(6,16,28,90);")
            self._overlay.setAttribute(Qt.WA_TransparentForMouseEvents)

        self._bottom_pos = QPoint(0, 0)
        self._mid_pos = QPoint(0, 0)

        self._anim = QPropertyAnimation(self._name_label, b"pos", self)
        self._anim.setDuration(220)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)

        self._relayout()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._relayout()

    def _relayout(self):
        size = self.size()
        if size.width() <= 0 or size.height() <= 0:
            return

        if self._bg_size != size:
            bg = _render_card_background(self.info.name, size)
            if not self._online:
                bg = _grayscale(bg)
            self._bg_label.setPixmap(bg)
            self._bg_size = QSize(size)
        self._bg_label.setGeometry(0, 0, size.width(), size.height())

        scrim_h = size.height() // 2
        self._scrim.setGeometry(0, size.height() - scrim_h, size.width(), scrim_h)

        self._bottom_pos = QPoint(
            0, size.height() - _STATUS_HEIGHT - _STATUS_MARGIN - _NAME_HEIGHT)
        self._mid_pos = QPoint(0, (size.height() - _NAME_HEIGHT) // 2)
        self._name_label.setGeometry(
            0, self._mid_pos.y() if self._hovered else self._bottom_pos.y(),
            size.width(), _NAME_HEIGHT)

        self._status_label.setGeometry(
            0, size.height() - _STATUS_HEIGHT - 6, size.width(), _STATUS_HEIGHT)

        if self._overlay is not None:
            self._overlay.setGeometry(0, 0, size.width(), size.height())
            self._overlay.raise_()
        self._scrim.raise_()
        self._name_label.raise_()
        self._status_label.raise_()

    def _set_border(self, color: str):
        self.setStyleSheet(f"""
            QFrame {{
                background: {Colors.BG_3};
                border: 1px solid {color};
                border-radius: 14px;
            }}
        """)

    def enterEvent(self, event):
        super().enterEvent(event)
        self._hovered = True
        if self._online:
            self._set_border(Colors.GRADIENT_1_END)
        self._animate_to(self._mid_pos)

    def leaveEvent(self, event):
        super().leaveEvent(event)
        self._hovered = False
        self._set_border(Colors.BORDER)
        self._animate_to(self._bottom_pos)

    def _animate_to(self, pos: QPoint):
        self._anim.stop()
        self._anim.setStartValue(self._name_label.pos())
        self._anim.setEndValue(pos)
        self._anim.start()

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if self._online and event.button() == Qt.LeftButton:
            self.clicked.emit(self.info.name, self.info.backend)


def _group_title(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setStyleSheet(f"color: {Colors.TEXT_DIM}; font-weight: 600; font-size: 11pt;")
    return lbl


def _vertical_divider() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.VLine)
    line.setFrameShadow(QFrame.Plain)
    line.setStyleSheet(f"color: {Colors.BORDER};")
    line.setFixedWidth(1)
    return line


class BackendSelectWindow(QWidget):

    backend_chosen = Signal(str, object)   # name, qi_backend

    def __init__(self, manager):
        super().__init__()
        self.manager = manager
        self._loaded = False

        root = QVBoxLayout(self)
        root.setContentsMargins(32, 28, 32, 24)
        root.setSpacing(14)

        header = QHBoxLayout()
        title = QLabel("Choose a backend")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet(f"color: {Colors.TEXT};")
        header.addWidget(title)
        header.addStretch(1)

        self._refresh_btn = QPushButton("⟳  Refresh")
        self._refresh_btn.setCursor(Qt.PointingHandCursor)
        self._refresh_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; color: {Colors.TEXT_DIM};
                border: 1px solid {Colors.BORDER}; border-radius: 8px; padding: 6px 12px;
            }}
            QPushButton:hover {{ color: {Colors.TEXT}; border-color: {Colors.ACCENT}; }}
        """)
        self._refresh_btn.clicked.connect(self._load_backends)
        header.addWidget(self._refresh_btn)

        # No separate "Home" button here — the persistent top bar already
        # has one, visible on this screen.

        root.addLayout(header)

        self._status_lbl = QLabel("")
        self._status_lbl.setWordWrap(True)
        self._status_lbl.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 9pt;")
        self._status_lbl.hide()
        root.addWidget(self._status_lbl)

        # Cards fill this area edge-to-edge (see _build_grid): every card is
        # an equal-stretch grid column reaching the full row height, so N
        # backends divide the available width into N equal vertical strips.
        # The scroll area is just a safety net for when there are too many
        # to fit at MIN_CARD_SIZE — normally no scrollbar appears at all.
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._scroll.setStyleSheet("background: transparent;")
        root.addWidget(self._scroll, stretch=1)

    def showEvent(self, event):
        super().showEvent(event)
        if not self._loaded:
            self._load_backends()

    def _load_backends(self):
        self._loaded = True
        try:
            self.manager.set_provider()
            self._status_lbl.hide()
        except Exception as exc:
            self._status_lbl.setText(f"Couldn't fetch the live backend list ({exc}).")
            self._status_lbl.show()

        devices, emulated = [], []
        for info in self.manager.backends:
            card = BackendCard(info)
            card.clicked.connect(self._on_card_clicked)
            (devices if backends.is_hardware(info.name) else emulated).append(card)
        emulated.sort(key=lambda card: not backends.is_local(card.info.name))  # Starmon-7 first

        # Replacing the scroll area's widget drops the previous one (and its
        # cards) along with it — simplest way to rebuild cleanly on refresh.
        self._scroll.setWidget(self._build_grid(devices, emulated))

    def _build_grid(self, devices: list, emulated: list) -> QWidget:
        content = QWidget()
        grid = QGridLayout(content)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(10)

        if not devices and not emulated:
            empty = QLabel("No backends found.")
            empty.setStyleSheet(f"color: {Colors.TEXT_MUTED};")
            grid.addWidget(empty, 0, 0)
            return content

        col = 0
        if devices:
            grid.addWidget(_group_title("Real hardware"), 0, col, 1, len(devices))
            for card in devices:
                grid.addWidget(card, 1, col)
                grid.setColumnStretch(col, 1)
                col += 1

        if devices and emulated:
            grid.addWidget(_vertical_divider(), 0, col, 2, 1)
            grid.setColumnStretch(col, 0)
            col += 1

        if emulated:
            grid.addWidget(_group_title("Simulators"), 0, col, 1, len(emulated))
            for card in emulated:
                grid.addWidget(card, 1, col)
                grid.setColumnStretch(col, 1)
                col += 1

        grid.setRowStretch(0, 0)
        grid.setRowStretch(1, 1)
        return content

    def _on_card_clicked(self, name: str, qi_backend):
        if backends.is_hardware(name) and not self._confirm_real_device(name):
            return
        self.backend_chosen.emit(name, qi_backend)

    def _confirm_real_device(self, name: str) -> bool:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Real quantum computer")
        box.setText(f"{name} is a real quantum computer.")
        box.setInformativeText(
            "Experiments on it are sent to Quantum Inspire's real hardware. Jobs "
            "wait in the device's queue, so a run can take significantly longer "
            "than on a simulator.\n\n"
            "Real hardware runs start from small random angles and use SPSA.")
        go = box.addButton("Use this device", QMessageBox.AcceptRole)
        box.addButton("Choose another", QMessageBox.RejectRole)
        box.setDefaultButton(go)
        box.exec()
        return box.clickedButton() is go
