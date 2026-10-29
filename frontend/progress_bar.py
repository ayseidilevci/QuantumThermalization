"""The Run screen's always-visible progress strip: what the run is doing now,
a track marked with the run's β, and the elapsed time."""

import time

from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from frontend.colors import Colors
from frontend.latex import set_tex, tex_pixmap

HEIGHT = 74
_END_COLORS = {"done": Colors.GOOD, "warn": Colors.WARN, "error": Colors.BAD}


def _beta_label(beta: float) -> str:
    """Mathtext, without $ delimiters."""
    return r"\beta \approx 0" if beta < 1e-6 else rf"\beta = {beta:g}"


def _clock(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes}:{seconds:02d}"


class _Track(QWidget):
    """The bar itself: stages as dots, filled up to `fraction`."""

    def __init__(self):
        super().__init__()
        self.labels = []
        self.fraction = 0.0
        self.stage = -1
        self.color = Colors.ACCENT
        self.finished = False
        self.setMinimumWidth(240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        left, right = 14.0, self.width() - 14.0
        y = self.height() / 2 - 6
        n = max(1, len(self.labels))
        xs = [left + (right - left) * i / n for i in range(n)]

        painter.setPen(QPen(QColor(Colors.BORDER), 6, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(QPointF(left, y), QPointF(right, y))
        if self.fraction > 0:
            painter.setPen(QPen(QColor(self.color), 6, Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(QPointF(left, y), QPointF(left + (right - left) * self.fraction, y))

        for i, (x, label) in enumerate(zip(xs, self.labels)):
            reached = i <= self.stage
            painter.setPen(QPen(QColor(self.color if reached else Colors.TEXT_MUTED), 2))
            painter.setBrush(QColor(self.color) if reached else QColor(Colors.BG_2))
            painter.drawEllipse(QPointF(x, y), 7, 7)
            pixmap = tex_pixmap(label, 11, Colors.TEXT if i == self.stage else Colors.TEXT_DIM)
            size = pixmap.deviceIndependentSize()
            painter.drawPixmap(QPointF(x - size.width() / 2, y + 10), pixmap)

        painter.end()


class RunProgressBar(QWidget):
    """start() before a run, then stage()/iteration()/status() as it goes,
    and finish() when it ends."""

    def __init__(self):
        super().__init__()
        self.setObjectName("runProgress")
        self.setFixedHeight(HEIGHT)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(f"""
            #runProgress {{ background: {Colors.BG_2}; border-top: 1px solid {Colors.BORDER}; }}
            #runProgress QWidget {{ background: transparent; }}
        """)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 8, 20, 8)
        layout.setSpacing(18)

        left = QVBoxLayout()
        left.setSpacing(2)
        self._title = QLabel("Ready")
        self._title.setStyleSheet(f"color: {Colors.TEXT}; font-weight: 600; font-size: 11pt;")
        self._detail = QLabel("")
        self._detail.setStyleSheet(f"color: {Colors.TEXT_DIM}; font-size: 9.5pt;")
        left.addWidget(self._title)
        left.addWidget(self._detail)
        left_box = QWidget()
        left_box.setFixedWidth(300)
        left_box.setLayout(left)
        layout.addWidget(left_box)

        self.track = _Track()
        layout.addWidget(self.track, stretch=1)

        right = QVBoxLayout()
        right.setSpacing(2)
        self._percent = QLabel("")
        self._percent.setAlignment(Qt.AlignRight)
        self._percent.setStyleSheet(f"color: {Colors.TEXT}; font-weight: 700; font-size: 13pt;")
        self._time = QLabel("")
        self._time.setAlignment(Qt.AlignRight)
        self._time.setStyleSheet(f"color: {Colors.TEXT_DIM}; font-size: 9.5pt;")
        right.addWidget(self._percent)
        right.addWidget(self._time)
        right_box = QWidget()
        right_box.setFixedWidth(190)
        right_box.setLayout(right)
        layout.addWidget(right_box)

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._update_time)
        self._started_at = None
        self._betas, self._max_iterations = [], 1
        self._stage_base = 0       # cumulative iteration count when the stage began
        self._last_iteration = 0

    # ---- run lifecycle ---------------------------------------------------
    def start(self, betas: list, max_iterations: int, live: bool = True):
        """`betas`: the run's stages in order. live=False for a saved run
        (no clock)."""
        self._betas = list(betas)
        self._max_iterations = max(1, max_iterations)
        self._stage_base = self._last_iteration = 0
        self.track.labels = [_beta_label(b) for b in self._betas]
        self.track.stage, self.track.fraction = -1, 0.0
        self.track.color, self.track.finished = Colors.ACCENT, False
        self._title.setText("Starting…")
        self._detail.setText("")
        self._started_at = time.monotonic() if live else None
        self._timer.start() if live else self._timer.stop()
        self._refresh()

    def stage(self, beta: float):
        """A new β stage began."""
        self.track.stage = min(self.track.stage + 1, len(self._betas) - 1)
        self._stage_base = self._last_iteration
        set_tex(self._title, f"Optimizing at ${_beta_label(beta)}$")
        self._detail.setText(f"Stage {self.track.stage + 1} of {len(self._betas)}"
                             if len(self._betas) > 1 else "")
        self._refresh()

    def iteration(self, cumulative: int):
        self._last_iteration = cumulative
        k = cumulative - self._stage_base
        beta = self._betas[self.track.stage] if 0 <= self.track.stage < len(self._betas) else None
        set_tex(self._title, f"${_beta_label(beta)}$ · iteration {k} of {self._max_iterations}"
                if beta is not None else f"Iteration {k}")
        self._refresh()

    def status(self, text: str):
        """What a cloud run is waiting on, e.g. "Iteration 13: Queued"."""
        self._detail.setText(text)

    def finish(self, kind: str, title: str):
        """kind: "done", "warn" or "error" (as the result popup)."""
        self._timer.stop()
        self.track.color = _END_COLORS[kind]
        if kind == "done":
            self.track.fraction, self.track.stage = 1.0, len(self._betas) - 1
            self.track.finished = True
        self._title.setText(title)
        self._detail.setText("")
        self._refresh(final=kind == "done")

    def show_saved(self, kind: str, title: str, saved: str):
        self.finish(kind, title)
        self._detail.setText(f"Saved run{f' from {saved}' if saved else ''}")

    # ---- drawing ---------------------------------------------------------
    def _progress(self) -> float:
        n = max(1, len(self._betas))
        if self.track.stage < 0:
            return 0.0
        within = min(1.0, (self._last_iteration - self._stage_base) / self._max_iterations)
        return min(1.0, (self.track.stage + within) / n)

    def _refresh(self, final: bool = False):
        if not final and not self.track.finished:
            self.track.fraction = max(self.track.fraction, self._progress())
        self._percent.setText(f"{self.track.fraction:.0%}")
        self._update_time()
        self.track.update()

    def _update_time(self):
        if self._started_at is None:
            self._time.setText("")
            return
        self._time.setText(f"{_clock(time.monotonic() - self._started_at)} elapsed")
