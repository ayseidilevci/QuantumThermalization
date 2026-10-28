"""Run-screen widgets for real-hardware runs: the device's live status
(sidebar) and the list of submitted jobs (Hardware jobs tab)."""

import threading
from datetime import datetime

from PySide6.QtCore import QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QFrame, QHBoxLayout, QHeaderView, QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QWidget,
)

from frontend.colors import Colors

# How often the device status is re-fetched from the QI API while a run is live.
STATUS_POLL_MS = 30_000

# compute_api_client.BackendStatus value -> (text, color, hint)
_DEVICE_STATES = {
    "idle": ("Idle", Colors.GOOD, "Ready to run jobs."),
    "executing": ("Busy", Colors.WARN, "Executing jobs; yours run in queue order."),
    "calibrating": ("Calibrating", Colors.WARN, "Jobs wait until calibration ends."),
    "benchmarking": ("Benchmarking", Colors.WARN, "Jobs wait until benchmarking ends."),
    "offline": ("Offline", Colors.BAD, "Jobs stay queued until the device is back."),
}

_JOB_STATUS_COLORS = {
    "Submitted": Colors.TEXT_DIM,
    "Queued": Colors.WARN,
    "Running on hardware": Colors.ACCENT,
    "Results received": Colors.GOOD,
    "Failed": Colors.BAD,
    "Cancelled": Colors.TEXT_MUTED,
    "Not cancelled": Colors.BAD,
}


class DeviceStatusWidget(QFrame):
    """Shows a QI device's live state (idle / busy / calibrating /
    benchmarking / offline), polled in the background while a run is live."""

    _fetched = Signal(str, str, str)   # status value, device message, error

    def __init__(self):
        super().__init__()
        self._manager = None
        self._name = None
        self._backend = None
        self._fetching = False

        self.setObjectName("deviceStatus")
        self.setStyleSheet(f"""
            #deviceStatus {{
                background: {Colors.BG_3}; border: 1px solid {Colors.BORDER};
                border-radius: 8px;
            }}
            #deviceStatus QLabel {{ background: transparent; }}
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(3)

        header = QLabel("Device status")
        header.setStyleSheet(f"color: {Colors.TEXT_DIM}; font-weight: 600; font-size: 9pt;")
        layout.addWidget(header)

        row = QHBoxLayout()
        row.setSpacing(6)
        self._dot = QLabel("●")
        self._state = QLabel("Checking…")
        self._state.setStyleSheet(f"color: {Colors.TEXT}; font-weight: 600;")
        row.addWidget(self._dot)
        row.addWidget(self._state, stretch=1)
        layout.addLayout(row)

        self._hint = QLabel()
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet(f"color: {Colors.TEXT_DIM}; font-size: 9pt;")
        layout.addWidget(self._hint)

        self._updated = QLabel()
        self._updated.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 8pt;")
        layout.addWidget(self._updated)

        self._timer = QTimer(self)
        self._timer.setInterval(STATUS_POLL_MS)
        self._timer.timeout.connect(self.refresh)
        self._fetched.connect(self._show)
        self._set_dot(Colors.TEXT_MUTED)

    def start(self, manager, name: str):
        if (manager, name) != (self._manager, self._name):
            self._backend = None
        self._manager, self._name = manager, name
        self._state.setText(f"{name}: checking…")
        self._hint.setText("")
        self._set_dot(Colors.TEXT_MUTED)
        self.refresh()
        self._timer.start()

    def stop(self):
        self._timer.stop()

    def refresh(self):
        if self._fetching or self._manager is None:
            return
        self._fetching = True
        threading.Thread(target=self._fetch, daemon=True).start()

    def _fetch(self):
        # A network round trip, so never on the GUI thread; the signal
        # hands the answer back to it.
        try:
            if self._backend is None:
                self._backend = self._manager.get_backend(self._name)
            status = self._backend.status
            value = str(getattr(status, "value", status)).lower()
            message = self._backend.message
            self._fetched.emit(value, message or "", "")
        except Exception as exc:
            self._fetched.emit("", "", str(exc))

    def _show(self, value: str, message: str, error: str):
        self._fetching = False
        if error:
            self._state.setText(f"{self._name}: status unavailable")
            self._hint.setText(error)
            self._set_dot(Colors.TEXT_MUTED)
        else:
            text, color, hint = _DEVICE_STATES.get(
                value, (value.title() or "Unknown", Colors.TEXT_MUTED, ""))
            self._state.setText(f"{self._name}: {text}")
            if message.strip().rstrip(":").strip().lower() == self._name.lower():
                message = ""   # QX's message is just its own name
            self._hint.setText("\n".join(part for part in (hint, message) if part))
            self._set_dot(color)
        self._updated.setText(f"Updated {datetime.now().strftime('%H:%M:%S')}")

    def _set_dot(self, color: str):
        self._dot.setStyleSheet(f"color: {color}; font-size: 12pt;")


class HardwareJobsView(QWidget):
    """One row per hardware job, updated live from OptimizationWorker's
    job_update signal."""

    _COLUMNS = ("#", "Purpose", "Job ID", "Status", "Submitted", "Last update")

    def __init__(self):
        super().__init__()
        self._rows = {}
        self._infos = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 14, 20, 20)
        layout.setSpacing(10)

        self.summary = QLabel()
        self.summary.setStyleSheet(f"color: {Colors.TEXT}; font-weight: 600; font-size: 12pt;")
        layout.addWidget(self.summary)

        note = QLabel(
            "Each optimizer iteration is one job on the device, and each waits in "
            "its queue. Stop deletes the waiting job from the queue; the Status "
            "column shows whether Quantum Inspire accepted that (hover for details).")
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {Colors.TEXT_DIM};")
        layout.addWidget(note)

        self.table = QTableWidget(0, len(self._COLUMNS))
        self.table.setHorizontalHeaderLabels(self._COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setShowGrid(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.setStyleSheet(f"""
            QTableWidget {{
                background: {Colors.BG_3}; color: {Colors.TEXT};
                border: 1px solid {Colors.BORDER}; border-radius: 8px;
            }}
            QTableWidget::item {{ padding: 4px 8px; }}
            QHeaderView::section {{
                background: {Colors.BG_2}; color: {Colors.TEXT_DIM}; border: none;
                padding: 6px 8px; font-weight: 600;
            }}
        """)
        layout.addWidget(self.table, stretch=1)
        self.clear()

    def jobs(self) -> list:
        """The latest info dict of every job, in table order."""
        return [self._infos[n] for n in sorted(self._infos)]

    def clear(self):
        self._rows.clear()
        self._infos = {}
        self.table.setRowCount(0)
        self._update_summary()

    def update_job(self, info: dict):
        n = info["n"]
        self._infos[n] = dict(info)
        if n not in self._rows:
            self._rows[n] = self.table.rowCount()
            self.table.insertRow(self._rows[n])
        row = self._rows[n]
        values = (str(n), info["purpose"], info["job_id"], info["status"],
                  info["submitted"], info["updated"])
        for col, value in enumerate(values):
            item = QTableWidgetItem(value)
            if col == 3:
                item.setForeground(QColor(_JOB_STATUS_COLORS.get(value, Colors.TEXT)))
                if info.get("detail"):
                    item.setToolTip(info["detail"])
            self.table.setItem(row, col, item)
        self.table.scrollToBottom()
        self._update_summary()

    def _update_summary(self):
        statuses = [self.table.item(r, 3).text() for r in range(self.table.rowCount())
                    if self.table.item(r, 3) is not None]
        if not statuses:
            self.summary.setText("No hardware jobs yet")
        else:
            counts = {s: statuses.count(s) for s in dict.fromkeys(statuses)}
            self.summary.setText(f"{len(statuses)} jobs  ·  " + "  ·  ".join(
                f"{count} {status.lower()}" for status, count in counts.items()))
