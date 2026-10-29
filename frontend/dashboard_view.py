"""The Run screen's "Live dashboard" tab for real-hardware runs: QuTech's
public Grafana dashboard for the chip (its latest per-qubit calibration),
embedded and reloaded every minute."""

from PySide6.QtCore import QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from frontend.colors import Colors

# dashboard.quantum-inspire.com/<backend> redirects to the backend's public
# dashboard on monitoring.qutech.support.
DASHBOARD_URL = "https://dashboard.quantum-inspire.com/"
RELOAD_MS = 60_000


def dashboard_url(backend_name: str) -> str:
    """E.g. "Tuna-17" -> .../tuna-17, "QX emulator" -> .../qx-emulator."""
    return DASHBOARD_URL + "-".join(backend_name.lower().split())


class DashboardView(QWidget):

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 14, 20, 20)
        layout.setSpacing(10)

        head = QHBoxLayout()
        self.caption = QLabel()
        self.caption.setWordWrap(True)
        self.caption.setStyleSheet(f"color: {Colors.TEXT_DIM};")
        head.addWidget(self.caption, stretch=1)
        button_style = f"""
            QPushButton {{ background: transparent; color: {Colors.ACCENT};
                          border: 1px solid {Colors.ACCENT}; border-radius: 8px; padding: 5px 12px; }}
            QPushButton:hover {{ background: rgba(135,206,235,0.12); }}
            QPushButton:disabled {{ color: {Colors.TEXT_MUTED}; border-color: {Colors.BORDER}; }}
        """
        # A link inside Grafana navigates the embedded page: Back undoes that,
        # Dashboard returns to the chip's dashboard.
        self.back_btn = QPushButton("←  Back")
        self.back_btn.setEnabled(False)
        home_btn = QPushButton("Dashboard")
        reload_btn = QPushButton("↻  Reload")
        open_btn = QPushButton("Open in browser ↗")
        for button in (self.back_btn, home_btn, reload_btn, open_btn):
            button.setStyleSheet(button_style)
            head.addWidget(button)
        layout.addLayout(head)

        self.web = QWebEngineView()
        self.web.page().setBackgroundColor(Colors.BG_1)
        layout.addWidget(self.web, stretch=1)

        self.back_btn.clicked.connect(self.web.back)
        home_btn.clicked.connect(lambda: self.web.setUrl(QUrl(self._url)))
        # The history updates after urlChanged, so check again once loaded.
        for signal in (self.web.urlChanged, self.web.loadFinished):
            signal.connect(lambda _: self.back_btn.setEnabled(self.web.history().canGoBack()))
        reload_btn.clicked.connect(self.web.reload)
        open_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(self._url)))
        # Grafana's panels don't all refresh themselves on a public dashboard.
        self._timer = QTimer(self, interval=RELOAD_MS)
        self._timer.timeout.connect(self.web.reload)
        self._url = DASHBOARD_URL

    def set_backend(self, backend_name: str):
        self._url = dashboard_url(backend_name)
        self.caption.setText(
            f"QuTech's public monitoring dashboard for {backend_name}: its latest calibration "
            f"(qubit coherence times and gate and readout errors). Reloads every minute. {self._url}")
        self.web.setUrl(QUrl(self._url))

    def showEvent(self, event):
        super().showEvent(event)
        self._timer.start()

    def hideEvent(self, event):
        super().hideEvent(event)
        self._timer.stop()
