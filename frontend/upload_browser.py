"""Home screen's "Load" flow: the experiment history — every experiment
run from this app, read from the hardcoded projects folder
(backend.experiment_manager.PROJECTS_PATH, one sub-folder per experiment)
— with a search box that filters it, shown between Home's Load click and
Setup. Clicking an experiment loads its settings into Setup.
frontend.upload's native file dialog stays available ("Browse…") for a
session .json saved elsewhere.
"""

from datetime import datetime

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QListWidget, QListWidgetItem, QMessageBox,
)

from backend.experiment_manager import ExperimentManager, ExperimentRecord
from frontend.colors import Colors
from frontend.session import SessionConfig, format_beta
from frontend.upload import load_session_dialog


def _row_widget(title: str, detail: list, subtitle: str) -> QWidget:
    row = QWidget()
    lay = QVBoxLayout(row)
    lay.setContentsMargins(10, 6, 10, 6)
    lay.setSpacing(2)
    title_lbl = QLabel(title)
    title_lbl.setStyleSheet(f"color: {Colors.TEXT}; font-weight: 600;")
    lay.addWidget(title_lbl)
    for line in detail:
        detail_lbl = QLabel(line)
        detail_lbl.setStyleSheet(f"color: {Colors.ACCENT}; font-size: 9pt;")
        detail_lbl.setWordWrap(True)
        lay.addWidget(detail_lbl)
    sub_lbl = QLabel(subtitle)
    sub_lbl.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 8pt;")
    sub_lbl.setWordWrap(True)
    lay.addWidget(sub_lbl)
    return row


def _describe(record: ExperimentRecord) -> tuple[str, list, str]:
    """(title, detail lines, subtitle) for one experiment's history row."""
    settings, results = record.settings, record.results
    when = (record.timestamp.strftime("%d %b %Y, %H:%M") if record.timestamp
            else record.folder.parent.name)
    title = f"{when}  ·  {record.backend_name}"

    beta = settings.get("beta")
    params = [
        f"β={format_beta(beta)}" if isinstance(beta, (int, float)) else None,
        settings.get("optimizer"),
        f"{settings['iterations']} iter" if "iterations" in settings else None,
    ]
    mapping = settings.get("qubit_mapping")
    if isinstance(mapping, dict) and mapping:
        params.append("qubits " + " ".join(
            f"{role}→{mapping[role]}" for role in ("A1", "A2", "B1", "B2") if role in mapping))

    outcome = []
    if isinstance(results.get("fidelity"), (int, float)):
        outcome.append(f"fidelity {results['fidelity']:.3f}")
    if isinstance(results.get("final_cost"), (int, float)):
        outcome.append(f"cost {results['final_cost']:.3f}")
    optimized = results.get("optimized_params")
    if isinstance(optimized, dict) and optimized:
        outcome.append("γ₁, γ₂, α₁, α₂ = " + ", ".join(
            f"{optimized[k]:.3f}" for k in ("gamma1", "gamma2", "alpha1", "alpha2")
            if k in optimized))
    if record.shots:
        outcome.append(f"{record.shots} shots")

    detail = ["  ·  ".join(p for p in params if p)]
    if outcome:
        detail.append("  ·  ".join(outcome))

    subtitle = str(record.folder)
    if record.timestamp:
        subtitle += f"  ·  {_friendly_time(record.timestamp)}"
    return title, detail, subtitle


def _friendly_time(dt: datetime) -> str:
    delta = datetime.now() - dt
    seconds = delta.total_seconds()
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)} min ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)} h ago"
    if seconds < 86400 * 30:
        return f"{int(seconds // 86400)} d ago"
    return dt.strftime("%Y-%m-%d")


class UploadBrowserWindow(QWidget):
    """Experiment history (filtered by a search box) shown between Home's
    Load click and Setup."""

    session_loaded = Signal(object)   # SessionConfig

    def __init__(self):
        super().__init__()
        self.experiments = ExperimentManager()
        self._rows = []   # (search text, title, detail, subtitle, record)

        root = QVBoxLayout(self)
        root.setContentsMargins(32, 28, 32, 24)
        root.setSpacing(14)

        title = QLabel("Past experiments")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet(f"color: {Colors.TEXT};")
        root.addWidget(title)

        folder_lbl = QLabel(f"Experiments in: {self.experiments.projects_path}")
        folder_lbl.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 9pt;")
        folder_lbl.setWordWrap(True)
        root.addWidget(folder_lbl)

        search_row = QHBoxLayout()
        search_row.setSpacing(8)

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Filter by date, backend, optimizer, β…")
        self.search_box.setStyleSheet(f"""
            QLineEdit {{
                background: {Colors.BG_3}; color: {Colors.TEXT};
                border: 1px solid {Colors.BORDER}; border-radius: 10px;
                padding: 10px 14px; font-size: 11pt;
            }}
            QLineEdit:focus {{ border: 1px solid {Colors.ACCENT}; }}
        """)
        self.search_box.textChanged.connect(self._apply_filter)
        search_row.addWidget(self.search_box, stretch=1)

        browse_btn = QPushButton("Browse…")
        browse_btn.setToolTip("Load a session .json file saved somewhere else")
        browse_btn.setCursor(Qt.PointingHandCursor)
        browse_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.BG_3}; color: {Colors.TEXT_DIM};
                border: 1px solid {Colors.BORDER}; border-radius: 10px;
                padding: 10px 16px; font-size: 10pt;
            }}
            QPushButton:hover {{ color: {Colors.TEXT}; border-color: {Colors.ACCENT}; }}
        """)
        browse_btn.clicked.connect(self._on_browse_manually)
        search_row.addWidget(browse_btn)

        root.addLayout(search_row)

        self.section_lbl = QLabel("Experiment history")
        self.section_lbl.setStyleSheet(f"color: {Colors.TEXT_DIM}; font-weight: 600;")
        root.addWidget(self.section_lbl)

        self.status_lbl = QLabel("")
        self.status_lbl.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 9pt;")
        self.status_lbl.setWordWrap(True)
        self.status_lbl.hide()
        root.addWidget(self.status_lbl)

        self.list = QListWidget()
        self.list.setStyleSheet(f"""
            QListWidget {{
                background: {Colors.BG_2}; border: 1px solid {Colors.BORDER};
                border-radius: 10px;
            }}
            QListWidget::item {{ border-bottom: 1px solid {Colors.BORDER}; }}
            QListWidget::item:selected {{ background: {Colors.BG_3}; }}
        """)
        self.list.setCursor(Qt.PointingHandCursor)
        self.list.itemClicked.connect(self._on_item_clicked)
        root.addWidget(self.list, stretch=1)

    def showEvent(self, event):
        super().showEvent(event)
        # Re-read on every visit so experiments run since last time show up.
        self._reload()

    # -----------------------------------------------------
    # History
    # -----------------------------------------------------

    def _reload(self):
        self._rows = []
        for record in self.experiments.list_experiments():
            title, detail, subtitle = _describe(record)
            search_text = " ".join([title, *detail, subtitle, record.project_name]).lower()
            self._rows.append((search_text, title, detail, subtitle, record))
        self._apply_filter()

    def _apply_filter(self):
        query = self.search_box.text().strip().lower()
        terms = query.split()
        self.list.clear()
        shown = 0
        for search_text, title, detail, subtitle, record in self._rows:
            if all(term in search_text for term in terms):
                self._add_row(title, detail, subtitle, record)
                shown += 1

        if not self._rows:
            self.status_lbl.setText(
                "No experiments yet — every run you start from Setup is saved here.")
        elif shown == 0:
            self.status_lbl.setText(f"No experiments match “{query}”.")
        else:
            self.status_lbl.setText("")
        self.status_lbl.setVisible(bool(self.status_lbl.text()))
        self.section_lbl.setText(
            f"Experiment history ({shown} of {len(self._rows)})" if query
            else f"Experiment history ({len(self._rows)})")

    def _add_row(self, title: str, detail: list, subtitle: str, record: ExperimentRecord):
        item = QListWidgetItem(self.list)
        item.setData(Qt.UserRole, record)
        widget = _row_widget(title, detail, subtitle)
        item.setSizeHint(widget.sizeHint())
        self.list.addItem(item)
        self.list.setItemWidget(item, widget)

    def _on_item_clicked(self, item: QListWidgetItem):
        record = item.data(Qt.UserRole)
        try:
            config = SessionConfig.from_json_dict(record.settings)
        except Exception as exc:
            QMessageBox.critical(
                self, "Couldn't load experiment",
                f"The settings in “{record.folder}” couldn't be read:\n{exc}")
            return
        self.session_loaded.emit(config)

    def _on_browse_manually(self):
        config = load_session_dialog(self)
        if config is not None:
            self.session_loaded.emit(config)
