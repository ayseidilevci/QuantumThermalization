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
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPushButton, QVBoxLayout,
    QWidget,
)

from backend.experiment_manager import RESULTS_FILENAME, ExperimentManager, ExperimentRecord
from frontend.colors import Colors
from frontend.latex import set_tex
from frontend.session import SessionConfig, format_beta, roles
from frontend.upload import open_saved_file_dialog


def _row_widget(title: str, detail: list, subtitle: str) -> QWidget:
    row = QWidget()
    lay = QVBoxLayout(row)
    lay.setContentsMargins(10, 6, 10, 6)
    lay.setSpacing(2)
    title_lbl = QLabel(title)
    title_lbl.setStyleSheet(f"color: {Colors.TEXT}; font-weight: 600;")
    lay.addWidget(title_lbl)
    for line in detail:
        detail_lbl = QLabel()
        detail_lbl.setStyleSheet(f"color: {Colors.ACCENT}; font-size: 9pt;")
        detail_lbl.setWordWrap(True)
        set_tex(detail_lbl, line, Colors.ACCENT)
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
    n = int(settings.get("n_spins", 2))
    params = [
        f"{n}+{n} qubits",
        rf"$\beta = {format_beta(beta)}$" if isinstance(beta, (int, float)) else None,
        settings.get("optimizer"),
        f"{settings['iterations']} iter" if "iterations" in settings else None,
    ]
    mapping = settings.get("qubit_mapping")
    if isinstance(mapping, dict) and mapping:
        params.append("qubits " + " ".join(
            f"{role}→{mapping[role]}" for role in roles(n) if role in mapping))

    outcome = []
    if isinstance(results.get("fidelity"), (int, float)):
        outcome.append(f"fidelity {results['fidelity']:.3f}")
    if isinstance(results.get("final_cost"), (int, float)):
        outcome.append(f"cost {results['final_cost']:.3f}")
    optimized = results.get("optimized_params")
    if isinstance(optimized, dict) and optimized:
        outcome.append(r"$\gamma_1, \gamma_2, \alpha_1, \alpha_2$ = " + ", ".join(
            f"{optimized[k]:.3f}" for k in ("gamma1", "gamma2", "alpha1", "alpha2")
            if k in optimized))
    if record.shots:
        outcome.append(f"{record.shots} shots")
    if (record.folder / RESULTS_FILENAME).exists():
        outcome.append("results saved — click to view")

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

    session_loaded = Signal(object)   # SessionConfig -> Setup
    results_loaded = Signal(object)   # saved results -> Run screen

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
        self.search_box.setPlaceholderText("Filter by date, backend, optimizer, beta…")
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

        browse_btn = QPushButton("Open file…")
        browse_btn.setToolTip("Open saved results (or settings) from a file")
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
            # Search the math as words: "$\beta = 1$" matches "beta = 1".
            plain = [line.replace("$", "").replace("\\", "") for line in detail]
            search_text = " ".join([title, *plain, subtitle, record.project_name]).lower()
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
        results = record.saved_results
        if results is not None:
            box = QMessageBox(self)
            box.setWindowTitle("Open experiment")
            box.setText("<b>Open this experiment</b>")
            box.setInformativeText(
                "View its results as they were when it finished, or use its settings "
                "to set up a new run.")
            view = box.addButton("View results", QMessageBox.AcceptRole)
            reuse = box.addButton("Use these settings", QMessageBox.ActionRole)
            box.addButton(QMessageBox.Cancel)
            box.setDefaultButton(view)
            box.exec()
            if box.clickedButton() is view:
                self.results_loaded.emit(results)
                return
            if box.clickedButton() is not reuse:
                return
        try:
            config = SessionConfig.from_json_dict(record.settings)
        except Exception as exc:
            QMessageBox.critical(
                self, "Couldn't load experiment",
                f"The settings in “{record.folder}” couldn't be read:\n{exc}")
            return
        self.session_loaded.emit(config)

    def _on_browse_manually(self):
        opened = open_saved_file_dialog(self)
        if opened is None:
            return
        kind, data = opened
        (self.results_loaded if kind == "results" else self.session_loaded).emit(data)
