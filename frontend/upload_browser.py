"""Home screen's "Load" flow: a search bar over the filesystem plus a
"previously used projects" history list, replacing a bare native
file-open dialog. Search walks a chosen root directory (defaulting to the
home directory, remembered across restarts — see the directory chooser in
UploadBrowserWindow) in a background thread (see _SearchWorker) so typing
doesn't freeze the UI; frontend.upload's native dialog stays available
here as a manual fallback for a single specific file.
"""

import json
import os
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QThread, QObject, Signal, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QListWidget, QListWidgetItem, QMessageBox, QFileDialog,
)

from frontend.colors import Colors
from frontend.session import SessionConfig, list_recent_sessions, get_search_root, set_search_root
from frontend.upload import parse_session_file, load_session_dialog

_MAX_RESULTS = 50
_MAX_DIRS_VISITED = 20000   # bail out of a pathological/huge root
_SEARCH_DEBOUNCE_MS = 300
_SKIP_DIR_NAMES = {"node_modules", "venv", "env", "__pycache__", "site-packages"}


class _SearchWorker(QObject):
    """Walks `root` in the background for *.json files whose name contains
    `query` (case-insensitive). Emits `finished(list[Path])` once done,
    aborted, or capped at _MAX_RESULTS/_MAX_DIRS_VISITED."""

    finished = Signal(list)

    def __init__(self, query: str, root: Path):
        super().__init__()
        self.query = query.lower()
        self.root = root
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        matches = []
        visited = 0
        for root, dirs, files in os.walk(self.root):
            if self._abort:
                return
            dirs[:] = [d for d in dirs if not d.startswith(".")
                       and d not in _SKIP_DIR_NAMES]
            visited += 1
            if visited > _MAX_DIRS_VISITED:
                break
            for name in files:
                if not name.lower().endswith(".json"):
                    continue
                if self.query and self.query not in name.lower():
                    continue
                matches.append(Path(root) / name)
                if len(matches) >= _MAX_RESULTS:
                    self.finished.emit(matches)
                    return
        self.finished.emit(matches)


def _row_widget(title: str, detail: str, subtitle: str) -> QWidget:
    row = QWidget()
    lay = QVBoxLayout(row)
    lay.setContentsMargins(10, 6, 10, 6)
    lay.setSpacing(2)
    title_lbl = QLabel(title)
    title_lbl.setStyleSheet(f"color: {Colors.TEXT}; font-weight: 600;")
    lay.addWidget(title_lbl)
    if detail:
        detail_lbl = QLabel(detail)
        detail_lbl.setStyleSheet(f"color: {Colors.ACCENT}; font-size: 9pt;")
        detail_lbl.setWordWrap(True)
        lay.addWidget(detail_lbl)
    sub_lbl = QLabel(subtitle)
    sub_lbl.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 8pt;")
    sub_lbl.setWordWrap(True)
    lay.addWidget(sub_lbl)
    return row


def _describe_session(path: Path) -> str:
    """A short 'Backend · β=.. · Optimizer · N iter' summary read straight
    from the file's own JSON, or "" if it doesn't look like one of our
    session files (the row still shows — just without this line — rather
    than fabricating a plausible-looking summary for an unrelated *.json
    the filesystem search happened to match, e.g. a package.json).
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "schema_version" not in data:
            return ""
        config = SessionConfig.from_json_dict(data)
        return (f"{config.backend_name}  ·  β={config.beta:.2f}  ·  "
                f"{config.optimizer}  ·  {config.iterations} iter")
    except Exception:
        return ""


def _friendly_time(iso_str: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_str)
    except ValueError:
        return iso_str
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
    """Search bar (browses the filesystem) over a "previously used
    projects" history list, shown between Home's Load click and Setup."""

    session_loaded = Signal(object)   # SessionConfig

    def __init__(self):
        super().__init__()
        self.thread = None
        self.worker = None
        self._search_root = get_search_root() or Path.home()

        root = QVBoxLayout(self)
        root.setContentsMargins(32, 28, 32, 24)
        root.setSpacing(14)

        title = QLabel("Load a session")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet(f"color: {Colors.TEXT};")
        root.addWidget(title)

        # -------------------------------------------------
        # Search root chooser — click the path to pick a different folder
        # to search; remembered across restarts (frontend.session).
        # -------------------------------------------------
        root_row = QHBoxLayout()
        root_row.setSpacing(6)
        searching_in_lbl = QLabel("Searching in:")
        searching_in_lbl.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 9pt;")
        root_row.addWidget(searching_in_lbl)

        self.root_path_btn = QPushButton(str(self._search_root))
        self.root_path_btn.setCursor(Qt.PointingHandCursor)
        self.root_path_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; color: {Colors.ACCENT}; border: none;
                text-decoration: underline; font-size: 9pt; text-align: left;
                padding: 0;
            }}
            QPushButton:hover {{ color: {Colors.ACCENT_HOVER}; }}
        """)
        self.root_path_btn.clicked.connect(self._on_choose_root)
        root_row.addWidget(self.root_path_btn, stretch=1)
        root.addLayout(root_row)

        search_row = QHBoxLayout()
        search_row.setSpacing(8)

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search for a session file by name…")
        self.search_box.setStyleSheet(f"""
            QLineEdit {{
                background: {Colors.BG_3}; color: {Colors.TEXT};
                border: 1px solid {Colors.BORDER}; border-radius: 10px;
                padding: 10px 14px; font-size: 11pt;
            }}
            QLineEdit:focus {{ border: 1px solid {Colors.ACCENT}; }}
        """)
        self.search_box.textChanged.connect(self._on_search_text_changed)
        search_row.addWidget(self.search_box, stretch=1)

        browse_btn = QPushButton("Browse…")
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

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self._run_search)

        self.section_lbl = QLabel("Previously used projects")
        self.section_lbl.setStyleSheet(f"color: {Colors.TEXT_DIM}; font-weight: 600;")
        root.addWidget(self.section_lbl)

        self.status_lbl = QLabel("")
        self.status_lbl.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 9pt;")
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
        if not self.search_box.text():
            self._show_recents()

    # -----------------------------------------------------
    # Search
    # -----------------------------------------------------

    def _on_search_text_changed(self, text: str):
        self._debounce.stop()
        if not text:
            self._stop_search()
            self._show_recents()
            return
        self.status_lbl.setText("Searching…")
        self.status_lbl.show()
        self._debounce.start(_SEARCH_DEBOUNCE_MS)

    def _run_search(self):
        query = self.search_box.text().strip()
        if not query:
            return
        self._stop_search()

        self.section_lbl.setText("Search results")
        self.list.clear()

        self.thread = QThread()
        self.worker = _SearchWorker(query, self._search_root)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._on_search_finished)
        self.worker.finished.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def _stop_search(self):
        if self.worker is not None:
            self.worker.abort()
        self.thread = None
        self.worker = None

    def _on_search_finished(self, paths: list):
        self.status_lbl.hide()
        if not paths:
            self.status_lbl.setText(f"No session files found under {self._search_root}.")
            self.status_lbl.show()
            return
        for path in paths:
            self._add_row(path.name, _describe_session(path), str(path), path)

    # -----------------------------------------------------
    # Recents
    # -----------------------------------------------------

    def _show_recents(self):
        self.section_lbl.setText("Previously used projects")
        self.list.clear()
        recents = list_recent_sessions()
        if not recents:
            self.status_lbl.setText(
                "No previously used sessions yet — save or load one to start a history.")
            self.status_lbl.show()
            return
        self.status_lbl.hide()
        for entry in recents:
            path = Path(entry["path"])
            subtitle = f"{path}  ·  {_friendly_time(entry['used_at'])}"
            self._add_row(path.name, _describe_session(path), subtitle, path)

    # -----------------------------------------------------
    # Shared row handling
    # -----------------------------------------------------

    def _add_row(self, title: str, detail: str, subtitle: str, path: Path):
        item = QListWidgetItem(self.list)
        item.setData(Qt.UserRole, path)
        widget = _row_widget(title, detail, subtitle)
        item.setSizeHint(widget.sizeHint())
        self.list.addItem(item)
        self.list.setItemWidget(item, widget)

    def _on_item_clicked(self, item: QListWidgetItem):
        path = item.data(Qt.UserRole)
        self._load_path(path)

    def _on_choose_root(self):
        chosen = QFileDialog.getExistingDirectory(
            self, "Choose a folder to search", str(self._search_root))
        if not chosen:
            return
        self._search_root = Path(chosen)
        self.root_path_btn.setText(chosen)
        set_search_root(chosen)
        if self.search_box.text().strip():
            self._run_search()

    def _on_browse_manually(self):
        config = load_session_dialog(self)
        if config is not None:
            self.session_loaded.emit(config)

    def _load_path(self, path: Path):
        try:
            config = parse_session_file(path)
        except Exception as exc:
            QMessageBox.critical(
                self, "Couldn't load session",
                f"“{path}” doesn't look like a valid session file:\n{exc}")
            return
        self.session_loaded.emit(config)
