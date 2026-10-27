"""Session file I/O: reading and writing the JSON a SessionConfig
round-trips through (see frontend/session.py for the schema), via native
file dialogs."""

import json

from PySide6.QtWidgets import QFileDialog, QMessageBox

from frontend.session import SessionConfig


def parse_session_file(path) -> SessionConfig:
    """Parse `path` into a SessionConfig. Raises on a missing/invalid file."""
    with open(path, "r", encoding="utf-8") as f:
        return SessionConfig.from_json_dict(json.load(f))


def load_session_dialog(parent):
    """Prompt for a session JSON file via the native file-open dialog and
    parse it into a SessionConfig.

    Returns the loaded SessionConfig, or None if the user cancelled or the
    file couldn't be parsed (a QMessageBox is shown in the latter case).
    """
    path, _ = QFileDialog.getOpenFileName(
        parent, "Load experiment session", "", "Session JSON (*.json)"
    )
    if not path:
        return None

    try:
        return parse_session_file(path)
    except Exception as exc:
        QMessageBox.critical(
            parent,
            "Couldn't load session",
            f"“{path}” doesn't look like a valid session file:\n{exc}",
        )
        return None


def save_session_dialog(parent, config: SessionConfig) -> bool:
    """Prompt for where to save `config` as session JSON via the native
    file-save dialog.

    Returns True on success, False if the user cancelled or the write failed
    (a QMessageBox is shown in the latter case).
    """
    path, _ = QFileDialog.getSaveFileName(
        parent, "Save experiment session", "session.json", "Session JSON (*.json)"
    )
    if not path:
        return False
    if not path.lower().endswith(".json"):
        path += ".json"

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config.to_json_dict(), f, indent=2)
        return True
    except OSError as exc:
        QMessageBox.critical(
            parent,
            "Couldn't save session",
            f"Couldn't write “{path}”:\n{exc}",
        )
        return False
