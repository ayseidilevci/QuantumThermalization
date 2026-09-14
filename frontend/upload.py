"""Home screen's "Load" action: restore a previously saved SessionConfig
from a JSON file (see frontend/session.py for the schema)."""

import json

from PySide6.QtWidgets import QFileDialog, QMessageBox

from frontend.session import SessionConfig


def load_session_dialog(parent):
    """Prompt for a session JSON file and parse it into a SessionConfig.

    Returns the loaded SessionConfig, or None if the user cancelled or the
    file couldn't be parsed (a QMessageBox is shown in the latter case).
    """
    path, _ = QFileDialog.getOpenFileName(
        parent, "Load experiment session", "", "Session JSON (*.json)"
    )
    if not path:
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return SessionConfig.from_json_dict(data)
    except Exception as exc:
        QMessageBox.critical(
            parent,
            "Couldn't load session",
            f"“{path}” doesn't look like a valid session file:\n{exc}",
        )
        return None
