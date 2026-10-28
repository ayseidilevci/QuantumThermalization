"""File I/O for the Run screen and Past Experiments: saving a run's full
results (reopened later in the Run screen) and loading either a results file
or a settings-only session file (see frontend/session.py), via native file
dialogs."""

import json
from pathlib import Path

from PySide6.QtWidgets import QFileDialog, QMessageBox

from frontend.session import SessionConfig

# Marks a results file (ExperimentWindow.snapshot) apart from a session file.
RESULTS_KIND = "tfd_results"


def parse_session_file(path) -> SessionConfig:
    """Parse `path` into a SessionConfig. Raises on a missing/invalid file."""
    with open(path, "r", encoding="utf-8") as f:
        return SessionConfig.from_json_dict(json.load(f))


def is_results(data) -> bool:
    return isinstance(data, dict) and data.get("kind") == RESULTS_KIND


def open_saved_file_dialog(parent):
    """Ask for a saved file and read it. Returns ("results", snapshot) for a
    results file, ("session", SessionConfig) for a settings-only file, or
    None if cancelled or unreadable (a QMessageBox is shown then)."""
    path, _ = QFileDialog.getOpenFileName(
        parent, "Open saved results or settings", "", "Saved experiment (*.json)")
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if is_results(data):
            return "results", data
        return "session", SessionConfig.from_json_dict(data)
    except Exception as exc:
        QMessageBox.critical(
            parent, "Couldn't open file",
            f"“{path}” isn't a saved experiment this app can read:\n{exc}")
        return None


def save_results_dialog(parent, snapshot: dict, export_figures) -> bool:
    """Ask where to save a run's results; write the results file there and
    every plot and its data next to it, in "<name>_figures/" (via
    export_figures(folder)). Returns True on success."""
    path, _ = QFileDialog.getSaveFileName(
        parent, "Save results", "tfd_results.json", "Saved experiment (*.json)")
    if not path:
        return False
    path = Path(path if path.lower().endswith(".json") else path + ".json")
    folder = path.with_name(path.stem + "_figures")
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(snapshot, f)
        export_figures(folder)
    except OSError as exc:
        QMessageBox.critical(parent, "Couldn't save results", f"Couldn't write “{path}”:\n{exc}")
        return False
    QMessageBox.information(
        parent, "Results saved",
        f"Saved the whole run to “{path.name}”: open it from Past Experiments → "
        f"Open file… to see it here again.\n\nEvery plot and its data are in “{folder.name}”.")
    return True
