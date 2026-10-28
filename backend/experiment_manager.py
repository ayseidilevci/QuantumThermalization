"""Experiment history on disk.

Every run's job record is stored by qi_utilities' StoreProjectRecord under
PROJECTS_PATH (its own hardcoded default), one folder per experiment:

    QuantumInspireProjects/<YYYYMMDD>/<HHMMSS>_<circuit name>/
        project_metadata_<timestamp>.json      (backend, project name, ...)
        job_idx_0__job_id_<id>/job_result_<timestamp>.json   (counts, shots)

That record knows nothing about this app's own settings (beta, optimizer,
qubit mapping, ...), so after storing it the Run screen also writes
SETTINGS_FILENAME into the same folder (see write_experiment_settings).
list_experiments() returns only folders that have that file — i.e.
experiments run from this app, which can be shown with their parameters
and loaded back into Setup.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Same location StoreProjectRecord defaults to.
PROJECTS_PATH = Path.home() / "Documents" / "QuantumInspireProjects"

SETTINGS_FILENAME = "tfd_experiment.json"

# The run's full results (curves, density matrices, counts, outcome), as the
# Run screen saves them, so Past Experiments can show the run again.
RESULTS_FILENAME = "tfd_results.json"


@dataclass
class ExperimentRecord:
    folder: Path
    project_name: str
    timestamp: datetime | None
    backend_name: str
    # The app's session settings (frontend.session.SessionConfig's JSON
    # form) plus a "results" dict — the contents of SETTINGS_FILENAME.
    settings: dict
    shots: int | None = None
    counts: dict = field(default_factory=dict)

    @property
    def results(self) -> dict:
        results = self.settings.get("results")
        return results if isinstance(results, dict) else {}

    @property
    def saved_results(self) -> dict | None:
        """The full results file (see RESULTS_FILENAME), if this run has one."""
        data = _read_json(self.folder / RESULTS_FILENAME)
        return data if isinstance(data, dict) else None


def write_experiment_settings(project_dir, settings: dict) -> Path:
    """Write the app's settings/results for one experiment into its
    project folder (the one StoreProjectRecord just created)."""
    path = Path(project_dir) / SETTINGS_FILENAME
    with path.open("w", encoding="utf-8") as file:
        json.dump(settings, file, indent=2)
    return path


def write_results(project_dir, snapshot: dict) -> Path:
    """Write the Run screen's full results (ExperimentWindow.snapshot) into
    the experiment's project folder."""
    path = Path(project_dir) / RESULTS_FILENAME
    with path.open("w", encoding="utf-8") as file:
        json.dump(snapshot, file)
    return path


def _read_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return None


def _parse_timestamp(value: str):
    try:
        return datetime.strptime(value, "%Y%m%d_%H%M%S")
    except (TypeError, ValueError):
        return None


class ExperimentManager:

    def __init__(self, projects_path: Path | None = None):
        self.projects_path = projects_path if projects_path is not None else PROJECTS_PATH

    def list_experiments(self) -> list[ExperimentRecord]:
        """Every experiment folder that has this app's settings file,
        newest first."""
        records = []

        if not self.projects_path.is_dir():
            return records

        for date_dir in self.projects_path.iterdir():
            if not date_dir.is_dir():
                continue

            for project_dir in date_dir.iterdir():
                if not project_dir.is_dir():
                    continue

                settings = _read_json(project_dir / SETTINGS_FILENAME)
                if not isinstance(settings, dict):
                    continue

                metadata_files = sorted(project_dir.glob("project_metadata_*.json"))
                metadata = (_read_json(metadata_files[0]) if metadata_files else None) or {}

                shots, counts = None, {}
                result_files = sorted(project_dir.glob("job_idx_*/job_result_*.json"))
                if result_files:
                    result = _read_json(result_files[0]) or {}
                    shots = result.get("Shots done")
                    counts = result.get("Counts") or {}

                records.append(ExperimentRecord(
                    folder=project_dir,
                    project_name=metadata.get("Project name", project_dir.name),
                    timestamp=_parse_timestamp(metadata.get("Project timestamp")),
                    backend_name=metadata.get(
                        "Backend name", settings.get("backend_name", "")),
                    settings=settings,
                    shots=shots,
                    counts=counts,
                ))

        records.sort(key=lambda r: r.timestamp or datetime.min, reverse=True)
        return records
