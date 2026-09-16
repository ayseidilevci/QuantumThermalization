import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class JobHistoryEntry:
    project_name: str
    timestamp: str
    backend_name: str
    job_id: int


class ExperimentManager:

    def __init__(
        self,
        projects_path: Path | None = None,
    ):
        self.projects_path = (
            projects_path
            if projects_path is not None
            else Path.home()
            / "Documents"
            / "QuantumInspireProjects"
        )

    def list_jobs(self) -> list[JobHistoryEntry]:
        """
        Return lightweight information about all locally
        stored Quantum Inspire jobs.

        Only project_metadata_*.json and the job directory
        name are inspected.
        """

        jobs = []

        if not self.projects_path.exists():
            return jobs

        for date_dir in self.projects_path.iterdir():

            if not date_dir.is_dir():
                continue

            for project_dir in date_dir.iterdir():

                if not project_dir.is_dir():
                    continue

                # Find the project metadata file.
                metadata_files = list(
                    project_dir.glob("project_metadata_*.json")
                )

                if not metadata_files:
                    continue

                metadata_path = metadata_files[0]

                try:
                    with metadata_path.open(
                        "r",
                        encoding="utf-8",
                    ) as file:
                        metadata = json.load(file)

                except (OSError, json.JSONDecodeError):
                    continue

                # Read the fields we actually want.
                project_name = metadata.get(
                    "Project name",
                    project_dir.name,
                )

                timestamp = metadata.get(
                    "Project timestamp",
                    "",
                )

                backend_name = metadata.get(
                    "Backend name",
                    "",
                )

                # Find the stored job directory.
                job_dirs = project_dir.glob(
                    "job_idx_*__job_id_*"
                )

                for job_dir in job_dirs:

                    marker = "__job_id_"

                    if marker not in job_dir.name:
                        continue

                    try:
                        job_id = int(
                            job_dir.name.split(marker, 1)[1]
                        )
                    except ValueError:
                        continue

                    jobs.append(
                        JobHistoryEntry(
                            project_name=project_name,
                            timestamp=timestamp,
                            backend_name=backend_name,
                            job_id=job_id,
                        )
                    )

        jobs.sort(
            key=lambda job: job.timestamp,
            reverse=True,
        )

        return jobs
