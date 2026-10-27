from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import threading
import webbrowser

from compute_api_client import ApiClient, JobsApi
from qi2_shared.client import config as qi_config
from qi2_shared.utils import run_async
from qiskit_quantuminspire.qi_provider import QIProvider

from qi_utilities.utility_functions.data_handling import StoreProjectRecord, RetrieveProjectRecord

from backend import starmon7


@dataclass
class BackendInfo:
    backend: Any
    name: str
    id: int
    status: Any
    # True for a device emulated on this machine (Starmon-7) rather than
    # served by the live QI API.
    local: bool = False


def _is_allowed_live_backend(name: str) -> bool:
    """Only the Tuna family is offered as a live backend; everything else
    (QI's own emulators, retired devices, anything added later) is excluded
    by default rather than by name-matching a blacklist."""
    normalized = "".join(ch for ch in name.lower() if ch.isalnum())
    return normalized in {"tuna5", "tuna9", "tuna17"}


# The live API reports a finer-grained status (see
# compute_api_client.BackendStatus: OFFLINE, IDLE, EXECUTING, CALIBRATING,
# BENCHMARKING) than the GUI needs. Collapse it to the three states the
# backend-selection cards actually color: green/online, yellow/busy,
# red/offline. Matched on status.value rather than importing the enum
# itself, so this file doesn't have to track compute_api_client's exact
# member set.
STATE_ONLINE = "online"
STATE_BUSY = "busy"
STATE_OFFLINE = "offline"


def classify_backend_status(status: Any) -> str:
    """Classify a live QI backend status into online / busy / offline."""
    value = str(getattr(status, "value", status)).lower()
    if value == "offline":
        return STATE_OFFLINE
    if value == "idle":
        return STATE_ONLINE
    return STATE_BUSY  # executing / calibrating / benchmarking


class QuantumInspireManager:

    def __init__(self):
        self.backends = []
        self.provider = None

        self.userID = None
        self.login_code = None

        self.STANDARD_BACKEND = starmon7.NAME

    def login(
        self,
        on_code: Callable[[str], None],
        on_complete: Callable[[int], None],
    ):
        """
        Start the Quantum Inspire OAuth device login in the background.

        on_code(code):
            Called when the QI verification code is available.

        on_complete(return_code):
            Called when the login process finishes.
            0 = success
            non-zero = failure
        """

        def worker():
            try:
                from quantuminspire.settings.user_settings import UserSettings
                from quantuminspire.managers.auth_manager import AuthManager
                from quantuminspire.utils.authentication import OauthDeviceSession
                from qi2_shared.settings import Url
                from qi2_shared.utils import run_async

                host = Url("https://api.quantum-inspire.com")

                # Load the persistent QI configuration.
                settings = UserSettings()

                # Use the SDK's AuthManager for authentication setup/storage.
                auth_manager = AuthManager(settings)

                # On a fresh installation, auths[host] does not exist yet.
                # The SDK normally creates it through _fetch_auth_settings().
                if host not in settings.auths:
                    run_async(auth_manager._fetch_auth_settings(host))

                auth_settings = settings.auths[host]

                # Start OAuth device authorization.
                auth_session = OauthDeviceSession(auth_settings)

                login_info = auth_session.initialize_authorization()

                # Get information for the GUI/browser.
                login_code = login_info["user_code"]
                login_url = login_info["verification_uri_complete"]

                self.login_code = login_code

                if on_code is not None:
                    on_code(login_code)

                # Open the browser exactly as the QI CLI does.
                webbrowser.open(login_url, new=2)

                # Wait for the user to finish authentication.
                tokens = auth_session.poll_for_tokens()

                # Use the SDK's existing token storage mechanism.
                # This also retrieves and stores the team member ID.
                auth_manager._store_tokens(host, tokens)

                # Keep the manager state in sync.
                self.userID = settings.auths[host].team_member_id

                # Create the authenticated provider.
                self.provider = QIProvider()
                self.provider.backends()

                if on_complete is not None:
                    on_complete(0)

            except Exception as exc:
                print(f"QI login failed: {exc}")

                self.provider = None
                self.userID = None

                if on_complete is not None:
                    on_complete(1)

        threading.Thread(
            target=worker,
            daemon=True,
        ).start()

    def check_login(self) -> bool:
        """
        Check whether the existing Quantum Inspire credentials
        can actually be used.

        Returns:
            True  -> existing authentication works
            False -> login is required
        """

        config_path = Path.home() / ".quantuminspire" / "config.json"

        if not config_path.exists():
            return False

        try:
            from quantuminspire.settings.user_settings import UserSettings
            from quantuminspire.managers.auth_manager import AuthManager
            from qi2_shared.settings import Url

            host = Url("https://api.quantum-inspire.com")

            settings = UserSettings()

            # Authentication configuration does not exist.
            if host not in settings.auths:
                return False

            auth_manager = AuthManager(settings)

            # This checks the stored access token and refreshes it
            # when necessary.
            auth_manager.refresh_tokens(host)

            self.userID = settings.auths[host].team_member_id

            # Verify that the API provider can be created.
            provider = QIProvider()
            provider.backends()

            self.provider = provider

            return True

        except Exception as exc:
            print(f"QI authentication check failed: {exc}")

            self.provider = None
            self.userID = None

            return False

    def set_provider(self):
        # Starmon-7 is emulated locally, so it's listed (and usable) even if
        # the live API can't be reached below.
        self.backends = [BackendInfo(
            backend=None, name=starmon7.NAME, id=-1, status="idle", local=True)]

        print("Connecting to Quantum Inspire...")

        self.provider = QIProvider()

        for backend in self.provider.backends():
            if not _is_allowed_live_backend(backend.name):
                continue
            backend_info = BackendInfo(
                backend=backend,
                name=backend.name,
                id=backend.id,
                status=backend.status,
            )

            self.backends.append(backend_info)

        print(f"Found {len(self.backends)} backends.")

        return self.provider

    def update_provider(self):
        self.provider = QIProvider()

    def backend_availability(self) -> dict[str, str]:
        """Map backend name -> "online"/"busy"/"offline" for every backend
        set_provider() last found. Call set_provider() first to refresh
        self.backends; this just classifies what's already there so the
        backend-selection cards can re-color without another network round
        trip for every repaint.
        """
        return {
            info.name: classify_backend_status(info.status)
            for info in self.backends
        }

    def submit_circuit(self, circuit, backend, wait=None):
        """
        Submit a Qiskit QuantumCircuit to the selected QI backend.

        wait(job) -> Result, if given, replaces job.result(): a real
        device's queue easily outlasts QIJob.result()'s 60 s default.
        """

        print("========================================")
        print("Submitting circuit to Quantum Inspire")
        print("========================================")

        print("\nCircuit:")
        print(circuit)

        print("\nBackend:")
        print(backend)

        try:
            if starmon7.is_starmon7(getattr(backend, "name", None)):
                circuit = starmon7.quantize_rotations(circuit)
                job = backend.run(circuit, shots=backend.options.shots)
            else:
                job = backend.run(circuit)

            print("\nJob submitted successfully.")

            print("\nWaiting for results...")
            results = wait(job) if wait else job.result()

            # A local SimulatorJob only fills circuits_run_data in result().
            print("Job ID:", job.circuits_run_data[0].job_id)

            print("\nResults:")
            print(results)

            print("\nResult data:")
            print(results.data())

            return job, results

        except Exception as exc:
            print("\nERROR submitting circuit:")
            print(exc)

            return None, None

    def get_backend(self, name=None):

        if name is None:
            name = self.STANDARD_BACKEND

        if starmon7.is_starmon7(name):
            return starmon7.make_backend()

        if self.provider is None:
            self.set_provider()        

        self.provider = QIProvider()

        return self.provider.get_backend(name)

    def cancel_job(self, job) -> tuple[bool, str]:
        """Try to withdraw a submitted batch job from the device's queue.

        QI has no "cancel" endpoint, but every circuit in a batch job is a
        separate job that can be deleted. Deleting them is attempted, then
        the jobs are read back so the answer reflects what QI actually did
        (it may refuse, e.g. once a job is already running). Returns
        (cancelled, detail).
        """
        job_ids = [data.job_id for data in getattr(job, "circuits_run_data", [])
                   if data.job_id is not None]
        if not job_ids:
            return False, "no QI job ids known for this job"

        async def delete_and_check():
            async with ApiClient(qi_config()) as client:
                api = JobsApi(client)
                errors = []
                for job_id in job_ids:
                    try:
                        await api.delete_job_jobs_id_delete(job_id)
                    except Exception as exc:
                        errors.append(f"job {job_id}: {getattr(exc, 'status', '')} "
                                      f"{getattr(exc, 'reason', None) or exc}".strip())
                page = await api.read_jobs_jobs_get(batch_job_id=job.batch_job_id)
                remaining = [item for item in page.items if item.id in job_ids]
                return errors, remaining

        try:
            errors, remaining = run_async(delete_and_check())
        except Exception as exc:
            return False, f"couldn't reach Quantum Inspire ({exc})"
        if not remaining:
            return True, "deleted from the queue"
        states = ", ".join(sorted({str(getattr(item.status, "value", item.status))
                                   for item in remaining}))
        detail = f"still at QI ({states})"
        if errors:
            detail += "; " + "; ".join(errors)
        return False, detail

    def get_job_id(self, job):
        return job.circuits_run_data[0].job_id

    def save_job(self, job):
        """Store the job record locally; returns the StoreProjectRecord,
        whose project_dir is the experiment's folder."""
        return StoreProjectRecord(job)
    