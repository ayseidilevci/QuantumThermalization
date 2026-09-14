import os
import re
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from qiskit_quantuminspire.qi_provider import QIProvider

from test_auth import check_login


@dataclass
class BackendInfo:
    backend: Any
    name: str
    id: int
    status: Any


class QuantumInspireManager:

    def __init__(self):
        self.backends = []
        self.provider = None

        self.userID = None
        self.login_code = None

        self.STANDARD_BACKEND = "QX emulator"

    def login(
        self,
        on_code: Callable[[str], None],
        on_complete: Callable[[int], None],
    ):
        """
        Start the Quantum Inspire CLI login in the background.

        on_code(code):
            called when the QI activation code is detected.

        on_complete(return_code):
            called when the login process finishes.
        """

        def worker():

            env = {
                **os.environ,
                "PYTHONUNBUFFERED": "1",
            }

            process = subprocess.Popen(
                ["qi", "login"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )

            if process.stdout is not None:

                while True:
                    line = process.stdout.readline()

                    if line == "":
                        break

                    line = line.strip()

                    code_match = re.search(
                        r"user_code=([A-Z0-9-]+)",
                        line,
                    )

                    if code_match:
                        self.login_code = code_match.group(1)

                        if on_code is not None:
                            on_code(self.login_code)

                    user_id_match = re.search(
                        r"member ID (\d+)",
                        line,
                    )

                    if user_id_match:
                        self.userID = int(
                            user_id_match.group(1)
                        )

            return_code = process.wait()

            if on_complete is not None:
                on_complete(return_code)

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
            from quantuminspire.utils.authentication import OauthDeviceSession

            # Load the persistent QI configuration
            settings = UserSettings()
            auth_settings = settings.default_auth_settings

            # No tokens stored
            if auth_settings.tokens is None:
                return False

            # Check the access token.
            # If it is expired/about to expire, this performs a refresh.
            session = OauthDeviceSession(auth_settings)
            token_info = session.refresh()

            # Update the settings with the token returned by refresh().
            auth_settings.tokens = token_info

            # Persist it to ~/.quantuminspire/config.json
            settings.save()

            # Store the user ID for our manager
            self.userID = auth_settings.team_member_id

            # Verify that the authenticated QI API is reachable
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
        print("Connecting to Quantum Inspire...")

        self.provider = QIProvider()

        self.backends = []

        for backend in self.provider.backends():

            backend_string = str(backend)

            name_match = re.search(
                r"name=([^,]+)",
                backend_string,
            )

            id_match = re.search(
                r"id=(\d+)",
                backend_string,
            )

            if name_match is None or id_match is None:
                continue

            name = name_match.group(1).strip()
            backend_id = int(id_match.group(1))

            status = backend.status

            backend_info = BackendInfo(
                backend=backend,
                name=name,
                id=backend_id,
                status=status,
            )

            self.backends.append(backend_info)

        print(f"Found {len(self.backends)} backends.")

        return self.provider

    def update_provider(self):
        self.provider = QIProvider()

    def submit_circuit(self, circuit, backend):
        """
        Submit a Qiskit QuantumCircuit to the selected QI backend.
        """

        print("========================================")
        print("Submitting circuit to Quantum Inspire")
        print("========================================")

        print("\nCircuit:")
        print(circuit)

        print("\nBackend:")
        print(backend)

        try:
            job = backend.run(circuit)

            print("\nJob submitted successfully.")
            print("Job ID:", job.circuits_run_data[0].job_id)

            print("\nWaiting for results...")
            results = job.result()

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
        #
        # if self.provider is None:
        #     self.set_provider()
        
        self.provider = QIProvider()

        return self.provider.get_backend(name)

