<<<<<<< HEAD
# QuantumBackend funtions

import subprocess

from qiskit_quantuminspire.qi_provider import QIProvider

import subprocess
import threading
import re 
import os
import time

class QuantumInspireManager:
    def __init__(self):
        """
        Create the Quantum Inspire provider after authentication has been completed.
        """
        self.backend = None
        self.provider = None
        self.userID = None
        self.login_code = None
        self.STANDARD_BACKEND = "QX Emulator"



    def login(self):

        def worker():

            print("WORKER: starting QI login")

            process = subprocess.Popen(
                ["qi", "login"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            print("WORKER: QI process started")

            if process.stdout is not None:
                while True:
                    line = process.stdout.readline()

                    if line == "":
                        break

                    print("WORKER QI:", line.strip())

            return_code = process.wait()

            print("WORKER: QI exited:", return_code)

        threading.Thread(
            target=worker,
            daemon=False
        ).start()

        print("MAIN: login() returned")


    def check_login(self) -> bool:
        """
        Check whether the local Quantum Inspire authentication configuration exists.
        """
        config_path = os.path.expanduser(
            "~/.quantuminspire/config.json")

        return os.path.exists(config_path)

    def set_provider(self):
        self.provider = QIProvider()

    def update_provider(self):
        print("Provider updated!")
        self.provider = QIProvider()

    #
    # def list_backends(self):
    #     backends = self.provider.backends()
    #     print(backends)
    #     return backends
         
    # # called when user wants to choose a backend
    # def get_backend_info(self):
    #
    #     # Show all current supported backends make the user choose:
    #     available_backend = self.list_backends() 
    #     ## show possible backends: 
    #     ## chosen_one = dropdown_menu(available_backend)
    #
    #     # Get Quantum Inspire backend chosen by the user:
    #     backend = self.provider.get_backend(self.STANDARD_BACKEND)
    #     # backend = provider.get_backend(str(chosen_one)) 
    #
    #     return


    # def status() -> None:
    #     return

    def submit(self, circuit, backend):
        return backend.run(circuit)

    def get_result(self, job):
        return job.result()


    # def set_backend() -> None:
    #     pass
=======
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import threading
import webbrowser

from qiskit_quantuminspire.qi_provider import QIProvider


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

>>>>>>> 5e6153b211fe233cbc42e90f5cf11aaa807bf908
