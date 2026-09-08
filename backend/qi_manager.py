# QuantumBackend funtions

import subprocess

from qiskit_quantuminspire.qi_provider import QIProvider

import subprocess
import threading
import os
import re

from dataclasses import dataclass
from typing import Any


@dataclass
class BackendInfo:
    backend: Any
    name: str
    id: int
    status: Any

class QuantumInspireManager:
    def __init__(self):
        """
        Create the Quantum Inspire provider after authentication has been completed.
        """
        self.backends = []
        self.provider = None
        self.userID = None
        self.login_code = None
        self.STANDARD_BACKEND = "QX Emulator"

    def login(self, on_complete):

        def worker():
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}

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

                    # Parse login code
                    code_match = re.search(
                        r"user_code=([A-Z0-9-]+)",
                        line
                    )

                    if code_match:
                        self.login_code = code_match.group(1)

                    # Parse user ID
                    user_id_match = re.search(
                        r"member ID (\d+)",
                        line
                    )

                    if user_id_match:
                        self.userID = int(user_id_match.group(1))

            return_code = process.wait()

            on_complete(return_code)

            print("QI: ", self.userID)
            print("QI: ", self.login_code)

        threading.Thread(
            target=worker,
            daemon=False
        ).start()

    def check_login(self):
        return os.path.exists("~/.quantuminspire/config.json")
        
    def set_provider(self):

        self.provider = QIProvider()

        for backend in self.provider.backends():
            backend_string = str(backend)

            name_match = re.search(
                r"name=([^,]+)",
                backend_string
            )

            id_match = re.search(
                r"id=(\d+)",
                backend_string
            )

            if name_match is None or id_match is None:
                continue

            name = name_match.group(1).strip()
            backend_id = int(id_match.group(1))

            # Get current backend status
            status = backend.status

            # Create our backend structure
            backend_info = BackendInfo(
                backend=backend,
                name=name,
                id=backend_id,
                status=status
            )

            self.backends.append(backend_info)

        # Print everything at the end
        for backend in self.backends:
            print(
                f"Backend: {backend.name}, "
                f"ID: {backend.id}, "
                f"Status: {backend.status}"
            )



    def update_provider(self):
        print("Provider updated!")
        self.provider = QIProvider()

    def submit(self, circuit, backend):
        return backend.run(circuit)

    def get_result(self, job):
        return job.result()

