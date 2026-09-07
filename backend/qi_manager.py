# QuantumBackend funtions

import subprocess

from qiskit_quantuminspire.qi_provider import QIProvider

import subprocess
import os

class QuantumInspireManager:
    def __init__(self):
        """
        Create the Quantum Inspire provider after authentication has been completed.
        """
        self.provider = QIProvider()
        self.backend = None
        self.STANDARD_BACKEND = "QX Emulator"

    def check_login(self) -> bool:
        """
        Check whether the local Quantum Inspire authentication configuration exists.
        """
        config_path = os.path.expanduser(
            "~/.quantuminspire/config.json")

        return os.path.exists(config_path)

    # def set_backend() -> None:
    #     pass

    def login(self) -> bool:
        """
        Launch the official Quantum Inspire login process.
        """
        subprocess.Popen(["qi", "login"])

        ## receive response from the terminal or browser
        return True

    def update_provider(self):
        print("Provider updated!")
        self.provider = QIProvider()


    def list_backends(self):
        backends = self.provider.backends()
        print(backends)
        return backends
         
    # called when user wants to choose a backend
    def get_backend_info(self):

        # Show all current supported backends make the user choose:
        available_backend = self.list_backends() 
        ## show possible backends: 
        ## chosen_one = dropdown_menu(available_backend)

        # Get Quantum Inspire backend chosen by the user:
        backend = self.provider.get_backend(self.STANDARD_BACKEND)
        # backend = provider.get_backend(str(chosen_one)) 

        return


    def status():
        return

    def submit(self, circuit, backend):
        return backend.run(circuit)

    def get_result(self, job):
        return job.result()
