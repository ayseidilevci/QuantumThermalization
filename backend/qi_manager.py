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
