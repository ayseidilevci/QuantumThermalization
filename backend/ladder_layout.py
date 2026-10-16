"""Drop-in replacement for the backend-selection lines at the top of
gibbs_tfd_reproduction/experiment.py. Works unchanged for any backend
name: "Tuna-5", "Tuna-9", "Tuna-17", or "Starmon-7".
 
Replaces this:
 
    backend = NoisySimulator("Starmon-7", ideal_simulation=True)
    initial_layout = [0, 2, 3, 5]
 
Repo layout this assumes:
 
    QuantumThermalization/
        backend/
            qi_manager.py
            ladder_layout.py
            starmon7.py
        gibbs_tfd_reproduction/
            experiment.py   <-- this file's contents go at the top of it
 
gibbs_tfd_reproduction/experiment.py uses bare sibling imports
("from ground_truth import ..."), which means it's run with
gibbs_tfd_reproduction/ as the script directory -- so backend/ (a
sibling folder, not a child) is not on sys.path by default. The sys.path
line below adds the project root so "from backend.qi_manager import ..."
resolves the same way it already does inside backend/qi_manager.py
itself.
"""
 
import os
import sys
import threading
 
# Make the project root importable so "backend.*" resolves, regardless
# of whether this script is run from gibbs_tfd_reproduction/ or
# elsewhere. Safe to run every time -- inserting a path twice is harmless.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
 
from backend.qi_manager import QuantumInspireManager
from backend.ladder_layout import get_initial_layout
 
# --- Pick the backend here. Nothing else below needs to change when you
# switch devices. ---
BACKEND_NAME = "Tuna-9"          # "Tuna-5" / "Tuna-9" / "Tuna-17" / "Starmon-7"
EXCLUDE_QUBITS = ()              # e.g. (8,) to skip Tuna-9's flagged Q8
 
 
manager = QuantumInspireManager()
 
if not manager.check_login():
    print("Not logged in -- starting Quantum Inspire login...")
    done = threading.Event()
 
    def on_code(code):
        print(f"Open the browser and enter verification code: {code}")
 
    def on_complete(result):
        if result != 0:
            print("Login failed.")
        done.set()
 
    manager.login(on_code=on_code, on_complete=on_complete)
    done.wait()  # blocks until the browser login finishes
 
    if manager.provider is None:
        raise RuntimeError("Quantum Inspire login did not succeed.")
 
backend = manager.get_backend(BACKEND_NAME)
initial_layout = get_initial_layout(backend, exclude_qubits=EXCLUDE_QUBITS)
 
print(f"{BACKEND_NAME}: using physical qubits {initial_layout} "
      f"(logical order A2, A1, B2, B1)")
 
# `backend` and `initial_layout` are now exactly what CostFunction(...)
# expects -- everything below this point in experiment.py is unchanged:
#
#   cost_function = CostFunction(varsigma=1.57, backend=backend,
#                                 initial_layout=initial_layout)
 