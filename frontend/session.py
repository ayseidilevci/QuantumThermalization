"""Experiment configuration shared across Home -> Setup -> Run.

Also defines the on-disk JSON schema used by Home's "Load" action (see
frontend/upload.py) so the two stay in lockstep. Bump SESSION_SCHEMA_VERSION
and branch on it in from_json_dict if the shape below ever changes.
"""

from dataclasses import dataclass
from typing import Optional

SESSION_SCHEMA_VERSION = 1

# The four ansatz roles a physical qubit can be assigned to — see
# gibbs_tfd_reproduction/tfd_circuit.py for the logical-qubit convention
# (q0=A2, q1=A1, q2=B2, q3=B1) that qubit_mapping below gets translated into.
ROLES = ("A1", "A2", "B1", "B2")


@dataclass
class SessionConfig:
    backend_name: str = "QX emulator"
    beta: float = 1.0
    optimizer: str = "COBYLA"
    iterations: int = 200

    # Initial guess for the variational optimizer, not fixed final angles —
    # minimize_cost() adjusts these during the run (see frontend/setup.py).
    gamma1: float = 0.0
    gamma2: float = 0.0
    alpha1: float = 0.0
    alpha2: float = 0.0

    # role -> physical qubit index, e.g. {"A1": 1, "A2": 0, "B1": 3, "B2": 2}.
    # None / incomplete until the user finishes the device panel on Setup.
    qubit_mapping: Optional[dict] = None

    def is_qubit_mapping_complete(self) -> bool:
        return self.qubit_mapping is not None and all(
            role in self.qubit_mapping for role in ROLES
        )

    def to_json_dict(self) -> dict:
        return {
            "schema_version": SESSION_SCHEMA_VERSION,
            "backend_name": self.backend_name,
            "beta": self.beta,
            "optimizer": self.optimizer,
            "iterations": self.iterations,
            "gamma1": self.gamma1,
            "gamma2": self.gamma2,
            "alpha1": self.alpha1,
            "alpha2": self.alpha2,
            "qubit_mapping": self.qubit_mapping,
        }

    @classmethod
    def from_json_dict(cls, data: dict) -> "SessionConfig":
        # schema_version is unused today (only version 1 exists) but is kept
        # in the file so a future loader can branch on it.
        return cls(
            backend_name=str(data.get("backend_name", "QX emulator")),
            beta=float(data.get("beta", 1.0)),
            optimizer=str(data.get("optimizer", "COBYLA")),
            iterations=int(data.get("iterations", 200)),
            gamma1=float(data.get("gamma1", 0.0)),
            gamma2=float(data.get("gamma2", 0.0)),
            alpha1=float(data.get("alpha1", 0.0)),
            alpha2=float(data.get("alpha2", 0.0)),
            qubit_mapping=data.get("qubit_mapping"),
        )
