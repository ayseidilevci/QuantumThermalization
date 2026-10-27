"""Experiment configuration shared across Home -> Setup -> Run.

Also defines the on-disk JSON schema used by Home's "Load" action (see
frontend/upload_browser.py) so the two stay in lockstep. Bump
SESSION_SCHEMA_VERSION and branch on it in from_json_dict if the shape
below ever changes.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

SESSION_SCHEMA_VERSION = 1

# Small local history of session files opened/saved from this app (see
# record_recent_session / list_recent_sessions). No longer shown on the Load
# screen — that lists experiment folders instead (backend.experiment_manager).
# Separate
# from qi_utilities' own job-record storage — this is just a most-recently-
# used list of *parameter* files, not submitted-job history.
_RECENTS_PATH = Path.home() / ".quantuminspire" / "gui_recent_sessions.json"
_MAX_RECENTS = 15


def _read_recents_raw() -> list:
    if not _RECENTS_PATH.exists():
        return []
    try:
        with _RECENTS_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def _write_recents_raw(entries: list) -> None:
    try:
        _RECENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _RECENTS_PATH.open("w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2)
    except OSError:
        pass  # a failed write here shouldn't block loading/saving a session


def list_recent_sessions() -> list:
    """Most-recent-first list of {"path": str, "used_at": iso-str} entries.
    Entries whose file has since been deleted/moved are silently dropped
    (and the pruned list re-saved) rather than shown as dead links.
    """
    entries = _read_recents_raw()
    alive = [e for e in entries if isinstance(e, dict) and "path" in e
             and Path(e["path"]).exists()]
    if len(alive) != len(entries):
        _write_recents_raw(alive)
    return alive


def record_recent_session(path) -> None:
    """Move `path` to the front of the recents list, creating the entry if
    it's new. Called on both a successful load and a save."""
    path_str = str(Path(path).resolve())
    entries = [e for e in _read_recents_raw() if e.get("path") != path_str]
    entries.insert(0, {
        "path": path_str,
        "used_at": datetime.now().isoformat(timespec="seconds"),
    })
    _write_recents_raw(entries[:_MAX_RECENTS])


# The four ansatz roles a physical qubit can be assigned to — see
# gibbs_tfd_reproduction/tfd_circuit.py for the logical-qubit convention
# (q0=A2, q1=A1, q2=B2, q3=B1) that qubit_mapping below gets translated into.
ROLES = ("A1", "A2", "B1", "B2")

# Default initial value for every variational angle. Not 0: the cost is
# even in the angles (C(x) == C(-x)), so all-zero angles are a stationary
# point at every beta, the correct answer at beta -> 0 but a saddle at
# beta > 0. Optimizers started exactly there often never leave it (COBYLA
# only escapes by shot-noise luck). 0.1 is the
# start gibbs_tfd_reproduction/experiment.py uses.
DEFAULT_START_ANGLE = 0.1

# Real-hardware runs always use SPSA: it estimates the gradient from two
# noisy cost evaluations per step and copes with shot noise far better than
# COBYLA's model-based steps (see noisy_optimizer_benchmark.py).
HARDWARE_OPTIMIZER = "SPSA"


def random_start_angles(rng) -> np.ndarray:
    """Random initial (γ₁, γ₂, α₁, α₂) for a cold start: each |angle| uniform
    in [0.1, π] with a random sign, which stays clear of the all-zero saddle
    (same draw as gibbs_tfd_reproduction/experiments/reproducibility_check.py)."""
    return rng.uniform(0.1, np.pi, size=4) * rng.choice([-1.0, 1.0], size=4)


# Selectable inverse temperatures, in units of 1/g (g = the transverse
# field of the Hamiltonian). beta = 0 itself can't be optimized — the cost's
# beta^-1.57 term diverges — so choosing 0 runs at BETA_ZERO instead.
BETA_CHOICES = (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0)
BETA_ZERO = 1e-8

# Boltzmann-constant unit systems for displaying T = 1 / (k_B beta):
# key -> (short label, k_B as text, k_B, energy unit of g). Outside natural
# units the coupling g is taken as 1 of that energy unit, so T is in kelvin.
KB_UNITS = {
    "natural": ("Natural", "k_B = 1", 1.0, None),
    "eV": ("eV/K", "k_B = 8.617×10⁻⁵ eV/K", 8.617333262e-5, "eV"),
    "GHz": ("GHz/K", "k_B/h = 20.84 GHz/K", 20.83661912, "GHz"),
    "J": ("J/K", "k_B = 1.381×10⁻²³ J/K", 1.380649e-23, "J"),
}
DEFAULT_KB_UNIT = "natural"

_SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def nearest_beta_choice(beta: float) -> float:
    """Snap a (possibly older, free-form) beta to the closest selectable value."""
    return min(BETA_CHOICES, key=lambda choice: abs(choice - beta))


def format_beta(beta: float) -> str:
    return "0" if beta <= BETA_ZERO else f"{beta:g}"


def _format_number(value: float) -> str:
    if 100 <= abs(value) < 1e4:
        return f"{value:.0f}"
    if 1e-2 <= abs(value) < 100:
        return f"{value:.3g}"
    mantissa, exponent = f"{value:.2e}".split("e")
    return f"{mantissa}×10{str(int(exponent)).translate(_SUPERSCRIPT)}"


def format_temperature(beta: float, kb_unit: str = DEFAULT_KB_UNIT) -> str:
    """T = 1 / (k_B beta) for display, e.g. "T = 0.5 g/k_B" or "T ≈ 1.16×10⁴ K"."""
    if beta <= BETA_ZERO:
        return "T → ∞"
    _, _, k_b, energy_unit = KB_UNITS.get(kb_unit, KB_UNITS[DEFAULT_KB_UNIT])
    if energy_unit is None:
        return f"T = {_format_number(1 / beta)} g/k_B"
    return f"T ≈ {_format_number(1 / (k_b * beta))} K"


@dataclass
class SessionConfig:
    backend_name: str = "Starmon-7"
    beta: float = 1.0
    # Only changes how the temperature is displayed, not the optimization.
    kb_unit: str = DEFAULT_KB_UNIT
    optimizer: str = "COBYLA"
    iterations: int = 200

    # Initial guess for the variational optimizer, not fixed final angles —
    # minimize_cost() adjusts these during the run (see frontend/setup.py).
    gamma1: float = DEFAULT_START_ANGLE
    gamma2: float = DEFAULT_START_ANGLE
    alpha1: float = DEFAULT_START_ANGLE
    alpha2: float = DEFAULT_START_ANGLE

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
            "kb_unit": self.kb_unit,
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
            backend_name=str(data.get("backend_name", "Starmon-7")),
            beta=float(data.get("beta", 1.0)),
            kb_unit=str(data.get("kb_unit", DEFAULT_KB_UNIT)),
            optimizer=str(data.get("optimizer", "COBYLA")),
            iterations=int(data.get("iterations", 200)),
            gamma1=float(data.get("gamma1", DEFAULT_START_ANGLE)),
            gamma2=float(data.get("gamma2", DEFAULT_START_ANGLE)),
            alpha1=float(data.get("alpha1", DEFAULT_START_ANGLE)),
            alpha2=float(data.get("alpha2", DEFAULT_START_ANGLE)),
            qubit_mapping=data.get("qubit_mapping"),
        )
