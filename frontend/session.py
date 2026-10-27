"""Experiment configuration shared across Home -> Setup -> Run.

Also defines the on-disk JSON schema used by Home's "Load" action (see
frontend/upload_browser.py) so the two stay in lockstep. Bump
SESSION_SCHEMA_VERSION and branch on it in from_json_dict if the shape
below ever changes.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

SESSION_SCHEMA_VERSION = 1

# Spins per system: n = 2 is the Sagastizabal et al. experiment (2+2 qubits);
# 3+3 and 4+4 use the n+n generalisation in tfd_study/ (model.py).
SPIN_CHOICES = (2, 3, 4)

# 3+3 and 4+4 run on Tuna-17 only. By connectivity alone Tuna-9 could host a
# 3+3 ladder too; it's left out by choice, so this is the one place to change.
MULTI_SPIN_BACKENDS = ("tuna17",)


def roles(n: int) -> tuple:
    """A1..An, B1..Bn: subsystem A's chain and its mirror B (B_i next to A_i)."""
    return tuple(f"A{i}" for i in range(1, n + 1)) + tuple(f"B{i}" for i in range(1, n + 1))


def spins_allowed(backend_name: str, n: int) -> bool:
    normalized = "".join(ch for ch in (backend_name or "").lower() if ch.isalnum())
    return n == 2 or normalized in MULTI_SPIN_BACKENDS

# Default initial value for every variational angle. Not 0: the cost is even
# in the angles, so all-zero angles are a saddle point for beta > 0 that
# optimizers often never leave.
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

    # Spins per system (see SPIN_CHOICES); the ansatz uses 2 * n_spins qubits.
    n_spins: int = 2

    # The optimizer's starting guess; it adjusts the angles during the run.
    gamma1: float = DEFAULT_START_ANGLE
    gamma2: float = DEFAULT_START_ANGLE
    alpha1: float = DEFAULT_START_ANGLE
    alpha2: float = DEFAULT_START_ANGLE

    # role -> physical qubit index, e.g. {"A1": 1, "A2": 0, "B1": 3, "B2": 2}.
    # None / incomplete until the user finishes the device panel on Setup.
    qubit_mapping: Optional[dict] = None

    def is_qubit_mapping_complete(self) -> bool:
        return self.qubit_mapping is not None and all(
            role in self.qubit_mapping for role in roles(self.n_spins)
        )

    def to_json_dict(self) -> dict:
        return {
            "schema_version": SESSION_SCHEMA_VERSION,
            "backend_name": self.backend_name,
            "beta": self.beta,
            "kb_unit": self.kb_unit,
            "optimizer": self.optimizer,
            "iterations": self.iterations,
            "n_spins": self.n_spins,
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
            n_spins=int(data.get("n_spins", 2)),
            gamma1=float(data.get("gamma1", DEFAULT_START_ANGLE)),
            gamma2=float(data.get("gamma2", DEFAULT_START_ANGLE)),
            alpha1=float(data.get("alpha1", DEFAULT_START_ANGLE)),
            alpha2=float(data.get("alpha2", DEFAULT_START_ANGLE)),
            qubit_mapping=data.get("qubit_mapping"),
        )
