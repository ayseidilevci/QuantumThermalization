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

# 3+3 and 4+4 run on Tuna-17 and the QX emulator. By connectivity alone Tuna-9
# could host a 3+3 ladder too; it's left out by choice, so this is the one
# place to change.
MULTI_SPIN_BACKENDS = ("tuna17", "qxemulator")


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


# Spread of the cloud runs' random starting angles around θ = 0 (rad), as
# the tfd-study branch's runs used.
RANDOM_START_SPREAD = 0.4


def random_start_angles(rng) -> np.ndarray:
    """Initial (γ₁, γ₂, α₁, α₂) for a cloud run: small random angles around
    θ = 0. θ = 0 is the exact β = 0 state (Bell pairs) and the solution at any
    β is a smooth deformation of it, so starting nearby lands in the right
    valley; fully random angles get stuck in poor local minima for 3+3 and
    4+4 (median fidelity 0.84 and 0.58 instead of 0.97 and 0.94, in a
    2048-shot simulation at β = 1). Not exactly 0: a saddle of the cost."""
    return rng.normal(0.0, RANDOM_START_SPREAD, size=4)


# Allowed maximum iterations: values typed or loaded outside are set to the nearest limit.
MIN_ITERATIONS = 5
MAX_ITERATIONS = 2000

# Inverse temperature β, unitless: energies in units of the
# Hamiltonian's own scale (g = 1) and k_B = 1, so β and T = 1/β are plain
# numbers. Any β in [BETA_ZERO, BETA_MAX] can be typed. β = 0 itself can't be
# optimized (the cost's β^-1.57 term diverges), so the minimum is just above it.
BETA_MAX = 5.0
BETA_ZERO = 1e-8


def clamp_beta(value: float) -> float:
    """β as the run uses it: clamped into [BETA_ZERO, BETA_MAX]."""
    return max(BETA_ZERO, min(BETA_MAX, float(value)))


def format_beta(beta: float) -> str:
    """Plain decimals (0.00000001, not 1e-08), up to 12 significant digits."""
    return np.format_float_positional(float(f"{beta:.12g}"), trim="-")


def format_temperature_tex(beta: float) -> str:
    """T = 1/β (unitless) as mathtext, without $ delimiters."""
    mantissa, _, exponent = f"{1 / beta:.4g}".partition("e")
    if exponent:
        number = (rf"10^{{{int(exponent)}}}" if mantissa == "1"
                  else rf"{mantissa}\times10^{{{int(exponent)}}}")
    else:
        number = mantissa
    return rf"T = 1/\beta = {number}"


@dataclass
class SessionConfig:
    backend_name: str = "Starmon-7"
    beta: float = 1.0
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
            beta=clamp_beta(data.get("beta", 1.0)),
            # Runs saved with an optimizer no longer offered (BFGS) reuse COBYLA.
            optimizer=data.get("optimizer") if data.get("optimizer") in ("COBYLA", "SPSA") else "COBYLA",
            iterations=min(MAX_ITERATIONS, max(MIN_ITERATIONS, int(data.get("iterations", 200)))),
            n_spins=int(data.get("n_spins", 2)),
            gamma1=float(data.get("gamma1", DEFAULT_START_ANGLE)),
            gamma2=float(data.get("gamma2", DEFAULT_START_ANGLE)),
            alpha1=float(data.get("alpha1", DEFAULT_START_ANGLE)),
            alpha2=float(data.get("alpha2", DEFAULT_START_ANGLE)),
            qubit_mapping=data.get("qubit_mapping"),
        )
