"""Real-backend three-spin TFD optimization for Quantum Inspire.

This is the data-acquisition/optimization runner for the three-spin TFD extension.
The objective is evaluated from measured expectation values on a Quantum Inspire
backend (QX emulator or Tuna-17); it is never silently evaluated using an ideal
Statevector. It saves raw counts and optimization traces only. Run plot_3spin.py
separately to generate per-beta cost-history and tomography figures.

Fixed logical-to-physical layout chosen from the supplied Tuna-17 connectivity
figure (logical qubit -> physical Tuna-17 qubit):
    q0=A1 -> Q0, q1=A2 -> Q2, q2=A3 -> Q5,
    q3=B1 -> Q1, q4=B2 -> Q4, q5=B3 -> Q8.

These six physical qubits embed the required open-chain six-qubit ladder:
A-chain Q0-Q2-Q5, B-chain Q1-Q4-Q8, with rungs Q0-Q1, Q2-Q4 and Q5-Q8.
The same initial_layout is used when selecting QX emulator, so the logical
layout is not changed between the two backends.

Readout mitigation is optional. When enabled, the script invokes the
QuantumInspireUtilities readout_correction.measure_ro_assignment_matrix, which
uses circuit_modifiers.apply_readout_circuit to perform mid-circuit calibration
measurements, then corrects measured X/Z distributions with
readout_correction.get_ro_corrected_multi_probs. Arbitrary mid-circuit
measurements are deliberately not inserted into the variational ansatz itself,
because those would collapse the state and change the objective being optimized.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.optimize import minimize

from qiskit import QuantumCircuit, transpile

# Logical qubits are ordered as q0=A1, q1=A2, q2=A3, q3=B1, q4=B2, q5=B3.
N_QUBITS = 6
N_A = 3
PARAMETERS_PER_LAYER = 4
# Logical q0..q5 -> physical Tuna-17 qubits Q0, Q2, Q5, Q1, Q4, Q8.
TUNA17_LAYOUT = (0, 2, 5, 1, 4, 8)
DEFAULT_BETAS = (1.0,)
DEFAULT_VARSIGMA = 2.05
DEFAULT_SHOTS = 1024
DEFAULT_MAXITER = 30
DEFAULT_OPTIMIZER = "COBYLA"
MAX_CIRCUITS_PER_BATCH = 5
TOMOGRAPHY_SETTINGS = tuple(itertools.product("XYZ", repeat=N_A))
BINARY_LABELS_6 = tuple(format(i, "06b") for i in range(2**N_QUBITS))
BINARY_LABELS_3 = tuple(format(i, "03b") for i in range(2**N_A))

# All two-qubit logical edges used by initial TFD preparation and the ansatz.
LOGICAL_TWO_QUBIT_EDGES = (
    (0, 1), (1, 2),       # A chain
    (3, 4), (4, 5),       # B chain
    (0, 3), (1, 4), (2, 5),  # A-B Bell-pair rungs / cross couplings
)


class TFD3SpinCircuit:
    """Construct the six-qubit variational TFD ansatz with p shared-angle layers.

    Each layer has four angles (gamma1, gamma2, alpha1, alpha2). Bell pairs are
    prepared only once at the beginning; each following variational layer has
    four independent angles. The open-chain circuit has only the seven logical
    two-qubit edges in LOGICAL_TWO_QUBIT_EDGES, all supported by TUNA17_LAYOUT.
    """

    N_QUBITS = 6
    PARAMETERS_PER_LAYER = 4

    def __init__(self, params: Sequence[float], p_layers: int | None = None,
                 periodic: bool = False):
        values = np.asarray(params, dtype=float)
        if values.ndim != 1 or values.size == 0 or values.size % PARAMETERS_PER_LAYER:
            raise ValueError("params must contain exactly 4 parameters per variational layer")
        inferred_layers = values.size // PARAMETERS_PER_LAYER
        self.p_layers = inferred_layers if p_layers is None else int(p_layers)
        if self.p_layers < 1:
            raise ValueError("p_layers must be >= 1")
        if values.size != PARAMETERS_PER_LAYER * self.p_layers:
            raise ValueError(
                f"Expected {PARAMETERS_PER_LAYER * self.p_layers} parameters for "
                f"p_layers={self.p_layers}; received {values.size}"
            )
        if periodic:
            raise NotImplementedError(
                "This hardware layout implements the open chain. Periodic A3-A1/B3-B1 "
                "edges are not part of the fixed Tuna-17 patch."
            )
        self.params = values.copy()
        self.periodic = False
        self.circuit = QuantumCircuit(N_QUBITS, name=f"TFD3Spin_p{self.p_layers}")
        self.build_circuit(self.circuit)

    def build_circuit(self, circuit: QuantumCircuit | None = None) -> QuantumCircuit:
        if circuit is None:
            circuit = self.circuit

        # |Phi+>_{A_i,B_i} = (|00> + |11>)/sqrt(2), for i=1,2,3.
        for a, b in zip((0, 1, 2), (3, 4, 5)):
            circuit.h(b)
            circuit.cx(b, a)
        circuit.barrier(*range(N_QUBITS))

        intra_chain_edges = ((0, 1), (1, 2), (3, 4), (4, 5))
        cross_edges = ((0, 3), (1, 4), (2, 5))

        for layer in range(self.p_layers):
            gamma1, gamma2, alpha1, alpha2 = self.params[
                PARAMETERS_PER_LAYER * layer: PARAMETERS_PER_LAYER * (layer + 1)
            ]
            circuit.rx(float(gamma1), range(N_QUBITS))
            circuit.barrier(*range(N_QUBITS))
            for q1, q2 in intra_chain_edges:
                circuit.rzz(float(gamma2), q1, q2)
            circuit.barrier(*range(N_QUBITS))
            for a, b in cross_edges:
                circuit.rxx(float(alpha1), a, b)
            circuit.barrier(*range(N_QUBITS))
            for a, b in cross_edges:
                circuit.rzz(float(alpha2), a, b)
            circuit.barrier(*range(N_QUBITS))

        return circuit

    def as_qiskit_circuit(self) -> QuantumCircuit:
        return self.circuit


# ---------- Backend/provider and fixed-layout validation ----------


def backend_name(backend: Any) -> str:
    name = getattr(backend, "name", None)
    if callable(name):
        name = name()
    return str(name if name is not None else backend)


def create_qi_backend(name: str):
    """Return a real Quantum Inspire SDK backend object; caller must have logged in."""
    try:
        from qiskit_quantuminspire.qi_provider import QIProvider
    except ImportError as exc:
        raise RuntimeError(
            "Quantum Inspire SDK not found. Install qiskit-quantuminspire in this Python environment."
        ) from exc
    provider = QIProvider()
    available = [backend_name(item) for item in provider.backends()]
    print("Quantum Inspire backends:", ", ".join(available))
    try:
        backend = provider.get_backend(name=name)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot select backend {name!r}. Available backends: {available}. "
            "Run `qi login` in this same Python environment first."
        ) from exc
    return backend


def _coupling_edges(backend: Any) -> set[frozenset[int]] | None:
    """Return undirected edges from the Qiskit backend target/coupling map, if available."""
    coupling_map = getattr(backend, "coupling_map", None)
    if coupling_map is None:
        target = getattr(backend, "target", None)
        if target is not None:
            try:
                coupling_map = target.build_coupling_map()
            except Exception:
                coupling_map = None
    if coupling_map is None:
        return None
    try:
        return {frozenset((int(edge[0]), int(edge[1]))) for edge in coupling_map.get_edges()}
    except Exception:
        return None


def validate_fixed_layout(backend: Any, layout: Sequence[int] = TUNA17_LAYOUT) -> None:
    """Fail early if the fixed six-qubit layout is not valid for the chosen backend."""
    layout = tuple(int(q) for q in layout)
    if len(layout) != N_QUBITS or len(set(layout)) != N_QUBITS or min(layout) < 0:
        raise ValueError(f"Layout must contain {N_QUBITS} distinct non-negative physical indices")
    n_backend_qubits = getattr(backend, "num_qubits", None)
    if n_backend_qubits is not None and max(layout) >= int(n_backend_qubits):
        raise ValueError(
            f"Fixed layout {layout} needs at least {max(layout) + 1} physical qubits, "
            f"but {backend_name(backend)} reports {n_backend_qubits}."
        )

    edges = _coupling_edges(backend)
    selected_name = backend_name(backend).lower()
    if edges is None:
        if "tuna-17" in selected_name:
            raise RuntimeError(
                "Could not read the Tuna-17 coupling map from the backend object; refusing to "
                "claim that the physical layout has been validated. Check your SDK/backend version."
            )
        # Some complete-connectivity backends expose no coupling map. QX emulator is all-to-all.
        print("Coupling-map metadata unavailable; no topology check was possible for this backend.")
        return

    required_physical_edges = {
        frozenset((layout[a], layout[b])) for a, b in LOGICAL_TWO_QUBIT_EDGES
    }
    missing = sorted(tuple(sorted(edge)) for edge in required_physical_edges - edges)
    if missing:
        if "tuna-17" in selected_name:
            raise RuntimeError(
                "The fixed logical-to-physical mapping does not match the Tuna-17 coupling map. "
                f"Missing physical couplings: {missing}. No job has been submitted."
            )
        print(
            "Warning: the chosen backend does not contain every direct edge in the fixed ladder "
            f"layout (missing={missing}); the transpiler may need routing SWAPs."
        )
    else:
        print("Fixed layout check passed: all seven ansatz/Bell-pair couplings exist on this backend.")


def logical_layout_description(layout: Sequence[int]) -> str:
    logical_names = ("A1", "A2", "A3", "B1", "B2", "B3")
    return ", ".join(f"q{i}={logical_names[i]} -> Q{physical}"
                      for i, physical in enumerate(layout))


# ---------- Measurement and expectation values ----------


def _counts_to_probability_vector(
    counts: dict,
    n_qubits: int,
    layout: Sequence[int] | None = None,
) -> np.ndarray:
    """Decode bit-mask-style backend counts into logical outcome probabilities.

    ``layout[i]`` maps logical qubit i to the physical wire index used by the
    transpiler. QI can return a shorter binary key when leading physical bits
    are zero, so all keys are interpreted as physical bit masks and remapped,
    regardless of their displayed string length.
    """
    if layout is None:
        if n_qubits > len(TUNA17_LAYOUT):
            raise ValueError(f"No default physical layout is available for {n_qubits} qubits")
        layout = TUNA17_LAYOUT[:n_qubits]
    layout = tuple(int(q) for q in layout[:n_qubits])
    if len(layout) != n_qubits or len(set(layout)) != n_qubits or min(layout, default=0) < 0:
        raise ValueError(f"layout must contain {n_qubits} distinct non-negative physical indices")

    probabilities = np.zeros(2**n_qubits, dtype=float)
    total = 0.0
    largest_physical_index = max(layout, default=-1)

    for key, count in counts.items():
        if isinstance(key, (int, np.integer)):
            physical_outcome = int(key)
        else:
            text = str(key).replace(" ", "").replace("_", "")
            if text.startswith("0x"):
                physical_outcome = int(text, 16)
            elif text.startswith("0b"):
                physical_outcome = int(text, 2)
            else:
                physical_outcome = int(text, 2)

        if physical_outcome < 0:
            raise ValueError(f"Negative measurement outcome {key!r} is invalid")
        if physical_outcome >> (largest_physical_index + 1):
            raise ValueError(
                f"Measurement outcome {key!r} sets a physical bit outside layout {layout}; "
                f"highest permitted physical index is {largest_physical_index}."
            )

        logical_outcome = 0
        for logical_q, physical_q in enumerate(layout):
            if (physical_outcome >> physical_q) & 1:
                logical_outcome |= 1 << logical_q

        probabilities[logical_outcome] += float(count)
        total += float(count)

    if total <= 0:
        raise ValueError("Backend returned zero total measurement shots")
    return probabilities / total


def _probability_dict(counts: dict, n_qubits: int) -> dict[str, float]:
    vector = _counts_to_probability_vector(counts, n_qubits)
    labels = tuple(format(i, f"0{n_qubits}b") for i in range(2**n_qubits))
    # This insertion order matches the utility functions' obtain_binary_list().
    return {label: float(vector[i]) for i, label in enumerate(labels)}


def _expectation(probabilities: np.ndarray, qubits: Sequence[int]) -> float:
    """Estimate a Pauli-product expectation from measurement probabilities."""
    total = 0.0
    for outcome, probability in enumerate(probabilities):
        parity = sum((outcome >> q) & 1 for q in qubits) & 1
        total += float(probability) * (-1.0 if parity else 1.0)
    return float(total)


def _make_cost_measurement_circuit(ansatz: QuantumCircuit, basis: str) -> QuantumCircuit:
    """Append global X-basis rotations or direct Z measurements to the ansatz."""
    if basis not in ("X", "Z"):
        raise ValueError("basis must be 'X' or 'Z'")
    qc = QuantumCircuit(N_QUBITS, N_QUBITS, name=f"cost_measure_{basis}")
    qc.compose(ansatz, qubits=list(range(N_QUBITS)), inplace=True)
    qc.barrier(*range(N_QUBITS))
    if basis == "X":
        qc.h(range(N_QUBITS))
        qc.barrier(*range(N_QUBITS))
    qc.measure(range(N_QUBITS), range(N_QUBITS))
    return qc


def submit_circuits_in_batches(
    circuits: Sequence[QuantumCircuit],
    backend: Any,
    layout: Sequence[int],
    shots: int,
    optimization_level: int,
    result_timeout: int = 3600,
    batch_size: int = MAX_CIRCUITS_PER_BATCH,
    memory: bool = False,
    label: str = "circuits",
) -> tuple[list[dict[str, int]], list[QuantumCircuit]]:
    """Transpile and submit circuits in batches of at most five.

    Counts and compiled circuits are returned in the same order as ``circuits``.
    ``backend.run`` is never given more than ``batch_size`` circuits at once.
    """
    circuit_list = list(circuits)
    if batch_size < 1 or batch_size > MAX_CIRCUITS_PER_BATCH:
        raise ValueError(f"batch_size must be between 1 and {MAX_CIRCUITS_PER_BATCH}")
    if not circuit_list:
        return [], []

    n_batches = (len(circuit_list) + batch_size - 1) // batch_size
    all_counts: list[dict[str, int]] = []
    all_compiled: list[QuantumCircuit] = []

    for batch_idx, start in enumerate(range(0, len(circuit_list), batch_size), start=1):
        batch = circuit_list[start:start + batch_size]
        compiled = transpile(
            batch,
            backend=backend,
            initial_layout=list(layout),
            optimization_level=optimization_level,
        )
        if isinstance(compiled, QuantumCircuit):
            compiled_batch = [compiled]
        else:
            compiled_batch = list(compiled)

        run_kwargs: dict[str, Any] = {"shots": int(shots)}
        if memory:
            run_kwargs["memory"] = True

        print(
            f"  [{label}] submitting batch {batch_idx}/{n_batches}: "
            f"{len(compiled_batch)} circuit(s)"
        )
        job = backend.run(compiled_batch, **run_kwargs)
        result = job.result(timeout=int(result_timeout))

        for circuit_idx in range(len(compiled_batch)):
            counts = result.get_counts(circuit_idx)
            all_counts.append({str(key): int(value) for key, value in counts.items()})
        all_compiled.extend(compiled_batch)

    if len(all_counts) != len(circuit_list):
        raise RuntimeError(
            f"Expected results for {len(circuit_list)} circuits but received {len(all_counts)}"
        )
    return all_counts, all_compiled


def prepare_output_directory(output_dir: Path, *, clear: bool = True) -> Path:
    """Create the output folder, clearing its contents on each run by default.

    To avoid accidental destructive paths, the directory must be inside the
    current working directory and cannot be the current directory itself.
    """
    requested = Path(output_dir).expanduser()
    resolved = requested.resolve()
    cwd = Path.cwd().resolve()
    forbidden = {Path("/").resolve(), Path.home().resolve(), cwd}
    if resolved in forbidden or cwd not in resolved.parents:
        raise ValueError(
            f"Refusing to use output directory {resolved}: choose a dedicated folder "
            "inside the repository, such as outputs/three_qubits."
        )
    if resolved.exists() and not resolved.is_dir():
        raise NotADirectoryError(f"Output path exists but is not a directory: {resolved}")

    resolved.mkdir(parents=True, exist_ok=True)
    if clear:
        for child in resolved.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
        print(f"[output] cleared existing contents of {resolved}")
    else:
        print(f"[output] keeping existing contents of {resolved}")
    return resolved


def _correct_with_qi_utilities(probabilities: np.ndarray, ro_matrix: np.ndarray,
                               qubit_list: Sequence[int]) -> np.ndarray:
    """Apply the utility repo's constrained readout correction to a distribution."""
    try:
        from qi_utilities.utility_functions.readout_correction import get_ro_corrected_multi_probs
    except ImportError as exc:
        raise RuntimeError(
            "Readout mitigation was requested, but qi-utilities is not importable. "
            "Install it with `pip install qi-utilities` in this environment."
        ) from exc
    n_qubits = len(qubit_list)
    labels = tuple(format(i, f"0{n_qubits}b") for i in range(2**n_qubits))
    probs_dict = {label: float(probabilities[i]) for i, label in enumerate(labels)}
    corrected_list = get_ro_corrected_multi_probs([probs_dict], ro_matrix, list(qubit_list))
    corrected_dict = corrected_list[0]
    corrected = np.asarray([float(corrected_dict[label]) for label in labels], dtype=float)
    corrected = np.clip(corrected, 0.0, None)
    total = corrected.sum()
    if not np.isfinite(total) or total <= 0:
        raise RuntimeError("QIUtilities readout correction returned an invalid probability vector")
    return corrected / total


class CostFunction3SpinBackend:
    """Evaluate C_varsigma(beta) from backend-measured expectation values.

    An evaluation submits two circuits in one QI job: one measured globally in X
    (to obtain X_A, X_B and XX_BA), and one measured in Z (to obtain ZZ_A, ZZ_B,
    and ZZ_BA). There is no ideal-statevector shortcut.
    """

    def __init__(self, backend: Any, varsigma: float = DEFAULT_VARSIGMA,
                 p_layers: int = 1, shots: int = DEFAULT_SHOTS,
                 optimizer: str = DEFAULT_OPTIMIZER,
                 layout: Sequence[int] = TUNA17_LAYOUT,
                 optimization_level: int = 1,
                 readout_mitigation: bool = False,
                 calibration_shots: int = 512,
                 calibration_interval: int = 10,
                 result_timeout: int = 3600):
        if not np.isfinite(varsigma):
            raise ValueError("varsigma must be finite")
        if int(p_layers) < 1:
            raise ValueError("p_layers must be >= 1")
        if int(shots) < 1:
            raise ValueError("shots must be positive")
        if int(calibration_shots) < 1:
            raise ValueError("calibration_shots must be positive")
        if int(calibration_interval) < 1:
            raise ValueError("calibration_interval must be positive")
        if optimizer.lower() not in {"cobyla", "nelder-mead", "neldermead"}:
            raise ValueError("optimizer must be COBYLA or Nelder-Mead")
        self.backend = backend
        self.varsigma = float(varsigma)
        self.p_layers = int(p_layers)
        self.shots = int(shots)
        self.optimizer = "Nelder-Mead" if optimizer.lower().replace("-", "") == "neldermead" else "COBYLA"
        self.layout = tuple(int(q) for q in layout)
        self.optimization_level = int(optimization_level)
        self.readout_mitigation = bool(readout_mitigation)
        self.calibration_shots = int(calibration_shots)
        self.calibration_interval = int(calibration_interval)
        self.result_timeout = int(result_timeout)
        self.cost_history: list[float] = []
        self.parameter_history: list[np.ndarray] = []
        self.expectation_history: list[dict[str, float]] = []
        self.raw_measurement_history: list[dict[str, Any]] = []
        self.ro_assignment_matrix: np.ndarray | None = None
        self.total_evaluations = 0
        self._calibration_runs = 0

        max_shots = getattr(backend, "max_shots", None)
        if max_shots is not None and self.shots > int(max_shots):
            raise ValueError(f"shots={self.shots} exceeds {backend_name(backend)} max_shots={max_shots}")
        validate_fixed_layout(backend, self.layout)

    def _transpile(self, circuits: QuantumCircuit | Sequence[QuantumCircuit]):
        return transpile(
            circuits,
            backend=self.backend,
            initial_layout=list(self.layout),
            optimization_level=self.optimization_level,
        )

    def _run_circuits(self, circuits: Sequence[QuantumCircuit], *, shots: int | None = None,
                      memory: bool = False):
        """Submit circuits in chunks of at most five and preserve input order."""
        return submit_circuits_in_batches(
            circuits=circuits,
            backend=self.backend,
            layout=self.layout,
            shots=self.shots if shots is None else int(shots),
            optimization_level=self.optimization_level,
            result_timeout=self.result_timeout,
            batch_size=MAX_CIRCUITS_PER_BATCH,
            memory=memory,
            label="cost evaluation",
        )

    def calibrate_readout(self) -> np.ndarray:
        """Measure a 6-qubit assignment matrix using QIUtilities mid-circuit calibration."""
        if not self.readout_mitigation:
            raise RuntimeError("Readout mitigation is disabled")
        try:
            from qi_utilities.utility_functions.readout_correction import measure_ro_assignment_matrix
        except ImportError as exc:
            raise RuntimeError(
                "Readout mitigation needs QuantumInspireUtilities. Install with `pip install qi-utilities`."
            ) from exc
        print(
            f"  [readout calibration] submitting mid-circuit assignment-matrix circuit "
            f"on physical qubits {list(self.layout)} ({self.calibration_shots} shots)"
        )
        matrix = measure_ro_assignment_matrix(
            backend=self.backend,
            qubit_list=list(self.layout),
            num_shots=self.calibration_shots,
        )
        if not isinstance(matrix, np.ndarray) or matrix.shape != (2**N_QUBITS, 2**N_QUBITS):
            raise RuntimeError(
                "QIUtilities did not return a 64x64 readout assignment matrix. "
                f"Received {type(matrix).__name__} with shape {getattr(matrix, 'shape', None)}."
            )
        self.ro_assignment_matrix = matrix.astype(float, copy=True)
        self._calibration_runs += 1
        return self.ro_assignment_matrix

    def _maybe_calibrate_readout(self):
        if not self.readout_mitigation:
            return
        if self.ro_assignment_matrix is None or (
            self.total_evaluations > 0 and self.total_evaluations % self.calibration_interval == 0
        ):
            self.calibrate_readout()

    @staticmethod
    def _estimate_expectations(p_x: np.ndarray, p_z: np.ndarray) -> dict[str, float]:
        return {
            "X_A": sum(_expectation(p_x, [q]) for q in (0, 1, 2)),
            "X_B": sum(_expectation(p_x, [q]) for q in (3, 4, 5)),
            "ZZ_A": _expectation(p_z, [0, 1]) + _expectation(p_z, [1, 2]),
            "ZZ_B": _expectation(p_z, [3, 4]) + _expectation(p_z, [4, 5]),
            "XX_BA": sum(_expectation(p_x, [a, b]) for a, b in ((0, 3), (1, 4), (2, 5))),
            "ZZ_BA": sum(_expectation(p_z, [a, b]) for a, b in ((0, 3), (1, 4), (2, 5))),
        }

    def cost_function(self, params: Sequence[float], beta: float) -> float:
        beta = float(beta)
        if not np.isfinite(beta) or beta <= 0:
            raise ValueError("Backend optimization requires finite beta > 0")
        params = np.asarray(params, dtype=float)
        if params.shape != (PARAMETERS_PER_LAYER * self.p_layers,):
            raise ValueError(f"Expected {PARAMETERS_PER_LAYER * self.p_layers} parameters, got {params.shape}")

        self._maybe_calibrate_readout()
        ansatz = TFD3SpinCircuit(params, p_layers=self.p_layers).as_qiskit_circuit()
        circuits = [_make_cost_measurement_circuit(ansatz, "X"),
                    _make_cost_measurement_circuit(ansatz, "Z")]
        circuit_counts, _compiled = self._run_circuits(circuits, shots=self.shots, memory=False)
        counts_x, counts_z = circuit_counts
        self.raw_measurement_history.append({
            "beta": beta,
            "evaluation_within_beta": len(self.cost_history) + 1,
            "global_evaluation": self.total_evaluations + 1,
            "parameters": params.tolist(),
            "X_counts": {str(key): int(value) for key, value in counts_x.items()},
            "Z_counts": {str(key): int(value) for key, value in counts_z.items()},
        })
        p_x = _counts_to_probability_vector(counts_x, N_QUBITS, self.layout)
        p_z = _counts_to_probability_vector(counts_z, N_QUBITS, self.layout)


        if self.readout_mitigation:
            if self.ro_assignment_matrix is None:
                raise RuntimeError("Readout matrix not available after calibration")
            p_x = _correct_with_qi_utilities(p_x, self.ro_assignment_matrix, self.layout)
            p_z = _correct_with_qi_utilities(p_z, self.ro_assignment_matrix, self.layout)

        values = self._estimate_expectations(p_x, p_z)
        weight = beta ** (-self.varsigma)
        cost = float(
            values["X_A"] + values["X_B"]
            + self.varsigma * (values["ZZ_A"] + values["ZZ_B"])
            - weight * (values["XX_BA"] + values["ZZ_BA"])
        )
        self.cost_history.append(cost)
        self.parameter_history.append(params.copy())
        self.expectation_history.append(values.copy())
        print(
            f"    eval {self.total_evaluations + 1:04d}: C={cost:+.7f} | "
            f"XA={values['X_A']:+.3f}, XB={values['X_B']:+.3f}, "
            f"ZZA={values['ZZ_A']:+.3f}, ZZB={values['ZZ_B']:+.3f}, "
            f"XXBA={values['XX_BA']:+.3f}, ZZBA={values['ZZ_BA']:+.3f}"
        )
        self.total_evaluations += 1
        return cost

    def minimize_cost(self, x0: Sequence[float], beta: float, maxiter: int = DEFAULT_MAXITER,
                      rhobeg: float = 0.5, tol: float = 0.02,
                      xatol: float = 0.02, fatol: float = 0.02):
        self.cost_history.clear()
        self.parameter_history.clear()
        self.expectation_history.clear()
        self.raw_measurement_history.clear()
        x0 = np.asarray(x0, dtype=float)
        expected_size = PARAMETERS_PER_LAYER * self.p_layers
        if x0.shape != (expected_size,):
            raise ValueError(f"x0 must have shape ({expected_size},) for p_layers={self.p_layers}")

        if self.optimizer == "COBYLA":
            # SciPy's current COBYLA implementation requires MAXFUN >= n_vars + 2.
            # SciPy maps the public maxiter option to this function-evaluation budget.
            effective_maxiter = max(int(maxiter), expected_size + 2)
            if effective_maxiter != int(maxiter):
                print(
                    f"  [optimizer note] COBYLA needs at least {expected_size + 2} "
                    f"function evaluations for {expected_size} parameters; "
                    f"raising maxiter from {int(maxiter)} to {effective_maxiter}."
                )
            options = {"maxiter": effective_maxiter, "rhobeg": float(rhobeg), "tol": float(tol)}
        else:
            options = {"maxiter": int(maxiter), "xatol": float(xatol), "fatol": float(fatol)}
        return minimize(
            self.cost_function,
            x0=x0,
            args=(float(beta),),
            method=self.optimizer,
            options=options,
        )


# ---------- Raw tomography measurement acquisition ----------


def _make_tomography_circuit(ansatz: QuantumCircuit, setting: Sequence[str]) -> QuantumCircuit:
    """Measure the three-qubit A register in a specified Pauli basis."""
    qc = QuantumCircuit(N_QUBITS, N_A, name="tomo_" + "".join(setting))
    qc.compose(ansatz, qubits=list(range(N_QUBITS)), inplace=True)
    for q, axis in enumerate(setting):
        if axis == "X":
            qc.h(q)
        elif axis == "Y":
            qc.sdg(q)
            qc.h(q)
        elif axis != "Z":
            raise ValueError(f"Unknown tomography axis {axis}")
    qc.barrier(*range(N_QUBITS))
    for q in range(N_A):
        qc.measure(q, q)
    return qc


def run_final_tomography(
    ansatz: QuantumCircuit,
    backend: Any,
    layout: Sequence[int],
    shots: int,
    optimization_level: int,
    result_timeout: int = 3600,
) -> dict[str, dict[str, Any]]:
    """Acquire and return raw counts for the 27 settings of A-register tomography.

    No density matrix reconstruction, Gibbs-state comparison, fidelity, or plots
    are calculated here. Those are performed offline by ``plot_3spin.py``.
    """
    circuits = [_make_tomography_circuit(ansatz, setting) for setting in TOMOGRAPHY_SETTINGS]
    print(
        f"  Tomography uses {len(circuits)} circuits; submitting in batches of at most "
        f"{MAX_CIRCUITS_PER_BATCH}."
    )
    counts_by_circuit, _compiled = submit_circuits_in_batches(
        circuits=circuits,
        backend=backend,
        layout=layout,
        shots=int(shots),
        optimization_level=optimization_level,
        result_timeout=result_timeout,
        batch_size=MAX_CIRCUITS_PER_BATCH,
        label="tomography",
    )

    raw_settings: dict[str, dict[str, Any]] = {}
    for idx, setting in enumerate(TOMOGRAPHY_SETTINGS):
        raw_counts = counts_by_circuit[idx]
        raw_settings["".join(setting)] = {
            "counts": raw_counts,
            "shots_returned": int(sum(raw_counts.values())),
        }
    return raw_settings


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    # Normalize field names and make nested values readable JSON strings.
    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict, tuple)) else value
                             for key, value in row.items()})


def run_extension_experiment(
    backend: Any,
    betas: Sequence[float] = DEFAULT_BETAS,
    varsigma: float = DEFAULT_VARSIGMA,
    p_layers: int = 1,
    optimizer: str = DEFAULT_OPTIMIZER,
    shots: int = DEFAULT_SHOTS,
    maxiter: int = DEFAULT_MAXITER,
    rhobeg: float = 0.5,
    tol: float = 0.02,
    xatol: float = 0.02,
    fatol: float = 0.02,
    layout: Sequence[int] = TUNA17_LAYOUT,
    optimization_level: int = 1,
    readout_mitigation: bool = False,
    calibration_shots: int = 512,
    calibration_interval: int = 10,
    tomography: bool = False,
    tomography_shots: int | None = None,
    output_dir: Path = Path("outputs/three_qubits"),
    clear_output: bool = True,
    initial_params: Sequence[float] | None = None,
    result_timeout: int = 3600,
) -> list[dict[str, Any]]:
    betas = tuple(sorted(set(float(beta) for beta in betas)))
    if not betas or any(not np.isfinite(beta) or beta <= 0 for beta in betas):
        raise ValueError("betas must be a non-empty collection of finite positive values")
    if p_layers < 1:
        raise ValueError("p_layers must be >= 1")
    if maxiter < 1 or shots < 1:
        raise ValueError("maxiter and shots must be positive")
    expected_parameters = PARAMETERS_PER_LAYER * int(p_layers)
    if initial_params is None:
        x0 = np.zeros(expected_parameters, dtype=float)
    else:
        x0 = np.asarray(initial_params, dtype=float)
        if x0.shape != (expected_parameters,):
            raise ValueError(f"initial_params must contain {expected_parameters} values")

    output_dir = prepare_output_directory(Path(output_dir), clear=clear_output)
    circuit_dir = output_dir / "circuits"
    circuit_dir.mkdir(parents=True, exist_ok=True)

    name = backend_name(backend)
    validate_fixed_layout(backend, layout)
    print("\n=== Three-spin TFD real-backend experiment ===")
    print(f"Backend: {name}")
    print(f"Logical-to-physical mapping: {logical_layout_description(layout)}")
    print(f"p_layers={p_layers}; variational parameters={expected_parameters}")
    print(f"varsigma={varsigma:g}; betas={betas}; optimizer={optimizer}")
    print(f"shots per measured circuit={shots}; optimizer maxiter={maxiter}")
    print("Every objective evaluation submits two measured circuits (global X and global Z).")
    print("The cost is calculated from the returned shot counts, not an ideal Statevector.")
    print(f"Readout mitigation: {'ON' if readout_mitigation else 'OFF'}")
    if tomography:
        print(f"Final A-register tomography: ON ({tomography_shots or shots} shots x 27 settings per beta)")
    else:
        print("Final A-register tomography: OFF (enable with --tomography)")

    model = CostFunction3SpinBackend(
        backend=backend, varsigma=varsigma, p_layers=p_layers, shots=shots,
        optimizer=optimizer, layout=layout, optimization_level=optimization_level,
        readout_mitigation=readout_mitigation, calibration_shots=calibration_shots,
        calibration_interval=calibration_interval, result_timeout=result_timeout,
    )

    results_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    raw_cost_measurements: list[dict[str, Any]] = []
    all_tomography: dict[str, Any] = {}

    for beta in betas:
        print(f"\n--- Optimizing beta={beta:g}, varsigma={varsigma:g}, p={p_layers} ---")
        optimization = model.minimize_cost(
            x0=x0, beta=beta, maxiter=maxiter, rhobeg=rhobeg, tol=tol,
            xatol=xatol, fatol=fatol,
        )
        params = np.asarray(optimization.x, dtype=float)
        x0 = params.copy()  # Warm-start the next beta.
        ansatz = TFD3SpinCircuit(params, p_layers=p_layers).as_qiskit_circuit()

        logical_circuit_path = circuit_dir / f"ansatz_beta_{beta:.6g}_p{p_layers}.txt"
        logical_circuit_path.write_text(str(ansatz.draw(output="text")), encoding="utf-8")
        # Save a transpiled measurement circuit for inspection of the fixed backend mapping.
        check_circuit = _make_cost_measurement_circuit(ansatz, "Z")
        transpiled_circuit = transpile(
            check_circuit, backend=backend, initial_layout=list(layout),
            optimization_level=optimization_level,
        )
        transpiled_path = circuit_dir / f"transpiled_Z_beta_{beta:.6g}_p{p_layers}.txt"
        transpiled_path.write_text(str(transpiled_circuit.draw(output="text")), encoding="utf-8")

        start_eval = model.total_evaluations - len(model.cost_history)
        for idx, (cost, parameters) in enumerate(
            zip(model.cost_history, model.parameter_history), start=1
        ):
            eval_no = start_eval + idx
            cost_rows.append({
                "beta": beta, "varsigma": varsigma, "p_layers": p_layers,
                "optimizer": model.optimizer, "evaluation_within_beta": idx,
                "global_evaluation": eval_no, "cost": float(cost),
                "parameters": parameters.tolist(),
            })
        raw_cost_measurements.extend(model.raw_measurement_history)

        row: dict[str, Any] = {
            "backend": name, "beta": beta, "varsigma": varsigma,
            "p_layers": p_layers, "optimizer": model.optimizer,
            "shots_per_cost_circuit": shots, "objective": float(optimization.fun),
            "nfev": int(getattr(optimization, "nfev", len(model.cost_history))),
            "nit": int(getattr(optimization, "nit", -1)),
            "optimizer_success": bool(optimization.success),
            "optimizer_message": str(optimization.message),
            "parameters": params.tolist(),
            "physical_layout": list(layout),
            "readout_mitigation": bool(readout_mitigation),
            "readout_calibration_runs": model._calibration_runs,
        }

        print(
            f"Optimized beta={beta:g}: C={float(optimization.fun):+.7f}; "
            f"nfev={row['nfev']}, success={optimization.success}; params={np.array2string(params, precision=5)}"
        )

        if tomography:
            tomo_shots = int(tomography_shots or shots)
            print(f"  Running 3-qubit A-register tomography: 27 settings x {tomo_shots} shots")
            raw_settings = run_final_tomography(
                ansatz, backend, layout, tomo_shots, optimization_level,
                result_timeout=result_timeout,
            )
            all_tomography[f"beta_{beta:.12g}"] = {
                "beta": float(beta),
                "shots_requested_per_setting": tomo_shots,
                "layout_logical_to_physical": [int(q) for q in layout],
                "measured_logical_qubits": [0, 1, 2],
                "settings": raw_settings,
            }
            print(f"  Saved raw A-register tomography counts for beta={beta:g} (27 settings).")
        results_rows.append(row)

    _write_csv(output_dir / "optimization_results.csv", results_rows)
    _write_csv(output_dir / "cost_history.csv", cost_rows)
    (output_dir / "raw_cost_measurements.json").write_text(
        json.dumps({
            "description": "Raw backend count dictionaries for both X- and Z-basis circuits at each optimizer evaluation.",
            "layout_logical_to_physical": [int(q) for q in layout],
            "records": raw_cost_measurements,
        }, indent=2), encoding="utf-8"
    )
    if all_tomography:
        (output_dir / "tomography_raw_counts.json").write_text(
            json.dumps({
                "description": "Raw backend counts for all 27 Pauli settings on the A register. No density-matrix reconstruction or fidelity has been calculated in this experiment run.",
                "backend": name,
                "layout_logical_to_physical": [int(q) for q in layout],
                "hamiltonian": {"model": "ZZ + ZZ + g( X + X + X )", "g": 1.0, "periodic": False},
                "betas": all_tomography,
            }, indent=2), encoding="utf-8"
        )
    if model.ro_assignment_matrix is not None:
        np.save(output_dir / "readout_assignment_matrix_6q.npy", model.ro_assignment_matrix)
    metadata = {
        "backend": name,
        "backend_mode": "Quantum Inspire QI backend; QX emulator is a simulator, Tuna-17 is hardware",
        "layout_logical_to_physical": {f"q{i}": int(q) for i, q in enumerate(layout)},
        "layout_description": logical_layout_description(layout),
        "logical_two_qubit_edges": [list(e) for e in LOGICAL_TWO_QUBIT_EDGES],
        "physical_two_qubit_edges": [sorted((int(layout[a]), int(layout[b]))) for a, b in LOGICAL_TWO_QUBIT_EDGES],
        "p_layers": int(p_layers), "parameters_per_layer": PARAMETERS_PER_LAYER,
        "betas": list(betas), "varsigma": float(varsigma),
        "optimizer": model.optimizer, "shots": int(shots), "maxiter": int(maxiter),
        "optimization_level": int(optimization_level),
        "readout_mitigation": bool(readout_mitigation),
        "calibration_shots": int(calibration_shots),
        "calibration_interval": int(calibration_interval),
        "tomography": bool(tomography), "tomography_shots": int(tomography_shots or shots),
        "expectation_measurement_bases": ["X", "Z"],
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nSaved results in: {output_dir.resolve()}")
    for filename in ("optimization_results.csv", "cost_history.csv", "raw_cost_measurements.json", "run_metadata.json"):
        print(f"  {output_dir / filename}")
    if all_tomography:
        print(f"  {output_dir / 'tomography_raw_counts.json'}")
    print("No plots or tomography fidelities were generated during the experiment.")
    if readout_mitigation:
        print("  Readout assignment calibration used QIUtilities mid-circuit measurement/reset circuits.")
    return results_rows


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-name", default="Tuna-17",
                        help='Quantum Inspire backend name, e.g. "Tuna-17" or "QX emulator"')
    parser.add_argument("--betas", nargs="+", type=float, default=list(DEFAULT_BETAS),
                        help="finite positive inverse temperatures; beta=0 is not supported by the measured objective")
    parser.add_argument("--varsigma", type=float, default=DEFAULT_VARSIGMA)
    parser.add_argument("--p-layers", type=int, default=1, help="variational depth p; uses 4*p angles")
    parser.add_argument("--optimizer", choices=("COBYLA", "Nelder-Mead"), default=DEFAULT_OPTIMIZER)
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    parser.add_argument("--maxiter", type=int, default=DEFAULT_MAXITER)
    parser.add_argument(
        "--initial-params", nargs="+", type=float, default=None,
        help="optional initial angles; provide exactly 4*p values (four for p=1)",
    )
    parser.add_argument("--rhobeg", type=float, default=0.5, help="COBYLA initial trust-region radius")
    parser.add_argument("--tol", type=float, default=0.02, help="COBYLA tolerance")
    parser.add_argument("--xatol", type=float, default=0.02, help="Nelder-Mead parameter convergence tolerance")
    parser.add_argument("--fatol", type=float, default=0.02, help="Nelder-Mead objective convergence tolerance")
    parser.add_argument("--optimization-level", type=int, choices=(0, 1, 2, 3), default=1)
    parser.add_argument("--readout-mitigation", action="store_true",
                        help="use QIUtilities mid-circuit assignment-matrix calibration and correction")
    parser.add_argument("--calibration-shots", type=int, default=512,
                        help="shots for each readout assignment-matrix calibration circuit")
    parser.add_argument("--calibration-interval", type=int, default=10,
                        help="refresh the assignment matrix every this many objective evaluations")
    parser.add_argument("--tomography", action="store_true",
                        help="after optimization, acquire raw counts for 27 Pauli settings on subsystem A; plot_3spin.py reconstructs/compares states")
    parser.add_argument("--tomography-shots", type=int, default=None,
                        help="shots per tomography setting (defaults to --shots)")
    parser.add_argument("--outdir", type=Path, default=Path("outputs/three_qubits"),
                        help="dedicated output directory; its contents are cleared before each run")
    parser.add_argument("--keep-output", action="store_true",
                        help="append/overwrite outputs without clearing existing files first")
    parser.add_argument("--result-timeout", type=int, default=3600, help="timeout in seconds for each QI job result")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.p_layers < 1:
        parser.error("--p-layers must be >= 1")
    expected_params = PARAMETERS_PER_LAYER * args.p_layers
    if args.initial_params is not None:
        if len(args.initial_params) != expected_params:
            parser.error(
                f"--initial-params requires exactly {expected_params} values for "
                f"--p-layers {args.p_layers}"
            )
        if not all(np.isfinite(value) for value in args.initial_params):
            parser.error("--initial-params values must be finite")
    if args.shots < 1 or args.maxiter < 1 or args.calibration_shots < 1:
        parser.error("--shots, --maxiter, and --calibration-shots must be positive")
    if args.tomography_shots is not None and args.tomography_shots < 1:
        parser.error("--tomography-shots must be positive")
    if any(not np.isfinite(beta) or beta <= 0 for beta in args.betas):
        parser.error("all beta values must be finite and positive")

    try:
        backend = create_qi_backend(args.backend_name)
        validate_fixed_layout(backend, TUNA17_LAYOUT)
        run_extension_experiment(
            backend=backend, betas=args.betas, varsigma=args.varsigma,
            p_layers=args.p_layers, optimizer=args.optimizer, shots=args.shots,
            maxiter=args.maxiter, rhobeg=args.rhobeg, tol=args.tol,
            xatol=args.xatol, fatol=args.fatol, layout=TUNA17_LAYOUT,
            optimization_level=args.optimization_level,
            readout_mitigation=args.readout_mitigation,
            calibration_shots=args.calibration_shots,
            calibration_interval=args.calibration_interval,
            tomography=args.tomography, tomography_shots=args.tomography_shots,
            output_dir=args.outdir, clear_output=not args.keep_output,
            initial_params=args.initial_params, result_timeout=args.result_timeout,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

