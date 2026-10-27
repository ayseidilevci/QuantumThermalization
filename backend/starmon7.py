"""Starmon-7: QuantumInspire's 7-qubit transmon device, emulated locally.

Starmon-7 is no longer served by the live QI API, so it runs here on
qi_utilities' NoisySimulator, which ships calibrated Starmon-7 parameters
(T1/T2, RB/IRB errors, readout assignment errors) and the device's real
coupling map. Specifications below are from
https://www.quantum-inspire.com/kbase/starmon-7-operational-specifics/

Topology: two rhombi with qubits on every vertex and couplers on every
edge, sharing the central qubit Q3:

        Q0      Q1
       /  \\    /  \\
     Q2    Q3     Q4
       \\  /  \\   /
        Q5      Q6
"""

from math import pi

NAME = "Starmon-7"

# Same pairs as the "Coupling map" in qi_utilities' backend_parameters.json.
EDGES = [(0, 2), (0, 3), (1, 3), (1, 4), (2, 5), (3, 5), (3, 6), (4, 6)]

NATIVE_SINGLE_QUBIT_GATES = (
    "X", "Y", "Z", "I", "Rx(θ)", "Ry(θ)", "X90", "Y90", "mX90", "mY90",
    "S", "Sdag", "T", "Tdag",
)
NATIVE_TWO_QUBIT_GATES = ("CZ",)
DECOMPOSED_GATES = ("Rz(θ)", "H", "CNOT", "SWAP")

SINGLE_QUBIT_GATE_NS = 20
CZ_GATE_NS = 60
MEASUREMENT_US = 1
INITIALIZATION_US = 150

# Rotation angles are quantized to the nearest multiple of pi/28.
ANGLE_QUANTUM = pi / 28

SPEC_SUMMARY = (
    f"7 transmon qubits · two rhombi sharing Q3 · native CZ ({CZ_GATE_NS} ns), "
    f"single-qubit gates {SINGLE_QUBIT_GATE_NS} ns · measurement {MEASUREMENT_US} µs, "
    f"init {INITIALIZATION_US} µs · rotation angles quantized to π/28 · "
    f"mid-circuit measurement supported"
)


def is_starmon7(name) -> bool:
    return bool(name) and "".join(ch for ch in name.lower() if ch.isalnum()) == "starmon7"


def make_backend(ideal_simulation: bool = False):
    """A NoisySimulator configured as Starmon-7 (noisy by default)."""
    from qi_utilities.device_simulation.simulators import NoisySimulator
    return NoisySimulator(NAME, ideal_simulation=ideal_simulation)


def quantize_rotations(circuit):
    """Round every single-qubit rotation angle to the nearest multiple of
    pi/28, as the Starmon-7 hardware does. Expects a circuit already
    transpiled to the device's native gates. Returns a new circuit."""
    from qiskit.circuit.library import RXGate, RYGate, RZGate
    gate_types = {"rx": RXGate, "ry": RYGate, "rz": RZGate}

    quantized = circuit.copy_empty_like()
    for instruction in circuit.data:
        gate_type = gate_types.get(instruction.operation.name)
        if gate_type is not None:
            angle = float(instruction.operation.params[0])
            instruction = instruction.replace(
                operation=gate_type(round(angle / ANGLE_QUANTUM) * ANGLE_QUANTUM))
        quantized.append(instruction)
    return quantized
