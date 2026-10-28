"""The three kinds of backend the app runs on:

  * local: the Starmon-7 noisy simulator, emulated on this machine
  * cloud simulator: Quantum Inspire's noiseless QX emulator
  * hardware: a real quantum computer (Tuna-9, Tuna-17)

Cloud simulator and hardware runs go through the same job pipeline (queued
jobs, the Hardware jobs tab, Stop and time-outs); only real hardware asks for
confirmation and is called a quantum computer.
"""

from backend import starmon7

QX_NAME = "QX emulator"

# Live Quantum Inspire backends offered; Tuna-5's star-shaped chip has no
# 2 x 2 ladder of coupled qubits, so it can't run the experiment.
OFFERED_LIVE_BACKENDS = ("tuna9", "tuna17", "qxemulator")


def normalized(name) -> str:
    return "".join(ch for ch in (name or "").lower() if ch.isalnum())


def is_local(name) -> bool:
    return starmon7.is_starmon7(name)


def is_qx(name) -> bool:
    return normalized(name) == normalized(QX_NAME)


def is_hardware(name) -> bool:
    return not is_local(name) and not is_qx(name)


def description(name) -> str:
    if is_local(name):
        return "Noisy simulator"
    if is_qx(name):
        return "Noiseless simulator"
    return "Real quantum computer"


def qx_mapping(n: int) -> dict:
    """The fixed qubits used on the QX emulator: physical qubit = the
    circuit's own (logical) qubit, in tfd_study's order A1..An, B1..Bn (every
    size runs through backend/ladder_tfd.py there). QX connects every qubit,
    so the choice doesn't matter, and its results come back indexed by
    physical qubit: only this layout makes them line up with the circuit's
    classical bits."""
    return {**{f"A{i}": i - 1 for i in range(1, n + 1)},
            **{f"B{i}": n + i - 1 for i in range(1, n + 1)}}
