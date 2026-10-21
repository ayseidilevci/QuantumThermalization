"""Utilities to print or save Qiskit circuit diagrams without leaking figures."""

from __future__ import annotations

from pathlib import Path


def print_circuit_text(circuit, *, fold: int = -1) -> None:
    """Print an ASCII/text diagram directly to the terminal."""
    print(circuit.draw(output="text", fold=fold))


def save_circuit_diagram(
    circuit,
    filename: str | Path,
    *,
    dpi: int = 200,
    background: str = "#EEEEEE",
    fold: int = -1,
) -> Path:
    """Save a circuit drawing to PNG/JPG/JPEG/SVG/PDF and close its figure.

    The figure is always closed after saving, avoiding Matplotlib's
    "More than 20 figures have been opened" warning when saving many circuits.
    """
    path = Path(filename)
    allowed = {".png", ".jpg", ".jpeg", ".svg", ".pdf"}
    suffix = path.suffix.lower()
    if suffix not in allowed:
        raise ValueError(f"Unsupported extension {suffix!r}; use one of {sorted(allowed)}")
    path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    fig = circuit.draw(
        output="mpl",
        style={"backgroundcolor": background},
        fold=fold,
    )
    try:
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=background)
    finally:
        plt.close(fig)
    return path

