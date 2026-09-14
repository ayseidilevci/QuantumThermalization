"""Shared matplotlib canvas + styling, ported from the `gui` branch's
tfd_gui.py (MplCanvas / style_axes) and restyled against frontend.colors.Colors
instead of that branch's own Pal palette, so every plot in this app matches
the rest of the Qt chrome."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QSizePolicy, QGroupBox, QVBoxLayout, QHBoxLayout, QPushButton,
)

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from frontend.colors import Colors


class MplCanvas(FigureCanvas):
    def __init__(self, width=5, height=3.2, use_layout=True):
        # constrained_layout cooperates with the colorbar the density-matrix
        # panel adds; it doesn't support 3D axes (a matplotlib limitation),
        # so that panel opts out via use_layout and positions itself manually.
        self.fig = Figure(figsize=(width, height), constrained_layout=use_layout)
        self.fig.patch.set_facecolor(Colors.BG_3)
        super().__init__(self.fig)
        self.setStyleSheet(f"background: {Colors.BG_3}; border: none;")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(120)
        self.setMinimumWidth(160)


def style_axes(ax, title=None, xlabel=None, ylabel=None):
    """Shared dark, scientific look for a 2D matplotlib axes: card-colored
    background, hairline spines/grid, muted ticks, accent title."""
    c = Colors
    ax.set_facecolor(c.BG_3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(c.GRID)
    ax.tick_params(colors=c.TEXT_MUTED, labelsize=9)
    ax.grid(True, color=c.GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=c.TEXT, fontsize=11, fontweight="bold", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=c.TEXT_DIM, fontsize=9.5)
    if ylabel:
        ax.set_ylabel(ylabel, color=c.TEXT_DIM, fontsize=9.5)
    return ax


def panel(title: str, widget: QWidget, on_save=None) -> QGroupBox:
    """A plot card: title, optional small Save button in the corner (wired
    to on_save), then the panel's content widget. Ported from the `gui`
    branch's tfd_gui.py."""
    box = QGroupBox(title)
    box.setStyleSheet(panel_card_stylesheet())
    lay = QVBoxLayout(box)
    if on_save is not None:
        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        bar.addStretch()
        save_btn = QPushButton("💾 Save")
        save_btn.setCursor(Qt.PointingHandCursor)
        save_btn.setStyleSheet(f"padding: 2px 10px; font-weight: 600; color: {Colors.TEXT};")
        save_btn.clicked.connect(on_save)
        bar.addWidget(save_btn)
        lay.addLayout(bar)
    lay.addWidget(widget)
    return box


def panel_card_stylesheet() -> str:
    """Stylesheet for a QGroupBox-style card wrapping a plot/panel, matching
    Colors instead of ad hoc per-screen colors."""
    c = Colors
    return f"""
        QGroupBox {{
            background: {c.BG_2};
            border: 1px solid {c.BORDER};
            border-radius: 12px;
            margin-top: 14px;
            padding: 8px;
            font-weight: 600;
            color: {c.TEXT};
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 12px;
            top: 2px;
            padding: 0 6px;
            color: {c.ACCENT};
        }}
    """
