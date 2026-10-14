"""Shared matplotlib canvas + styling, ported from the `gui` branch's
tfd_gui.py (MplCanvas / style_axes) and restyled against frontend.colors.Colors
instead of that branch's own Pal palette, so every plot in this app matches
the rest of the Qt chrome."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QSizePolicy, QGroupBox, QVBoxLayout, QHBoxLayout, QPushButton,
    QFrame, QLabel,
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

    def wheelEvent(self, event):
        event.ignore()


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


def panel(title: str, widget: QWidget, on_save=None) -> QFrame:
    c = Colors
    box = QFrame()
    box.setObjectName("card")
    box.setStyleSheet(f"""
        QFrame#card {{
            background: {c.BG_2}; border: 1px solid {c.BORDER}; border-radius: 14px;
        }}
        QLabel#cardTitle {{
            color: {c.TEXT}; font-size: 12pt; font-weight: 700;
            background: transparent; border: none;
        }}
        QFrame#cardDivider {{ background: {c.BORDER}; border: none; }}
    """)
    lay = QVBoxLayout(box)
    lay.setContentsMargins(18, 14, 18, 16)
    lay.setSpacing(10)

    head = QHBoxLayout()
    head.setContentsMargins(0, 0, 0, 0)
    heading = QLabel(title)
    heading.setObjectName("cardTitle")
    head.addWidget(heading)
    head.addStretch()
    if on_save is not None:
        save_btn = QPushButton("💾  Save")
        save_btn.setCursor(Qt.PointingHandCursor)
        save_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; color: {c.ACCENT};
                border: 1px solid {c.ACCENT}; border-radius: 8px;
                padding: 5px 16px; font-weight: 600;
            }}
            QPushButton:hover {{ background: rgba(135,206,235,0.12); }}
        """)
        save_btn.clicked.connect(on_save)
        head.addWidget(save_btn)
    lay.addLayout(head)

    divider = QFrame()
    divider.setObjectName("cardDivider")
    divider.setFixedHeight(1)
    lay.addWidget(divider)

    lay.addWidget(widget, stretch=1)
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
