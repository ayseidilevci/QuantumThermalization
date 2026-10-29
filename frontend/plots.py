"""Shared matplotlib canvas, axis styling and card panel, styled with
frontend.colors.Colors so every plot matches the rest of the app."""

import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QStackedWidget, QVBoxLayout, QWidget,
)

from frontend.colors import Colors

# Math in plot labels ($...$) in Computer Modern, like the rest of the app's
# math (see frontend/latex.py).
matplotlib.rcParams["mathtext.fontset"] = "cm"


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


class FlipView(QStackedWidget):
    """A measured plot and, flipped over, the exact target it should match,
    sharing one card; flip_button() goes in the card's head."""

    def __init__(self, measured: MplCanvas, exact: MplCanvas):
        super().__init__()
        self.measured, self.exact = measured, exact
        self.addWidget(measured)
        self.addWidget(exact)
        self.button = QPushButton()
        self.button.setCursor(Qt.PointingHandCursor)
        self.button.setStyleSheet(f"""
            QPushButton {{
                background: transparent; color: {Colors.ACCENT}; border: 1px solid {Colors.ACCENT};
                border-radius: 8px; padding: 4px 12px; font-weight: 600;
            }}
            QPushButton:hover {{ background: rgba(135,206,235,0.12); }}
        """)
        self.button.clicked.connect(self.flip)
        self.show_measured()

    @property
    def exact_shown(self) -> bool:
        return self.currentWidget() is self.exact

    def canvases(self) -> list:
        return [self.measured, self.exact]

    def show_measured(self):
        self.setCurrentWidget(self.measured)
        self.button.setText("⇄  Exact target")
        self.button.setToolTip("Show the exact result this experiment aims for")

    def flip(self):
        if self.exact_shown:
            self.show_measured()
        else:
            self.setCurrentWidget(self.exact)
            self.button.setText("⇄  Measured")
            self.button.setToolTip("Back to the measured result")
        self.currentWidget().draw_idle()

    def draw_idle(self):
        self.currentWidget().draw_idle()
