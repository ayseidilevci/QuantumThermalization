"""
TFD / Gibbs-state preparation GUI  —  Quantum Inspire front-end (PySide6)
=========================================================================

Front-end skeleton for the variational finite-temperature state-preparation
workflow (Sagastizabal et al., npj Quantum Information 7:130, 2021), targeting
the Quantum Inspire platform (https://www.quantum-inspire.com).

WHAT IS REAL HERE
-----------------
  * The full GUI: a two-step Quantum Inspire login (account email/password,
    then an optional API token used for job submission) followed by a
    single control panel with parameter inputs + start button, and four
    panels (real backend chip schematic, Qiskit-style circuit, live loss
    curve, density matrix) all fit in one box as a 2x2 grill of
    drag-to-resize panes.
  * The threading + signal plumbing that streams the loss value into the plot
    on every optimizer iteration WITHOUT freezing the UI.
  * The qubit-connectivity panel shows the actual chip schematic images
    shipped in Appdata/Img/ (Tuna-5, Tuna-9, Tuna-17) for the selected backend,
    switching and rescaling live as you change the Backend dropdown —
    no synthesized/placeholder graph.

WHAT IS STUBBED  (search for the tag  ===== BACKEND STUB =====)
---------------------------------------------------------------
  1. authenticate_credentials() -> validate email/password against Quantum Inspire
  2. build_circuit()        -> build the real Qiskit circuit for given params
  3. compute_density_matrix()-> real tomography / density matrix
  4. VariationalWorker.run()-> real optimizer loop submitting jobs to QI

Every stub returns plausible dummy data so the GUI runs end-to-end today.
Drop your Python in at each tag and the front-end needs no changes.

RUN
---
  pip install PySide6 matplotlib numpy
  pip install qiskit pylatexenc     # optional: renders the real circuit panel
  python tfd_gui.py
"""

import sys
import time
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

from PySide6.QtCore import Qt, QThread, Signal, QObject, QSize, QRectF
from PySide6.QtGui import QPalette, QColor, QFont, QPixmap, QImage, QPainter
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout,
    QHBoxLayout, QFormLayout, QComboBox, QDoubleSpinBox, QGroupBox, QSplitter,
    QStackedWidget, QMessageBox, QSizePolicy, QSpinBox, QDialog, QDialogButtonBox,
    QFileDialog,
)

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.cm as cm
import matplotlib.colors as mcolors

try:
    # Optional: pip install qiskit pylatexenc (the latter is what qc.draw
    # ('mpl') needs). Everything degrades to a text placeholder without it.
    from qiskit import QuantumCircuit
except ImportError:
    QuantumCircuit = None


# =========================================================================
#  Look & feel — a cool, dark "instrument panel" theme shared by the whole
#  app: Qt chrome (via the stylesheet below) AND every matplotlib panel
#  (via the style_axes() helper further down) draw from these same colors,
#  so the widgets and the plots read as one coherent piece of software.
# =========================================================================

class Pal:
    PAGE       = "#0e1420"   # window background — deep slate navy, not black
    SURFACE    = "#161d2e"   # card / panel background
    SURFACE_HI = "#1c2540"   # raised surface (inputs, hovered rows)
    BORDER     = "#2a3350"   # hairline borders
    ACCENT     = "#4c8dff"   # primary accent — cool blue
    ACCENT_HI  = "#6ea2ff"   # accent hover
    ACCENT_DIM = "#2f4b85"   # accent, subdued (e.g. disabled)
    TEAL       = "#2dd4bf"   # secondary accent — used sparingly
    AMBER      = "#f5a524"   # warm counterpoint — stop / warning
    TEXT       = "#eaf0ff"   # primary ink
    TEXT_DIM   = "#93a0c2"   # secondary ink
    TEXT_MUTED = "#5c6788"   # muted / axis labels
    GRID       = "#242e49"   # gridlines / hairlines on plots
    GOOD       = "#22c55e"
    BAD        = "#f2596a"


def build_stylesheet() -> str:
    p = Pal
    return f"""
        QWidget {{
            background: {p.PAGE};
            color: {p.TEXT};
            font-family: "Segoe UI", "Cantarell", sans-serif;
            font-size: 10.5pt;
        }}
        QMainWindow, QStackedWidget {{ background: {p.PAGE}; }}

        QGroupBox {{
            background: {p.SURFACE};
            border: 1px solid {p.BORDER};
            border-radius: 12px;
            margin-top: 16px;
            padding: 10px 8px 8px 8px;
            font-weight: 600;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 14px;
            top: 2px;
            padding: 0 6px;
            color: {p.ACCENT_HI};
        }}

        QLabel {{ background: transparent; color: {p.TEXT_DIM}; }}

        QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox {{
            background: {p.SURFACE_HI};
            border: 1px solid {p.BORDER};
            border-radius: 8px;
            padding: 5px 8px;
            color: {p.TEXT};
            selection-background-color: {p.ACCENT};
        }}
        QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus {{
            border: 1px solid {p.ACCENT};
        }}
        QComboBox::drop-down {{ border: none; width: 22px; }}
        QComboBox QAbstractItemView {{
            background: {p.SURFACE_HI};
            border: 1px solid {p.BORDER};
            selection-background-color: {p.ACCENT_DIM};
            color: {p.TEXT};
            outline: none;
        }}
        QSpinBox::up-button, QSpinBox::down-button,
        QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
            width: 16px; border: none; background: transparent;
        }}

        QPushButton {{
            background: {p.SURFACE_HI};
            border: 1px solid {p.BORDER};
            border-radius: 10px;
            padding: 8px 16px;
            font-weight: 600;
        }}
        QPushButton:hover {{ border: 1px solid {p.ACCENT}; }}
        QPushButton:disabled {{ color: {p.TEXT_MUTED}; }}

        QPushButton#startBtn {{
            background: {p.ACCENT};
            color: #071022;
            border: none;
        }}
        QPushButton#startBtn:hover {{ background: {p.ACCENT_HI}; }}
        QPushButton#startBtn:disabled {{ background: {p.ACCENT_DIM}; color: {p.TEXT_MUTED}; }}

        QPushButton#stopBtn {{
            background: transparent;
            color: {p.AMBER};
            border: 1px solid {p.AMBER};
        }}
        QPushButton#stopBtn:hover {{ background: rgba(245,165,36,0.12); }}
        QPushButton#stopBtn:disabled {{ color: {p.TEXT_MUTED}; border-color: {p.BORDER}; }}

        QPushButton#loginBtn {{
            background: {p.ACCENT};
            color: #071022;
            border: none;
        }}
        QPushButton#loginBtn:hover {{ background: {p.ACCENT_HI}; }}

        QSplitter::handle {{ background: {p.PAGE}; }}
        QSplitter::handle:hover {{ background: {p.ACCENT_DIM}; }}
        QSplitter::handle:horizontal {{ width: 8px; }}
        QSplitter::handle:vertical {{ height: 8px; }}

        QScrollBar:vertical, QScrollBar:horizontal {{
            background: {p.PAGE}; border: none; margin: 0; border-radius: 5px;
        }}
        QScrollBar::handle {{ background: {p.BORDER}; border-radius: 5px; min-height: 24px; }}
        QScrollBar::handle:hover {{ background: {p.ACCENT_DIM}; }}
        QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
    """


# =========================================================================
#  BACKEND STUBS
#  Replace the bodies of these with your real Quantum Inspire / Qiskit code.
#  Keep the signatures and return types and the GUI keeps working unchanged.
# =========================================================================

def authenticate_credentials(email: str, password: str) -> bool:
    """===== BACKEND STUB 1: LOGIN =====
    Validate the user's Quantum Inspire account email + password.

    Real version (roughly):
        from quantuminspire.util.authentication import ...   # QI SDK
        # or POST to the QI auth endpoint and store the returned session
    Return True on success, False on failure.
    """
    time.sleep(0.4)                       # pretend we hit the network
    return bool(email.strip()) and "@" in email and bool(password)


def register_api_token(token: str) -> bool:
    """===== BACKEND STUB 1b: API TOKEN =====
    Called once the user is logged in and enters their Quantum Inspire API
    token (used to submit jobs / authenticate SDK calls against a backend).
    The token is optional at login time — jobs just can't be submitted to a
    real backend until one is set.

    Real version (roughly):
        from quantuminspire.credentials import save_account
        save_account(token)   # or however the QI SDK wants it stored
    Return True if the token looks usable, False otherwise.
    """
    time.sleep(0.2)
    return True                            # accept anything for now, incl. blank/skip


def build_circuit(beta: float, n_steps: int = 1):
    """===== BACKEND STUB 2: CIRCUIT =====
    Build and return a real qiskit.QuantumCircuit for the TFD ansatz at this
    beta — a real QuantumCircuit object, not a placeholder, so the panel
    always draws it with qc.draw('mpl', ...); it's the gate *angles* that are
    dummy (toy functions of beta) rather than the optimizer's actual
    variational parameters.

    Layout: qubits 0,1 = subsystem A (A1, A2); qubits 2,3 = the mirror copy
    B (B1, B2) — the same 4-transmon square as the Tuna-5 backend.
    Sequence, per the paper: Bell pairs -> U_intra(gamma) -> U_inter(alpha)
    -> readout/tomography rotation, repeated n_steps times.

    Real version: swap `gamma`/`alpha` below for the optimizer's actual
    parameters (VariationalWorker already carries them each iteration), and/or
    swap the gate set for the backend's native gates. The GUI needs no
    changes either way — it only ever calls qc.draw('mpl').

    Returns None if qiskit isn't installed, and the circuit panel falls back
    to a placeholder rather than crashing.
    """
    if QuantumCircuit is None:
        return None

    A1, A2, B1, B2 = 0, 1, 2, 3
    qc = QuantumCircuit(4, 4, name=f"TFD ansatz (β={beta:.2f})")

    # 1) Bell pairs between each site and its mirror copy — the β=0
    #    (infinite-temperature) starting state.
    for a, b in ((A1, B1), (A2, B2)):
        qc.h(a)
        qc.cx(a, b)
    qc.barrier()

    # 2) U_intra(gamma) + 3) U_inter(alpha): both toy functions of beta, so
    #    the drawn circuit visibly changes as you adjust beta in Run
    #    parameters — swap for the real variational angles when wired up.
    for step in range(max(1, n_steps)):
        frac = (step + 1) / max(1, n_steps)
        gamma = beta * 0.5 * frac
        alpha = beta * 0.25 * frac
        qc.rxx(gamma, A1, A2)
        qc.rxx(gamma, B1, B2)
        qc.ryy(alpha, A1, B2)
        qc.ryy(alpha, A2, B1)
        qc.barrier()

    # 4) Readout / tomography rotation, then measure.
    qc.measure(range(4), range(4))
    return qc


def compute_density_matrix(beta: float) -> np.ndarray:
    """===== BACKEND STUB 3: DENSITY MATRIX =====
    Return the 4x4 (single-system) density matrix from tomography.

    Real version: reconstruct rho_Exp from two-qubit state tomography results.
    Below: a toy beta-dependent Hermitian PSD matrix so the plot looks alive.
    """
    dim = 4
    diag = np.exp(-beta * np.linspace(0, 1.2, dim))
    diag = diag / diag.sum()
    rho = np.diag(diag).astype(complex)
    coh = 0.12 * np.tanh(beta)            # coherences grow with beta, as in paper
    for i in range(dim):
        for j in range(i + 1, dim):
            rho[i, j] = coh * (0.6 ** (j - i))
            rho[j, i] = np.conj(rho[i, j])
    return rho


class VariationalWorker(QObject):
    """===== BACKEND STUB 4: OPTIMIZER LOOP =====
    Runs on a background QThread. Emits `progress(iteration, loss)` every
    optimizer step so the GUI can stream the loss curve, `rho_progress(rho)`
    periodically so the density-matrix panel updates live as the state
    converges, then `finished(rho)` with the final result.

    Real version: replace the loop body with your variational optimizer
    (e.g. scikit-optimize) that, each iteration, submits a job to Quantum
    Inspire, reads back measurements, evaluates the engineered cost function
    C(beta), and reports the loss. Emit `progress` each iteration; emit
    `rho_progress` whenever you have a fresh (partial or full) tomography
    estimate of rho_Exp to show the live convergence of the state.
    """
    progress = Signal(int, float)         # (iteration, loss)
    rho_progress = Signal(object)         # live density-matrix estimate (np.ndarray)
    finished = Signal(object)             # final density matrix (np.ndarray)
    log = Signal(str)

    def __init__(self, beta: float, optimizer: str, n_iters: int):
        super().__init__()
        self.beta = beta
        self.optimizer = optimizer
        self.n_iters = n_iters
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        self.log.emit(f"Starting optimization  |  beta={self.beta}  "
                      f"optimizer={self.optimizer}")
        rng = np.random.default_rng(int(self.beta * 100) + 7)
        loss = 4.0                                    # cost ~ -<H_BA>, see paper
        target = -4.0 + 3.0 * np.exp(-self.beta)      # dummy convergence target
        # Cap the number of density-matrix redraws (the 3D bar plot is the
        # most expensive one to render) regardless of how many iterations run.
        rho_every = max(1, self.n_iters // 40)
        for it in range(self.n_iters):
            if self._abort:
                self.log.emit("Aborted.")
                break
            # ---- real work goes here: submit QI job, evaluate C(beta) ----
            time.sleep(0.05)                          # simulate a job round-trip
            loss += (target - loss) * 0.08 + rng.normal(0, 0.15) * (1 - it / self.n_iters)
            self.progress.emit(it, float(loss))
            if it % rho_every == 0 or it == self.n_iters - 1:
                # ---- real work goes here: partial-tomography estimate of ----
                # ---- rho_Exp from measurements taken so far ----
                progress_frac = (it + 1) / self.n_iters
                live_beta = self.beta * progress_frac  # state "cools" toward beta as it converges
                self.rho_progress.emit(compute_density_matrix(live_beta))
        self.log.emit("Optimization complete. Running tomography...")
        rho = compute_density_matrix(self.beta)       # STUB 3
        self.finished.emit(rho)


# =========================================================================
#  Reusable matplotlib canvas
# =========================================================================

class MplCanvas(FigureCanvas):
    def __init__(self, width=5, height=3.2, use_layout=True):
        # constrained_layout (not tight_layout) cooperates with the colorbar
        # the density-matrix panel adds. It doesn't support 3D axes though
        # (matplotlib limitation), so the rho panel opts out via use_layout
        # and positions itself manually instead.
        self.fig = Figure(figsize=(width, height), constrained_layout=use_layout)
        # Match the card background exactly so the rectangular canvas blends
        # into the rounded QGroupBox around it, instead of showing as a
        # mismatched pale/black rectangle poking out past the corners.
        self.fig.patch.set_facecolor(Pal.SURFACE)
        super().__init__(self.fig)
        self.setStyleSheet(f"background: {Pal.SURFACE}; border: none;")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(120)
        self.setMinimumWidth(160)


def panel(title: str, widget: QWidget, on_save=None) -> QGroupBox:
    """A grill card: title, optional small "Save" button in the corner
    (wired to on_save), then the panel's content widget.
    """
    box = QGroupBox(title)
    lay = QVBoxLayout(box)
    if on_save is not None:
        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        bar.addStretch()
        save_btn = QPushButton("💾 Save")
        save_btn.setCursor(Qt.PointingHandCursor)
        save_btn.setStyleSheet("padding: 2px 10px; font-weight: 600;")
        save_btn.clicked.connect(on_save)
        bar.addWidget(save_btn)
        lay.addLayout(bar)
    lay.addWidget(widget)
    return box


def style_axes(ax, title=None, xlabel=None, ylabel=None):
    """Apply the shared dark, scientific look to a 2D matplotlib axes:
    card-colored background, hairline spines/grid, muted ticks, accent title.
    """
    p = Pal
    ax.set_facecolor(p.SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(p.GRID)
    ax.tick_params(colors=p.TEXT_MUTED, labelsize=9)
    ax.grid(True, color=p.GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=p.TEXT, fontsize=11, fontweight="bold", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=p.TEXT_DIM, fontsize=9.5)
    if ylabel:
        ax.set_ylabel(ylabel, color=p.TEXT_DIM, fontsize=9.5)
    return ax


# =========================================================================
#  Real backend chip schematics (Appdata/Img/) — shown as-is instead of a
#  synthesized connectivity graph.
# =========================================================================

DATA_DIR = Path(__file__).resolve().parent / "Appdata" / "Img"

# Maps a Backend dropdown entry to its real chip-layout image. Add a new
# backend by dropping its schematic/photo in Appdata/Img/ and adding a row
# here — no other code needs to change.
BACKEND_IMAGES = {
    "Tuna-5 (hardware)":  DATA_DIR / "1766393003-tuna-5_schematic.svg",
    "Tuna-9 (hardware)":  DATA_DIR / "1765992515-tuna9logo_v1.webp",
    "Tuna-17 (hardware)": DATA_DIR / "1772648307-tuna-17_schematic-svg.svg",
}

_SVG_NS = "{http://www.w3.org/2000/svg}"


def _extract_svg_labels(renderer: QSvgRenderer, path: Path):
    """Qt's SVG module doesn't paint the tiny qubit-number/"TC" <text> labels
    these Inkscape-exported schematics use (a QtSvg limitation, not missing
    data) — it still tracks each element's geometry, though, so pull the
    label strings straight from the XML and ask the renderer where each
    one's id ends up in the final image; BackendImageView draws them back on
    itself with a font Qt can actually render. Returns [] if anything about
    a given file doesn't match this shape — the schematic then just renders
    without labels rather than raising.
    """
    labels = []
    try:
        # transformForElement()/boundsOnElement() report positions in the
        # SVG's own viewBox units, not the defaultSize() pixel space render()
        # actually fills — those only coincide when width/height are unitless
        # (as in the Tuna-5 file). Tuna-17 declares its canvas in millimetres
        # with a much smaller viewBox, so without this correction every label
        # lands scaled down toward the top-left corner. Rescale viewBox units
        # -> defaultSize() pixels explicitly so both cases work.
        vbox = renderer.viewBoxF()
        default = renderer.defaultSize()
        if vbox.width() > 0 and vbox.height() > 0:
            ux, uy = default.width() / vbox.width(), default.height() / vbox.height()
        else:
            ux = uy = 1.0

        root = ET.parse(str(path)).getroot()
        for el in root.iter(f"{_SVG_NS}text"):
            eid = el.get("id")
            text = "".join(el.itertext()).strip()
            if not eid or not text or not renderer.elementExists(eid):
                continue
            r = renderer.transformForElement(eid).mapRect(renderer.boundsOnElement(eid))
            rect = QRectF(r.x() * ux, r.y() * uy, r.width() * ux, r.height() * uy)
            labels.append((text, rect))
    except Exception:
        return []
    return labels


class BackendImageView(QLabel):
    """Displays the actual chip schematic for the selected backend — an SVG
    re-rendered crisply at the panel's current size, or a raster image
    (e.g. the Tuna-9 photo/logo) scaled to fit — instead of a synthesized
    connectivity graph. Swap backends and it redraws immediately; resize the
    grill panel and the SVG re-renders sharp at the new size.
    """

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(120)
        self.setMinimumWidth(160)
        self.setStyleSheet(f"background: {Pal.SURFACE}; border: none;")
        self._svg = None
        self._svg_labels = []
        self._raster = None

    def set_backend(self, name: str):
        path = BACKEND_IMAGES.get(name)
        if path is None or not path.exists():
            self._svg = None
            self._svg_labels = []
            self._raster = None
            self.setPixmap(QPixmap())
            self.setText(f"No schematic available for “{name}”.")
            self.setStyleSheet(f"background: {Pal.SURFACE}; color: {Pal.TEXT_MUTED}; "
                                f"border: none; padding: 24px;")
            return
        self.setStyleSheet(f"background: {Pal.SURFACE}; border: none;")
        self.setText("")
        if path.suffix.lower() in (".svg", ".svgz"):
            self._svg = QSvgRenderer(str(path))
            self._svg_labels = _extract_svg_labels(self._svg, path)
            self._raster = None
        else:
            self._svg = None
            self._svg_labels = []
            self._raster = QPixmap(str(path))
        self._redraw()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._redraw()

    def _redraw(self):
        size = self.size()
        if size.width() < 2 or size.height() < 2:
            return
        if self._svg is not None and self._svg.isValid():
            base = self._svg.defaultSize()
            if base.width() <= 0 or base.height() <= 0:
                base = size
            scale = min(size.width() / base.width(), size.height() / base.height())
            target = QSize(max(1, int(base.width() * scale)), max(1, int(base.height() * scale)))
            img = QImage(target, QImage.Format_ARGB32_Premultiplied)
            img.fill(QColor(Pal.SURFACE))
            painter = QPainter(img)
            self._svg.render(painter)
            self._paint_labels(painter, scale)
            painter.end()
            self.setPixmap(QPixmap.fromImage(img))
        elif self._raster is not None and not self._raster.isNull():
            self.setPixmap(self._raster.scaled(
                size, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _paint_labels(self, painter: QPainter, scale: float):
        """Redraw the schematic's qubit-index / tunable-coupler labels
        ourselves (see _extract_svg_labels) at the position Qt says the
        original <text> element occupies, scaled to the current render.
        """
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.TextAntialiasing)
        font = QFont("Segoe UI")
        for text, rect in self._svg_labels:
            is_tc = text.upper() == "TC"
            h_px = max(7.0, rect.height() * scale * (0.75 if is_tc else 0.95))
            font.setPixelSize(int(round(h_px)))
            font.setBold(not is_tc)
            painter.setFont(font)
            painter.setPen(QColor(Pal.TEXT_MUTED if is_tc else Pal.TEXT))
            cx, cy = rect.center().x() * scale, rect.center().y() * scale
            w = max(rect.width() * scale, h_px * len(text))
            box = QRectF(cx - w / 2, cy - h_px, w, h_px * 2)
            painter.drawText(box, Qt.AlignCenter, text)


# =========================================================================
#  Login screen
# =========================================================================

class LoginScreen(QWidget):
    """Two-step Quantum Inspire sign-in: account email/password first, then
    (once signed in) an optional API token box used for job submission.
    """
    logged_in = Signal(str)          # emits the API token (may be blank)

    def __init__(self):
        super().__init__()
        self.api_token = ""
        outer = QVBoxLayout(self)
        outer.addStretch()

        self.card = QGroupBox("Quantum Inspire login")
        self.card.setMaximumWidth(420)
        self.card.setMinimumWidth(360)
        card_v = QVBoxLayout(self.card)
        card_v.setContentsMargins(18, 24, 18, 18)
        card_v.setSpacing(4)

        # ---- step 1: account credentials ----------------------------------
        self.step1 = QWidget()
        f1 = QFormLayout(self.step1)
        f1.setContentsMargins(0, 0, 0, 0)
        f1.setVerticalSpacing(12)

        self.email = QLineEdit()
        self.email.setPlaceholderText("you@example.com")
        self.email.setMinimumHeight(32)

        self.password = QLineEdit()
        self.password.setPlaceholderText("Password")
        self.password.setEchoMode(QLineEdit.Password)
        self.password.setMinimumHeight(32)

        self.status = QLabel("")
        self.status.setStyleSheet(f"color: {Pal.BAD};")
        self.status.setWordWrap(True)

        self.login_btn = QPushButton("Log in")
        self.login_btn.setObjectName("loginBtn")
        self.login_btn.setMinimumHeight(38)
        self.login_btn.clicked.connect(self._attempt_login)
        self.email.returnPressed.connect(lambda: self.password.setFocus())
        self.password.returnPressed.connect(self._attempt_login)

        f1.addRow("Email:", self.email)
        f1.addRow("Password:", self.password)
        f1.addRow(self.login_btn)
        f1.addRow(self.status)
        card_v.addWidget(self.step1)

        # ---- step 2: optional API token, shown after a successful login ---
        self.step2 = QWidget()
        f2 = QFormLayout(self.step2)
        f2.setContentsMargins(0, 0, 0, 0)
        f2.setVerticalSpacing(12)

        self.signed_in_lbl = QLabel("")
        self.signed_in_lbl.setStyleSheet(f"color: {Pal.GOOD}; font-weight: 600;")
        self.signed_in_lbl.setWordWrap(True)

        self.token = QLineEdit()
        self.token.setPlaceholderText("Optional — paste your API token, or add it later")
        self.token.setEchoMode(QLineEdit.Password)
        self.token.setMinimumHeight(32)

        self.token_hint = QLabel("Needed to submit jobs to a real backend; "
                                  "simulators work without one.")
        self.token_hint.setStyleSheet(f"color: {Pal.TEXT_MUTED}; font-size: 9pt;")
        self.token_hint.setWordWrap(True)

        self.continue_btn = QPushButton("Continue")
        self.continue_btn.setObjectName("loginBtn")
        self.continue_btn.setMinimumHeight(38)
        self.continue_btn.clicked.connect(self._continue)
        self.token.returnPressed.connect(self._continue)

        f2.addRow(self.signed_in_lbl)
        f2.addRow("API token:", self.token)
        f2.addRow(self.token_hint)
        f2.addRow(self.continue_btn)
        card_v.addWidget(self.step2)
        self.step2.hide()

        row = QHBoxLayout()
        row.addStretch(); row.addWidget(self.card); row.addStretch()
        outer.addLayout(row)
        outer.addStretch()

    def _attempt_login(self):
        self.login_btn.setEnabled(False)
        self.status.setStyleSheet(f"color: {Pal.TEXT_DIM};")
        self.status.setText("Authenticating...")
        QApplication.processEvents()
        ok = authenticate_credentials(self.email.text(), self.password.text())   # STUB 1
        self.login_btn.setEnabled(True)
        if ok:
            self.status.setText("")
            self.card.setTitle("Quantum Inspire")
            self.signed_in_lbl.setText(f"Signed in as {self.email.text()}")
            self.step1.hide()
            self.step2.show()
            self.token.setFocus()
        else:
            self.status.setStyleSheet(f"color: {Pal.BAD};")
            self.status.setText("Login failed — check your email and password.")

    def _continue(self):
        self.continue_btn.setEnabled(False)
        QApplication.processEvents()
        register_api_token(self.token.text())      # STUB 1b
        self.api_token = self.token.text()
        self.continue_btn.setEnabled(True)
        self.logged_in.emit(self.api_token)


# =========================================================================
#  Control panel  (single box, all four panels laid out as an adjustable grill)
# =========================================================================

class ControlPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.thread = None
        self.worker = None
        self.loss_x, self.loss_y = [], []
        self._last_qc = None            # last-drawn circuit, for the Save button
        self._last_rho = None           # last-drawn density matrix, ditto

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # ---- controls row -------------------------------------------------
        controls = QGroupBox("Run parameters")
        cform = QHBoxLayout(controls)

        pform = QFormLayout()
        self.backend = QComboBox()
        self.backend.addItems([*BACKEND_IMAGES.keys(), "QX simulator"])
        self.backend.currentTextChanged.connect(self.draw_connectivity)

        self.beta = QDoubleSpinBox()
        self.beta.setRange(0.0, 5.0); self.beta.setSingleStep(0.1); self.beta.setValue(1.0)
        self.beta.setDecimals(2)

        self.optimizer = QComboBox()
        self.optimizer.addItems(["Gradient-Based Random-Tree (skopt)",
                                 "Nelder-Mead", "COBYLA", "SPSA"])

        self.iters = QSpinBox()
        self.iters.setRange(10, 2000); self.iters.setValue(200)

        pform.addRow("Backend:", self.backend)
        pform.addRow("Inverse temperature \u03b2:", self.beta)
        pform.addRow("Optimizer:", self.optimizer)
        pform.addRow("Iterations:", self.iters)
        cform.addLayout(pform)

        # ---- API token status + add/update, always reachable here even if
        # ---- it was skipped at login -------------------------------------
        self.api_token = ""
        tokcol = QVBoxLayout()
        tokcol.addStretch()
        self.token_status_lbl = QLabel()
        self.token_status_lbl.setWordWrap(True)
        self.token_status_lbl.setMaximumWidth(180)
        self.token_btn = QPushButton()
        self.token_btn.clicked.connect(self._open_token_dialog)
        tokcol.addWidget(self.token_status_lbl)
        tokcol.addWidget(self.token_btn)
        tokcol.addStretch()
        cform.addLayout(tokcol)
        self._refresh_token_ui()

        cform.addStretch()
        btncol = QVBoxLayout()
        self.start_btn = QPushButton("\u25b6  Start")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.setMinimumHeight(48)
        self.start_btn.clicked.connect(self.on_start)
        self.stop_btn = QPushButton("\u25a0  Stop")
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.setMinimumHeight(48)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.on_stop)
        btncol.addWidget(self.start_btn)
        btncol.addWidget(self.stop_btn)
        cform.addLayout(btncol)
        root.addWidget(controls)

        self.statuslbl = QLabel("Idle.")
        self.statuslbl.setStyleSheet(f"color:{Pal.TEXT_MUTED}; padding:2px;")
        root.addWidget(self.statuslbl)

        # ---- adjustable 2x2 grill of panels, all in one box ---------------
        # Nested splitters instead of a scroll area: every pane is visible at
        # once, and the user drags the bars between them (both the row divider
        # and the two column dividers) to resize any cell, like an adjustable
        # grill, rather than scrolling through a stack of separate windows.
        self.conn_view = BackendImageView()
        self.circ_canvas = MplCanvas()
        self.loss_canvas = MplCanvas()
        self.rho_canvas = MplCanvas(height=4.0, use_layout=False)

        top_row = QSplitter(Qt.Horizontal)
        top_row.addWidget(panel("Qubit connectivity (backend chip layout)", self.conn_view))
        top_row.addWidget(panel("Circuit schematic", self.circ_canvas,
                                 on_save=self._save_circuit))
        top_row.setSizes([1, 1])

        bottom_row = QSplitter(Qt.Horizontal)
        bottom_row.addWidget(panel("Loss minimization (live)", self.loss_canvas,
                                    on_save=self._save_loss))
        bottom_row.addWidget(panel("Density matrix  \u03c1_Exp", self.rho_canvas,
                                    on_save=self._save_density))
        bottom_row.setSizes([1, 1])

        grill = QSplitter(Qt.Vertical)
        grill.addWidget(top_row)
        grill.addWidget(bottom_row)
        grill.setSizes([1, 1])
        self.grill = grill

        for sp in (top_row, bottom_row, grill):
            sp.setHandleWidth(8)
            sp.setChildrenCollapsible(False)

        root.addWidget(grill, stretch=1)

        # initial static draws
        self.draw_connectivity()
        self.draw_circuit()
        self.init_loss_plot()
        self.draw_density(compute_density_matrix(self.beta.value()))

    # ---- API token (settable at login, or any time from here) ------------
    def set_api_token(self, token: str):
        self.api_token = token or ""
        self._refresh_token_ui()

    def _refresh_token_ui(self):
        if self.api_token:
            masked = "•" * max(0, len(self.api_token) - 4) + self.api_token[-4:]
            self.token_status_lbl.setText(f"✓ API token set\n{masked}")
            self.token_status_lbl.setStyleSheet(f"color: {Pal.GOOD};")
            self.token_btn.setText("Update token")
        else:
            self.token_status_lbl.setText("⚠ No API token\n(simulators only)")
            self.token_status_lbl.setStyleSheet(f"color: {Pal.AMBER};")
            self.token_btn.setText("Add token")

    def _open_token_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Quantum Inspire API token")
        dlg.setMinimumWidth(380)
        lay = QVBoxLayout(dlg)

        info = QLabel("Paste your Quantum Inspire API token. It's needed to "
                       "submit jobs to a real hardware backend — simulators "
                       "work without one.")
        info.setWordWrap(True)
        info.setStyleSheet(f"color: {Pal.TEXT_DIM};")

        field = QLineEdit(self.api_token)
        field.setEchoMode(QLineEdit.Password)
        field.setPlaceholderText("API token")
        field.setMinimumHeight(32)

        btns = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        field.returnPressed.connect(dlg.accept)

        lay.addWidget(info)
        lay.addWidget(field)
        lay.addWidget(btns)

        if dlg.exec() == QDialog.Accepted:
            register_api_token(field.text())        # STUB 1b
            self.set_api_token(field.text())

    # ---- drawing helpers -------------------------------------------------
    def draw_connectivity(self):
        # Real chip schematic for the selected backend (Appdata/Img/) — no synthesized
        # connectivity graph. See BACKEND_IMAGES to add a new backend's image.
        self.conn_view.set_backend(self.backend.currentText())

    def draw_circuit(self):
        p = Pal
        qc = build_circuit(self.beta.value(), 1)                # STUB 2
        self._last_qc = qc
        fig = self.circ_canvas.fig
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_facecolor(p.SURFACE)
        if qc is None:
            ax.text(0.5, 0.5,
                    "qiskit isn't installed\n\n"
                    "pip install qiskit pylatexenc\n"
                    "to render the real circuit here via qc.draw('mpl').",
                    ha="center", va="center", fontsize=10, color=p.TEXT_DIM,
                    bbox=dict(boxstyle="round,pad=0.8", facecolor=p.SURFACE_HI,
                              edgecolor=p.BORDER, linewidth=1))
            ax.axis("off")
        else:
            # Real Qiskit circuit, drawn onto this axis and themed to match
            # the rest of the app — fall back to qiskit's default style if a
            # future qiskit version renames one of these style keys.
            style = {
                "backgroundcolor": p.SURFACE,
                "linecolor": p.TEXT_DIM,
                "textcolor": p.TEXT,
                "gatetextcolor": "#071022",
                "subtextcolor": p.TEXT_MUTED,
                "creglinecolor": p.TEXT_MUTED,
                "gatefacecolor": p.ACCENT,
                "barrierfacecolor": p.GRID,
            }
            try:
                qc.draw(output="mpl", ax=ax, style=style)
            except Exception:
                qc.draw(output="mpl", ax=ax)
        self.circ_canvas.draw()

    def init_loss_plot(self):
        p = Pal
        self.loss_x, self.loss_y = [], []
        fig = self.loss_canvas.fig
        fig.clear()
        ax = fig.add_subplot(111)
        self._loss_ax = ax
        (self._loss_line,) = ax.plot([], [], "-", color=p.ACCENT, linewidth=2.2,
                                      solid_capstyle="round")
        (self._loss_head,) = ax.plot([], [], "o", color=p.ACCENT_HI,
                                      markersize=6, markeredgecolor=p.SURFACE, markeredgewidth=1.2)
        self._loss_fill = None
        style_axes(ax, title="Loss minimization", xlabel="Iteration", ylabel="Cost  C(\u03b2)")
        self.loss_canvas.draw()

    def draw_density(self, rho: np.ndarray):
        p = Pal
        self._last_rho = rho
        fig = self.rho_canvas.fig
        fig.clear()
        fig.subplots_adjust(left=0.02, right=0.88, bottom=0.05, top=0.92)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor(p.SURFACE)
        # 3D pane chrome: dark panes + hairline edges instead of matplotlib's
        # default pale-grey box, so the plot sits flush with the dark theme.
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.set_pane_color((0, 0, 0, 0))
            pane._axinfo["grid"]["color"] = mcolors.to_rgba(p.GRID, 0.6)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.line.set_color(p.GRID)
            axis.label.set_color(p.TEXT_DIM)
        ax.tick_params(colors=p.TEXT_MUTED)

        n = rho.shape[0]
        xpos, ypos = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        xpos = xpos.ravel(); ypos = ypos.ravel()
        heights = np.abs(rho).ravel()
        phases = np.angle(rho).ravel()
        norm_phase = (phases + np.pi) / (2 * np.pi)
        colors = cm.twilight(norm_phase)
        ax.bar3d(xpos, ypos, np.zeros_like(heights), 0.7, 0.7, heights,
                  color=colors, shade=True, edgecolor=p.PAGE, linewidth=0.6)
        labels = ["00", "01", "10", "11"]
        ax.set_xticks(np.arange(n) + 0.35); ax.set_xticklabels(labels)
        ax.set_yticks(np.arange(n) + 0.35); ax.set_yticklabels(labels)
        ax.set_zlabel("|\u03c1|")
        ax.set_title("Density matrix (height = magnitude, colour = phase)",
                      color=p.TEXT, fontsize=11, fontweight="bold", pad=6)

        # cyclic phase colorbar (-\u03c0..\u03c0), the accessible key to the color channel
        sm = cm.ScalarMappable(cmap=cm.twilight, norm=mcolors.Normalize(vmin=-np.pi, vmax=np.pi))
        cbar = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.02, ticks=[-np.pi, 0, np.pi])
        cbar.ax.set_yticklabels(["\u2212\u03c0", "0", "\u03c0"], color=p.TEXT_DIM)
        cbar.set_label("phase", color=p.TEXT_DIM, fontsize=9)
        cbar.outline.set_edgecolor(p.BORDER)
        self.rho_canvas.draw()

    # ---- save buttons (one per grill panel) -------------------------------
    def _save_circuit(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save circuit", "circuit.png",
            "PNG image (*.png);;OpenQASM 2 (*.qasm)")
        if not path:
            return
        if path.lower().endswith(".qasm"):
            if self._last_qc is None:
                QMessageBox.warning(self, "Nothing to export",
                                     "qiskit isn't installed, so there's no circuit "
                                     "object to export — see the circuit panel.")
                return
            try:
                from qiskit import qasm2
                Path(path).write_text(qasm2.dumps(self._last_qc), encoding="utf-8")
            except Exception as e:
                QMessageBox.warning(self, "Export failed", str(e))
        else:
            self.circ_canvas.fig.savefig(path, facecolor=self.circ_canvas.fig.get_facecolor())

    def _save_loss(self):
        if not self.loss_x:
            QMessageBox.information(self, "Nothing to save", "Run an optimization first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save loss curve", "loss.csv", "CSV data (*.csv);;PNG image (*.png)")
        if not path:
            return
        if path.lower().endswith(".png"):
            self.loss_canvas.fig.savefig(path, facecolor=self.loss_canvas.fig.get_facecolor())
        else:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["iteration", "loss"])
                w.writerows(zip(self.loss_x, self.loss_y))

    def _save_density(self):
        if self._last_rho is None:
            QMessageBox.information(self, "Nothing to save", "No density matrix yet.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save density matrix", "density_matrix.npy",
            "NumPy array (*.npy);;CSV data (*.csv);;PNG image (*.png)")
        if not path:
            return
        lower = path.lower()
        if lower.endswith(".png"):
            self.rho_canvas.fig.savefig(path, facecolor=self.rho_canvas.fig.get_facecolor())
        elif lower.endswith(".npy"):
            np.save(path, self._last_rho)
        else:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["# real part"])
                w.writerows(self._last_rho.real.tolist())
                w.writerow(["# imag part"])
                w.writerows(self._last_rho.imag.tolist())

    # ---- run control -----------------------------------------------------
    def on_start(self):
        self.draw_connectivity()
        self.draw_circuit()
        self.init_loss_plot()

        self.thread = QThread()
        self.worker = VariationalWorker(self.beta.value(),
                                        self.optimizer.currentText(),
                                        self.iters.value())      # STUB 4
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_progress)
        self.worker.rho_progress.connect(self.draw_density)
        self.worker.finished.connect(self.on_finished)
        self.worker.log.connect(self.statuslbl.setText)
        self.worker.finished.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.thread.start()

    def on_progress(self, it: int, loss: float):
        self.loss_x.append(it); self.loss_y.append(loss)
        self._loss_line.set_data(self.loss_x, self.loss_y)
        self._loss_head.set_data([it], [loss])
        if self._loss_fill is not None:
            self._loss_fill.remove()
        self._loss_fill = self._loss_ax.fill_between(
            self.loss_x, self.loss_y, min(self.loss_y), color=Pal.ACCENT, alpha=0.12, linewidth=0)
        self._loss_ax.relim(); self._loss_ax.autoscale_view()
        self.loss_canvas.draw_idle()

    def on_finished(self, rho):
        self.draw_density(rho)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def on_stop(self):
        if self.worker:
            self.worker.abort()
        self.stop_btn.setEnabled(False)


# =========================================================================
#  Main window (stacked: login -> control panel)
# =========================================================================

class MainWindow(QStackedWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TFD / Gibbs-state preparation — Quantum Inspire")
        self.resize(900, 950)

        self.login = LoginScreen()
        self.panel = ControlPanel()
        self.addWidget(self.login)
        self.addWidget(self.panel)

        self.login.logged_in.connect(self._on_logged_in)

    def _on_logged_in(self, token: str):
        self.panel.set_api_token(token)
        self.setCurrentWidget(self.panel)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")           # a neutral base the stylesheet fully overrides
    app.setFont(QFont("Segoe UI", 10))

    # Dark QPalette too (not just the stylesheet) so native bits Qt renders
    # itself — menus, tooltips, the login card's focus ring — stay dark
    # instead of flashing the OS's light default.
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(Pal.PAGE))
    pal.setColor(QPalette.WindowText, QColor(Pal.TEXT))
    pal.setColor(QPalette.Base, QColor(Pal.SURFACE_HI))
    pal.setColor(QPalette.AlternateBase, QColor(Pal.SURFACE))
    pal.setColor(QPalette.Text, QColor(Pal.TEXT))
    pal.setColor(QPalette.Button, QColor(Pal.SURFACE_HI))
    pal.setColor(QPalette.ButtonText, QColor(Pal.TEXT))
    pal.setColor(QPalette.ToolTipBase, QColor(Pal.SURFACE_HI))
    pal.setColor(QPalette.ToolTipText, QColor(Pal.TEXT))
    pal.setColor(QPalette.Highlight, QColor(Pal.ACCENT))
    pal.setColor(QPalette.HighlightedText, QColor("#071022"))
    app.setPalette(pal)
    app.setStyleSheet(build_stylesheet())

    w = MainWindow()
    w.show()
    sys.exit(app.exec())