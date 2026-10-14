"""Run screen. Rebuilt from the original minimal ExperimentWindow (a single
placeholder QLabel + hardcoded angles) into a real variational run:
OptimizationWorker drives an exact, noiseless cost evaluation (statevector
simulation via TFDCircuit — the same formula as
gibbs_tfd_reproduction/verify_exact_cost.py) on a background QThread
(mirroring the gui branch's tfd_gui.py VariationalWorker QObject/QThread/
Signal pattern) so the UI streams the loss curve and a live
reduced-density-matrix estimate without freezing, then submits the
optimized circuit to the selected Quantum Inspire backend.

Deliberately NOT gibbs_tfd_reproduction.cost_function.CostFunction: that
module was rewritten to build a real measurement circuit and run it on a
backend (NoisySimulator), with no callback support and no configurable
method/maxiter — each cost evaluation now takes seconds, and scipy can't
stream intermediate steps to it at all. Fine for the reproduction
pipeline's own scripts; wrong tool for an interactive GUI that needs to
stream dozens of steps smoothly in under a second.
"""

import csv
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator

from PySide6.QtCore import Qt, QThread, QObject, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout, QLabel, QPushButton,
    QPlainTextEdit, QFileDialog, QMessageBox, QTabWidget, QTabBar, QGroupBox,
)
from frontend.plots import MplCanvas, style_axes, panel, panel_card_stylesheet
from frontend.device_panel import QubitTopologyWidget, KNOWN_BACKEND_TOPOLOGY, _lookup_by_name

from scipy.optimize import minimize

from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import partial_trace, DensityMatrix, state_fidelity, Pauli

from gibbs_tfd_reproduction.tfd_circuit import TFDCircuit
from gibbs_tfd_reproduction.ground_truth import GibbsState, default_subsystem_hamiltonian

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent / "gibbs_tfd_reproduction"))
from gibbs_tfd_reproduction.cost_function import CostFunction
from qi_utilities.device_simulation.simulators import NoisySimulator

from backend import starmon7
from backend.experiment_manager import write_experiment_settings
from frontend.colors import Colors
from frontend.session import SessionConfig
from frontend.upload import save_session_dialog
from frontend.plots import MplCanvas, style_axes, panel

SIDEBAR_WIDTH = 300
BOTTOM_BAR_HEIGHT = 200

# Logical-qubit order TFDCircuit builds on (see tfd_circuit.py): q0=A2,
# q1=A1, q2=B2, q3=B1. A role->physical-qubit mapping from the device panel
# becomes a transpile() initial_layout in this order.
_LOGICAL_ROLE_ORDER = ("A2", "A1", "B2", "B1")

# Purely cosmetic: the exact statevector cost is sub-millisecond per
# evaluation, so a run finishes before the loss curve or density-matrix
# bars have any chance to visibly animate. A short pause per iteration
# (on the worker thread, not the GUI thread) makes the live updates
# actually watchable during a demo instead of just snapping to the final
# result. Tune down/remove for real runs where speed matters more.
_STEP_DELAY_SECONDS = 0.06

# Phase colormap for the density-matrix bars: a light/dark ramp in the app's
# blue instead of a rainbow. Dark at ±π, light at 0 — cyclic, so -π and π
# (the same phase) get the same color, and for the real matrices here
# positive entries read light and negative ones dark.
_PHASE_DARK = "#2B5A80"
_PHASE_LIGHT = "#D6F1FB"
_PHASE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "phase_light_dark", [_PHASE_DARK, _PHASE_LIGHT, _PHASE_DARK])


def _legend_above(ax, ncol=2):
    """Legend in a strip above the plot area, so it never sits on the
    curve, the β markers, or the tick numbers."""
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=ncol,
              fontsize=8, frameon=False, labelcolor=Colors.TEXT_DIM,
              borderaxespad=0.0, handlelength=1.8)


def _nice_limits(lo, hi, nbins=5, integer=False):
    if lo == hi:
        pad = abs(lo) * 0.1 or 0.1
        lo, hi = lo - pad, hi + pad
    ticks = MaxNLocator(nbins=nbins, integer=integer).tick_values(lo, hi)
    return ticks[0], ticks[-1]


def _exact_cost(params, beta, varsigma=1.57):
    """Exact, noiseless cost — same formula as
    gibbs_tfd_reproduction/verify_exact_cost.py's exact_cost, reimplemented
    here (not imported) since that module uses bare imports that only
    resolve when gibbs_tfd_reproduction/ itself is on sys.path, and has a
    top-level print() side effect on import. Takes `params` as a single
    array (scipy's calling convention) rather than four positional args.
    """
    g1, g2, a1, a2 = params
    sv = TFDCircuit(g1, g2, a1, a2).get_state_vector()
    exp_X_A = sv.expectation_value(Pauli("IIXI")).real + sv.expectation_value(Pauli("IIIX")).real
    exp_X_B = sv.expectation_value(Pauli("XIII")).real + sv.expectation_value(Pauli("IXII")).real
    exp_ZZ_A = sv.expectation_value(Pauli("IIZZ")).real
    exp_ZZ_B = sv.expectation_value(Pauli("ZZII")).real
    exp_XX_BA = sv.expectation_value(Pauli("XIXI")).real + sv.expectation_value(Pauli("IXIX")).real
    exp_ZZ_BA = sv.expectation_value(Pauli("ZIZI")).real + sv.expectation_value(Pauli("IZIZ")).real
    return (exp_X_A + exp_X_B + varsigma * (exp_ZZ_A + exp_ZZ_B)
            - beta ** (-varsigma) * (exp_XX_BA + exp_ZZ_BA))


class _Aborted(Exception):
    """Raised from inside scipy's optimizer callback to unwind minimize()
    cleanly when the user hits Stop (verified against COBYLA/Nelder-Mead/
    Powell)."""


class OptimizationWorker(QObject):
    """Runs on a background QThread. Emits `progress(iteration, cost)` every
    optimizer step, `rho_progress(rho)` with the live reduced density matrix
    of subsystem A (an exact partial trace of the simulated ansatz state,
    not a placeholder), then `finished(result_dict)` once done.
    """

    progress = Signal(int, float)
    fidelity_progress = Signal(int, float)
    rho_progress = Signal(object)
    finished = Signal(object)
    log = Signal(str)
    step = Signal(float)

    def __init__(self, manager, config: SessionConfig):
        super().__init__()
        self.manager = manager
        self.config = config
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        config = self.config
        x0 = np.array([config.gamma1, config.gamma2, config.alpha1, config.alpha2])
        iteration = [0]
        betas = [b for b in (1e-8, 0.5, 1, 5) if b < config.beta] + [config.beta]
        current_beta = [betas[0]]

        # Computed once up front rather than per-iteration or only at the
        # end: the true Gibbs state depends only on beta, not on the
        # optimizer's current guess, so the live fidelity trace below and
        # the final result further down can share this one reference.
        try:
            target = DensityMatrix(
                GibbsState(default_subsystem_hamiltonian()).gibbs_density_matrix(config.beta))
        except Exception as exc:
            self.log.emit(f"Couldn't compute the true Gibbs state: {exc}")
            target = None

        # The cost is always evaluated on an ideal Starmon-7 simulation. When
        # Starmon-7 is also the chosen backend, use the user's own qubit
        # assignment; otherwise keep CostFunction's default valid layout
        # (a mapping for another chip needn't fit Starmon-7's coupling map).
        cost_kwargs = {}
        if starmon7.is_starmon7(config.backend_name) and config.is_qubit_mapping_complete():
            cost_kwargs["initial_layout"] = [
                config.qubit_mapping[role] for role in _LOGICAL_ROLE_ORDER]
        cost_fn = CostFunction(
            varsigma=1.57,
            backend=NoisySimulator(starmon7.NAME, ideal_simulation=True),
            **cost_kwargs,
        )

        # The live density-matrix preview is by far the most expensive thing
        # to redraw (a full 3D bar plot + colorbar rebuild, ~20-100ms each),
        # and it runs on the UI thread since Qt marshals the signal there.
        # Emitting it on every single callback call made the whole run look
        # hung: for a typical 100-iteration run this alone accounted for ~90%
        # of the wall-clock time, all spent redrawing on the main thread
        # instead of anything the optimizer needed. Cap it to a bounded
        # number of live previews across the whole run instead — the actual
        # optimization (this is a local statevector simulation, not a
        # hardware round-trip) finishes in well under a second regardless of
        # iteration count. Fidelity itself is just linear algebra on a 4x4
        # matrix (no redraw of an expensive 3D plot involved) so, unlike the
        # density preview, it doesn't need the same throttling.
        rho_every = max(1, config.iterations // 20)

        def callback(xk):
            if self._abort:
                raise _Aborted()
            iteration[0] += 1
            bound = CostFunction.cost_lower_bound(current_beta[0])
            cost = (cost_fn.cost_history[-1] - bound) / abs(bound)
            self.progress.emit(iteration[0], float(cost))
            try:
                rho_a = cost_fn.tomography_history[-1]
                if iteration[0] % rho_every == 0:
                    self.rho_progress.emit(rho_a)
                tgt = DensityMatrix(
                    GibbsState(default_subsystem_hamiltonian()).gibbs_density_matrix(current_beta[0]))
                fid = state_fidelity(DensityMatrix(rho_a), tgt, validate=False)
                self.fidelity_progress.emit(iteration[0], float(fid))
            except Exception:
                pass  # a live preview glitch shouldn't kill the optimization
            time.sleep(_STEP_DELAY_SECONDS)

        self.log.emit(
            f"Starting optimization  |  beta={config.beta}  optimizer={config.optimizer}")

        try:
            for b in betas:
                current_beta[0] = b
                self.step.emit(b)
                self.log.emit(f"Optimizing at beta={b}")
                result = minimize(
                    cost_fn.cost_function, x0=x0, args=(b,), method=config.optimizer,
                    options={'maxiter': config.iterations}, callback=callback,
                )
                x0 = result.x
            final_params = result.x
        except _Aborted:
            self.log.emit("Aborted.")
            self.finished.emit({"aborted": True})
            return
        except Exception as exc:
            self.log.emit(f"Optimization failed: {exc}")
            self.finished.emit({"aborted": True, "error": str(exc)})
            return

        final_circuit = TFDCircuit(*final_params)
        final_cost = float(cost_fn.cost_function(final_params, config.beta))
        rho_final = cost_fn.tomography_history[-1]

        # How good the prepared state actually is, not just how low the
        # (engineered, proxy) cost got: fidelity against the true Gibbs
        # state ρ ∝ exp(-βH) — the same ground-truth check
        # gibbs_tfd_reproduction/experiment.py runs after optimizing.
        # Reuses `target` computed once above rather than recomputing it.
        fidelity = None
        if target is not None:
            try:
                fidelity = float(state_fidelity(DensityMatrix(rho_final), target, validate=False))
            except Exception as exc:
                self.log.emit(f"Couldn't compute final fidelity: {exc}")

        local = starmon7.is_starmon7(config.backend_name)
        self.log.emit(
            "Optimization complete. Running circuit on the Starmon-7 emulator..." if local
            else "Optimization complete. Submitting circuit to Quantum Inspire...")
        submitted = False
        try:
            backend = self.manager.get_backend(config.backend_name)
            initial_layout = [config.qubit_mapping[role] for role in _LOGICAL_ROLE_ORDER]
            qc = final_circuit.as_qiskit_circuit()
            if local:
                # The local emulator returns counts only for measured qubits,
                # and rebuilds circuits with default-named registers — so
                # measure into a plain "c" register, not measure_all()'s "meas".
                measured = QuantumCircuit(qc.num_qubits, qc.num_qubits)
                measured.compose(qc, inplace=True)
                measured.measure(range(qc.num_qubits), range(qc.num_qubits))
                qc = measured
            transpiled = transpile(qc, backend, initial_layout=initial_layout)
            job, _results = self.manager.submit_circuit(transpiled, backend)
            submitted = job is not None
            if submitted:
                try:
                    record = self.manager.save_job(job)
                    # The job record has no app settings; store them (and the
                    # optimizer's outcome) alongside it for the Load screen's
                    # experiment history.
                    write_experiment_settings(record.project_dir, {
                        **config.to_json_dict(),
                        "results": {
                            "optimized_params": dict(zip(
                                ("gamma1", "gamma2", "alpha1", "alpha2"),
                                (float(p) for p in final_params))),
                            "final_cost": final_cost,
                            "fidelity": fidelity,
                            "job_id": str(self.manager.get_job_id(job)),
                        },
                    })
                    self.log.emit(f"Saved to {record.project_dir}")
                except Exception as exc:
                    self.log.emit(f"Job ran but couldn't be saved locally: {exc}")
            self.log.emit("Submitted." if submitted else "Submission returned no job.")
        except Exception as exc:
            self.log.emit(f"Submission failed: {exc}")

        self.finished.emit({
            "aborted": False,
            "params": final_params,
            "cost": final_cost,
            "rho": rho_final,
            "fidelity": fidelity,
            "submitted": submitted,
        })


class ExperimentWindow(QWidget):

    back_to_setup = Signal()

    def __init__(self, manager):
        super().__init__()
        self.manager = manager
        self.thread = None
        self.worker = None
        self.config = None
        self.loss_x, self.loss_y = [], []
        self.fidelity_x, self.fidelity_y = [], []
        self._last_qc = None
        self._last_rho = None
        self._last_params = None
        self._target_rho = None
        self._ideal_zmax = None
        self._graphs = {}
        self._tab_pages = {}

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # -------------------------------------------------
        # Left sidebar — recap, stop, status log
        # -------------------------------------------------

        sidebar = QWidget()
        sidebar.setFixedWidth(SIDEBAR_WIDTH)
        sidebar.setStyleSheet(f"background-color: {Colors.BG_2};")
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(20, 28, 20, 20)
        side.setSpacing(10)

        title = QLabel("Run")
        title_font = QFont()
        title_font.setPointSize(15)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet(f"color: {Colors.TEXT};")
        side.addWidget(title)

        self.recap_form = QFormLayout()
        self.recap_form.setVerticalSpacing(6)
        self._recap_labels = {}
        for key in ("Backend", "β", "Optimizer", "Iterations", "γ, α (initial)", "Qubits"):
            lbl = QLabel("–")
            lbl.setWordWrap(True)
            lbl.setStyleSheet(f"color: {Colors.TEXT};")
            self._recap_labels[key] = lbl
            self.recap_form.addRow(f"{key}:", lbl)
        side.addLayout(self.recap_form)

        self.qubit_view = QubitTopologyWidget()
        self.qubit_view.set_read_only(True)
        self.qubit_view.setFixedHeight(190)
        self.qubit_view.hide()
        side.addWidget(self.qubit_view)

        side.addSpacing(6)
        self.result_lbl = QLabel("")
        self.result_lbl.setWordWrap(True)
        self.result_lbl.setStyleSheet(f"color: {Colors.GOOD}; font-weight: 600;")
        side.addWidget(self.result_lbl)

        self.save_session_btn = QPushButton("💾  Save session")
        self.save_session_btn.setCursor(Qt.PointingHandCursor)
        self.save_session_btn.setEnabled(False)
        self.save_session_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; color: {Colors.TEXT_DIM};
                border: 1px solid {Colors.BORDER}; border-radius: 10px;
                padding: 8px;
            }}
            QPushButton:hover {{ color: {Colors.TEXT}; border-color: {Colors.ACCENT}; }}
            QPushButton:disabled {{ color: {Colors.TEXT_MUTED}; border-color: {Colors.BORDER}; }}
        """)
        self.save_session_btn.clicked.connect(self._on_save_session)
        side.addWidget(self.save_session_btn)

        side.addStretch(0)
        side.addSpacing(6)

        self.stop_btn = QPushButton("■  Stop")
        self.stop_btn.setMinimumHeight(40)
        self.stop_btn.setCursor(Qt.PointingHandCursor)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; color: {Colors.WARN};
                border: 1px solid {Colors.WARN}; border-radius: 10px; font-weight: bold;
            }}
            QPushButton:hover {{ background: rgba(254,138,39,0.12); }}
            QPushButton:disabled {{ color: {Colors.TEXT_MUTED}; border-color: {Colors.BORDER}; }}
        """)
        self.stop_btn.clicked.connect(self.stop)
        side.addWidget(self.stop_btn)

        self.restart_btn = QPushButton("↻  Run again")
        self.restart_btn.setMinimumHeight(40)
        self.restart_btn.setCursor(Qt.PointingHandCursor)
        self.restart_btn.setEnabled(False)
        self.restart_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; color: {Colors.ACCENT};
                border: 1px solid {Colors.ACCENT}; border-radius: 10px; font-weight: bold;
            }}
            QPushButton:hover {{ background: rgba(135,206,235,0.12); }}
            QPushButton:disabled {{ color: {Colors.TEXT_MUTED}; border-color: {Colors.BORDER}; }}
        """)
        self.restart_btn.clicked.connect(self._on_restart)
        side.addWidget(self.restart_btn)

        status_lbl = QLabel("Status")
        status_lbl.setStyleSheet(f"color: {Colors.TEXT_DIM}; font-weight: 600;")
        side.addWidget(status_lbl)

        self.status_log = QPlainTextEdit()
        self.status_log.setReadOnly(True)
        self.status_log.setMaximumBlockCount(500)
        self.status_log.setStyleSheet(f"""
            QPlainTextEdit {{
                background: {Colors.BG_3}; color: {Colors.TEXT_DIM};
                border: 1px solid {Colors.BORDER}; border-radius: 8px;
                font-family: monospace; font-size: 9pt;
            }}
        """)
        self.status_log.setFixedHeight(140)
        side.addWidget(self.status_log)

        back_btn = QPushButton("←  Back to Setup")
        back_btn.setCursor(Qt.PointingHandCursor)
        back_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; color: {Colors.TEXT_DIM};
                border: 1px solid {Colors.BORDER}; border-radius: 10px; padding: 8px;
            }}
            QPushButton:hover {{ color: {Colors.TEXT}; border-color: {Colors.ACCENT}; }}
        """)
        back_btn.clicked.connect(self.back_to_setup.emit)
        side.addWidget(back_btn)

        root.addWidget(sidebar)

        # -------------------------------------------------
        # Right main area
        # -------------------------------------------------

        overview = QWidget()
        main_layout = QVBoxLayout(overview)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(10)

        # One-line summary of the run's settings above the results grid
        # (angles stay in the sidebar recap only).
        params_bar = QWidget()
        params_bar.setObjectName("paramsBar")
        params_bar.setStyleSheet(f"""
            #paramsBar {{
                background: {Colors.BG_2}; border: 1px solid {Colors.BORDER};
                border-radius: 10px;
            }}
            QLabel {{ background: transparent; }}
        """)
        params_row = QHBoxLayout(params_bar)
        params_row.setContentsMargins(16, 8, 16, 8)
        params_row.setSpacing(28)
        self._param_strip_labels = {}
        for key in ("Backend", "β", "Optimizer", "Iterations", "Qubits"):
            lbl = QLabel()
            lbl.setTextFormat(Qt.RichText)
            lbl.setText(f"<span style='color:{Colors.TEXT_DIM};'>{key}</span>&nbsp;&nbsp;–")
            self._param_strip_labels[key] = lbl
            params_row.addWidget(lbl)
        params_row.addStretch()
        main_layout.addWidget(params_bar)

        self.loss_canvas = MplCanvas()
        self.fidelity_canvas = MplCanvas()
        self.ideal_rho_canvas = MplCanvas(height=4.0, use_layout=False)
        self.rho_canvas = MplCanvas(height=4.0, use_layout=False)

        # 2x2 results grid: live curves on top, ideal vs. observed ρ_A
        # side by side underneath.
        grid = QGridLayout()
        grid.setSpacing(10)
        def _card(key, title, canvas, save):
            card = panel(title, canvas)
            self._graphs[key] = (title, canvas, save, card)
            return card

        grid = QGridLayout()
        grid.setSpacing(10)
        grid.addWidget(_card("fidelity", "Fidelity to target (live)", self.fidelity_canvas,
                             self._save_fidelity), 0, 0)
        grid.addWidget(_card("loss", "Cost function (live)", self.loss_canvas,
                             self._save_loss), 0, 1)
        grid.addWidget(_card("ideal", "Ideal density matrix (true Gibbs state)", self.ideal_rho_canvas,
                             self._save_ideal_density), 1, 0)
        grid.addWidget(_card("measured", "Measured density matrix (tomography)", self.rho_canvas,
                             self._save_density), 1, 1)
        for i in (0, 1):
            grid.setRowStretch(i, 1)
            grid.setColumnStretch(i, 1)
        main_layout.addLayout(grid, stretch=1)

        circuit_toggle = QPushButton("▸  Circuit diagram")
        circuit_toggle.setCheckable(True)
        circuit_toggle.setCursor(Qt.PointingHandCursor)
        circuit_toggle.setStyleSheet(f"""
            QPushButton {{
                text-align: left; background: {Colors.BG_2}; color: {Colors.TEXT_DIM};
                border: 1px solid {Colors.BORDER}; border-radius: 8px; padding: 6px 10px;
            }}
        """)
        main_layout.addWidget(circuit_toggle)

        self.circ_canvas = MplCanvas(height=2.6)
        self.circuit_card = _card("circuit", "Circuit (reference)", self.circ_canvas,
                                  self._save_circuit)
        self.circuit_card.hide()
        main_layout.addWidget(self.circuit_card)

        def _toggle_circuit(checked):
            circuit_toggle.setText(("▾" if checked else "▸") + "  Circuit diagram")
            self.circuit_card.setVisible(checked)
        circuit_toggle.toggled.connect(_toggle_circuit)

        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.tabBar().setDrawBase(False)
        self.tabs.setStyleSheet(f"""
            QTabWidget::pane {{ border: none; }}
            QTabBar::tab {{
                background: {Colors.BG_2}; color: {Colors.TEXT_DIM};
                padding: 8px 16px; margin-right: 2px; min-width: 90px;
                border-top-left-radius: 10px; border-top-right-radius: 10px;
            }}
            QTabBar::tab:selected {{
                background: {Colors.BG_3}; color: {Colors.TEXT};
                border-bottom: 2px solid {Colors.ACCENT};
            }}
            QTabBar::tab:hover:!selected {{ background: {Colors.BG_3}; }}
        """)
        self.tabs.addTab(overview, "Overview")
        self.tabs.tabBar().setTabButton(0, QTabBar.RightSide, None)
        self.tabs.tabBar().setTabButton(0, QTabBar.LeftSide, None)
        self.tabs.currentChanged.connect(self._rehome)
        self.tabs.tabCloseRequested.connect(self._close_tab)
        for key in self._graphs:
            self._make_clickable(key)
        root.addWidget(self.tabs, stretch=1)

        self.init_loss_plot()
        self.init_fidelity_plot()
        self._draw_empty_density()
        self._draw_ideal_density()

    # -----------------------------------------------------
    # Public API (driven by main.py)
    # -----------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self.thread is not None and self.thread.isRunning()

    def start_run(self, config: SessionConfig):
        self.config = config
        self._last_params = None
        self.save_session_btn.setEnabled(False)
        self.restart_btn.setEnabled(False)
        self._update_recap(config)
        known = _lookup_by_name(config.backend_name, KNOWN_BACKEND_TOPOLOGY)
        if known is not None and config.qubit_mapping:
            self.qubit_view.set_topology(
                len(known["coords"]), {frozenset(p) for p in known["edges"]}, known["coords"])
            self.qubit_view.set_assignment(config.qubit_mapping)
            self.qubit_view.show()
        else:
            self.qubit_view.hide()
        self.result_lbl.setText("")
        self.status_log.clear()
        self.init_loss_plot()
        self.init_fidelity_plot()
        self._draw_circuit_for(
            [config.gamma1, config.gamma2, config.alpha1, config.alpha2])

        try:
            self._target_rho = DensityMatrix(
                GibbsState(default_subsystem_hamiltonian()).gibbs_density_matrix(config.beta)).data
        except Exception as exc:
            self._on_log(f"Couldn't compute the true Gibbs state: {exc}")
            self._target_rho = None
        self._last_rho = None
        self._draw_ideal_density()
        self._draw_empty_density()

        self.thread = QThread()
        self.worker = OptimizationWorker(self.manager, config)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._on_progress)
        self.worker.fidelity_progress.connect(self._on_fidelity_progress)
        self.worker.rho_progress.connect(self._draw_density)
        self.worker.log.connect(self._on_log)
        self.worker.finished.connect(self._on_finished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.step.connect(self._on_step)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self._on_thread_finished)

        self.stop_btn.setEnabled(True)
        self.thread.start()

    def _on_thread_finished(self):
        # self.thread is a plain, unparented QThread(), so Python's own
        # refcounting owns it — dropping our last reference below triggers
        # *immediate* C++ destruction via shiboken, independent of (and
        # potentially racing ahead of) the deleteLater() already queued
        # above. QThread.finished firing means run() has returned, but
        # wait() is the actual guarantee the OS thread has fully joined;
        # skipping it was producing a real
        # "QThread: Destroyed while thread '' is still running" abort,
        # reliably reachable once something (e.g. the Run-again button)
        # creates a new run shortly after the previous one finished.
        if self.thread is not None:
            self.thread.wait()
        self.thread = None
        self.worker = None

    def stop(self):
        if self.worker is not None:
            self.worker.abort()
        self.stop_btn.setEnabled(False)

    def _on_restart(self):
        # Re-run the same config from scratch — same starting angles, same
        # backend/beta/optimizer/iterations — so the live loss/density
        # animation plays again without a trip back through Setup. Guarded
        # by is_running too, not just the button's enabled state, since
        # start_run() tears down and replaces self.thread/self.worker and
        # doing that while the previous run is still live would orphan it
        # mid-flight.
        if self.config is not None and not self.is_running:
            self.start_run(self.config)
    
    def _make_clickable(self, key):
        canvas = self._graphs[key][1]
        press = {}

        def on_press(ev):
            press["xy"] = (ev.x, ev.y)

        def on_release(ev):
            p = press.get("xy")
            if p is None or self.tabs.currentIndex() != 0:
                return
            if abs(ev.x - p[0]) < 5 and abs(ev.y - p[1]) < 5:
                self._open_tab(key)

        canvas.mpl_connect("button_press_event", on_press)
        canvas.mpl_connect("button_release_event", on_release)
        canvas.setCursor(Qt.PointingHandCursor)

    def _open_tab(self, key):
        if key in self._tab_pages:
            self.tabs.setCurrentWidget(self._tab_pages[key][0])
            return
        title, canvas, save, card = self._graphs[key]
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(20, 14, 20, 20)
        frame = QGroupBox(title)
        frame.setStyleSheet(panel_card_stylesheet())
        frame_lay = QVBoxLayout(frame)
        frame_lay.setContentsMargins(16, 14, 16, 16)
        frame_lay.setSpacing(10)
        bar = QHBoxLayout()
        bar.addStretch()
        save_btn = QPushButton("💾  Save")
        save_btn.setCursor(Qt.PointingHandCursor)
        save_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; color: {Colors.ACCENT};
                border: 1px solid {Colors.ACCENT}; border-radius: 8px;
                padding: 5px 16px; font-weight: 600;
            }}
            QPushButton:hover {{ background: rgba(135,206,235,0.12); }}
        """)
        save_btn.clicked.connect(save)
        bar.addWidget(save_btn)
        frame_lay.addLayout(bar)
        body = QVBoxLayout()
        frame_lay.addLayout(body, stretch=1)
        lay.addWidget(frame)
        self._tab_pages[key] = (page, body)
        idx = self.tabs.addTab(page, title.split(" (")[0])
        self.tabs.setCurrentIndex(idx)

    def _rehome(self, index):
        page = self.tabs.widget(index)
        for key, (title, canvas, save, card) in self._graphs.items():
            entry = self._tab_pages.get(key)
            if entry is not None and entry[0] is page:
                if canvas.parent() is not entry[1].parentWidget():
                    entry[1].addWidget(canvas)
                canvas.setCursor(Qt.ArrowCursor)
            else:
                if canvas.parent() is not card:
                    card.layout().addWidget(canvas)
                canvas.setCursor(Qt.PointingHandCursor)
            canvas.draw_idle()

    def _close_tab(self, index):
        if index == 0:
            return
        page = self.tabs.widget(index)
        key = next(k for k, (p, _) in self._tab_pages.items() if p is page)
        _, canvas, _, card = self._graphs[key]
        card.layout().addWidget(canvas)
        del self._tab_pages[key]
        self.tabs.removeTab(index)
        page.deleteLater()

    # -----------------------------------------------------
    # Recap / logging
    # -----------------------------------------------------

    def _update_recap(self, config: SessionConfig):
        self._recap_labels["Backend"].setText(config.backend_name)
        self._recap_labels["β"].setText(f"{config.beta:.2f}")
        self._recap_labels["Optimizer"].setText(config.optimizer)
        self._recap_labels["Iterations"].setText(str(config.iterations))
        self._recap_labels["γ, α (initial)"].setText(
            f"γ₁={config.gamma1:.2f}  γ₂={config.gamma2:.2f}\n"
            f"α₁={config.alpha1:.2f}  α₂={config.alpha2:.2f}")
        mapping = config.qubit_mapping or {}
        qubits = "  ".join(f"{role}→q{mapping[role]}" for role in ("A1", "A2", "B1", "B2")
                           if role in mapping) or "–"
        self._recap_labels["Qubits"].setText(qubits)

        strip_values = {
            "Backend": config.backend_name,
            "β": f"{config.beta:.2f}",
            "Optimizer": config.optimizer,
            "Iterations": str(config.iterations),
            "Qubits": qubits,
        }
        for key, value in strip_values.items():
            self._param_strip_labels[key].setText(
                f"<span style='color:{Colors.TEXT_DIM};'>{key}</span>&nbsp;&nbsp;"
                f"<b style='color:{Colors.TEXT};'>{value}</b>")

    def _on_log(self, message: str):
        self.status_log.appendPlainText(message)

    def _on_step(self, beta: float):
        for ax, xs in ((self._loss_ax, self.loss_x), (self._fidelity_ax, self.fidelity_x)):
            x = (xs[-1] + 0.5) if xs else 0
            ax.axvline(x, color=Colors.TEXT_MUTED, linestyle=":", linewidth=1.2)
            # Label sits just right of the line (inside the plot), so the
            # first one at x=0 doesn't land on the y-axis tick numbers.
            mantissa, _, exponent = f"{beta:g}".partition("e")
            beta_tex = (rf"10^{{{int(exponent)}}}" if mantissa == "1" else
                        rf"{mantissa}\times10^{{{int(exponent)}}}") if exponent else mantissa
            ax.annotate(rf"$\beta={beta_tex}$",xy=(x, 0.97), xycoords=ax.get_xaxis_transform(),
                        xytext=(3, 0), textcoords="offset points",
                        color=Colors.TEXT_DIM, fontsize=7.5, rotation=90,
                        ha="left", va="top",
                        bbox=dict(boxstyle="round,pad=0.15", facecolor=Colors.BG_3,
                                  edgecolor="none", alpha=0.85))

    def _on_finished(self, result: dict):
        self.stop_btn.setEnabled(False)
        self.restart_btn.setEnabled(True)  # retry allowed whether this run succeeded, failed, or was stopped
        if result.get("aborted"):
            return
        params = result["params"]
        self._last_params = params
        self.save_session_btn.setEnabled(True)
        self._draw_circuit_for(params)
        # Always draw the true converged result, regardless of where the
        # throttled live preview (see OptimizationWorker.run) last landed.
        self._draw_density(result["rho"])
        fidelity = result.get("fidelity")
        if fidelity is not None:
            # Make sure the live curve's last point matches the reported
            # result exactly, even if result.x differs slightly from the
            # last xk the callback saw.
            final_it = (self.fidelity_x[-1] + 1) if self.fidelity_x else 1
            self._on_fidelity_progress(final_it, fidelity)
        fidelity_line = (
            f"Fidelity to true Gibbs state: {fidelity:.4f}\n" if fidelity is not None
            else "")
        self.result_lbl.setText(
            f"Converged  |  final cost={result['cost']:.4f}\n"
            f"γ₁={params[0]:.3f} γ₂={params[1]:.3f} "
            f"α₁={params[2]:.3f} α₂={params[3]:.3f}\n"
            + fidelity_line
            + ("Submitted to backend." if result.get("submitted")
               else "Submission did not complete — see status log."))

    # -----------------------------------------------------
    # Plotting
    # -----------------------------------------------------

    def init_loss_plot(self):
        self.loss_x, self.loss_y = [], []
        fig = self.loss_canvas.fig
        fig.clear()
        ax = fig.add_subplot(111)
        self._loss_ax = ax
        (self._loss_line,) = ax.plot([], [], "-", color=Colors.ACCENT, linewidth=2.2,
                                      solid_capstyle="round")
        (self._loss_head,) = ax.plot([], [], "o", color=Colors.ACCENT_HOVER,
                                      markersize=6, markeredgecolor=Colors.BG_3,
                                      markeredgewidth=1.2)
        self._loss_fill = None
        self._loss_line.set_label("Cost")
        # No in-plot title: the panel card already says "Cost function", and
        # the legend strip goes where a title would be.
        style_axes(ax, xlabel="Iteration", ylabel="Relative gap to lower bound")
        self._loss_bound = 0.0
        ax.axhline(0.0, color=Colors.GOOD, linestyle="--", linewidth=1.8, label="Lower bound")
        _legend_above(ax)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.loss_canvas.draw()

    def _on_progress(self, it: int, cost: float):
        self.loss_x.append(it)
        self.loss_y.append(cost)
        self._loss_line.set_data(self.loss_x, self.loss_y)
        self._loss_head.set_data([it], [cost])
        if self._loss_fill is not None:
            self._loss_fill.remove()
        self._loss_fill = self._loss_ax.fill_between(
            self.loss_x, self.loss_y, min(self.loss_y),
            color=Colors.ACCENT, alpha=0.12, linewidth=0)
        self._loss_ax.set_xlim(*_nice_limits(0, max(self.loss_x), integer=True))
        lo = min(self.loss_y) if self._loss_bound is None else min(min(self.loss_y), self._loss_bound)
        self._loss_ax.set_ylim(*_nice_limits(lo, max(self.loss_y)))
        self.loss_canvas.draw_idle()

    def init_fidelity_plot(self):
        self.fidelity_x, self.fidelity_y = [], []
        fig = self.fidelity_canvas.fig
        fig.clear()
        ax = fig.add_subplot(111)
        self._fidelity_ax = ax
        self._fidelity_expected_line = ax.axhline(
            1.0, color=Colors.GOOD, alpha=0.55, linewidth=2, linestyle="--",
            zorder=3, label="Expected (ideal)")
        self._fidelity_gap_fill = None
        (self._fidelity_line,) = ax.plot([], [], "-", color=Colors.ACCENT, linewidth=2.2,
                                          solid_capstyle="round", zorder=2, label="Observed")
        (self._fidelity_head,) = ax.plot([], [], "o", color=Colors.ACCENT_HOVER,
                                          markersize=6, markeredgecolor=Colors.BG_3,
                                          markeredgewidth=1.2, zorder=2)
        style_axes(ax, xlabel="Iteration", ylabel="Fidelity")
        ax.set_ylim(0, 1.05)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        _legend_above(ax)
        self.fidelity_canvas.draw()

    def _on_fidelity_progress(self, it: int, fid: float):
        self.fidelity_x.append(it)
        self.fidelity_y.append(fid)
        self._fidelity_line.set_data(self.fidelity_x, self.fidelity_y)
        self._fidelity_head.set_data([it], [fid])
        if self._fidelity_gap_fill is not None:
            self._fidelity_gap_fill.remove()
        self._fidelity_gap_fill = self._fidelity_ax.fill_between(
            self.fidelity_x, self.fidelity_y, 1.0,
            color=Colors.GOOD, alpha=0.12, linewidth=0, zorder=1)
        lo, _ = _nice_limits(min(self.fidelity_y), 1.0)
        lo = max(lo, 0.0)  # fidelity can't go below 0
        self._fidelity_ax.set_ylim(lo, 1.0 + (1.0 - lo) * 0.05)  # small headroom above the 1.0 line
        self._fidelity_ax.set_xlim(*_nice_limits(0, max(self.fidelity_x), integer=True))
        self.fidelity_canvas.draw_idle()

    def _draw_empty_density(self):
        self._render_density(self.rho_canvas, np.eye(4, dtype=complex) / 4,
                             title=r"$\rho_A$  (waiting for first estimate)", zmax=self._density_zmax())

    def _draw_density(self, rho: np.ndarray):
        """Connected to the worker's live rho_progress and called from
        _on_finished with the converged result."""
        self._last_rho = rho
        zmax = self._density_zmax()
        if zmax != self._ideal_zmax:
            # Keep both 3D plots on the same |ρ| scale so they compare
            # bar-for-bar.
            self._draw_ideal_density()
        self._render_density(self.rho_canvas, rho, title=r"$\rho_A$  (observed, live)", zmax=zmax)

    def _draw_ideal_density(self):
        self._ideal_zmax = self._density_zmax()
        if self._target_rho is None:
            rho, title = np.eye(4, dtype=complex) / 4, r"$\rho_A^{\mathrm{ideal}}$  (unavailable)"
        else:
            rho, title = self._target_rho, r"$\rho_A^{\mathrm{ideal}} = e^{-\beta H} / Z$"
        self._render_density(self.ideal_rho_canvas, rho, title=title, zmax=self._ideal_zmax)

    def _density_zmax(self) -> float:
        """Shared z-axis top for the ideal and observed plots: the larger
        matrix's biggest |entry|, rounded up to 0.1."""
        peak = 0.25
        for rho in (self._target_rho, self._last_rho):
            if rho is not None:
                peak = max(peak, float(np.abs(rho).max()))
        return min(1.0, np.ceil(peak * 10 - 1e-9) / 10)

    def _render_density(self, canvas: MplCanvas, rho: np.ndarray, title: str, zmax: float):
        fig = canvas.fig
        fig.clear()
        fig.subplots_adjust(left=0.0, right=0.80, bottom=0.04, top=0.94)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor(Colors.BG_3)
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.set_pane_color((0, 0, 0, 0))
            pane._axinfo["grid"]["color"] = mcolors.to_rgba(Colors.GRID, 0.6)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.line.set_color(Colors.GRID)
            axis.label.set_color(Colors.TEXT_DIM)
        ax.tick_params(colors=Colors.TEXT_MUTED, labelsize=8, pad=1)

        n = rho.shape[0]
        xpos, ypos = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        xpos = xpos.ravel(); ypos = ypos.ravel()
        heights = np.abs(rho).ravel()
        phases = np.angle(rho).ravel()
        norm_phase = (phases + np.pi) / (2 * np.pi)
        colors = _PHASE_CMAP(norm_phase)
        ax.bar3d(xpos, ypos, np.zeros_like(heights), 0.7, 0.7, heights,
                  color=colors, shade=True, edgecolor=Colors.BG_1, linewidth=0.6)
        labels = ["00", "01", "10", "11"][:n]
        ax.set_xticks(np.arange(n) + 0.35); ax.set_xticklabels(labels)
        ax.set_yticks(np.arange(n) + 0.35); ax.set_yticklabels(labels)
        ax.set_zlim(0, zmax)
        # Few, short z ticks so they don't run into each other or the label.
        ax.zaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.zaxis.set_major_formatter("{x:.1f}")
        ax.set_zlabel(r"$|\rho_{ij}|$", labelpad=2, fontsize=10)
        ax.text2D(0.02, 0.98, title, transform=ax.transAxes, fontsize=12,
                   color=Colors.TEXT, va="top")
        # No in-plot title via ax.set_title(): the panel() card above already
        # headers it, and a 3D-axes title doesn't respect the panel's width
        # (it overflows past the canvas edges once the panel gets narrow).

        # Colorbar in its own fixed slot at the right edge, clear of the 3D
        # axes' tick numbers.
        sm = cm.ScalarMappable(cmap=_PHASE_CMAP, norm=mcolors.Normalize(vmin=-np.pi, vmax=np.pi))
        cax = fig.add_axes([0.85, 0.22, 0.03, 0.56])
        cbar = fig.colorbar(sm, cax=cax, ticks=[-np.pi, 0, np.pi])
        cbar.ax.set_yticklabels(["−π", "0", "π"], color=Colors.TEXT_DIM, fontsize=8)
        cbar.set_label("phase", color=Colors.TEXT_DIM, fontsize=8, labelpad=2)
        cbar.outline.set_edgecolor(Colors.BORDER)
        canvas.draw_idle()

    def _draw_circuit_for(self, params):
        qc = TFDCircuit(*params).as_qiskit_circuit()
        self._last_qc = qc
        fig = self.circ_canvas.fig
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_facecolor(Colors.BG_3)
        style = {
            "backgroundcolor": Colors.BG_3,
            "linecolor": Colors.TEXT_DIM,
            "textcolor": Colors.TEXT,
            "gatetextcolor": "#06101C",
            "subtextcolor": Colors.TEXT_MUTED,
            "creglinecolor": Colors.TEXT_MUTED,
            "gatefacecolor": Colors.ACCENT,
            "barrierfacecolor": Colors.GRID,
            # Qiskit colors gates per name via displaycolor, which overrides
            # gatefacecolor: same light/dark blues as the density plots —
            # single-qubit gates light, two-qubit gates dark.
            "displaycolor": {
                inst.operation.name: (
                    [_PHASE_LIGHT, "#06101C"] if inst.operation.num_qubits == 1
                    else [_PHASE_DARK, Colors.TEXT])
                for inst in qc.data
            },
        }
        # Wires labelled by role instead of q0..q3: one single-qubit register
        # per role (Qiskit labels a size-1 register by its name alone), in
        # TFDCircuit's logical order, drawn top-to-bottom as A1 A2 B1 B2.
        # Only the drawing is relabelled; _last_qc (QASM export) keeps "q".
        labelled = QuantumCircuit(*(QuantumRegister(1, role) for role in _LOGICAL_ROLE_ORDER))
        labelled.compose(qc, inplace=True)
        wire_order = [_LOGICAL_ROLE_ORDER.index(role) for role in ("A1", "A2", "B1", "B2")]
        try:
            labelled.draw(output="mpl", ax=ax, style=style, wire_order=wire_order)
        except Exception:
            labelled.draw(output="mpl", ax=ax)
        self.circ_canvas.draw()

    # -----------------------------------------------------
    # Save actions
    # -----------------------------------------------------

    def _on_save_session(self):
        if self.config is None:
            return
        # Save the *converged* angles, not the starting guess this run
        # began from — reloading this session should resume from the
        # result, not repeat the optimization from scratch.
        params = self._last_params
        config = replace(
            self.config,
            gamma1=float(params[0]), gamma2=float(params[1]),
            alpha1=float(params[2]), alpha2=float(params[3]),
        ) if params is not None else self.config
        save_session_dialog(self, config)

    def _save_circuit(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save circuit", "circuit.png",
            "PNG image (*.png);;OpenQASM 2 (*.qasm)")
        if not path:
            return
        if path.lower().endswith(".qasm"):
            if self._last_qc is None:
                QMessageBox.warning(self, "Nothing to export", "No circuit drawn yet.")
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
                w.writerow(["iteration", "cost"])
                w.writerows(zip(self.loss_x, self.loss_y))

    def _save_fidelity(self):
        if not self.fidelity_x:
            QMessageBox.information(self, "Nothing to save", "Run an optimization first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save fidelity curve", "fidelity.csv", "CSV data (*.csv);;PNG image (*.png)")
        if not path:
            return
        if path.lower().endswith(".png"):
            self.fidelity_canvas.fig.savefig(path, facecolor=self.fidelity_canvas.fig.get_facecolor())
        else:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["iteration", "fidelity"])
                w.writerows(zip(self.fidelity_x, self.fidelity_y))

    def _save_density(self):
        self._save_density_matrix(self._last_rho, self.rho_canvas, "density_matrix.npy")

    def _save_ideal_density(self):
        self._save_density_matrix(self._target_rho, self.ideal_rho_canvas,
                                  "ideal_density_matrix.npy")

    def _save_density_matrix(self, displayed_rho, canvas, default_name):
        if displayed_rho is None:
            QMessageBox.information(self, "Nothing to save", "No density matrix yet.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save density matrix", default_name,
            "NumPy array (*.npy);;CSV data (*.csv);;PNG image (*.png)")
        if not path:
            return
        lower = path.lower()
        if lower.endswith(".png"):
            canvas.fig.savefig(path, facecolor=canvas.fig.get_facecolor())
        elif lower.endswith(".npy"):
            np.save(path, displayed_rho)
        else:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["# real part"])
                w.writerows(displayed_rho.real.tolist())
                w.writerow(["# imag part"])
                w.writerows(displayed_rho.imag.tolist())