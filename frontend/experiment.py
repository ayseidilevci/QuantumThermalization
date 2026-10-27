"""Run screen: runs the optimization on a background QThread
(OptimizationWorker) and streams the cost, fidelity and density matrix to
live plots, then submits the optimized circuit to the chosen backend.

2+2 runs use gibbs_tfd_reproduction's CostFunction: an ideal Starmon-7
simulation on the emulator, the device itself on real hardware. 3+3 and
4+4 runs use backend/ladder_tfd.py, on Tuna-17 only.
"""

import csv
import json
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

# cost_function.py imports its siblings bare ("from tfd_circuit import ...").
sys.path.append(str(Path(__file__).resolve().parent.parent / "gibbs_tfd_reproduction"))

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.ticker import MaxNLocator
from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFileDialog, QFormLayout, QGridLayout, QHBoxLayout, QLabel, QMessageBox, QPushButton, QScrollArea,
    QTabBar, QTabWidget, QVBoxLayout, QWidget,
)
from qi_utilities.device_simulation.simulators import NoisySimulator
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.providers.jobstatus import JobStatus
from qiskit.quantum_info import DensityMatrix, state_fidelity
from scipy.optimize import minimize

from backend import starmon7
from backend.experiment_manager import PROJECTS_PATH, write_experiment_settings
from backend.ladder_tfd import LadderTFD, loss_lower_bound
from backend.spsa import spsa_minimize
from frontend.colors import Colors
from frontend.device_panel import KNOWN_BACKEND_TOPOLOGY, QubitTopologyWidget, _lookup_by_name
from frontend.hardware_panels import DeviceStatusWidget, HardwareJobsView
from frontend.plots import MplCanvas, panel, style_axes
from frontend.session import (
    DEFAULT_START_ANGLE, HARDWARE_OPTIMIZER, SessionConfig, format_beta, format_temperature,
    random_start_angles, roles,
)
from frontend.upload import save_session_dialog
from gibbs_tfd_reproduction.cost_function import CostFunction
from gibbs_tfd_reproduction.tfd_circuit import TFDCircuit
from tfd_study import model as tfd_model

SIDEBAR_WIDTH = 300

# Logical-qubit order TFDCircuit builds on (see tfd_circuit.py): q0=A2,
# q1=A1, q2=B2, q3=B1. A role->physical-qubit mapping from the device panel
# becomes a transpile() initial_layout in this order.
_LOGICAL_ROLE_ORDER = ("A2", "A1", "B2", "B1")

# Pause per emulator iteration so the live plots visibly animate; the
# noiseless simulation is otherwise near-instant.
_STEP_DELAY_SECONDS = 0.06

# How often a queued hardware job's status is polled, and how the Qiskit
# job states read in the Hardware jobs tab.
_POLL_SECONDS = 5
_JOB_STATUS_TEXT = {
    JobStatus.INITIALIZING: "Submitted",
    JobStatus.QUEUED: "Queued",
    JobStatus.VALIDATING: "Queued",
    JobStatus.RUNNING: "Running on hardware",
    JobStatus.DONE: "Results received",
    JobStatus.ERROR: "Failed",
    JobStatus.CANCELLED: "Cancelled",
}


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")

# Phase colormap for the density bars: cyclic light/dark blue, light at 0
# and dark at ±π.
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


class _Aborted(Exception):
    """Raised from the optimizer's callback or a job wait to unwind the run
    when the user hits Stop."""


class OptimizationWorker(QObject):
    """Runs on a background QThread. Emits `progress(iteration, cost)` every
    optimizer step, `rho_progress(rho)` with subsystem A's measured density
    matrix, then `finished(result_dict)` once done.
    """

    progress = Signal(int, float)
    fidelity_progress = Signal(int, float)
    rho_progress = Signal(object)
    finished = Signal(object)
    log = Signal(str)
    step = Signal(float)
    # One dict per hardware job state change: n, purpose, job_id, status,
    # submitted, updated (see _wait_for_hardware_job).
    job_update = Signal(object)

    def __init__(self, manager, config: SessionConfig):
        super().__init__()
        self.manager = manager
        self.config = config
        self._abort = False
        self._job_count = 0
        self._job_purpose = ""
        # What happened to the job that was waiting when Stop was pressed.
        self._stop_detail = None

    def abort(self):
        self._abort = True

    def _make_cost_function(self, local: bool):
        """The cost runs on an ideal Starmon-7 simulation for the emulator and
        on the device itself for real hardware. Returns (cost function, max
        circuits per job — None when unlimited)."""
        config = self.config
        layout = ([config.qubit_mapping[role] for role in _LOGICAL_ROLE_ORDER]
                  if config.is_qubit_mapping_complete() else None)
        if local:
            kwargs = {"initial_layout": layout} if layout else {}
            backend = NoisySimulator(starmon7.NAME, ideal_simulation=True)
            return CostFunction(varsigma=1.57, backend=backend, **kwargs), None
        if layout is None:
            raise ValueError("assign all four qubits (A1, A2, B1, B2) on the Setup screen first")
        backend = self.manager.get_backend(config.backend_name)
        max_circuits = backend.get_backend_type().max_jobs_per_batch_job
        cost_fn = CostFunction(varsigma=1.57, backend=backend, initial_layout=layout,
                               wait_for_result=self._wait_for_hardware_job)
        return cost_fn, max_circuits

    def _make_ladder(self, local: bool) -> LadderTFD:
        """3+3 / 4+4: the n+n experiment of tfd_study, on the device itself."""
        config, n = self.config, self.config.n_spins
        if local:
            raise ValueError(f"{n}+{n} qubits needs Tuna-17; the Starmon-7 emulator has 7 qubits")
        if not config.is_qubit_mapping_complete():
            raise ValueError(f"assign all {2 * n} qubits on the Setup screen first")
        layout = ([config.qubit_mapping[f"A{i}"] for i in range(1, n + 1)]
                  + [config.qubit_mapping[f"B{i}"] for i in range(1, n + 1)])
        backend = self.manager.get_backend(config.backend_name)

        def wait(job, purpose):
            self._job_purpose = purpose
            return self._wait_for_hardware_job(job)

        return LadderTFD(n, backend, layout, wait,
                         backend.get_backend_type().max_jobs_per_batch_job)

    def _wait_for_hardware_job(self, job):
        """Poll a submitted hardware job until its results are in, reporting
        each state change (job_update) and honouring Stop. Replaces
        QIJob.result()'s default wait, which gives up after 60 s."""
        self._job_count += 1
        info = {"n": self._job_count, "purpose": self._job_purpose,
                "job_id": str(getattr(job, "batch_job_id", None) or job.job_id()),
                "status": "Submitted", "submitted": _now(), "updated": _now()}
        self.job_update.emit(dict(info))
        self.log.emit(f"Job {info['job_id']} submitted ({info['purpose']}).")

        def set_status(text):
            if text != info["status"]:
                info.update(status=text, updated=_now())
                self.job_update.emit(dict(info))
                self.log.emit(f"Job {info['job_id']}: {text}.")

        connection_error = False
        while True:
            if self._abort:
                cancelled, detail = self.manager.cancel_job(job)
                info["detail"] = detail
                set_status("Cancelled" if cancelled else "Not cancelled")
                self._stop_detail = (
                    f"Job {info['job_id']} was {detail}." if cancelled else
                    f"Job {info['job_id']} could not be cancelled: {detail}. "
                    f"It will still run on {self.config.backend_name}.")
                self.log.emit(self._stop_detail)
                raise _Aborted()
            try:
                status = job.status()
                connection_error = False
            except Exception as exc:
                # A dropped connection mustn't lose the job: keep polling.
                if not connection_error:
                    self.log.emit(f"Couldn't check job {info['job_id']} ({exc}); retrying.")
                connection_error = True
                status = None
            if status is not None:
                if status == JobStatus.DONE:
                    break
                set_status(_JOB_STATUS_TEXT.get(status, status.name.title()))
                if status in (JobStatus.ERROR, JobStatus.CANCELLED):
                    raise RuntimeError(f"job {info['job_id']} ended as {status.name}")
            self._sleep_unless_aborted(_POLL_SECONDS)

        result = job.result(wait_for_results=False)
        set_status("Results received" if result.success else "Failed")
        return result

    def _sleep_unless_aborted(self, seconds):
        end = time.monotonic() + seconds
        while not self._abort and time.monotonic() < end:
            time.sleep(0.2)

    def run(self):
        config = self.config
        local = starmon7.is_starmon7(config.backend_name)
        optimizer = config.optimizer if local else HARDWARE_OPTIMIZER
        if optimizer != config.optimizer:
            # Setup already enforces this; a restored session could still
            # carry another optimizer.
            self.log.emit(f"Real hardware runs use {HARDWARE_OPTIMIZER} "
                          f"(suggested for noisy setups), not {config.optimizer}.")
        x0 = np.array([config.gamma1, config.gamma2, config.alpha1, config.alpha2])
        if np.allclose(x0, 0.0):
            # All-zero angles are a saddle of the cost at beta > 0 (see
            # DEFAULT_START_ANGLE), so step off it rather than get stuck.
            x0 = np.full(4, DEFAULT_START_ANGLE)
            self.log.emit(
                f"All starting angles are 0, a saddle point of the cost for beta > 0 "
                f"-- starting from {DEFAULT_START_ANGLE} instead.")
        iteration = [0]
        if local:
            # Chain experiment: warm-start up a ladder of betas, each step
            # starting from the previous step's optimum.
            betas = [b for b in (1e-8, 0.5, 1, 5) if b < config.beta] + [config.beta]
        else:
            # Cold start: one optimization straight at the target beta from
            # random angles. Far fewer hardware jobs than the chain, for
            # practically the same result.
            betas = [config.beta]
            x0 = random_start_angles(np.random.default_rng())
            self.log.emit(
                "Cold start from random angles  γ₁={:.3f}  γ₂={:.3f}  α₁={:.3f}  α₂={:.3f}"
                .format(*x0))
        current_beta = [betas[0]]
        start_params = x0.copy()

        # Append-only log of every iteration, so a crash or Stop loses nothing.
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        live_dir = PROJECTS_PATH / "_live_runs"
        live_dir.mkdir(parents=True, exist_ok=True)
        self.live_log_path = live_dir / f"{config.backend_name}_{run_tag}.jsonl"
        self.log.emit(f"Live progress log: {self.live_log_path}")

        n = config.n_spins
        try:
            target = DensityMatrix(tfd_model.gibbs_state(n, config.beta))
        except Exception as exc:
            self.log.emit(f"Couldn't compute the true Gibbs state: {exc}")
            target = None

        # 2+2: Sagastizabal et al.'s circuits (CostFunction). 3+3 / 4+4: the
        # n+n generalisation (LadderTFD), whose fidelity is measured once, by
        # tomography at the end: 3^n settings each iteration would be 6 or 17
        # extra jobs per step.
        ladder = cost_fn = max_circuits = None
        try:
            if n == 2:
                cost_fn, max_circuits = self._make_cost_function(local)
            else:
                ladder = self._make_ladder(local)
        except Exception as exc:
            self.log.emit(f"Couldn't set up {config.backend_name}: {exc}")
            self.finished.emit({"aborted": True, "error": str(exc)})
            return
        if not local:
            self.log.emit(
                f"Every cost evaluation runs on {config.backend_name}: one job per "
                f"iteration, each waiting in the device's queue. Follow them in the "
                f"Hardware jobs tab.")
        if ladder is not None:
            self.log.emit(f"{n}+{n} qubits on {ladder.layout}: the fidelity is measured "
                          f"by tomography at the end ({3 ** n} circuits).")

        # Redrawing the 3D density preview is slow and happens on the GUI thread,
        # so emulator runs draw at most ~20 previews.
        rho_every = max(1, config.iterations // 20) if local else 1

        # Every evaluation of the current beta step as (params, cost, rho_A), so
        # the live plots show the optimizer's accepted point xk, not its rejected
        # trial points.
        evaluations = []

        def evaluate(params, beta):
            cost = cost_fn.cost_function(params, beta)
            evaluations.append((np.array(params, dtype=float), cost, cost_fn.tomography_history[-1]))
            return cost

        def evaluation_at(xk):
            dists = [np.linalg.norm(params - xk) for params, _, _ in evaluations]
            i = int(np.argmin(dists))
            # Fall back to the latest evaluation if xk somehow wasn't evaluated.
            return evaluations[i] if dists[i] < 1e-9 else evaluations[-1]

        def report(xk, current_cost, rho_a):
            """Live plots + on-disk log for the optimizer's current point."""
            iteration[0] += 1
            if local:
                bound = CostFunction.cost_lower_bound(current_beta[0])
                self.progress.emit(iteration[0], float((current_cost - bound) / abs(bound)))
            else:
                # Real hardware: plot the cost exactly as the device measured it.
                self.progress.emit(iteration[0], float(current_cost))
            fid = None
            try:
                if rho_a is not None:
                    if iteration[0] % rho_every == 0:
                        self.rho_progress.emit(rho_a)
                    tgt = DensityMatrix(tfd_model.gibbs_state(n, current_beta[0]))
                    fid = state_fidelity(DensityMatrix(rho_a), tgt, validate=False)
                    self.fidelity_progress.emit(iteration[0], float(fid))
            except Exception:
                pass  # a live preview glitch shouldn't kill the optimization

            try:
                with open(self.live_log_path, "a") as f:
                    f.write(json.dumps({
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "beta": current_beta[0],
                        "iteration": iteration[0],
                        "gamma1": float(xk[0]), "gamma2": float(xk[1]),
                        "alpha1": float(xk[2]), "alpha2": float(xk[3]),
                        "cost": float(current_cost),
                        "fidelity": fid,
                    }) + "\n")
            except Exception:
                pass  # a logging hiccup shouldn't kill the optimization
            if local:
                time.sleep(_STEP_DELAY_SECONDS)

        def callback(xk):
            if self._abort:
                raise _Aborted()
            _, current_cost, rho_a = evaluation_at(np.asarray(xk, dtype=float))
            report(xk, current_cost, rho_a)

        def spsa_callback(xk):
            # SPSA's iterate was already reported by spsa_batch.
            if self._abort:
                raise _Aborted()

        def spsa_batch(points, current, beta):
            """One SPSA round in as few jobs as the backend allows: the probe
            points plus — for the live plots — the current iterate with its
            tomography. On Tuna (5 circuits per job) that's one job."""
            requests = [(np.asarray(p, dtype=float), False) for p in points]
            if current is not None:
                requests.append((np.asarray(current, dtype=float), True))
            jobs, size = [[]], 0
            for request in requests:
                need = 2 if request[1] else 1
                if max_circuits and jobs[-1] and size + need > max_circuits:
                    jobs.append([])
                    size = 0
                jobs[-1].append(request)
                size += need
            purpose = "SPSA calibration" if current is None else f"Iteration {iteration[0] + 1}"
            out = []
            for k, chunk in enumerate(jobs):
                self._job_purpose = purpose + (f" ({k + 1}/{len(jobs)})" if len(jobs) > 1 else "")
                out.extend(cost_fn.evaluate_batch(
                    [params for params, _ in chunk], beta,
                    tomography_for=[i for i, (_, tomo) in enumerate(chunk) if tomo]))
            if current is not None:
                current_cost, rho_a = out[-1]
                report(current, current_cost, rho_a)
            return [cost for cost, _ in out[:len(points)]]

        def ladder_batch(points, current, beta):
            """3+3 / 4+4: SPSA's probe points only (2 circuits each, so a step
            is one job); the live cost is the mean of the two probe losses,
            since measuring the iterate too would need a second job."""
            purpose = "SPSA calibration" if current is None else f"Iteration {iteration[0] + 1}"
            losses = ladder.losses(points, beta, purpose)
            if current is not None:
                report(current, float(np.mean(losses)), None)
            return losses

        self.log.emit(
            f"Starting optimization  |  beta={config.beta}  optimizer={optimizer}")

        try:
            for b in betas:
                current_beta[0] = b
                self.step.emit(b)
                self.log.emit(f"Optimizing at beta={b}")
                evaluations.clear()
                if ladder is not None:
                    result = spsa_minimize(
                        lambda points, current, b=b: ladder_batch(points, current, b), x0,
                        maxiter=config.iterations, callback=spsa_callback)
                elif optimizer == "SPSA":
                    result = spsa_minimize(
                        lambda points, current, b=b: spsa_batch(points, current, b), x0,
                        maxiter=config.iterations, callback=spsa_callback)
                else:
                    options = {'maxiter': config.iterations}
                    if optimizer == "COBYLA":
                        # Same initial step as CostFunction.minimize_cost. The
                        # default 1.0 stalls more often near the zero-angle saddle.
                        options['rhobeg'] = 0.5
                    result = minimize(
                        evaluate, x0=x0, args=(b,), method=optimizer,
                        options=options, callback=callback,
                    )
                x0 = result.x
            final_params = result.x
            if ladder is not None:
                rho_final, final_cost = ladder.tomography(final_params, config.beta, "Final tomography")
            else:
                self._job_purpose = "Final state"
                final_cost, rho_final = cost_fn.evaluate_batch(
                    [final_params], config.beta, tomography_for=(0,))[0]
            final_cost = float(final_cost)
        except _Aborted:
            self.log.emit(f"Aborted. Progress up to this point is saved in {self.live_log_path}")
            self.finished.emit({"aborted": True, "stop_detail": self._stop_detail})
            return
        except Exception as exc:
            self.log.emit(f"Optimization failed: {exc}")
            self.log.emit(f"Progress up to this point is saved in {self.live_log_path}")
            self.finished.emit({"aborted": True, "error": str(exc)})
            return


        # Fidelity to the true Gibbs state: the real measure of success (the cost
        # is only a proxy).
        fidelity = None
        if target is not None:
            try:
                fidelity = float(state_fidelity(DensityMatrix(rho_final), target, validate=False))
            except Exception as exc:
                self.log.emit(f"Couldn't compute final fidelity: {exc}")

        self.log.emit(
            "Optimization complete. Running circuit on the Starmon-7 emulator..." if local
            else "Optimization complete. Submitting circuit to Quantum Inspire...")
        submitted = False
        submit_error = None
        # Set before the try, so a failed submission still reaches finished.emit.
        real_counts = None
        try:
            if ladder is not None:
                backend, initial_layout = ladder.backend, ladder.layout
                qc = ladder.counts_circuit(final_params)
            else:
                backend = self.manager.get_backend(config.backend_name) if local else cost_fn.backend
                initial_layout = [config.qubit_mapping[role] for role in _LOGICAL_ROLE_ORDER]
                qc = TFDCircuit(*final_params).as_qiskit_circuit()
            # TFDCircuit adds no measurements: measure every qubit into a plain "c"
            # register, which the Starmon-7 emulator expects.
            if ladder is None:   # LadderTFD's counts circuit is already measured
                measured = QuantumCircuit(qc.num_qubits, qc.num_qubits)
                measured.compose(qc, inplace=True)
                measured.measure(range(qc.num_qubits), range(qc.num_qubits))
                qc = measured
            transpiled = transpile(qc, backend, initial_layout=initial_layout)
            self._job_purpose = "Final counts"
            job, job_results = self.manager.submit_circuit(
                transpiled, backend, wait=None if local else self._wait_for_hardware_job)
            submitted = job is not None

            if submitted and job_results is not None:
                try:
                    real_counts = job_results.get_counts()
                except Exception as exc:
                    self.log.emit(f"Couldn't read back measured counts: {exc}")

            if submitted:
                try:
                    record = self.manager.save_job(job)
                    # The job record has no app settings; store them with the outcome for
                    # Past Experiments.
                    write_experiment_settings(record.project_dir, {
                        **config.to_json_dict(),
                        "results": {
                            "optimized_params": dict(zip(
                                ("gamma1", "gamma2", "alpha1", "alpha2"),
                                (float(p) for p in final_params))),
                            "final_cost": final_cost,
                            "fidelity": fidelity,
                            "protocol": "chain" if local else "cold start",
                            "optimizer_used": optimizer,
                            "start_params": dict(zip(
                                ("gamma1", "gamma2", "alpha1", "alpha2"),
                                (float(p) for p in start_params))),
                            "job_id": str(self.manager.get_job_id(job)),
                            "live_log": str(self.live_log_path),
                        },
                    })
                    self.log.emit(f"Saved to {record.project_dir}")
                except Exception as exc:
                    self.log.emit(f"Job ran but couldn't be saved locally: {exc}")
            if not submitted:
                submit_error = "the backend returned no job"
            self.log.emit("Submitted." if submitted else "Submission returned no job.")
        except Exception as exc:
            submit_error = str(exc)
            self.log.emit(f"Submission failed: {exc}")

        self.finished.emit({
            "aborted": False,
            "params": final_params,
            "cost": final_cost,
            "rho": rho_final,
            "fidelity": fidelity,
            "submitted": submitted,
            "submit_error": submit_error,
            "real_counts": real_counts,
            "stop_detail": self._stop_detail,
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
        self._last_real_counts = None
        self._last_ideal_probs = None
        self._last_params = None
        self._target_rho = None
        self._ideal_zmax = None
        self._graphs = {}
        self._tab_pages = {}
        self._hardware = False
        self._spins = 2

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # -------------------------------------------------
        # Left sidebar — recap, device status, result, controls
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
        for key in ("Backend", "System", "β", "Optimizer", "Max iterations", "γ, α (initial)", "Qubits"):
            lbl = QLabel("–")
            lbl.setWordWrap(True)
            lbl.setStyleSheet(f"color: {Colors.TEXT};")
            self._recap_labels[key] = lbl
            self.recap_form.addRow(f"{key}:", lbl)
        side.addLayout(self.recap_form)

        # Real-hardware runs only (see start_run).
        self.device_status = DeviceStatusWidget()
        self.device_status.hide()
        side.addWidget(self.device_status)

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
        for key in ("Backend", "System", "β", "Optimizer", "Max iterations", "Qubits"):
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
        self.ideal_counts_canvas = MplCanvas(height=3.4)
        self.real_counts_canvas = MplCanvas(height=3.4)
        self.fidelity_canvas.setMinimumHeight(340)
        self.loss_canvas.setMinimumHeight(340)
        self.ideal_rho_canvas.setMinimumHeight(420)
        self.rho_canvas.setMinimumHeight(420)
        self.ideal_counts_canvas.setMinimumHeight(340)
        self.real_counts_canvas.setMinimumHeight(340)

        # 3x2 results grid; clicking a card opens it as its own tab (see _card).
        grid = QGridLayout()
        grid.setSpacing(10)
        self._grid = grid
        def _card(key, title, canvas, save):
            card = panel(title, canvas)
            self._graphs[key] = (title, canvas, save, card)
            return card

        grid.addWidget(_card("fidelity", "Fidelity to target (live)", self.fidelity_canvas,
                             self._save_fidelity), 0, 0)
        grid.addWidget(_card("loss", "Cost function (live)", self.loss_canvas,
                             self._save_loss), 0, 1)
        grid.addWidget(_card("ideal", "Ideal density matrix (true Gibbs state)", self.ideal_rho_canvas,
                             self._save_ideal_density), 1, 0)
        grid.addWidget(_card("measured", "Measured density matrix (tomography)", self.rho_canvas,
                             self._save_density), 1, 1)
        grid.addWidget(_card("ideal_counts", "Ideal measurement histogram (exact statevector)",
                             self.ideal_counts_canvas, self._save_ideal_counts), 2, 0)
        grid.addWidget(_card("real_counts", "Emulated counts (Starmon-7 noise model)",
                             self.real_counts_canvas, self._save_real_counts), 2, 1)
        for i in (0, 1, 2):
            grid.setRowStretch(i, 1)
        for i in (0, 1):
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
        self.circ_canvas.setMinimumHeight(340)
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
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"""
            QScrollArea {{ border: none; background: transparent; }}
            QScrollArea > QWidget > QWidget {{ background: transparent; }}
            QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
            QScrollBar::handle:vertical {{
                background: {Colors.BORDER}; border-radius: 5px; min-height: 30px;
            }}
            QScrollBar::handle:vertical:hover {{ background: {Colors.TEXT_MUTED}; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
        """)
        scroll.setWidget(overview)
        self.tabs.addTab(scroll, "Overview")
        self.tabs.tabBar().setTabButton(0, QTabBar.RightSide, None)
        self.tabs.tabBar().setTabButton(0, QTabBar.LeftSide, None)
        self.tabs.currentChanged.connect(self._rehome)
        self.tabs.tabCloseRequested.connect(self._close_tab)

        # Inserted next to Overview for real-hardware runs (see start_run).
        self.jobs_view = HardwareJobsView()
        for key in self._graphs:
            self._make_clickable(key)
        root.addWidget(self.tabs, stretch=1)

        self.init_loss_plot()
        self.init_fidelity_plot()
        self._draw_empty_density()
        self._draw_ideal_density()
        self._render_counts_pair(None, None)

    # -----------------------------------------------------
    # Public API (driven by main.py)
    # -----------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self.thread is not None and self.thread.isRunning()

    def start_run(self, config: SessionConfig):
        self.config = config
        self._show_hardware_panels(not starmon7.is_starmon7(config.backend_name))
        self._last_params = None
        self.save_session_btn.setEnabled(False)
        self.restart_btn.setEnabled(False)
        self._update_recap(config)
        known = _lookup_by_name(config.backend_name, KNOWN_BACKEND_TOPOLOGY)
        if known is not None and config.qubit_mapping:
            self.qubit_view.set_topology(
                len(known["coords"]), {frozenset(p) for p in known["edges"]}, known["coords"])
            self.qubit_view.set_spins(config.n_spins)
            self.qubit_view.set_assignment(config.qubit_mapping)
            self.qubit_view.show()
        else:
            self.qubit_view.hide()
        self.result_lbl.setText("")
        self.result_lbl.setStyleSheet(f"color: {Colors.GOOD}; font-weight: 600;")
        self.init_loss_plot()
        self.init_fidelity_plot()
        self._draw_circuit_for(
            [config.gamma1, config.gamma2, config.alpha1, config.alpha2])

        try:
            self._target_rho = tfd_model.gibbs_state(config.n_spins, config.beta)
        except Exception as exc:
            self._on_log(f"Couldn't compute the true Gibbs state: {exc}")
            self._target_rho = None
        self._last_rho = None
        self._draw_ideal_density()
        self._draw_empty_density()

        # Ideal histogram for the starting angles (emulator only); this run's
        # measured counts fill the other side when it finishes.
        self._last_real_counts = None
        ideal_probs = None
        if not self._hardware:
            try:
                ideal_probs = TFDCircuit(
                    config.gamma1, config.gamma2, config.alpha1, config.alpha2,
                ).get_state_vector().probabilities_dict()
            except Exception as exc:
                self._on_log(f"Couldn't compute the ideal measurement histogram: {exc}")
        self._last_ideal_probs = ideal_probs
        self._render_counts_pair(ideal_probs, None)

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
        self.worker.job_update.connect(self.jobs_view.update_job)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self._on_thread_finished)

        self.stop_btn.setEnabled(True)
        self.thread.start()

    def _on_thread_finished(self):
        # Wait for the OS thread to end before dropping the last reference:
        # Python would destroy the QThread at once, aborting with "QThread:
        # Destroyed while thread is still running".
        if self.thread is not None:
            self.thread.wait()
        self.thread = None
        self.worker = None
        self.device_status.stop()

    def _show_hardware_panels(self, hardware: bool):
        """Device status in the sidebar and a Hardware jobs tab next to
        Overview, for real-hardware runs only. A hardware run shows only what
        the device measured: the statevector histogram is hidden, the
        measured counts take the full row, and the reference density matrix
        is labelled as the exact target it is."""
        self._hardware = hardware
        self._spins = self.config.n_spins
        ideal_counts_card = self._graphs["ideal_counts"][3]
        real_counts_card = self._graphs["real_counts"][3]
        self._grid.removeWidget(real_counts_card)
        if hardware:
            ideal_counts_card.hide()
            self._grid.addWidget(real_counts_card, 2, 0, 1, 2)
        else:
            ideal_counts_card.show()
            self._grid.addWidget(real_counts_card, 2, 1)
        self._set_card_title("real_counts", f"Measured counts ({self.config.backend_name})"
                             if hardware else "Emulated counts (Starmon-7 noise model)")
        self._set_card_title("measured", f"Measured density matrix (tomography on "
                             f"{self.config.backend_name})" if hardware
                             else "Measured density matrix (tomography)")
        self._set_card_title("ideal", "Target density matrix (exact Gibbs state)")
        self._set_card_title("loss", f"Measured cost on {self.config.backend_name} (live)"
                             if hardware else "Cost function (live)")
        self._set_card_title("fidelity", "Fidelity to target (tomography at the end)"
                             if self._spins > 2 else "Fidelity to target (live)")
        self.jobs_view.clear()
        jobs_index = self.tabs.indexOf(self.jobs_view)
        if hardware:
            self.device_status.start(self.manager, self.config.backend_name)
            self.device_status.show()
            if jobs_index < 0:
                jobs_index = self.tabs.insertTab(1, self.jobs_view, "Hardware jobs")
                self.tabs.tabBar().setTabButton(jobs_index, QTabBar.RightSide, None)
                self.tabs.tabBar().setTabButton(jobs_index, QTabBar.LeftSide, None)
        else:
            self.device_status.stop()
            self.device_status.hide()
            if jobs_index >= 0:
                self.tabs.removeTab(jobs_index)

    def _set_card_title(self, key: str, title: str):
        card = self._graphs[key][3]
        card.findChild(QLabel, "cardTitle").setText(title)
        self._graphs[key] = (title,) + self._graphs[key][1:]

    def stop(self):
        if self.worker is not None:
            self.worker.abort()
        self.stop_btn.setEnabled(False)

    def _on_restart(self):
        # Re-run the same config. Guarded by is_running: start_run replaces the
        # thread and worker, which would orphan a live run.
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
        holder = QWidget()
        body = QVBoxLayout(holder)
        body.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(panel(title, holder, on_save=save))
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
        page = self.tabs.widget(index)
        if index == 0 or page is self.jobs_view:
            return
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
        # What actually runs: real hardware always cold-starts from random
        # angles with HARDWARE_OPTIMIZER (see OptimizationWorker.run).
        local = starmon7.is_starmon7(config.backend_name)
        optimizer = config.optimizer if local else HARDWARE_OPTIMIZER
        self._recap_labels["Backend"].setText(config.backend_name)
        beta_text = f"{format_beta(config.beta)}   ({format_temperature(config.beta, config.kb_unit)})"
        self._recap_labels["β"].setText(beta_text)
        self._recap_labels["Optimizer"].setText(optimizer)
        self._recap_labels["Max iterations"].setText(str(config.iterations))
        self._recap_labels["γ, α (initial)"].setText(
            f"γ₁={config.gamma1:.2f}  γ₂={config.gamma2:.2f}\n"
            f"α₁={config.alpha1:.2f}  α₂={config.alpha2:.2f}" if local
            else "random (cold start)")
        mapping = config.qubit_mapping or {}
        n = config.n_spins
        self._recap_labels["System"].setText(f"{n}+{n} qubits")
        qubits = "  ".join(f"{role}→q{mapping[role]}" for role in roles(n)
                           if role in mapping) or "–"
        self._recap_labels["Qubits"].setText(qubits)

        strip_values = {
            "Backend": config.backend_name,
            "System": f"{n}+{n}",
            "β": beta_text,
            "Optimizer": optimizer,
            "Max iterations": str(config.iterations),
            "Qubits": qubits,
        }
        for key, value in strip_values.items():
            self._param_strip_labels[key].setText(
                f"<span style='color:{Colors.TEXT_DIM};'>{key}</span>&nbsp;&nbsp;"
                f"<b style='color:{Colors.TEXT};'>{value}</b>")

    def _on_log(self, message: str):
        # No on-screen log: outcomes show in the sidebar's result text and
        # the Hardware jobs tab. Keep the trail on the console for debugging.
        print(message, flush=True)

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
            if result.get("error"):
                self._show_outcome(f"Run failed: {result['error']}", Colors.BAD)
            else:
                self._show_outcome("Stopped.\n" + (result.get("stop_detail") or
                                   "No job was waiting at Quantum Inspire."), Colors.WARN)
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
            # End the live curve on the reported final fidelity.
            final_it = (self.fidelity_x[-1] + 1) if self.fidelity_x else 1
            self._on_fidelity_progress(final_it, fidelity)
        fidelity_line = (
            f"Fidelity to true Gibbs state: {fidelity:.4f}\n" if fidelity is not None
            else "")
        self.result_lbl.setText(
            f"Finished  |  final cost={result['cost']:.4f}\n"
            f"γ₁={params[0]:.3f} γ₂={params[1]:.3f} "
            f"α₁={params[2]:.3f} α₂={params[3]:.3f}\n"
            + fidelity_line
            + ("Final counts received." if result.get("submitted")
               else f"Final counts failed: {result.get('submit_error') or 'unknown error'}")
            + (f"\n{result['stop_detail']}" if result.get("stop_detail") else ""))

        real_counts = result.get("real_counts")
        self._last_real_counts = real_counts
        ideal_probs = None
        if not self._hardware:
            try:
                ideal_probs = TFDCircuit(*params).get_state_vector().probabilities_dict()
            except Exception as exc:
                self._on_log(f"Couldn't compute the ideal measurement histogram: {exc}")
        self._last_ideal_probs = ideal_probs
        self._render_counts_pair(ideal_probs, real_counts)

    def _show_outcome(self, text: str, color: str):
        self.result_lbl.setStyleSheet(f"color: {color}; font-weight: 600;")
        self.result_lbl.setText(text)

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
        if self._hardware and self.config is not None:
            # Raw cost as measured on the device; the exact minimum is only
            # a reference line.
            self._loss_line.set_label("Measured cost C" if self._spins == 2
                                      else "Measured cost C (mean of SPSA's ± probes)")
            style_axes(ax, xlabel="Iteration", ylabel="Cost C (measured)")
            self._loss_bound = loss_lower_bound(self._spins, self.config.beta)
            ax.axhline(self._loss_bound, color=Colors.GOOD, linestyle="--", linewidth=1.8,
                       label="Theoretical minimum")
        else:
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
        ax.axhline(
            1.0, color=Colors.GOOD, alpha=0.55, linewidth=2, linestyle="--",
            zorder=3, label="Perfect (F = 1)")
        self._fidelity_gap_fill = None
        (self._fidelity_line,) = ax.plot([], [], "-", color=Colors.ACCENT, linewidth=2.2,
                                          solid_capstyle="round", zorder=2,
                                          label=f"Measured on {self.config.backend_name}"
                                          if self._hardware and self.config is not None
                                          else "Observed")
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
        dim = 2 ** self._spins
        self._render_density(self.rho_canvas, np.eye(dim, dtype=complex) / dim,
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
            dim = 2 ** self._spins
            rho, title = np.eye(dim, dtype=complex) / dim, r"$\rho_A^{\mathrm{ideal}}$  (unavailable)"
        else:
            rho, title = self._target_rho, r"$\rho_A^{\mathrm{ideal}} = e^{-\beta H} / Z$"
        self._render_density(self.ideal_rho_canvas, rho, title=title, zmax=self._ideal_zmax)

    def _render_counts_pair(self, ideal_probs: dict | None, real_counts: dict | None):
        """Ideal (exact statevector) vs. real (hardware-measured) histogram,
        drawn on the SAME bitstring ordering and the SAME y-scale (both
        panels show probability/fraction-of-shots, 0 to a shared max) so
        they're genuinely comparable bar-for-bar -- the same spirit as the
        ideal/measured density-matrix pair sharing one zmax above. There's
        no phase information here (that's what the density-matrix pair is
        for); this is purely the measured/predicted population over each of
        the 2^4 computational-basis outcomes.

        Ordering is picked from whichever side has data (real takes
        priority once it exists, since that's the thing you actually care
        about reading off), capped at the top 24 bitstrings so a device
        with many unused register slots doesn't produce a wall of
        near-zero bars.
        """
        if real_counts:
            order = [k for k, _ in sorted(real_counts.items(), key=lambda kv: kv[1], reverse=True)][:24]
        elif ideal_probs:
            order = [k for k, _ in sorted(ideal_probs.items(), key=lambda kv: kv[1], reverse=True)][:24]
        else:
            order = []

        real_total = sum(real_counts.values()) if real_counts else 0
        ideal_values = [ideal_probs.get(b, 0.0) for b in order] if ideal_probs else []
        real_values = ([real_counts.get(b, 0) / real_total for b in order] if real_counts and real_total
                        else [])
        ymax = max([0.05] + ideal_values + real_values) * 1.15

        def _draw(canvas, values, title, empty_message):
            fig = canvas.fig
            fig.clear()
            ax = fig.add_subplot(111)
            ax.set_facecolor(Colors.BG_3)
            if not order or not values:
                ax.text(0.5, 0.5, empty_message, transform=ax.transAxes,
                        ha="center", va="center", color=Colors.TEXT_MUTED, fontsize=10)
                ax.set_xticks([]); ax.set_yticks([])
            else:
                ax.bar(range(len(order)), values, color=Colors.ACCENT, width=0.7)
                ax.set_xticks(range(len(order)))
                ax.set_xticklabels(order, rotation=60, ha="right", fontsize=8)
                ax.set_ylim(0, ymax)
                style_axes(ax, xlabel="Measured bitstring", ylabel="Probability")
            ax.set_title(title, color=Colors.TEXT_DIM, fontsize=9, pad=8)
            for spine in ax.spines.values():
                spine.set_color(Colors.BORDER)
            fig.tight_layout()
            canvas.draw_idle()

        _draw(self.ideal_counts_canvas, ideal_values,
              r"$|\langle b|\psi_{\mathrm{ideal}}\rangle|^2$",
              "No angles to compute yet.")
        # Starmon-7 is a local noisy emulation, not hardware: say so.
        if self._hardware:
            source, empty = (f"Measured on {self.config.backend_name}",
                             f"No counts from {self.config.backend_name} yet.\n"
                             f"They appear once the final-counts job returns.")
        else:
            source, empty = ("Emulated (Starmon-7 noise model)",
                             "No emulated counts yet.\nThey appear once the run finishes.")
        _draw(self.real_counts_canvas, real_values,
              f"{source}: {real_total} shots" if real_total else source, empty)

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
        bits = max(1, (n - 1).bit_length())
        # Every basis state up to 8x8; every 4th at 16x16, where all of them overlap.
        step = 4 if n > 8 else 1
        labels = [format(i, f"0{bits}b") if i % step == 0 else "" for i in range(n)]
        ax.set_xticks(np.arange(n) + 0.35); ax.set_xticklabels(labels)
        ax.set_yticks(np.arange(n) + 0.35); ax.set_yticklabels(labels)
        if n > 4:
            ax.tick_params(axis="x", labelsize=6 if n > 8 else 7)
            ax.tick_params(axis="y", labelsize=6 if n > 8 else 7)
        ax.set_zlim(0, zmax)
        # Few, short z ticks so they don't run into each other or the label.
        ax.zaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.zaxis.set_major_formatter("{x:.1f}" if zmax > 0.5 else "{x:.2f}")
        ax.set_zlabel(r"$|\rho_{ij}|$", labelpad=2, fontsize=10)
        ax.text2D(0.02, 0.98, title, transform=ax.transAxes, fontsize=12,
                   color=Colors.TEXT, va="top")
        # No ax.set_title(): the card already titles it, and a 3D title
        # overflows narrow panels.

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
        n = self._spins
        if n == 2:
            qc = TFDCircuit(*params).as_qiskit_circuit()
            role_order = _LOGICAL_ROLE_ORDER
        else:
            # tfd_study/model.py's n+n ansatz; its qubits are A1..An, B1..Bn.
            qc = tfd_model.bound_ansatz(n, 1, params)
            role_order = roles(n)
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
        # Wires labelled by role instead of q0, q1, ...: one single-qubit
        # register per role (Qiskit labels a size-1 register by its name
        # alone), in the circuit's logical order, drawn top-to-bottom as
        # A1..An B1..Bn.
        # Only the drawing is relabelled; _last_qc (QASM export) keeps "q".
        labelled = QuantumCircuit(*(QuantumRegister(1, role) for role in role_order))
        labelled.compose(qc, inplace=True)
        wire_order = [role_order.index(role) for role in roles(n)]
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

    def _save_ideal_counts(self):
        if not self._last_ideal_probs:
            QMessageBox.information(
                self, "Nothing to save", "No angles computed yet.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save ideal histogram", "ideal_probabilities.csv",
            "CSV data (*.csv);;PNG image (*.png)")
        if not path:
            return
        if path.lower().endswith(".png"):
            self.ideal_counts_canvas.fig.savefig(
                path, facecolor=self.ideal_counts_canvas.fig.get_facecolor())
        else:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["bitstring", "probability"])
                w.writerows(sorted(self._last_ideal_probs.items(),
                                    key=lambda kv: kv[1], reverse=True))

    def _save_real_counts(self):
        if not self._last_real_counts:
            QMessageBox.information(
                self, "Nothing to save",
                "No measured counts yet — they appear once a run finishes.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save measured counts", "measured_counts.csv",
            "CSV data (*.csv);;PNG image (*.png)")
        if not path:
            return
        if path.lower().endswith(".png"):
            self.real_counts_canvas.fig.savefig(
                path, facecolor=self.real_counts_canvas.fig.get_facecolor())
        else:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["bitstring", "shots"])
                w.writerows(sorted(self._last_real_counts.items(),
                                    key=lambda kv: kv[1], reverse=True))

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