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
import traceback
from dataclasses import replace
from datetime import datetime
from pathlib import Path

# cost_function.py imports its siblings bare ("from tfd_circuit import ...").
sys.path.append(str(Path(__file__).resolve().parent.parent / "gibbs_tfd_reproduction"))

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.ticker import MaxNLocator
from PySide6.QtCore import QObject, QRect, Qt, QThread, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPixmap
from PySide6.QtWidgets import (
    QFileDialog, QFormLayout, QGridLayout, QHBoxLayout, QLabel, QMessageBox, QPushButton, QScrollArea,
    QTabBar, QTabWidget, QVBoxLayout, QWidget,
)
from qi_utilities.device_simulation.simulators import NoisySimulator
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.providers.jobstatus import JobStatus
from qiskit.quantum_info import DensityMatrix, state_fidelity
from scipy.optimize import minimize

from backend import backends, ladder_tfd, starmon7
from backend.experiment_manager import PROJECTS_PATH, write_experiment_settings, write_results
from backend.ladder_tfd import LadderTFD, loss_lower_bound
from backend.spsa import spsa_minimize
from frontend.colors import Colors
from frontend.device_panel import KNOWN_BACKEND_TOPOLOGY, QubitTopologyWidget, _lookup_by_name, qx_ladder
from frontend.hardware_panels import DeviceStatusWidget, HardwareJobsView
from frontend.help_button import HelpButton
from frontend.latex import font_px, inline_tex, set_tex, tex_label
from frontend.plots import MplCanvas, panel, style_axes
from frontend.progress_bar import RunProgressBar
from frontend.result_card import FinalFidelityView
from frontend.session import (
    DEFAULT_START_ANGLE, HARDWARE_OPTIMIZER, SessionConfig, cold_start_angles, format_beta,
    format_temperature_tex, roles,
)
from frontend.upload import RESULTS_KIND, save_results_dialog
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

# A job still unfinished after JOB_TIMEOUT is cancelled and the run ends (the
# device may be offline or its queue stuck); so does a run that can't reach
# Quantum Inspire for CONNECTION_TIMEOUT.
JOB_TIMEOUT = 6 * 3600
CONNECTION_TIMEOUT = 10 * 60
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


def _status_icon(color: str, symbol: str, size: int = 48) -> QPixmap:
    """A filled circle with a symbol, for the result popup."""
    ratio = 2
    pixmap = QPixmap(size * ratio, size * ratio)
    pixmap.setDevicePixelRatio(ratio)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(color))
    painter.drawEllipse(2, 2, size - 4, size - 4)
    font = QFont()
    font.setPixelSize(int(size * 0.55))
    font.setBold(True)
    painter.setFont(font)
    painter.setPen(QColor(Colors.BG_1))
    painter.drawText(QRect(0, 0, size, size), Qt.AlignCenter, symbol)
    painter.end()
    return pixmap


def _outcome_color(outcome: dict) -> str:
    return {"done": Colors.GOOD, "warn": Colors.WARN, "error": Colors.BAD}[outcome["kind"]]


def _outcome_html(outcome: dict) -> str:
    return (f"<p style='font-size:12pt; font-weight:700; color:{_outcome_color(outcome)}; "
            f"margin-bottom:6px;'>{outcome['title']}</p>{outcome['html']}")


def _result_rows(rows) -> str:
    """(name, value) rows as a two-column table for the result box."""
    return ("<table cellspacing='0' cellpadding='2'>" + "".join(
        f"<tr><td style='color:{Colors.TEXT_DIM}; padding-right:10px;'>{name}</td>"
        f"<td style='color:{Colors.TEXT};'>{value}</td></tr>" for name, value in rows)
        + "</table>")


def uses_ladder(config: SessionConfig) -> bool:
    """Whether a run uses the n+n experiment of backend/ladder_tfd.py: for
    3+3 and 4+4, and for every size on the QX emulator. 2+2 elsewhere uses
    the paper's own circuits (CostFunction), which measure the qubits several
    times mid-circuit; QX reports results by qubit, so those repeated
    measurements can't be told apart there. The ladder circuits measure each
    qubit once, at the end; their fidelity comes from a tomography at the
    end, since 3^n settings every iteration would cost too many jobs."""
    return config.n_spins > 2 or backends.is_qx(config.backend_name)


def chain_betas(beta: float) -> list:
    """The chain experiment's β steps up to the target (1e-8 stands in for β = 0)."""
    return [b for b in (1e-8, 0.5, 1, 5) if b < beta] + [beta]


def _plot_help(config: SessionConfig, ladder: bool) -> tuple:
    """The "?" explanations of the Fidelity (or Final fidelity) and Cost
    cards for this run."""
    n, device = config.n_spins, config.backend_name
    if ladder:
        fidelity = (
            "<b>Measured once, at the end</b><br>"
            "The fidelity compares the prepared state with the exact Gibbs state, so it "
            f"needs the whole state from tomography: $3^{n} = {3 ** n}$ measurement settings. SPSA "
            "only needs the cost, so the tomography runs once, after the last iteration, "
            "at the final angles. Until then, follow the cost plot.<br><br>"
            r"Purity $\mathrm{Tr}\,\rho^2$ shows how mixed the state is: a thermal state is mixed, so a "
            "purity well above the target's means the state isn't thermal enough.")
        if backends.is_qx(device) and n == 2:
            fidelity += (
                "<br><br><b>Why not live, as on Starmon-7?</b><br>The paper's 2+2 circuit "
                "measures qubits mid-circuit, and QX reports results by qubit number, which "
                "mixes those measurements up. So every QX run uses circuits measured only at "
                "the end.")
        cost = (
            "<b>What each point is</b><br>"
            r"The average cost at SPSA's two probe points, $\theta \pm c\Delta$, not at $\theta$ "
            r"itself: measuring $\theta$ too would need a second job per iteration. It's noisy "
            r"because the probes sit slightly off $\theta$ and every estimate comes from a finite "
            r"number of shots.<br><br>The dashed line is the lowest cost possible at this $\beta$: "
            "expect the "
            "curve to approach it and level off a little above it. SPSA always runs the "
            "maximum number of iterations; the final angles are the average of the last "
            "quarter of them.")
    else:
        fidelity = (
            "<b>Measured every iteration</b><br>"
            "Each iteration also measures the current state by tomography (9 measurement "
            "settings) and compares it with the exact Gibbs state. $F = 1$ is a perfect match.")
        if backends.is_local(device):
            cost = (
                "<b>What each point is</b><br>"
                r"How far the cost is above its lowest possible value at the current $\beta$, as a "
                r"fraction: 0 means the optimizer reached the minimum. Each $\beta$ stage of the chain "
                "experiment has its own minimum, so the curve jumps when a new stage begins.")
        else:
            cost = (
                "<b>What each point is</b><br>"
                f"The cost measured on {device} at the current angles. Gate and readout noise "
                "and the finite number of shots keep it above the dashed theoretical minimum.")
    return fidelity, cost


def _protocol_text(config: SessionConfig, optimizer: str) -> str:
    """Why this run uses a chain experiment (Starmon-7) or a cold start (QX
    emulator, real hardware), for the box above the plots."""
    beta = format_beta(config.beta)
    title = f"<b style='color:{Colors.TEXT};'>"
    if backends.is_local(config.backend_name):
        betas = chain_betas(config.beta)
        if len(betas) == 1:
            return (f"{title}Single step at $\\beta \\approx 0$.</b> The optimizer starts from "
                    f"the known $\\beta = 0$ solution, so no chain of $\\beta$ steps is needed.")
        steps = r" \to ".join(r"\approx 0" if b < 1e-6 else format_beta(b) for b in betas)
        return (f"{title}Chain experiment.</b> The optimizer begins at the "
                f"known $\\beta \\approx 0$ solution and steps $\\beta {steps}$, each step starting "
                f"from the previous optimum. On this local simulator every cost evaluation "
                f"is nearly free, so the extra steps cost nothing, and following the solution "
                f"as $\\beta$ grows keeps {optimizer} away from poor local minima.")
    noise = ("shot noise" if backends.is_qx(config.backend_name)
             else "gate and shot noise")
    return (f"{title}Cold start.</b> One optimization straight at $\\beta = {beta}$, from small "
            f"random angles near $\\theta = 0$ (the $\\beta = 0$ solution, so the optimizer starts in the "
            f"right valley). On {config.backend_name} every cost evaluation is a job waiting in "
            f"Quantum Inspire's queue, and a chain would repeat the whole optimization at "
            f"every $\\beta$ step; our tests showed the cold start reaches practically the same "
            f"result. {optimizer} is used because it copes best with {noise}.")


class _Aborted(Exception):
    """Raised from the optimizer's callback or a job wait to unwind the run
    when the user hits Stop."""


class _JobTimedOut(Exception):
    """A job didn't finish within JOB_TIMEOUT."""


class _ConnectionLost(Exception):
    """Quantum Inspire couldn't be reached for CONNECTION_TIMEOUT."""


class _JobFailed(Exception):
    """The device reported a job as failed."""


def _duration(seconds: float) -> str:
    return f"{seconds / 3600:g} hours" if seconds >= 3600 else f"{seconds / 60:g} minutes"


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
        backend_type = backend.get_backend_type()
        # The QX emulator allows at most 2048 shots per circuit.
        cost_fn = CostFunction(varsigma=1.57, backend=backend, initial_layout=layout,
                               shots=min(8192, backend_type.max_number_of_shots),
                               wait_for_result=self._wait_for_hardware_job)
        return cost_fn, backend_type.max_jobs_per_batch_job

    def _make_ladder(self, local: bool) -> LadderTFD:
        """3+3 / 4+4: the n+n experiment of tfd_study, on the device itself."""
        config, n = self.config, self.config.n_spins
        if local:
            raise ValueError(f"{n}+{n} qubits needs Tuna-17 or the QX emulator; Starmon-7 has 7 qubits")
        if not config.is_qubit_mapping_complete():
            raise ValueError(f"assign all {2 * n} qubits on the Setup screen first")
        layout = ([config.qubit_mapping[f"A{i}"] for i in range(1, n + 1)]
                  + [config.qubit_mapping[f"B{i}"] for i in range(1, n + 1)])
        backend = self.manager.get_backend(config.backend_name)

        def wait(job, purpose):
            self._job_purpose = purpose
            return self._wait_for_hardware_job(job)

        backend_type = backend.get_backend_type()
        return LadderTFD(n, backend, layout, wait, backend_type.max_jobs_per_batch_job,
                         shots=min(ladder_tfd.SHOTS, backend_type.max_number_of_shots))

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

        submitted_at = time.monotonic()
        unreachable_since = None
        while True:
            if time.monotonic() - submitted_at > JOB_TIMEOUT:
                cancelled, detail = self.manager.cancel_job(job)
                info["detail"] = detail
                set_status("Timed out")
                raise _JobTimedOut(
                    f"Job {info['job_id']} ({info['purpose']}) didn't finish within "
                    f"{_duration(JOB_TIMEOUT)}. "
                    + (f"It was {detail}." if cancelled else
                       f"It could not be cancelled ({detail}), so it may still run."))
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
                unreachable_since = None
            except Exception as exc:
                # A dropped connection mustn't lose the job: keep polling, up
                # to CONNECTION_TIMEOUT.
                if unreachable_since is None:
                    unreachable_since = time.monotonic()
                    self.log.emit(f"Couldn't check job {info['job_id']} ({exc}); retrying.")
                elif time.monotonic() - unreachable_since > CONNECTION_TIMEOUT:
                    set_status("Connection lost")
                    raise _ConnectionLost(
                        f"Quantum Inspire couldn't be reached for {_duration(CONNECTION_TIMEOUT)} "
                        f"while waiting for job {info['job_id']} ({exc}).")
                status = None
            if status is not None:
                if status == JobStatus.DONE:
                    break
                set_status(_JOB_STATUS_TEXT.get(status, status.name.title()))
                if status in (JobStatus.ERROR, JobStatus.CANCELLED):
                    raise _JobFailed(f"Job {info['job_id']} ({info['purpose']}) ended as "
                                     f"{status.name.lower()} on {self.config.backend_name}.")
            self._sleep_unless_aborted(_POLL_SECONDS)

        result = job.result(wait_for_results=False)
        set_status("Results received" if result.success else "Failed")
        return result

    def _write_error_log(self, exc: Exception) -> str | None:
        """The technical details of a failed run (traceback, Quantum Inspire's
        messages), for reporting; the popup itself stays in plain language."""
        path = self.live_log_path.with_name(self.live_log_path.stem + "_error.txt")
        try:
            path.write_text(f"{datetime.now().isoformat(timespec='seconds')}  "
                            f"{self.config.backend_name}\n{exc}\n\n{traceback.format_exc()}",
                            encoding="utf-8")
            return str(path)
        except OSError:
            return None

    @staticmethod
    def _failure_reason(exc: Exception) -> str:
        if isinstance(exc, _Aborted):
            return "stopped"
        if isinstance(exc, _JobTimedOut):
            return "timeout"
        if isinstance(exc, _ConnectionLost):
            return "connection"
        if isinstance(exc, _JobFailed) or str(exc).startswith("Backend job failed"):
            return "job_failed"
        return "error"

    def _sleep_unless_aborted(self, seconds):
        end = time.monotonic() + seconds
        while not self._abort and time.monotonic() < end:
            time.sleep(0.2)

    def run(self):
        config = self.config
        if backends.is_qx(config.backend_name):
            # See backends.qx_mapping; also covers sessions saved with another pick.
            mapping = backends.qx_mapping(config.n_spins)
            if config.qubit_mapping != mapping:
                self.log.emit(f"QX emulator: using qubits {sorted(mapping.values())}.")
            config = self.config = replace(config, qubit_mapping=mapping)
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
            betas = chain_betas(config.beta)
        else:
            # Cold start: one optimization straight at the target beta, from
            # small random angles near θ = 0 (see cold_start_angles). Far fewer
            # cloud jobs than the chain.
            betas = [config.beta]
            x0 = cold_start_angles(np.random.default_rng())
            self.log.emit(
                "Cold start near θ = 0  γ₁={:.3f}  γ₂={:.3f}  α₁={:.3f}  α₂={:.3f}"
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

        # The paper's circuits (CostFunction) or the n+n experiment (LadderTFD):
        # see uses_ladder.
        ladder = cost_fn = max_circuits = None
        try:
            if uses_ladder(config):
                ladder = self._make_ladder(local)
            else:
                cost_fn, max_circuits = self._make_cost_function(local)
        except Exception as exc:
            self.log.emit(f"Couldn't set up {config.backend_name}: {exc}")
            self.finished.emit({
                "aborted": True, "reason": "setup",
                # Our own checks explain themselves; anything else is a connection problem.
                "message": str(exc) if isinstance(exc, ValueError) else "",
                "error_log": self._write_error_log(exc), "iterations": 0})
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
        except Exception as exc:
            reason = self._failure_reason(exc)
            self.log.emit(f"Run ended ({reason}): {exc}")
            self.log.emit(f"Progress up to this point is saved in {self.live_log_path}")
            self.finished.emit({
                "aborted": True, "reason": reason,
                "message": str(exc) if reason == "timeout" else "",
                "error_log": None if reason == "stopped" else self._write_error_log(exc),
                "stop_detail": self._stop_detail, "iterations": iteration[0],
                "live_log": str(self.live_log_path),
            })
            return


        # Fidelity to the true Gibbs state: the real measure of success (the cost
        # is only a proxy).
        fidelity = None
        warnings = []
        if target is not None:
            try:
                fidelity = float(state_fidelity(DensityMatrix(rho_final), target, validate=False))
            except Exception as exc:
                warnings.append(f"The final fidelity couldn't be computed: {exc}")
        else:
            warnings.append("The final fidelity couldn't be computed: no target state.")

        self.log.emit(
            "Optimization complete. Running circuit on the Starmon-7 emulator..." if local
            else "Optimization complete. Submitting circuit to Quantum Inspire...")
        submitted = False
        submit_error = None
        project_dir = None
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
            if not job_results.success:
                raise _JobFailed(f"the final counts job failed on {config.backend_name}: "
                                 f"{getattr(job_results, 'system_messages', '') or job_results.status}")
            submitted = True
            real_counts = job_results.get_counts()

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
                    project_dir = str(record.project_dir)
                    self.log.emit(f"Saved to {record.project_dir}")
                except Exception as exc:
                    warnings.append(f"The run couldn't be saved for Past Experiments: {exc}")
        except _Aborted:
            submit_error = "you stopped the run before they came back"
        except Exception as exc:
            submit_error = str(exc)
        if submit_error:
            self.log.emit(f"Final counts failed: {submit_error}")

        self.finished.emit({
            "aborted": False,
            "iterations": iteration[0],
            "warnings": warnings,
            "params": final_params,
            "cost": final_cost,
            "rho": rho_final,
            "fidelity": fidelity,
            "submitted": submitted,
            "submit_error": submit_error,
            "real_counts": real_counts,
            "stop_detail": self._stop_detail,
            "project_dir": project_dir,
        })


class ExperimentWindow(QWidget):

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
        self._hover = {}
        self._result_box = None
        # Every plot update of the run in order (see _record), so a saved run
        # can be redrawn exactly; and the outcome shown in the sidebar.
        self._events = []
        self._replaying = False
        self._outcome = None
        self._hardware = False
        self._spins = 2
        self._ladder = False

        # Sidebar and results side by side, the progress strip below both.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        content = QWidget()
        root = QHBoxLayout(content)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        outer.addWidget(content, stretch=1)
        self.progress_bar = RunProgressBar()
        outer.addWidget(self.progress_bar)

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
        for key, label in (("Backend", "Backend"), ("System", "System"),
                           ("Temperature", "Temperature"), ("Optimizer", "Optimizer"),
                           ("Max iterations", "Max iterations"),
                           ("Angles", r"$\gamma, \alpha$ (initial)"), ("Qubits", "Qubits")):
            lbl = QLabel("–")
            lbl.setWordWrap(True)
            lbl.setStyleSheet(f"color: {Colors.TEXT};")
            self._recap_labels[key] = lbl
            self.recap_form.addRow(tex_label(f"{label}:"), lbl)
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
        # The run's outcome (see _show_outcome), in its own box.
        self.result_lbl = QLabel("")
        self.result_lbl.setObjectName("resultBox")
        self.result_lbl.setWordWrap(True)
        self.result_lbl.setTextFormat(Qt.RichText)
        self.result_lbl.hide()
        side.addWidget(self.result_lbl)

        side.addStretch(1)

        def small_button(text, color, hover):
            button = QPushButton(text)
            button.setCursor(Qt.PointingHandCursor)
            button.setEnabled(False)
            button.setFixedHeight(32)
            button.setStyleSheet(f"""
                QPushButton {{
                    background: transparent; color: {color}; border: 1px solid {color};
                    border-radius: 8px; padding: 0 8px; font-weight: 600;
                }}
                QPushButton:hover {{ background: {hover}; }}
                QPushButton:disabled {{ color: {Colors.TEXT_MUTED}; border-color: {Colors.BORDER}; }}
            """)
            return button

        self.save_session_btn = small_button("💾 Save", Colors.TEXT_DIM, "rgba(255,255,255,0.06)")
        self.save_session_btn.setToolTip("Save the whole run (every plot, the density matrices, "
                                         "counts and outcome) to reopen it here later")
        self.save_session_btn.clicked.connect(self._on_save_results)
        self.stop_btn = small_button("■ Stop", Colors.WARN, "rgba(254,138,39,0.12)")
        self.stop_btn.clicked.connect(self.stop)
        self.restart_btn = small_button("↻ Run again", Colors.ACCENT, "rgba(135,206,235,0.12)")
        self.restart_btn.clicked.connect(self._on_restart)
        buttons = QHBoxLayout()
        buttons.setSpacing(6)
        for button in (self.save_session_btn, self.stop_btn, self.restart_btn):
            buttons.addWidget(button)
        side.addLayout(buttons)

        root.addWidget(sidebar)

        # -------------------------------------------------
        # Right main area
        # -------------------------------------------------

        overview = QWidget()
        main_layout = QVBoxLayout(overview)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(10)

        # Why this run uses a chain experiment or a cold start (the run's
        # settings are in the sidebar recap).
        self.protocol_lbl = QLabel()
        self.protocol_lbl.setObjectName("protocolBox")
        self.protocol_lbl.setTextFormat(Qt.RichText)
        self.protocol_lbl.setWordWrap(True)
        self.protocol_lbl.setStyleSheet(f"""
            #protocolBox {{
                background: {Colors.BG_2}; border: 1px solid {Colors.BORDER};
                border-radius: 10px; padding: 10px 16px; color: {Colors.TEXT_DIM};
            }}
        """)
        main_layout.addWidget(self.protocol_lbl)

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
        self._plot_help = {}

        def _card(key, title, canvas, save):
            card = panel(title, canvas)
            self._graphs[key] = (title, canvas, save, card)
            if key in ("fidelity", "loss"):
                # "?" right after the title; its text is set per run.
                self._plot_help[key] = HelpButton(width=340)
                card.layout().itemAt(0).layout().insertWidget(1, self._plot_help[key])
            return card

        grid.addWidget(_card("fidelity", "Fidelity to target (live)", self.fidelity_canvas,
                             self._save_fidelity), 0, 0)
        grid.addWidget(_card("loss", "Cost function (live)", self.loss_canvas,
                             self._save_loss), 0, 1)
        # Replaces the fidelity plot on runs that measure it only at the end.
        self.final_fidelity = FinalFidelityView()
        self.final_card = panel("Final fidelity (tomography at the end)", self.final_fidelity)
        self._plot_help["final"] = HelpButton(width=340)
        self.final_card.layout().itemAt(0).layout().insertWidget(1, self._plot_help["final"])
        self.final_card.hide()
        grid.addWidget(self.final_card, 0, 0)
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
        for key, canvas in (("loss", self.loss_canvas), ("fidelity", self.fidelity_canvas)):
            canvas.mpl_connect("motion_notify_event", lambda e, k=key: self._on_hover(e, k))
            canvas.mpl_connect("figure_leave_event", lambda e, k=key: self._on_hover(e, k))
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
        if backends.is_qx(config.backend_name):
            # The qubits the run really uses (see backends.qx_mapping).
            config = replace(config, qubit_mapping=backends.qx_mapping(config.n_spins))
        self._prepare_view(config, live=True)

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
        self.worker.job_update.connect(
            lambda info: self.progress_bar.status(f"{info['purpose']}: {info['status']}"))
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self._on_thread_finished)

        self.stop_btn.setEnabled(True)
        self.thread.start()

    def _prepare_view(self, config: SessionConfig, live: bool):
        """Reset every panel for `config`: before a run starts, or before a
        saved run is redrawn (live=False: no device polling)."""
        if self._result_box is not None:
            # The previous run's popup mustn't overwrite this run's sidebar.
            self._result_box.finished.disconnect()
            self._result_box.close()
            self._result_box = None
        self.config = config
        self._events = []
        self._outcome = None
        local = starmon7.is_starmon7(config.backend_name)
        self._show_hardware_panels(not local, live)
        self.progress_bar.start(chain_betas(config.beta) if local else [config.beta],
                                config.iterations, live)
        self._last_params = None
        self.save_session_btn.setEnabled(False)
        self.restart_btn.setEnabled(False)
        self._update_recap(config)
        if backends.is_qx(config.backend_name):
            coords, edges = qx_ladder(config.n_spins)
            topology = (2 * config.n_spins, edges, coords)
        else:
            known = _lookup_by_name(config.backend_name, KNOWN_BACKEND_TOPOLOGY)
            topology = None if known is None else (
                len(known["coords"]), {frozenset(p) for p in known["edges"]}, known["coords"])
        if topology is not None and config.qubit_mapping:
            self.qubit_view.set_spins(config.n_spins)
            self.qubit_view.set_topology(*topology)
            self.qubit_view.set_assignment(config.qubit_mapping)
            self.qubit_view.show()
        else:
            self.qubit_view.hide()
        self.result_lbl.setText("")
        self.result_lbl.hide()
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
        self.final_fidelity.reset(
            3 ** config.n_spins,
            float(np.real(np.trace(self._target_rho @ self._target_rho)))
            if self._target_rho is not None else None)

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

    def _on_thread_finished(self):
        # Wait for the OS thread to end before dropping the last reference:
        # Python would destroy the QThread at once, aborting with "QThread:
        # Destroyed while thread is still running".
        if self.thread is not None:
            self.thread.wait()
        self.thread = None
        self.worker = None
        self.device_status.stop()

    def _show_hardware_panels(self, hardware: bool, live: bool = True):
        """Device status in the sidebar and a jobs tab next to Overview, for
        cloud runs (QX emulator or real hardware) only. A cloud run shows only
        what the backend returned: the statevector histogram is hidden, the
        returned counts take the full row, and the reference density matrix
        is labelled as the exact target it is."""
        self._hardware = hardware
        self._spins = self.config.n_spins
        self._ladder = uses_ladder(self.config)
        device = self.config.backend_name
        qx = backends.is_qx(device)
        returned = "Simulated" if qx else "Measured"
        ideal_counts_card = self._graphs["ideal_counts"][3]
        real_counts_card = self._graphs["real_counts"][3]
        self._grid.removeWidget(real_counts_card)
        if hardware:
            ideal_counts_card.hide()
            self._grid.addWidget(real_counts_card, 2, 0, 1, 2)
        else:
            ideal_counts_card.show()
            self._grid.addWidget(real_counts_card, 2, 1)
        self._set_card_title("real_counts", f"{returned} counts ({device}{', noiseless' if qx else ''})"
                             if hardware else "Emulated counts (Starmon-7 noise model)")
        self._set_card_title("measured", f"{returned} density matrix (tomography on {device})"
                             if hardware else "Measured density matrix (tomography)")
        self._set_card_title("ideal", "Target density matrix (exact Gibbs state)")
        self._set_card_title("loss", f"Cost on {device} (live)" if hardware else "Cost function (live)")
        # Ladder runs measure the fidelity once, at the end: a Final fidelity
        # card instead of a plot with a single point.
        self._graphs["fidelity"][3].setVisible(not self._ladder)
        self.final_card.setVisible(self._ladder)
        fidelity_help, cost_help = _plot_help(self.config, self._ladder)
        self._plot_help["final" if self._ladder else "fidelity"].set_help(fidelity_help)
        self._plot_help["loss"].set_help(cost_help)
        self.jobs_view.clear()
        jobs_index = self.tabs.indexOf(self.jobs_view)
        if hardware:
            if live:
                self.device_status.start(self.manager, self.config.backend_name)
            else:
                self.device_status.stop()
            self.device_status.setVisible(live)
            if jobs_index < 0:
                jobs_index = self.tabs.insertTab(1, self.jobs_view, "")
                self.tabs.tabBar().setTabButton(jobs_index, QTabBar.RightSide, None)
                self.tabs.tabBar().setTabButton(jobs_index, QTabBar.LeftSide, None)
            self.tabs.setTabText(jobs_index, "Cloud jobs" if qx else "Hardware jobs")
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
        set_tex(self._recap_labels["Temperature"],
                rf"$\beta = {format_beta(config.beta)}$<br>"
                f"${format_temperature_tex(config.beta, config.kb_unit)}$")
        self._recap_labels["Optimizer"].setText(optimizer)
        self._recap_labels["Max iterations"].setText(str(config.iterations))
        set_tex(self._recap_labels["Angles"],
                rf"$\gamma_1 = {config.gamma1:.2f}$  $\gamma_2 = {config.gamma2:.2f}$<br>"
                rf"$\alpha_1 = {config.alpha1:.2f}$  $\alpha_2 = {config.alpha2:.2f}$" if local
                else "random, near 0<br>(cold start)")
        mapping = config.qubit_mapping or {}
        n = config.n_spins
        self._recap_labels["System"].setText(f"{n}+{n} qubits")
        qubits = "  ".join(f"{role}→q{mapping[role]}" for role in roles(n)
                           if role in mapping) or "–"
        self._recap_labels["Qubits"].setText(qubits)

        set_tex(self.protocol_lbl, _protocol_text(config, optimizer), Colors.TEXT_DIM)

    def _on_log(self, message: str):
        # No on-screen log: outcomes show in the sidebar's result text and
        # the Hardware jobs tab. Keep the trail on the console for debugging.
        print(message, flush=True)

    def _record(self, *event):
        if not self._replaying:
            self._events.append(list(event))

    def _on_step(self, beta: float):
        self._record("step", float(beta))
        self.progress_bar.stage(beta)
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
        self.save_session_btn.setEnabled(bool(self._events) or not result.get("aborted"))
        if result.get("aborted"):
            self._show_failure(result)
            return
        params = result["params"]
        self._last_params = params
        self._draw_circuit_for(params)
        # Always draw the true converged result, regardless of where the
        # throttled live preview (see OptimizationWorker.run) last landed.
        self._draw_density(result["rho"])
        fidelity = result.get("fidelity")
        if fidelity is not None:
            # End the live curve on the reported final fidelity (a ladder run
            # has no curve: its one point goes at the last iteration).
            final_it = ((self.fidelity_x[-1] + 1) if self.fidelity_x
                        else max(1, result.get("iterations") or 1))
            self._on_fidelity_progress(final_it, fidelity)
        rho = np.asarray(result["rho"])
        purity = float(np.real(np.trace(rho @ rho)))
        target_purity = (float(np.real(np.trace(self._target_rho @ self._target_rho)))
                         if self._target_rho is not None else None)
        self._show_completion(result, params, fidelity, purity, target_purity)

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
        if result.get("project_dir"):
            # So Past Experiments can show this run again.
            try:
                write_results(result["project_dir"], self.snapshot())
            except OSError as exc:
                self._on_log(f"Couldn't save the results for Past Experiments: {exc}")

    def _show_completion(self, result, params, fidelity, purity, target_purity):
        """Popup with the headline numbers; once it's closed, the full result
        stays in the sidebar. Problems that didn't stop the run are listed."""
        fid_txt = f"{fidelity:.1%}" if fidelity is not None else "unknown"
        problems = list(result.get("warnings") or [])
        if not result.get("submitted"):
            problems.append(f"The final counts failed: {result.get('submit_error') or 'unknown error'}.")
        if result.get("stop_detail"):
            problems.append(result["stop_detail"])

        informative = (r"Fidelity is the overlap with the target Gibbs state; purity $\mathrm{Tr}\,\rho^2$ shows "
                       "how mixed the prepared state is")
        informative += (f" (the target's is {target_purity:.1%})." if target_purity is not None else ".")
        if problems:
            informative += "\n\nSome parts didn't complete:\n" + "\n".join(f"• {p}" for p in problems)

        sidebar = _result_rows([
            ("Fidelity", f"<b>{fid_txt}</b>"),
            ("Purity", f"<b>{purity:.1%}</b>"
             + (f" <span style='color:{Colors.TEXT_DIM};'>(target {target_purity:.1%})</span>"
                if target_purity is not None else "")),
            ("Final cost", f"{result['cost']:.4f}"),
            (r"$\gamma_1, \gamma_2$", f"{params[0]:.3f}, {params[1]:.3f}"),
            (r"$\alpha_1, \alpha_2$", f"{params[2]:.3f}, {params[3]:.3f}"),
            ("Counts", "received" if result.get("submitted") else "failed"),
        ]) + "".join(f"<p style='color:{Colors.WARN}; margin-top:6px;'>{p}</p>"
                     for p in problems if not p.startswith("The final counts"))
        self._show_result(
            "done", "Experiment completed",
            f"The experiment has been completed. We reached {fid_txt} fidelity and "
            f"{purity:.1%} purity.", informative, sidebar)

    def _show_failure(self, result):
        """Popup for a run that didn't complete, in plain language: what
        happened, how far it got and what to do next. Technical details are
        in the error-log file it points to. Then a short version in the
        sidebar."""
        reason, message = result.get("reason", "error"), result.get("message", "")
        device = self.config.backend_name if self.config else "the backend"
        n = result.get("iterations", 0)
        after = (f"after {n} iteration{'s' if n != 1 else ''}" if n
                 else "before the first iteration finished")

        if reason == "stopped":
            kind, title = "warn", "Experiment stopped"
            text = f"You stopped the experiment {after}."
            advice = result.get("stop_detail") or "No job was waiting at Quantum Inspire."
        elif reason == "timeout":
            kind, title = "warn", "Queue time-out"
            text = (f"A job waited longer than {_duration(JOB_TIMEOUT)} on {device}, so the "
                    f"experiment stopped {after}.")
            advice = (f"{message}\n\n{device} may be offline, calibrating or very busy: check "
                      f"its status on the Backend screen and try again later.")
        elif reason == "connection":
            kind, title = "warn", "Connection lost"
            text = (f"Quantum Inspire couldn't be reached for {_duration(CONNECTION_TIMEOUT)}, "
                    f"so the experiment stopped {after}.")
            advice = ("Check your internet connection, and log in again if your session "
                      "expired. The job that was waiting may still run.")
        elif reason == "job_failed":
            kind, title = "error", f"Job failed on {device}"
            text = f"Quantum Inspire reported that a job failed on {device} {after}."
            advice = (f"{device} may be calibrating or having a problem: check its status on "
                      f"the Backend screen and try again later.")
        elif reason == "setup":
            kind, title = "error", "Experiment couldn't start"
            text = "The experiment couldn't start."
            advice = (message[:1].upper() + message[1:] + ("" if message.endswith(".") else ".")
                      if message else
                      f"{device} couldn't be reached. Check your internet connection, and log "
                      f"in to Quantum Inspire again if your session expired.")
        else:
            kind, title = "error", "Experiment failed"
            text = f"Something unexpected went wrong {after}, so the experiment stopped."
            advice = "Please try again. If it keeps happening, report it with the error log below."

        notes = []
        if n and result.get("live_log"):
            notes.append(f"Progress so far is saved in {result['live_log']}.")
        if result.get("error_log"):
            notes.append(f"Error log, for reporting: {result['error_log']}")
        informative = advice + "".join(f"\n\n{note}" for note in notes)
        self._show_result(kind, title, text, informative, advice.replace("\n\n", "<br>"))

    def _show_result(self, kind, title, text, informative, sidebar_html):
        """Non-blocking result popup with a colored status icon: green when
        the run completed, orange when it was stopped, red on an error.
        Closing it puts `title` and `sidebar_html` in the sidebar."""
        color = {"done": Colors.GOOD, "warn": Colors.WARN, "error": Colors.BAD}[kind]
        box = QMessageBox(self)
        box.setIconPixmap(_status_icon(color, {"done": "✓", "warn": "!", "error": "✕"}[kind]))
        box.setWindowTitle(title)
        # macOS doesn't show a message box's window title, so it leads the text.
        box.setText(f"<p style='font-size:15pt; font-weight:700; margin-bottom:6px;'>{title}</p>"
                    f"<p style='font-weight:400;'>{text}</p>")
        box.setInformativeText(inline_tex(informative.replace("\n", "<br>"),
                                          font_px(box.font().pointSizeF())))
        box.setStandardButtons(QMessageBox.Close)
        self._outcome = {"kind": kind, "title": title, "html": sidebar_html}
        self.progress_bar.finish(kind, title)
        box.finished.connect(lambda _: self._show_outcome(self._outcome))
        self._result_box = box
        box.open()

    def _show_outcome(self, outcome: dict):
        """The outcome box in the sidebar, edged in the outcome's colour. Its
        html keeps its $...$ (as saved with a run): typeset here."""
        self.result_lbl.setStyleSheet(f"""
            #resultBox {{
                background: {Colors.BG_2}; color: {Colors.TEXT}; font-size: 10.5pt;
                border: 1px solid {Colors.BORDER}; border-left: 3px solid {_outcome_color(outcome)};
                border-radius: 8px; padding: 10px 12px;
            }}
        """)
        set_tex(self.result_lbl, _outcome_html(outcome), Colors.TEXT)
        self.result_lbl.show()

    # -----------------------------------------------------
    # Plotting
    # -----------------------------------------------------

    def _add_hover(self, key, ax):
        """Highlight + value label for the iteration under the cursor (see
        _on_hover); recreated whenever the plot is reset."""
        label = ax.annotate("", xy=(0, 0), xytext=(12, 12), textcoords="offset points",
                            fontsize=9, color=Colors.TEXT, zorder=10,
                            bbox=dict(boxstyle="round,pad=0.4", facecolor=Colors.BG_2,
                                      edgecolor=Colors.ACCENT, linewidth=1))
        label.set_visible(False)
        (dot,) = ax.plot([], [], "o", markersize=10, markerfacecolor="none",
                         markeredgecolor=Colors.TEXT, markeredgewidth=1.6, zorder=9)
        self._hover[key] = {"ax": ax, "label": label, "dot": dot, "index": None}

    def _on_hover(self, event, key):
        hover = self._hover.get(key)
        if hover is None:
            return
        xs, ys = (self.loss_x, self.loss_y) if key == "loss" else (self.fidelity_x, self.fidelity_y)
        canvas = self.loss_canvas if key == "loss" else self.fidelity_canvas
        if event.inaxes is not hover["ax"] or not xs or event.xdata is None:
            if hover["index"] is not None:
                hover["index"] = None
                hover["label"].set_visible(False)
                hover["dot"].set_data([], [])
                canvas.draw_idle()
            return
        i = min(range(len(xs)), key=lambda k: abs(xs[k] - event.xdata))
        if i == hover["index"]:
            return
        hover["index"] = i
        x, y = xs[i], ys[i]
        hover["dot"].set_data([x], [y])
        label = hover["label"]
        label.xy = (x, y)
        label.set_text(self._hover_text(key, x, y))
        # Keep the label inside the plot: left of the point in the right half.
        right_half = event.x > (hover["ax"].bbox.x0 + hover["ax"].bbox.x1) / 2
        label.set_position((-12 if right_half else 12, 12))
        label.set_horizontalalignment("right" if right_half else "left")
        label.set_visible(True)
        canvas.draw_idle()

    def _hover_text(self, key, x, y):
        if key == "fidelity":
            return f"Iteration {x}\nFidelity $F = {y:.4f}$ ({y:.1%})"
        if not self._hardware:
            return f"Iteration {x}\n{y:.1%} above the minimum"
        probes = r" (mean of $\pm$ probes)" if self._ladder else ""
        return f"Iteration {x}\nCost $C = {y:.4f}${probes}"

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
            self._loss_line.set_label("Measured cost $C$" if not self._ladder
                                      else r"Measured cost $C$ (mean of SPSA's $\pm$ probes)")
            style_axes(ax, xlabel="Iteration", ylabel="Cost $C$ (measured)")
            self._loss_bound = loss_lower_bound(self._spins, self.config.beta)
            ax.axhline(self._loss_bound, color=Colors.GOOD, linestyle="--", linewidth=1.8,
                       label="Theoretical minimum")
        else:
            style_axes(ax, xlabel="Iteration", ylabel="Relative gap to lower bound")
            self._loss_bound = 0.0
            ax.axhline(0.0, color=Colors.GOOD, linestyle="--", linewidth=1.8, label="Lower bound")
        _legend_above(ax)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        self._add_hover("loss", ax)
        self.loss_canvas.draw()

    def _on_progress(self, it: int, cost: float):
        self._record("cost", int(it), float(cost))
        self.progress_bar.iteration(it)
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
            zorder=3, label="Perfect ($F = 1$)")
        self._fidelity_gap_fill = None
        (self._fidelity_line,) = ax.plot([], [], "-", color=Colors.ACCENT, linewidth=2.2,
                                          solid_capstyle="round", zorder=2,
                                          label=(f"{'Simulated' if backends.is_qx(self.config.backend_name) else 'Measured'}"
                                                 f" on {self.config.backend_name}")
                                          if self._hardware and self.config is not None
                                          else "Observed")
        (self._fidelity_head,) = ax.plot([], [], "o", color=Colors.ACCENT_HOVER,
                                          markersize=6, markeredgecolor=Colors.BG_3,
                                          markeredgewidth=1.2, zorder=2)
        style_axes(ax, xlabel="Iteration", ylabel="Fidelity $F$")
        ax.set_ylim(0, 1.05)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        _legend_above(ax)
        self._add_hover("fidelity", ax)
        self.fidelity_canvas.draw()

    def _on_fidelity_progress(self, it: int, fid: float):
        self._record("fidelity", int(it), float(fid))
        if self._ladder:
            self.final_fidelity.fidelity(fid)
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
        if self._ladder:
            self.final_fidelity.purity(float(np.real(np.trace(rho @ rho))))
        zmax = self._density_zmax()
        if zmax != self._ideal_zmax:
            # Keep both 3D plots on the same |ρ| scale so they compare
            # bar-for-bar.
            self._draw_ideal_density()
        title = r"$\rho_A$  (final tomography)" if self._ladder else r"$\rho_A$  (observed, live)"
        self._render_density(self.rho_canvas, rho, title=title, zmax=zmax)

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
        if not self._ladder:
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

    def snapshot(self) -> dict:
        """Everything needed to show this run again (show_saved_results)."""
        def matrix(rho):
            return None if rho is None else [np.real(rho).tolist(), np.imag(rho).tolist()]

        return {
            "kind": RESULTS_KIND, "schema_version": 1,
            "saved": datetime.now().isoformat(timespec="seconds"),
            "config": self.config.to_json_dict(),
            "events": self._events,
            "params": None if self._last_params is None else [float(x) for x in self._last_params],
            "rho": matrix(self._last_rho),
            "target_rho": matrix(self._target_rho),
            "counts": self._last_real_counts,
            "ideal_probs": ({k: float(v) for k, v in self._last_ideal_probs.items()}
                            if self._last_ideal_probs else None),
            "outcome": self._outcome,
            "jobs": self.jobs_view.jobs(),
        }

    def show_saved_results(self, snap: dict):
        """Redraw a saved run (see snapshot) exactly as it ended."""
        def matrix(m):
            return None if m is None else np.array(m[0]) + 1j * np.array(m[1])

        self._prepare_view(SessionConfig.from_json_dict(snap["config"]), live=False)
        self._replaying = True
        handlers = {"step": self._on_step, "cost": self._on_progress,
                    "fidelity": self._on_fidelity_progress}
        for name, *args in snap.get("events", []):
            handlers[name](*args)
        self._replaying = False
        self._events = [list(e) for e in snap.get("events", [])]

        if snap.get("target_rho") is not None:
            self._target_rho = matrix(snap["target_rho"])
        self._draw_ideal_density()
        if snap.get("rho") is not None:
            self._draw_density(matrix(snap["rho"]))
        if snap.get("params"):
            self._last_params = np.array(snap["params"])
            self._draw_circuit_for(self._last_params)
        self._last_real_counts = snap.get("counts")
        self._last_ideal_probs = snap.get("ideal_probs")
        self._render_counts_pair(self._last_ideal_probs, self._last_real_counts)
        for job in snap.get("jobs", []):
            self.jobs_view.update_job(job)
        self._outcome = snap.get("outcome")
        saved = snap.get("saved", "").replace("T", " ")
        if self._outcome:
            self._show_outcome(self._outcome)
            self.progress_bar.show_saved(self._outcome["kind"], self._outcome["title"], saved)
        self.protocol_lbl.setText(
            f"<b style='color:{Colors.ACCENT};'>Saved results{f' from {saved}' if saved else ''}.</b> "
            f"Run again repeats this experiment with the same settings.<br>"
            + self.protocol_lbl.text())
        self.stop_btn.setEnabled(False)
        self.restart_btn.setEnabled(True)
        self.save_session_btn.setEnabled(True)

    def export_figures(self, folder: Path):
        """Every plot as PNG and its data as CSV, into `folder`."""
        folder.mkdir(parents=True, exist_ok=True)
        canvases = {"fidelity": self.fidelity_canvas, "cost": self.loss_canvas,
                    "density_target": self.ideal_rho_canvas, "density_measured": self.rho_canvas,
                    "counts_measured": self.real_counts_canvas, "circuit": self.circ_canvas}
        if not self._hardware:
            canvases["counts_ideal"] = self.ideal_counts_canvas
        if self._ladder:
            del canvases["fidelity"]
            self.final_card.grab().save(str(folder / "fidelity.png"))
        for name, canvas in canvases.items():
            canvas.fig.savefig(folder / f"{name}.png", dpi=150, facecolor=canvas.fig.get_facecolor())

        def write_csv(name, header, rows):
            with open(folder / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(rows)

        write_csv("fidelity", ["iteration", "fidelity"], zip(self.fidelity_x, self.fidelity_y))
        cost_header = "cost" if self._hardware else "relative_gap_to_minimum"
        write_csv("cost", ["iteration", cost_header], zip(self.loss_x, self.loss_y))
        for name, rho in (("density_target", self._target_rho), ("density_measured", self._last_rho)):
            if rho is not None:
                rows = [["real"]] + np.real(rho).tolist() + [["imaginary"]] + np.imag(rho).tolist()
                write_csv(name, [f"col {j}" for j in range(rho.shape[1])], rows)
        if self._last_real_counts:
            write_csv("counts_measured", ["bitstring", "count"],
                      sorted(self._last_real_counts.items(), key=lambda kv: kv[1], reverse=True))
        if self._last_ideal_probs:
            write_csv("counts_ideal", ["bitstring", "probability"],
                      sorted(self._last_ideal_probs.items(), key=lambda kv: kv[1], reverse=True))

    def _on_save_results(self):
        if self.config is not None:
            save_results_dialog(self, self.snapshot(), self.export_figures)

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