"""The Run screen's "How it works" tab: the variational loop between the
classical computer and the quantum computer as a diagram. Click a part to
read, flash-card style, what it does in this run."""

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QFrame, QLabel, QScrollArea, QSizePolicy, QVBoxLayout, QWidget

from backend import backends, ladder_tfd
from frontend.colors import Colors
from frontend.latex import set_tex
from frontend.session import SessionConfig, format_beta

_NODES = ("classical", "angles", "quantum", "cost")

# As in frontend/experiment.py: CostFunction's default shots, its cap on the
# cloud, and backend/ladder_tfd.py's.
_COSTFUNCTION_SHOTS = 8192
_QX_SHOTS = 2048


def _quantum_title(device: str) -> str:
    """Starmon-7 and the QX emulator are simulations, not quantum computers."""
    return "Quantum computer" if backends.is_hardware(device) else "Quantum computer simulation"


def _kind(device: str) -> str:
    return "local" if backends.is_local(device) else "qx" if backends.is_qx(device) else "hardware"


def _shots(device: str, ladder: bool) -> str:
    if backends.is_qx(device):
        return f"{_QX_SHOTS} shots"
    if ladder:
        return f"{ladder_tfd.SHOTS} shots"
    return f"{_COSTFUNCTION_SHOTS} shots" if backends.is_local(device) else f"up to {_COSTFUNCTION_SHOTS} shots"


def _subtitles(config: SessionConfig, optimizer: str, ladder: bool) -> dict:
    """The small captions drawn on the diagram, per backend."""
    device, qubits = config.backend_name, 2 * config.n_spins
    quantum = {
        "local": f"{device}, simulated here\nnoiseless, {qubits} qubits",
        "qx": f"{device}, in the cloud\nnoiseless, {qubits} qubits",
        "hardware": f"the {device} chip: every cost\nis measured here, {qubits} qubits",
    }[_kind(device)]
    measured = "X and Z, measured mid-circuit" if not ladder else "all-Z and all-X circuits"
    return {
        "classical": (f"{optimizer}: picks the next angles\nfrom the measured costs"
                      if backends.is_hardware(device) else f"{optimizer}\non this computer"),
        "quantum": quantum,
        "angles": "a direct call, no queue" if backends.is_local(device) else "a job, waits in the queue",
        "cost": f"{measured}, {_shots(device, ladder)}",
    }


def _iteration_steps(config: SessionConfig, optimizer: str, ladder: bool) -> list:
    """What one iteration does on this backend, in order (frontend/experiment.py's worker)."""
    n, device = config.n_spins, config.backend_name
    where = {"local": f"a noiseless simulation of {device} on this computer",
             "qx": f"the {device} in the cloud",
             "hardware": f"the {device} chip"}[_kind(device)]
    shots = _shots(device, ladder)
    if ladder:
        return [
            r"SPSA draws a random direction $\Delta$ and two probe points, $\theta \pm c_k\Delta$.",
            f"Four circuits, all-Z and all-X at each probe, every qubit measured once at the end, "
            f"go to {where} as one job ({shots}). It waits in Quantum Inspire's queue: see the "
            "jobs tab.",
            "The counts are readout-corrected with the latest calibration (an all-0 and an all-1 "
            "circuit, its own job, repeated every 30 minutes).",
            "The cost at each probe follows from the X and Z parities; the cost plot shows their "
            "mean, since measuring θ itself would need a second job.",
            r"SPSA steps downhill along $\Delta$. No fidelity yet: that needs the "
            rf"$3^{n} = {3 ** n}$-setting tomography at the end.",
        ]
    cost_circuit = (f"one circuit on {where} ({shots}) that prepares the state and measures it in X, "
                    "resets, prepares it again and measures in Z, then measures all-0 and all-1 for "
                    "the readout calibration (refreshed every 10 evaluations)")
    tomography = ("a 9-setting tomography of A (X, Y, Z on A1 and A2, chained in one circuit the "
                  "same way), for the live density matrix and fidelity")
    if optimizer == "SPSA":
        steps = [
            r"SPSA draws a random direction $\Delta$ and two probe points, $\theta \pm c_k\Delta$.",
            f"Each probe's cost: {cost_circuit}.",
            f"At the current angles θ, the cost again (the cost plot) and {tomography}.",
            r"SPSA steps downhill along $\Delta$.",
        ]
        if not backends.is_local(device):
            steps.insert(3, "All four circuits go as one job, which waits in Quantum Inspire's "
                            "queue: see the jobs tab.")
        return steps
    return [
        "COBYLA proposes the next angles from a linear model of the costs so far, within its "
        "trust region.",
        f"Their cost: {cost_circuit}.",
        f"At the same angles, {tomography}.",
        "COBYLA keeps the point if the cost fell and shrinks its trust region if not; it stops "
        "by itself once the steps get small.",
    ]


def _end_steps(config: SessionConfig, ladder: bool) -> list:
    n, device = config.n_spins, config.backend_name
    local = backends.is_local(device)
    if ladder:
        jobs = "" if local else ", in jobs of up to 5 circuits"
        final = [f"Tomography at the final angles: $3^{n} = {3 ** n}$ circuits{jobs}, readout-corrected, "
                 "give ρ_A, the final fidelity and purity, and the final cost."]
    else:
        final = ["The cost and the 9-setting tomography once more, at the final angles: the final "
                 "fidelity and purity."]
    counts = {
        "local": f"Final counts: the optimized circuit, measured in Z, on the {device} noise model. "
                 "It's the only noisy step: the optimization above is noiseless.",
        "qx": f"Final counts: the optimized circuit, measured in Z, as one more job on the {device}.",
        "hardware": f"Final counts: the optimized circuit, measured in Z, as one more job on the "
                    f"{device} chip.",
    }[_kind(device)]
    return final + [counts, "Everything is saved in the project folder (the link in the sidebar)."]


def _explanations(config: SessionConfig, optimizer: str, ladder: bool) -> dict:
    """Node key -> (heading, rich text with $...$ math) for this run."""
    n, device, beta = config.n_spins, config.backend_name, format_beta(config.beta)
    kind = _kind(device)
    method = {
        "SPSA": "It nudges all four angles at once along a random direction "
                r"$\Delta \in \{\pm 1\}^4$, measures the cost on both sides, "
                r"$C(\theta \pm c_k\Delta)$, and steps downhill: "
                r"$\theta \leftarrow \theta - a_k\,\frac{C_+ - C_-}{2c_k}\,\Delta$. "
                "Two cost measurements per step, however many angles, and its shrinking "
                "steps average the shot noise away. Before the first step it measures a few "
                "random directions to size its steps; then it runs every iteration, and the "
                "result is the average of the last quarter of the steps.",
        "COBYLA": "It fits a local linear model to the measured costs and moves within a "
                  "trust region that shrinks as it converges. No gradients, few cost "
                  "evaluations, and it stops on its own once the steps get small; noise "
                  "in the cost can mislead it.",
    }[optimizer]
    why = {
        "local": "On Starmon-7 you choose the optimizer.",
        "qx": "Cloud runs always use SPSA: every cost is a queued job with shot noise, and SPSA "
              "needs only two per step and tolerates the noise.",
        "hardware": "Real hardware always uses SPSA: every cost is a queued job with gate, readout "
                    "and shot noise, and SPSA needs only two per step and tolerates the noise.",
    }[kind]
    start = ("from the angles set in Setup" if kind == "local"
             else r"from the random angles drawn in Setup, near $\theta = 0$")
    quantum = {
        "local": f"Nothing leaves this computer: the {2 * n} qubits of {device} are simulated "
                 "here, <b>without noise</b> during the optimization, so the only noise is from the "
                 f"finite number of shots ({_shots(device, ladder)}). The {device} noise model is "
                 "applied once, to the final counts.",
        "qx": f"Quantum Inspire's {device} simulates the {2 * n} qubits in the cloud, "
              f"<b>without noise</b> ({_shots(device, ladder)} per circuit, its maximum). "
              "Every cost is a job that waits in the emulator's queue: see the Cloud jobs tab.",
        "hardware": f"The real {device} chip, in Quantum Inspire's cloud, on the {2 * n} qubits you "
                    f"picked. Gate, readout and shot noise ({_shots(device, ladder)}). Every cost "
                    "is a job that waits in the chip's queue; its calibration is in the Live "
                    "dashboard tab.",
    }[kind]
    if ladder:
        circuits = ("Per point, two circuits measure every qubit once, at the end: all in Z, "
                    "and all in X. "
                    + ("QX reports results by qubit, so it can't use the paper's circuit, which "
                       "measures the same qubits several times. " if kind == "qx" else ""))
        fidelity = (rf"The fidelity needs the whole state: a $3^{n} = {3 ** n}$-setting tomography, "
                    "run once, at the end.")
    else:
        circuits = ("Per point, the paper's single circuit: prepare, measure in X, reset, prepare "
                    "again, measure in Z, reset, then the readout calibration (all-0, all-1), all "
                    "by mid-circuit measurement. ")
        fidelity = ("The fidelity is measured live: a 9-setting tomography of A at the current "
                    "angles, every iteration.")
    plot = (r"The cost plot shows how far it is above its lowest possible value, "
            r"$(C - C_{\min}) / |C_{\min}|$." if kind == "local"
            else "The cost plot shows it as measured, with the lowest possible value dashed.")
    return {
        "classical": (
            f"Classical computer: {optimizer}",
            "Your computer runs the optimizer. It only ever sees numbers: it sends angles "
            f"to the {_quantum_title(device).lower()} and gets the cost back. It starts {start}. "
            f"{why}<br><br><b>{optimizer}.</b> {method}"),
        "angles": (
            r"Angles $\theta = (\gamma_1, \gamma_2, \alpha_1, \alpha_2)$",
            r"The four knobs of the circuit. $\gamma_1$ and $\gamma_2$ (the $R_X$ and $R_{ZZ}$ "
            r"rotations inside A and inside B) shape each system like its Hamiltonian $H$; "
            r"$\alpha_1$ and $\alpha_2$ (the $R_{XX}$ and $R_{ZZ}$ rotations between A and B) set "
            "how entangled the two are, which is what makes half A thermal.<br><br>"
            + ("They're passed straight to the simulator, with no queue. " if kind == "local" else
               "They're compiled into circuits for the qubits you picked and sent as a job. ")
            + rf"This run started at $\gamma_1 = {config.gamma1:.2f}$, $\gamma_2 = {config.gamma2:.2f}$, "
            rf"$\alpha_1 = {config.alpha1:.2f}$, $\alpha_2 = {config.alpha2:.2f}$."),
        "quantum": (
            f"{_quantum_title(device)}: {device}",
            f"Prepares the {2 * n}-qubit state ({n} spins per system): Bell pairs between A and B (the $\\beta = 0$ "
            f"thermofield double), then the rotations set by the angles.<br><br>{quantum}"
            f"<br><br>{circuits}{fidelity}"),
        "cost": (
            r"Cost $C(\theta)$",
            "The number the optimizer minimises, from the readout-corrected X and Z measurements:"
            r"$$C = \langle X_A + X_B\rangle + s\langle ZZ_A + ZZ_B\rangle"
            r" - \beta^{-s}\langle H_{AB}\rangle$$"
            rf"with $s = 1.57$ and $\beta = {beta}$. Its minimum is (close to) the thermofield "
            f"double, whose half A is the Gibbs state. {plot}"),
    }


def _general_explanations() -> dict:
    """Node key -> (heading, rich text) for the Home screen: only what holds on
    every backend (Starmon-7, QX emulator, Tuna-9, Tuna-17)."""
    return {
        "classical": (
            "Classical computer: the optimizer",
            "Your computer runs the optimizer. It never sees the quantum state, only the cost: "
            "it sends angles, gets back the cost measured on the backend you chose, and picks "
            "better angles from it, iteration after iteration. A quantum chip can only run "
            "circuits, so this step is always classical.<br><br>"
            "<b>SPSA</b> runs on every backend: it measures the cost on both sides of a random "
            "direction, two measurements per step whatever the number of angles, and copes "
            "with noisy measurements. On Starmon-7 you can also pick <b>COBYLA</b>."),
        "angles": (
            r"Angles $\theta = (\gamma_1, \gamma_2, \alpha_1, \alpha_2)$",
            r"The four knobs of the circuit. $\gamma_1$ and $\gamma_2$ act inside A and inside B "
            r"and shape each copy like its Hamiltonian $H$; $\alpha_1$ and $\alpha_2$ act between A "
            "and B and set how entangled they are, which is what makes half A thermal. Setup "
            "draws random starting angles."),
        "quantum": (
            "Quantum computer or simulation",
            "Prepares two copies A and B, entangled in Bell pairs, applies the rotations set by "
            "the angles, and measures the result many times in the X and Z bases.<br><br>"
            "On <b>Starmon-7</b> and the <b>QX emulator</b> the circuit is simulated; on "
            "<b>Tuna-9</b> and <b>Tuna-17</b> it runs on the real chip. Either way the cost "
            "comes from counted measurement outcomes, not from the exact state.<br><br>"
            "Once a run is set up, the Run screen's <b>How it works</b> tab shows each step "
            "for the backend you picked."),
        "cost": (
            r"Cost $C(\theta)$",
            "The single number the optimizer minimises, computed from the X and Z measurements:"
            r"$$C = \langle X_A + X_B\rangle + s\langle ZZ_A + ZZ_B\rangle"
            r" - \beta^{-s}\langle H_{AB}\rangle$$"
            r"with $s = 1.57$ and the $\beta$ you choose. Its minimum is (close to) the "
            "thermofield double, whose half A is the Gibbs state. The fidelity, from "
            "tomography, checks how close the prepared state got."),
    }


class _Diagram(QWidget):
    """The loop itself; emits the clicked node's key."""

    clicked = Signal(str)

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(260)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMouseTracking(True)
        self.subtitles = {key: "" for key in _NODES}
        self.quantum_title = "Quantum computer"
        self.selected = "classical"
        self._hover = None

    def _rects(self) -> dict:
        w, h = self.width(), self.height()
        box_w, box_h = min(230.0, w * 0.3), 110.0
        pill_w, pill_h = min(240.0, w * 0.32), 46.0
        mid_y = h / 2
        return {
            "classical": QRectF(20, mid_y - box_h / 2, box_w, box_h),
            "quantum": QRectF(w - 20 - box_w, mid_y - box_h / 2, box_w, box_h),
            "angles": QRectF(w / 2 - pill_w / 2, 12, pill_w, pill_h),
            "cost": QRectF(w / 2 - pill_w / 2, h - 12 - pill_h, pill_w, pill_h),
        }

    def mouseMoveEvent(self, event):
        hover = next((k for k, r in self._rects().items() if r.contains(event.position())), None)
        if hover != self._hover:
            self._hover = hover
            self.setCursor(Qt.PointingHandCursor if hover else Qt.ArrowCursor)
            self.update()

    def mousePressEvent(self, event):
        key = next((k for k, r in self._rects().items() if r.contains(event.position())), None)
        if key:
            self.selected = key
            self.update()
            self.clicked.emit(key)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        rects = self._rects()

        # The loop: classical -> angles -> quantum -> cost -> classical.
        pen = QPen(QColor(Colors.TEXT_MUTED), 2.2)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        c, q, a, k = rects["classical"], rects["quantum"], rects["angles"], rects["cost"]
        legs = [
            (QPointF(c.center().x(), c.top()), QPointF(a.left(), a.center().y()), "up"),
            (QPointF(a.right(), a.center().y()), QPointF(q.center().x(), q.top()), "down"),
            (QPointF(q.center().x(), q.bottom()), QPointF(k.right(), k.center().y()), "down"),
            (QPointF(k.left(), k.center().y()), QPointF(c.center().x(), c.bottom()), "up"),
        ]
        for start, end, _ in legs:
            path = QPainterPath(start)
            corner = QPointF(start.x(), end.y()) if start.x() in (c.center().x(), q.center().x()) \
                else QPointF(end.x(), start.y())
            path.quadTo(corner, end)
            p.drawPath(path)
            # Arrowhead at `end`, pointing along the last tangent.
            direction = end - corner
            length = max((direction.x() ** 2 + direction.y() ** 2) ** 0.5, 1e-6)
            ux, uy = direction.x() / length, direction.y() / length
            tip = end
            left = QPointF(tip.x() - 9 * ux + 5 * uy, tip.y() - 9 * uy - 5 * ux)
            right = QPointF(tip.x() - 9 * ux - 5 * uy, tip.y() - 9 * uy + 5 * ux)
            head = QPainterPath(tip)
            head.lineTo(left)
            head.lineTo(right)
            head.closeSubpath()
            p.fillPath(head, QColor(Colors.TEXT_MUTED))

        titles = {"classical": "Classical computer", "quantum": self.quantum_title,
                  "angles": "angles θ  →", "cost": "←  cost C(θ)"}
        for key, rect in rects.items():
            selected, hover = key == self.selected, key == self._hover
            accent = Colors.WARN if key == "quantum" else Colors.ACCENT
            p.setPen(QPen(QColor(accent if selected or hover else Colors.BORDER), 2 if selected else 1.4))
            fill = QColor(accent)
            fill.setAlpha(46 if selected else 22 if hover else 0)
            p.setBrush(QColor(Colors.BG_3))
            radius = 14 if key in ("classical", "quantum") else rect.height() / 2
            p.drawRoundedRect(rect, radius, radius)
            p.setBrush(fill)
            p.drawRoundedRect(rect, radius, radius)

            font = QFont()
            font.setBold(True)
            font.setPointSizeF(12 if key in ("classical", "quantum") else 11)
            p.setFont(font)
            p.setPen(QColor(Colors.TEXT))
            if key in ("classical", "quantum"):
                icon = "🖥" if key == "classical" else "⚛"
                title_rect = QRectF(rect.left(), rect.top() + 14, rect.width(), 26)
                title = f"{icon}  {titles[key]}"
                # Shrink a long title ("Quantum computer simulation") to fit the box.
                while QFontMetrics(font).horizontalAdvance(title) > rect.width() - 16 and font.pointSizeF() > 8:
                    font.setPointSizeF(font.pointSizeF() - 0.5)
                    p.setFont(font)
                p.drawText(title_rect, Qt.AlignHCenter | Qt.AlignVCenter, title)
                font.setBold(False)
                font.setPointSizeF(10)
                p.setFont(font)
                p.setPen(QColor(accent))
                p.drawText(QRectF(rect.left() + 8, rect.top() + 44, rect.width() - 16, rect.height() - 52),
                           Qt.AlignHCenter | Qt.AlignTop | Qt.TextWordWrap, self.subtitles[key])
            else:
                p.drawText(rect, Qt.AlignCenter, titles[key])
                # What travels along this leg, on this backend: below the angles
                # pill, above the cost pill.
                font.setBold(False)
                font.setPointSizeF(9.5)
                p.setFont(font)
                p.setPen(QColor(Colors.TEXT_DIM))
                # Kept between the two boxes, elided if it doesn't fit.
                left = rects["classical"].right() + 8
                width = rects["quantum"].left() - 8 - left
                top = rect.bottom() + 4 if key == "angles" else rect.top() - 22
                text = QFontMetrics(font).elidedText(self.subtitles[key], Qt.ElideRight, int(width))
                p.drawText(QRectF(left, top, width, 18), Qt.AlignCenter, text)

        font = QFont()
        font.setPointSizeF(9)
        p.setFont(font)
        p.setPen(QColor(Colors.TEXT_MUTED))
        p.drawText(QRectF(0, self.height() / 2 - 10, self.width(), 20), Qt.AlignCenter,
                   "click a part of the loop")
        p.end()


class LoopDiagram(QWidget):
    """set_run() before a run (or a saved run's replay); set_general() for
    the backend-independent version on the Home screen, which sits inside
    Home's own scroll area (scrollable=False)."""

    def __init__(self, scrollable: bool = True):
        super().__init__()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        content = QWidget()
        if scrollable:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.NoFrame)
            scroll.setWidget(content)
            outer.addWidget(scroll)
        else:
            outer.addWidget(content)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(*((20, 16, 20, 20) if scrollable else (0, 0, 0, 0)))
        layout.setSpacing(14)

        self.intro = QLabel()
        self.intro.setWordWrap(True)
        self.intro.setStyleSheet(f"color: {Colors.TEXT_DIM};")
        layout.addWidget(self.intro)

        self.diagram = _Diagram()
        self.diagram.clicked.connect(self._show)
        layout.addWidget(self.diagram)

        self.card = QFrame()
        self.card.setObjectName("loopCard")
        self.card.setStyleSheet(f"""
            #loopCard {{ background: {Colors.BG_2}; border: 1px solid {Colors.BORDER};
                         border-radius: 14px; }}
            #loopCard QLabel {{ background: transparent; border: none; }}
        """)
        self.heading, self.body = self._card(self.card)
        layout.addWidget(self.card)

        # The run's workflow, step by step, for the chosen backend.
        self.steps_card = QFrame()
        self.steps_card.setObjectName("loopCard")
        self.steps_card.setStyleSheet(self.card.styleSheet())
        self.steps_heading, self.steps_body = self._card(self.steps_card)
        layout.addWidget(self.steps_card)
        layout.addStretch()
        self._texts = {}

    @staticmethod
    def _card(frame: QFrame) -> tuple:
        card_layout = QVBoxLayout(frame)
        card_layout.setContentsMargins(20, 16, 20, 18)
        card_layout.setSpacing(8)
        heading = QLabel()
        heading.setStyleSheet(f"color: {Colors.TEXT}; font-size: 13pt; font-weight: 700;")
        body = QLabel()
        body.setWordWrap(True)
        body.setStyleSheet(f"color: {Colors.TEXT_DIM}; font-size: 11pt;")
        card_layout.addWidget(heading)
        card_layout.addWidget(body)
        return heading, body

    def set_run(self, config: SessionConfig, optimizer: str, ladder: bool):
        device = config.backend_name
        self._texts = _explanations(config, optimizer, ladder)
        self.diagram.subtitles = _subtitles(config, optimizer, ladder)
        self.diagram.quantum_title = _quantum_title(device)
        self.diagram.update()
        hardware_note = (
            f" Every cost is measured on the {device} chip; your computer only does the "
            f"optimizer's arithmetic between jobs, which a quantum chip can't run."
            if backends.is_hardware(device) else "")
        self.intro.setText(
            f"The variational loop on {device}: the classical computer proposes angles, the "
            f"{_quantum_title(device).lower()} prepares and measures the state, and the measured cost goes back "
            "to the classical computer, iteration after iteration. Click a part to see what it "
            f"does in this run; the steps below follow one iteration.{hardware_note}")

        def numbered(steps):
            return "<br>".join(f"<b>{i}.</b> {step}" for i, step in enumerate(steps, 1))

        set_tex(self.steps_heading, f"The workflow on {device}", Colors.TEXT)
        set_tex(self.steps_body,
                f"<b style='color:{Colors.ACCENT};'>Every iteration</b><br>"
                f"{numbered(_iteration_steps(config, optimizer, ladder))}<br><br>"
                f"<b style='color:{Colors.ACCENT};'>After the last iteration</b><br>"
                f"{numbered(_end_steps(config, ladder))}",
                Colors.TEXT_DIM)
        self._show(self.diagram.selected)

    def set_general(self):
        self._texts = _general_explanations()
        self.diagram.quantum_title = "Quantum computer or simulation"
        self.diagram.subtitles = {
            "classical": "the optimizer: picks the next\nangles from the measured costs",
            "quantum": "simulated (Starmon-7, QX)\nor a real chip (Tuna)",
            "angles": "the circuit's four rotation angles",
            "cost": "from X and Z measurement counts",
        }
        self.diagram.update()
        self.intro.hide()
        self.steps_card.hide()
        # Home's text is larger than the Run screen's.
        self.heading.setStyleSheet(f"color: {Colors.TEXT}; font-size: 15pt; font-weight: 700;")
        self.body.setStyleSheet(f"color: {Colors.TEXT_DIM}; font-size: 13pt;")
        self._show(self.diagram.selected)

    def _show(self, key: str):
        if key not in self._texts:
            return
        heading, body = self._texts[key]
        set_tex(self.heading, heading, Colors.TEXT)
        set_tex(self.body, body, Colors.TEXT_DIM)
