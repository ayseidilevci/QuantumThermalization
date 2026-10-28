"""Setup screen's device panel: a toggle between a static chip schematic and
an interactive qubit-topology widget, where the user picks subsystem A (the
chain A1..An) of the TFD ansatz by dragging or clicking, with the mirrored
subsystem B (B1..Bn) filled in automatically.

For the known backends (Tuna-9, Tuna-17, Starmon-7, QX emulator) node
coordinates and connectivity are hardcoded in KNOWN_BACKEND_TOPOLOGY, read off
the schematics in Appdata/Img/. Any other backend falls back to the live QIBackend's coupling
map (backend_topology) on a plain grid (simple_grid_layout).
"""

from math import ceil, sqrt
from pathlib import Path

from PySide6.QtCore import QPointF, QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsPathItem, QGraphicsScene, QGraphicsSimpleTextItem,
    QGraphicsView, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QStackedLayout, QVBoxLayout, QWidget,
)

from backend import backends, starmon7
from frontend.colors import Colors
from frontend.session import roles


def required_pairs(n: int) -> list:
    """Role pairs that must be coupled on the chip: the A chain, the B chain
    and the rungs A_i–B_i (a 2 x n ladder)."""
    chain = [(f"{s}{i}", f"{s}{i + 1}") for s in "AB" for i in range(1, n)]
    return chain + [(f"A{i}", f"B{i}") for i in range(1, n + 1)]


def qx_ladder(n: int):
    """(coords, edges) drawing the QX emulator's fixed ladder (see
    backends.qx_mapping): A1..An on top, B1..Bn below, rungs between."""
    mapping = backends.qx_mapping(n)
    coords = {q: ((int(role[1:]) - 1 - (n - 1) / 2) * 110, -60 if role[0] == "A" else 60)
              for role, q in mapping.items()}
    edges = {frozenset((mapping[a], mapping[b])) for a, b in required_pairs(n)}
    return coords, edges


def _role_color(role: str) -> str:
    # Subsystem A vs its mirror B, so the two halves are told apart.
    return Colors.ACCENT if role.startswith("A") else Colors.WARN


# =========================================================================
#  Image mode — real backend chip schematics (Appdata/Img/)
# =========================================================================

DATA_DIR = Path(__file__).resolve().parent.parent / "Appdata" / "Img"

# Keys are matched against the live backend name case-insensitively/fuzzily
# (see _lookup_by_name) since the exact string the QI API returns for a given
# chip isn't guaranteed to match these display labels verbatim.
BACKEND_IMAGES = {
    "tuna9": DATA_DIR / "1765992515-tuna9logo_v1.webp",
    "tuna17": DATA_DIR / "1772648307-tuna-17_schematic-svg.svg",
    "starmon7": DATA_DIR / "starmon-7_schematic.svg",
}

# Hardcoded, real per-chip topology: on-screen coordinates (matching each
# schematic in Appdata/Img/, y increasing downward same as the images) and
# real connectivity (which pairs are actually coupled — the "TC" tunable
# couplers in the pictures are the physical link for each edge here, not a
# separate addressable qubit). Keyed the same fuzzy way as BACKEND_IMAGES.
#
# Tuna-9 / Tuna-17: a diamond/checkerboard lattice — every qubit couples to
# its diagonal neighbors one lattice-step away (see the schematics).
# Starmon-7: two rhombi (0-2-5-3 and 1-3-6-4) sharing the central qubit 3;
# edges come from backend/starmon7.py.
# QX emulator: 10 qubits with all-to-all connectivity, shown as a 2 x 5 grid
# whose couplings are enough for any 2+2, 3+3 or 4+4 ladder.
KNOWN_BACKEND_TOPOLOGY = {
    "tuna9": {
        "coords": {
            0: (0, -160),
            1: (-80, -80),
            2: (80, -80),
            3: (-160, 0),
            4: (0, 0),
            5: (160, 0),
            6: (-80, 80),
            7: (80, 80),
            8: (0, 160),
        },
        "edges": [
            (0, 1), (0, 2),
            (1, 3), (1, 4), (2, 4), (2, 5),
            (3, 6), (4, 6), (4, 7), (5, 7),
            (6, 8), (7, 8),
        ],
    },
    "tuna17": {
        "coords": {
            0: (-38, -116), 1: (-77, -77), 2: (0, -77), 3: (77, -77),
            4: (-38, -39), 5: (39, -39), 6: (116, -39),
            7: (-77, 0), 8: (0, 0), 9: (77, 0),
            10: (-116, 39), 11: (-38, 39), 12: (38, 39),
            13: (-77, 77), 14: (0, 77), 15: (77, 77),
            16: (38, 116),
        },
        "edges": [
            (0, 1), (0, 2),
            (1, 4), (2, 4), (2, 5), (3, 5), (3, 6),
            (4, 7), (4, 8), (5, 8), (5, 9), (6, 9),
            (7, 10), (7, 11), (8, 11), (8, 12), (9, 12),
            (10, 13), (11, 13), (11, 14), (12, 14), (12, 15),
            (14, 16), (15, 16),
        ],
    },
    "starmon7": {
        "coords": {
            0: (-80, -80), 1: (80, -80),
            2: (-160, 0), 3: (0, 0), 4: (160, 0),
            5: (-80, 80), 6: (80, 80),
        },
        "edges": starmon7.EDGES,
    },
    "qxemulator": {
        "coords": {q: ((q % 5 - 2) * 90, -50 if q < 5 else 50) for q in range(10)},
        "edges": [(q, q + 1) for q in (0, 1, 2, 3, 5, 6, 7, 8)] + [(q, q + 5) for q in range(5)],
    },
}


def simple_grid_layout(num_qubits: int, spacing: float = 90.0):
    """Fallback node layout for a backend we have no real coordinates for: a
    plain row-major grid, which stays readable however many edges there are."""
    if num_qubits <= 0:
        return {}
    ncols = ceil(sqrt(num_qubits))
    nrows = ceil(num_qubits / ncols)
    coords = {}
    for i in range(num_qubits):
        row, col = divmod(i, ncols)
        x = (col - (ncols - 1) / 2) * spacing
        y = (row - (nrows - 1) / 2) * spacing
        coords[i] = (x, y)
    return coords


def _normalize(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _lookup_by_name(name: str, table: dict):
    """Fuzzy, case/format-insensitive lookup of `name` against `table`'s
    keys (substring match either way) — real QI backend names aren't
    guaranteed to match our display labels exactly."""
    norm = _normalize(name)
    for key, value in table.items():
        if key in norm or norm in key:
            return value
    return None


class BackendImageView(QLabel):
    """Displays the real chip schematic for the selected backend — an SVG
    re-rendered crisply at the panel's current size, or a raster image
    scaled to fit — instead of a synthesized connectivity graph.
    """

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(120)
        self.setMinimumWidth(160)
        self.setStyleSheet(f"background: {Colors.BG_3}; border: none;")
        self._svg = None
        self._raster = None

    def set_backend(self, name: str):
        path = _lookup_by_name(name, BACKEND_IMAGES) if name else None
        if path is None or not path.exists():
            self._svg = None
            self._raster = None
            self.setPixmap(QPixmap())
            self.setText(f"No schematic available for “{name}”.")
            self.setStyleSheet(f"background: {Colors.BG_3}; color: {Colors.TEXT_MUTED}; "
                                f"border: none; padding: 24px;")
            return
        self.setStyleSheet(f"background: {Colors.BG_3}; border: none;")
        self.setText("")
        if path.suffix.lower() in (".svg", ".svgz"):
            self._svg = QSvgRenderer(str(path))
            self._raster = None
        else:
            self._svg = None
            self._raster = QPixmap(str(path))
        self._redraw()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._redraw()

    def _redraw(self):
        # QtSvg draws the schematics' own text labels; nothing is redrawn here.
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
            img.fill(QColor(Colors.BG_3))
            painter = QPainter(img)
            self._svg.render(painter)
            painter.end()
            self.setPixmap(QPixmap.fromImage(img))
        elif self._raster is not None and not self._raster.isNull():
            self.setPixmap(self._raster.scaled(
                size, Qt.KeepAspectRatio, Qt.SmoothTransformation))


# =========================================================================
#  Real backend topology (live from the QI API, not guessed)
# =========================================================================

def backend_topology(qi_backend):
    """Return (num_qubits, edges) for a QIBackend, edges as a set of
    frozenset({i, j}) pairs. Falls back to treating the device as fully
    connected if the API reports no restricted topology (e.g. a simulator).
    """
    num_qubits = qi_backend.num_qubits
    coupling_map = qi_backend.target.build_coupling_map()
    if coupling_map is None:
        edges = {frozenset((i, j)) for i in range(num_qubits)
                  for j in range(num_qubits) if i != j}
    else:
        edges = {frozenset(pair) for pair in coupling_map.get_edges()}
    return num_qubits, edges


# =========================================================================
#  Interactive mode — clickable qubit topology
# =========================================================================

_NODE_RADIUS = 22.0


def find_mirror_chain(a_chain: list, edges: set):
    """Subsystem B for a chosen A chain: qubits B1..Bn, distinct from A and
    each other, with every A_i–B_i and B_i–B_(i+1) coupled — the mirrored
    half of the 2 x n ladder the TFD ansatz needs. Returns None if the device
    has none. Lowest-numbered first, so the same A always gives the same B."""
    def neighbors(q):
        return sorted(next(iter(e - {q})) for e in edges if q in e)

    def extend(b_chain):
        i = len(b_chain)
        if i == len(a_chain):
            return b_chain
        for b in neighbors(a_chain[i]):
            if b in a_chain or b in b_chain:
                continue
            if b_chain and frozenset((b_chain[-1], b)) not in edges:
                continue
            found = extend(b_chain + [b])
            if found:
                return found
        return None

    return extend([])


class QubitTopologyWidget(QGraphicsView):
    """The user picks subsystem A's chain A1..An only; subsystem B is filled
    in automatically (see find_mirror_chain). Two ways to pick A:

    - drag: press on a qubit (A1) and drag through coupled qubits A2..An,
      then release (dragging back over the previous qubit undoes a step);
    - click: click A1..An one by one, each coupled to the one before
      (clicking the last pick again undoes it; clicking any qubit once A is
      complete starts a new pick from there).

    Emits `mapping_changed(dict)` on every change (role -> physical qubit
    index; incomplete/impossible selections emit {}), and
    `status_changed(str)` with the current inline hint/validation text.
    """

    mapping_changed = Signal(dict)
    status_changed = Signal(str)

    def __init__(self):
        super().__init__()
        self.setRenderHint(QPainter.Antialiasing)
        self.setStyleSheet(f"background: {Colors.BG_3}; border: none;")
        self.setScene(QGraphicsScene(self))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(160)

        self._edges = set()
        self._coords = {}
        self._n = 2                    # spins per system
        self._a_path = []              # qubits picked for A1, A2, ... so far
        self._assignment = {}          # role -> qubit index
        self._problem = None           # why the current pick can't work, if so
        self._problem_is_hint = False  # True: a rejected click, the pick stays open
        self._node_items = {}          # qubit index -> QGraphicsEllipseItem
        self._label_items = {}         # qubit index -> QGraphicsSimpleTextItem (role badge)

        self._drag_path = None         # qubits a drag has passed through
        self._drag_line = None         # rubber-band line while dragging
        self._dragging = False         # moved off the start qubit since press
        self._read_only = False

    # ---- public API --------------------------------------------------
    def set_topology(self, num_qubits: int, edges: set, coords: dict = None):
        self._edges = edges
        self._coords = coords if coords else simple_grid_layout(num_qubits)
        self._set_path([])

    def set_spins(self, n: int):
        """Spins per system; a new size clears the selection."""
        if n != self._n:
            self._n = n
            self._set_path([])

    def current_mapping(self) -> dict:
        complete = len(self._assignment) == 2 * self._n and not self._missing_pairs()
        return dict(self._assignment) if complete else {}

    def clear_selection(self):
        """Drop the whole role assignment."""
        self._set_path([])

    def set_assignment(self, mapping: dict):
        """Restore a role -> qubit assignment directly (used when loading a
        saved session), skipping roles/qubits that no longer fit this
        topology (or collide with another role) rather than raising."""
        valid, used = {}, set()
        for role in roles(self._n):
            qubit = mapping.get(role)
            if qubit is None or qubit not in self._coords or qubit in used:
                continue
            valid[role] = qubit
            used.add(qubit)
        self._a_path = [valid[f"A{i}"] for i in range(1, self._n + 1) if f"A{i}" in valid]
        self._assignment = valid
        self._problem = None
        self._redraw_scene()
        self._emit_state()

    # ---- interaction ---------------------------------------------------
    # A press on a qubit starts a potential drag through coupled qubits;
    # releasing without moving counts as a click (see _click_qubit).
    def mousePressEvent(self, event):
        if self._read_only:
            return
        if event.button() == Qt.LeftButton:
            start = self._node_at(self.mapToScene(event.pos()))
            self._drag_path = [start] if start is not None else None
            self._dragging = False
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_path is not None:
            scene_pos = self.mapToScene(event.pos())
            over = self._node_at(scene_pos)
            path = self._drag_path
            if over is not None and over != path[-1]:
                self._dragging = True
                if len(path) > 1 and over == path[-2]:
                    path.pop()                          # dragged back: undo a step
                elif (over not in path and len(path) < self._n
                      and frozenset((path[-1], over)) in self._edges):
                    path.append(over)
            elif over is None and len(path) == 1:
                self._dragging = True
            if self._dragging:
                # Snap the line's end to a qubit when hovering one.
                end = QPointF(*self._coords[over]) if over is not None else scene_pos
                self._show_drag_line(end)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._drag_path is not None:
            path, self._drag_path = self._drag_path, None
            end = self._node_at(self.mapToScene(event.pos()))
            self._hide_drag_line()
            if not self._dragging:
                self._click_qubit(path[0])
            elif len(path) > 1:
                self._set_path(path)
            elif end is not None and end != path[0]:
                self._set_path(path, problem=f"Qubits {path[0]} and {end} aren't coupled, "
                                              f"so they can't be A1–A2.")
            # Released on empty space with nothing picked: nothing changes.
            self._dragging = False
        super().mouseReleaseEvent(event)

    def _show_drag_line(self, end: QPointF):
        if self._drag_line is None:
            self._drag_line = QGraphicsPathItem()
            self._drag_line.setPen(QPen(QColor(Colors.ACCENT), 3.0, Qt.DashLine, Qt.RoundCap))
            self._drag_line.setZValue(3)
            self.scene().addItem(self._drag_line)
        line = QPainterPath(QPointF(*self._coords[self._drag_path[0]]))
        for q in self._drag_path[1:]:
            line.lineTo(QPointF(*self._coords[q]))
        line.lineTo(end)
        self._drag_line.setPath(line)

    def _hide_drag_line(self):
        if self._drag_line is not None:
            self.scene().removeItem(self._drag_line)
            self._drag_line = None

    def _node_at(self, scene_pos: QPointF):
        for idx, (x, y) in self._coords.items():
            if (scene_pos.x() - x) ** 2 + (scene_pos.y() - y) ** 2 <= _NODE_RADIUS ** 2:
                return idx
        return None

    def _click_qubit(self, qubit: int):
        path = self._a_path
        if not path or len(path) == self._n or (self._problem and not self._problem_is_hint):
            # Nothing picked yet, A complete, or a dead end: start over here.
            self._set_path([qubit])
        elif qubit == path[-1]:
            self._set_path(path[:-1])                   # undo the last pick
        elif qubit in path:
            role = f"A{path.index(qubit) + 1}"
            self._set_path(path, problem=f"Qubit {qubit} is already {role}.", keep=True)
        elif frozenset((path[-1], qubit)) not in self._edges:
            self._set_path(path, keep=True, problem=(
                f"Qubit {qubit} isn't coupled to A{len(path)} (qubit {path[-1]}); "
                f"A{len(path) + 1} must be."))
        else:
            self._set_path(path + [qubit])

    def _set_path(self, path: list, problem: str = None, keep: bool = False):
        """Set the picked A chain; once it's complete, fill in B or record
        why this A can't work. keep=True leaves the pick open after a
        rejected click, so the user can try another qubit."""
        self._a_path = list(path)
        self._assignment = {f"A{i + 1}": q for i, q in enumerate(path)}
        self._problem = problem
        self._problem_is_hint = keep
        if problem is None and len(path) == self._n:
            chain = find_mirror_chain(path, self._edges)
            if chain is None:
                a_txt = "–".join(map(str, path))
                self._problem = (
                    f"Impossible: no qubits on this device mirror A = {a_txt} "
                    f"(needs B_i coupled to each A_i, and B1..B{self._n} coupled in a chain).")
            else:
                self._assignment.update({f"B{i + 1}": q for i, q in enumerate(chain)})
        self._redraw_scene()
        self._emit_state()

    def set_read_only(self, value: bool = True):
        self._read_only = value
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setMinimumHeight(0)
        self._fit()

    def _fit(self):
        if self._read_only and not self.scene().sceneRect().isEmpty():
            self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()

    # ---- drawing ---------------------------------------------------------
    def _redraw_scene(self):
        scene = self.scene()
        scene.clear()
        self._drag_line = None   # deleted along with everything else
        self._node_items.clear()
        self._label_items.clear()

        if not self._coords:
            return

        role_at = {q: r for r, q in self._assignment.items()}
        missing_sets = {frozenset(pair) for pair in self._missing_pairs()}
        required_sets = {frozenset(pair) for pair in required_pairs(self._n)}

        # connectivity edges: muted by default, colored where the ladder
        # needs them (green when coupled, red when the chosen roles aren't).
        for pair in self._edges:
            i, j = tuple(pair)
            if i not in self._coords or j not in self._coords:
                continue
            color, width = QColor(Colors.BORDER), 1.4
            if i in role_at and j in role_at and frozenset((role_at[i], role_at[j])) in required_sets:
                bad = frozenset((role_at[i], role_at[j])) in missing_sets
                color, width = QColor(Colors.BAD if bad else Colors.GOOD), 2.6
            x1, y1 = self._coords[i]
            x2, y2 = self._coords[j]
            line = QGraphicsLineItem(x1, y1, x2, y2)
            line.setPen(QPen(color, width))
            line.setZValue(0)
            scene.addItem(line)

        # required pairs that aren't real edges at all (only reachable via a
        # restored saved mapping), drawn dashed/BAD.
        for r1, r2 in self._missing_pairs():
            q1, q2 = self._assignment[r1], self._assignment[r2]
            if frozenset((q1, q2)) in self._edges or q1 not in self._coords or q2 not in self._coords:
                continue
            line = QGraphicsLineItem(*self._coords[q1], *self._coords[q2])
            line.setPen(QPen(QColor(Colors.BAD), 2.0, Qt.DashLine))
            scene.addItem(line)

        # A complete A chain with no mirror B: mark the chain itself BAD.
        if self._problem and not self._problem_is_hint and len(self._a_path) > 1:
            for q1, q2 in zip(self._a_path, self._a_path[1:]):
                coupled = frozenset((q1, q2)) in self._edges
                line = QGraphicsLineItem(*self._coords[q1], *self._coords[q2])
                line.setPen(QPen(QColor(Colors.BAD), 2.6 if coupled else 2.0,
                                 Qt.SolidLine if coupled else Qt.DashLine))
                line.setZValue(0.5)
                scene.addItem(line)

        # nodes
        for idx, (x, y) in self._coords.items():
            role = role_at.get(idx)
            fill = QColor(_role_color(role)) if role else QColor(Colors.BG_2)
            pen_color = QColor(Colors.TEXT) if role else QColor(Colors.BORDER)
            ellipse = QGraphicsEllipseItem(
                x - _NODE_RADIUS, y - _NODE_RADIUS, _NODE_RADIUS * 2, _NODE_RADIUS * 2)
            ellipse.setBrush(QBrush(fill))
            ellipse.setPen(QPen(pen_color, 1.6))
            ellipse.setZValue(1)
            scene.addItem(ellipse)
            self._node_items[idx] = ellipse

            idx_label = QGraphicsSimpleTextItem(str(idx))
            idx_label.setBrush(QBrush(QColor(Colors.TEXT if role else Colors.TEXT_DIM)))
            idx_label.setZValue(2)
            br = idx_label.boundingRect()
            idx_label.setPos(x - br.width() / 2, y - br.height() / 2 - 8)
            scene.addItem(idx_label)

            if role:
                role_label = QGraphicsSimpleTextItem(role)
                font = role_label.font()
                font.setBold(True)
                role_label.setFont(font)
                role_label.setBrush(QBrush(QColor("#06101C")))
                role_label.setZValue(2)
                br2 = role_label.boundingRect()
                role_label.setPos(x - br2.width() / 2, y - br2.height() / 2 + 6)
                scene.addItem(role_label)

        scene.setSceneRect(scene.itemsBoundingRect().adjusted(-30, -30, 30, 30))
        self._fit()

    def _missing_pairs(self):
        if len(self._assignment) != 2 * self._n:
            return []
        return [(r1, r2) for r1, r2 in required_pairs(self._n)
                if frozenset((self._assignment[r1], self._assignment[r2])) not in self._edges]

    def _emit_state(self):
        n, path = self._n, self._a_path
        if self._problem and not self._problem_is_hint:
            self.status_changed.emit(self._problem + " Pick another A.")
            self.mapping_changed.emit({})
            return

        if not path:
            status = (f"Pick subsystem A: drag a path through {n} coupled qubits "
                      f"(A1 → A{n}), or click them one by one. Subsystem B is chosen "
                      f"automatically.")
        elif len(path) < n:
            picked = ", ".join(f"A{i + 1} = {q}" for i, q in enumerate(path))
            status = (f"{picked}. Click a qubit coupled to qubit {path[-1]} for "
                      f"A{len(path) + 1} (or click qubit {path[-1]} again to undo).")
        elif len(self._assignment) != 2 * n:
            # Only reachable via set_assignment() with a partial saved mapping.
            status = "Incomplete assignment — pick subsystem A again."
        elif self._missing_pairs():
            pairs_txt = ", ".join(f"{r1}–{r2}" for r1, r2 in self._missing_pairs())
            status = f"Invalid: {pairs_txt} must be adjacent on this device."
        else:
            a = "–".join(str(self._assignment[f"A{i}"]) for i in range(1, n + 1))
            b = "–".join(str(self._assignment[f"B{i}"]) for i in range(1, n + 1))
            self.status_changed.emit(
                f"Valid: A = {a}, B = {b} (chosen automatically). Drag or click again "
                f"to pick a different A.")
            self.mapping_changed.emit(dict(self._assignment))
            return
        if self._problem:
            status = self._problem + " " + status
        self.status_changed.emit(status)
        self.mapping_changed.emit({})


# =========================================================================
#  DevicePanel — the widget Setup actually embeds: mode toggle + status line
# =========================================================================

class DevicePanel(QWidget):

    mapping_changed = Signal(dict)

    def __init__(self):
        super().__init__()
        self._backend_name = None
        self._had_selection = False
        self._qx = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        toggle_row = QHBoxLayout()
        self.image_btn = QPushButton("Schematic")
        self.interactive_btn = QPushButton("Interactive")
        for btn in (self.image_btn, self.interactive_btn):
            btn.setCheckable(True)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: {Colors.BG_2}; color: {Colors.TEXT_DIM};
                    border: 1px solid {Colors.BORDER}; border-radius: 8px;
                    padding: 5px 14px; font-weight: 600;
                }}
                QPushButton:checked {{ background: {Colors.ACCENT}; color: #06101C; }}
            """)
        self.interactive_btn.setChecked(True)
        self.image_btn.clicked.connect(lambda: self._set_mode("image"))
        self.interactive_btn.clicked.connect(lambda: self._set_mode("interactive"))
        toggle_row.addWidget(self.image_btn)
        toggle_row.addWidget(self.interactive_btn)
        toggle_row.addStretch()

        self.reset_btn = QPushButton("🔄  Reset selection")
        self.reset_btn.setCursor(Qt.PointingHandCursor)
        self.reset_btn.setMinimumHeight(36)
        self.reset_btn.setStyleSheet(f"""
            QPushButton {{
                background: rgba(254,138,39,0.15); color: {Colors.WARN};
                border: 2px solid {Colors.WARN}; border-radius: 10px;
                padding: 6px 18px; font-size: 11pt; font-weight: bold;
            }}
            QPushButton:hover {{ background: {Colors.WARN}; color: #06101C; }}
            QPushButton:pressed {{ background: #E07415; color: #06101C; }}
        """)
        self.reset_btn.clicked.connect(self.clear_selection)
        toggle_row.addWidget(self.reset_btn)
        root.addLayout(toggle_row)

        stack_holder = QWidget()
        self._stack = QStackedLayout(stack_holder)
        self.image_view = BackendImageView()
        self.topology_view = QubitTopologyWidget()
        self._stack.addWidget(self.topology_view)
        self._stack.addWidget(self.image_view)
        root.addWidget(stack_holder, stretch=1)

        # Device specifications, shown only for backends we have them for
        # (currently Starmon-7 — see backend/starmon7.py).
        self.spec_label = QLabel("")
        self.spec_label.setWordWrap(True)
        self.spec_label.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 9pt; padding: 2px;")
        self.spec_label.hide()
        root.addWidget(self.spec_label)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(f"color: {Colors.TEXT_DIM}; padding: 2px;")
        root.addWidget(self.status_label)

        self.topology_view.status_changed.connect(self.status_label.setText)
        self.topology_view.mapping_changed.connect(self.mapping_changed.emit)
        self.topology_view.mapping_changed.connect(self._note_selection_made)

        self._set_mode("interactive")

    def _set_mode(self, mode: str):
        self.image_btn.setChecked(mode == "image")
        self.interactive_btn.setChecked(mode == "interactive")
        self._stack.setCurrentWidget(self.image_view if mode == "image" else self.topology_view)
        self.reset_btn.setVisible(mode == "interactive" and not self._qx)

    def clear_selection(self):
        self.topology_view.clear_selection()

    def set_backend(self, name: str, qi_backend=None):
        """Called by Setup whenever the backend dropdown changes. For one of
        the known chips (see KNOWN_BACKEND_TOPOLOGY) both the layout
        and the connectivity are the hardcoded real ones — no live lookup,
        no parsing, no guessing. For anything else, `qi_backend` (the live
        QIBackend, if available — see backend_topology) supplies real
        connectivity and a plain grid is used for positions; pass None if it
        couldn't be fetched (offline / not logged in) for a generic
        fully-connected 4-qubit placeholder instead.
        """
        self.image_view.set_backend(name)

        if starmon7.is_starmon7(name):
            self.spec_label.setText(
                f"{starmon7.SPEC_SUMMARY}.\nNative gates: "
                f"{', '.join(starmon7.NATIVE_SINGLE_QUBIT_GATES + starmon7.NATIVE_TWO_QUBIT_GATES)}"
                f"  ·  decomposed: {', '.join(starmon7.DECOMPOSED_GATES)}"
                f"  ·  emulated locally with calibrated noise.")
            self.spec_label.show()
        elif backends.is_qx(name):
            self.spec_label.setText(
                "Quantum Inspire's noiseless cloud simulator: 10 qubits, all connected to "
                "each other, up to 2048 shots per circuit.")
            self.spec_label.show()
        else:
            self.spec_label.hide()

        if self._had_selection and name != self._backend_name:
            self.status_label.setText(
                "Backend changed — qubit selection cleared.")
        self._backend_name = name
        self._qx = backends.is_qx(name)
        self.reset_btn.setVisible(self.interactive_btn.isChecked() and not self._qx)
        if self._qx:
            self._show_qx_ladder(self.topology_view._n)
            self._had_selection = False
            return

        known = _lookup_by_name(name, KNOWN_BACKEND_TOPOLOGY) if name else None
        if known is not None:
            coords = known["coords"]
            num_qubits = len(coords)
            edges = {frozenset(pair) for pair in known["edges"]}
        elif qi_backend is not None:
            num_qubits, edges = backend_topology(qi_backend)
            coords = None
        else:
            num_qubits, edges = 4, {frozenset((i, j)) for i in range(4)
                                     for j in range(4) if i != j}
            coords = None

        self.topology_view._read_only = False
        self.topology_view.set_topology(num_qubits, edges, coords)
        self._had_selection = False
        self.mapping_changed.emit({})

    def current_mapping(self) -> dict:
        return self.topology_view.current_mapping()

    def set_spins(self, n: int):
        if self._qx:
            self._show_qx_ladder(n)
        else:
            self.topology_view.set_spins(n)

    def _show_qx_ladder(self, n: int):
        """QX emulator: no picking. Every qubit is connected, and the run
        always uses backends.qx_mapping (see there why), so that ladder is
        shown, fixed: A on top, B below."""
        view = self.topology_view
        coords, edges = qx_ladder(n)
        view._n = n
        view.set_topology(2 * n, edges, coords)
        view.set_assignment(backends.qx_mapping(n))
        view._read_only = True
        self.status_label.setText(
            f"The QX emulator connects every qubit, so the ladder is fixed: qubits "
            f"0–{2 * n - 1} in the circuit's own order.")

    def set_assignment(self, mapping: dict):
        if not self._qx:   # QX's ladder is fixed (see _show_qx_ladder)
            self.topology_view.set_assignment(mapping)

    def _note_selection_made(self, mapping: dict):
        self._had_selection = bool(mapping)
