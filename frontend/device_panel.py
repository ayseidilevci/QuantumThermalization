"""Setup screen's device panel: a toggle between a static chip-schematic
image (ported from the `gui` branch's tfd_gui.py BackendImageView /
BACKEND_IMAGES) and a new interactive qubit-topology widget that lets the
user pick subsystem A (A1/A2) of the TFD ansatz by dragging or clicking,
with the mirrored subsystem B (B1/B2) filled in automatically.

Real chips are fixed, published hardware, not something to re-derive at
runtime — so for every backend we have a reference schematic for (Tuna-5,
Tuna-9, Tuna-17, Starmon-7), both the on-screen node *coordinates* and the real
*connectivity* are hardcoded directly in KNOWN_BACKEND_TOPOLOGY below,
read straight off the schematics in Appdata/Img/, rather than parsed from
the SVG at runtime or fetched from the live API. (An earlier version tried
both: parsing qubit-label positions out of the SVG at runtime, and a
generic graph-layout algorithm for anything else — both were more clever
than they needed to be and each introduced its own bugs. Direct, boring
data wins here.)

Only a backend that ISN'T one of the known chips (an unfamiliar/live
backend name we don't ship a schematic for) falls back to querying the live
QIBackend's own coupling map for connectivity (see backend_topology) and a
plain row-major grid for positions (see simple_grid_layout) — still no
fancy layout tricks, just a readable default for the unknown case.
"""

from math import ceil, sqrt
from pathlib import Path

from PySide6.QtCore import Qt, Signal, QSize, QPointF
from PySide6.QtGui import QColor, QPixmap, QImage, QPainter, QPen, QBrush
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QStackedLayout,
    QSizePolicy, QGraphicsView, QGraphicsScene, QGraphicsEllipseItem,
    QGraphicsLineItem, QGraphicsSimpleTextItem,
)

from backend import starmon7
from frontend.colors import Colors

ROLES = ("A1", "A2", "B1", "B2")
REQUIRED_ADJACENT_PAIRS = (("A1", "A2"), ("B1", "B2"), ("A1", "B1"), ("A2", "B2"))

# Roles A1/A2 (subsystem A) vs B1/B2 (subsystem B) are color-coded separately
# so the two mirrored halves of the ansatz are visually distinguishable.
_ROLE_COLOR = {
    "A1": Colors.ACCENT, "A2": Colors.ACCENT,
    "B1": Colors.WARN, "B2": Colors.WARN,
}


# =========================================================================
#  Image mode — real backend chip schematics (Appdata/Img/), ported from
#  gui.py's BackendImageView/BACKEND_IMAGES and restyled to Colors.
# =========================================================================

DATA_DIR = Path(__file__).resolve().parent.parent / "Appdata" / "Img"

# Keys are matched against the live backend name case-insensitively/fuzzily
# (see _lookup_by_name) since the exact string the QI API returns for a given
# chip isn't guaranteed to match these display labels verbatim.
BACKEND_IMAGES = {
    "tuna5": DATA_DIR / "1766393003-tuna-5_schematic.svg",
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
# Tuna-5: an X — hub qubit 2 coupled to each of the 4 others.
# Tuna-9 / Tuna-17: a diamond/checkerboard lattice — every qubit couples to
# its diagonal neighbors one lattice-step away (see the schematics).
# Starmon-7: two rhombi (0-2-5-3 and 1-3-6-4) sharing the central qubit 3;
# edges come from backend/starmon7.py.
KNOWN_BACKEND_TOPOLOGY = {
    "tuna5": {
        "coords": {
            0: (-51, -87),
            1: (52, -87),
            2: (0, -28),
            3: (-52, 88),
            4: (52, 88),
        },
        "edges": [(2, 0), (2, 1), (2, 3), (2, 4)],
    },
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
}


def simple_grid_layout(num_qubits: int, spacing: float = 90.0):
    """Fallback node layout for a backend we have no real coordinates for:
    a plain row-major grid. Deliberately not graph-aware — an earlier
    version of this tried a spectral (Laplacian eigenvector) embedding to
    make connected qubits land next to each other, which is correct for a
    clean 2D lattice but turns into a dense, unreadable tangle on anything
    more connected (a simulator's near-complete coupling, say). A plain grid
    stays readable regardless of how many edges there are.
    """
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
        # Qt's SVG renderer already draws these schematics' <text> labels
        # (qubit indices, "TC" couplers) natively — an earlier version of
        # this widget re-drew them manually on the assumption QtSvg couldn't,
        # which just doubled every label. Render straight through instead.
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


def find_mirror_pair(a1: int, a2: int, edges: set):
    """Subsystem B for a chosen subsystem A: qubits (B1, B2), distinct from
    A1/A2, with A1–B1, A2–B2 and B1–B2 all coupled — the mirrored half the
    TFD ansatz needs. Returns None if the device has no such pair. If
    several exist, the lowest-numbered one is taken, so the same A always
    gives the same B."""
    def neighbors(q):
        return sorted(next(iter(e - {q})) for e in edges if q in e)

    for b1 in neighbors(a1):
        if b1 in (a1, a2):
            continue
        for b2 in neighbors(a2):
            if b2 in (a1, a2, b1):
                continue
            if frozenset((b1, b2)) in edges:
                return b1, b2
    return None


class QubitTopologyWidget(QGraphicsView):
    """The user picks subsystem A only; subsystem B is filled in
    automatically (see find_mirror_pair). Two ways to pick A:

    - drag: press on a qubit (A1), drag a line to a coupled qubit (A2) and
      release;
    - click: click A1, then A2, one by one (clicking A1 again un-picks it;
      clicking any qubit once A is complete starts a new pick from there).

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

        self._num_qubits = 0
        self._edges = set()
        self._coords = {}
        self._assignment = {}          # role -> qubit index
        self._problem = None           # why the picked A has no valid B, if so
        self._node_items = {}          # qubit index -> QGraphicsEllipseItem
        self._label_items = {}         # qubit index -> QGraphicsSimpleTextItem (role badge)

        self._drag_from = None         # qubit a press started on
        self._drag_line = None         # rubber-band line while dragging
        self._dragging = False         # moved off the start qubit since press

    # ---- public API --------------------------------------------------
    def set_topology(self, num_qubits: int, edges: set, coords: dict = None):
        self._num_qubits = num_qubits
        self._edges = edges
        self._coords = coords if coords else simple_grid_layout(num_qubits)
        self._assignment = {}
        self._problem = None
        self._redraw_scene()
        self._emit_state()

    def current_mapping(self) -> dict:
        return dict(self._assignment) if len(self._assignment) == len(ROLES) else {}

    def clear_selection(self):
        """Drop the whole role assignment (same end state as clicking A1
        twice)."""
        self._assignment = {}
        self._problem = None
        self._redraw_scene()
        self._emit_state()

    def set_assignment(self, mapping: dict):
        """Restore a role -> qubit assignment directly (used when loading a
        saved session), skipping any role/qubit that no longer fits this
        topology (or collides with another role) rather than raising."""
        valid = {}
        used_qubits = set()
        for role in ROLES:
            qubit = mapping.get(role)
            if qubit is None or qubit not in self._coords or qubit in used_qubits:
                continue
            valid[role] = qubit
            used_qubits.add(qubit)
        self._assignment = valid
        self._problem = None
        self._redraw_scene()
        self._emit_state()

    # ---- interaction ---------------------------------------------------
    # A press on a qubit starts a potential drag. Releasing on a different
    # qubit picks A1 (start) -> A2 (end); releasing on the same qubit counts
    # as a click (see _click_qubit).
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_from = self._node_at(self.mapToScene(event.pos()))
            self._dragging = False
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_from is not None:
            scene_pos = self.mapToScene(event.pos())
            over = self._node_at(scene_pos)
            if over != self._drag_from:
                self._dragging = True
            if self._dragging:
                # Snap the line's end to a qubit when hovering one.
                end = QPointF(*self._coords[over]) if over is not None else scene_pos
                self._show_drag_line(end)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._drag_from is not None:
            start, end = self._drag_from, self._node_at(self.mapToScene(event.pos()))
            self._drag_from = None
            self._hide_drag_line()
            if end == start and not self._dragging:
                self._click_qubit(start)
            elif end is not None and end != start:
                self._pick_subsystem_a(start, end)
            # Released on empty space: drag cancelled, nothing changes.
            self._dragging = False
        super().mouseReleaseEvent(event)

    def _show_drag_line(self, end: QPointF):
        if self._drag_line is None:
            self._drag_line = QGraphicsLineItem()
            self._drag_line.setPen(QPen(QColor(Colors.ACCENT), 3.0, Qt.DashLine, Qt.RoundCap))
            self._drag_line.setZValue(3)
            self.scene().addItem(self._drag_line)
        x, y = self._coords[self._drag_from]
        self._drag_line.setLine(x, y, end.x(), end.y())

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
        a1, a2 = self._assignment.get("A1"), self._assignment.get("A2")
        if a1 is None or a2 is not None:
            # Nothing picked yet, or A already complete: start over here.
            self._assignment = {"A1": qubit}
            self._problem = None
        elif qubit == a1:
            self._assignment = {}
            self._problem = None
        else:
            self._pick_subsystem_a(a1, qubit)
            return
        self._redraw_scene()
        self._emit_state()

    def _pick_subsystem_a(self, a1: int, a2: int):
        """Set A1/A2 and fill in B automatically, or record why that A
        can't work on this device."""
        self._assignment = {"A1": a1, "A2": a2}
        self._problem = None
        if frozenset((a1, a2)) not in self._edges:
            self._problem = f"Impossible: qubits {a1} and {a2} aren't coupled, so they can't be A1–A2."
        else:
            pair = find_mirror_pair(a1, a2, self._edges)
            if pair is None:
                self._problem = (
                    f"Impossible: no qubit pair on this device mirrors A = {a1}–{a2} "
                    f"(needs B1 next to {a1}, B2 next to {a2}, and B1–B2 coupled).")
            else:
                self._assignment["B1"], self._assignment["B2"] = pair
        self._redraw_scene()
        self._emit_state()

    # ---- drawing ---------------------------------------------------------
    def _redraw_scene(self):
        scene = self.scene()
        scene.clear()
        self._drag_line = None   # deleted along with everything else
        self._node_items.clear()
        self._label_items.clear()

        if not self._coords:
            return

        missing_pairs = self._missing_pairs()
        complete = len(self._assignment) == len(ROLES)
        required_sets = {frozenset(pair) for pair in REQUIRED_ADJACENT_PAIRS}
        missing_sets = {frozenset(pair) for pair in missing_pairs}

        # connectivity edges: muted by default, highlighted once a full
        # (valid or invalid) assignment exists.
        for pair in self._edges:
            i, j = tuple(pair)
            if i not in self._coords or j not in self._coords:
                continue
            role_i = next((r for r, q in self._assignment.items() if q == i), None)
            role_j = next((r for r, q in self._assignment.items() if q == j), None)
            color = QColor(Colors.BORDER)
            width = 1.4
            if role_i and role_j and frozenset((role_i, role_j)) in required_sets:
                is_missing = frozenset((role_i, role_j)) in missing_sets
                color = QColor(Colors.BAD if is_missing else Colors.GOOD)
                width = 2.6
            x1, y1 = self._coords[i]
            x2, y2 = self._coords[j]
            line = QGraphicsLineItem(x1, y1, x2, y2)
            line.setPen(QPen(color, width))
            line.setZValue(0)
            scene.addItem(line)

        # required pairs that aren't real edges at all (drawn dashed/BAD so
        # the user can see what's missing even when nothing connects them).
        if complete:
            for r1, r2 in missing_pairs:
                q1, q2 = self._assignment[r1], self._assignment[r2]
                if frozenset((q1, q2)) in self._edges:
                    continue  # already drawn above as a real, invalid edge
                if q1 not in self._coords or q2 not in self._coords:
                    continue
                x1, y1 = self._coords[q1]
                x2, y2 = self._coords[q2]
                pen = QPen(QColor(Colors.BAD), 2.0, Qt.DashLine)
                line = QGraphicsLineItem(x1, y1, x2, y2)
                line.setPen(pen)
                scene.addItem(line)

        # A picked but impossible: if A1/A2 aren't even coupled, show the
        # requested link dashed/BAD; otherwise the A1–A2 edge itself is
        # marked BAD (there's just no mirror B for it).
        a1, a2 = self._assignment.get("A1"), self._assignment.get("A2")
        if self._problem and a1 is not None and a2 is not None:
            x1, y1 = self._coords[a1]
            x2, y2 = self._coords[a2]
            coupled = frozenset((a1, a2)) in self._edges
            line = QGraphicsLineItem(x1, y1, x2, y2)
            line.setPen(QPen(QColor(Colors.BAD), 2.6 if coupled else 2.0,
                             Qt.SolidLine if coupled else Qt.DashLine))
            line.setZValue(0.5)
            scene.addItem(line)

        # nodes
        for idx, (x, y) in self._coords.items():
            role = next((r for r, q in self._assignment.items() if q == idx), None)
            fill = QColor(_ROLE_COLOR[role]) if role else QColor(Colors.BG_2)
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

    def _missing_pairs(self):
        if len(self._assignment) != len(ROLES):
            return []
        missing = []
        for r1, r2 in REQUIRED_ADJACENT_PAIRS:
            q1, q2 = self._assignment[r1], self._assignment[r2]
            if frozenset((q1, q2)) not in self._edges:
                missing.append((r1, r2))
        return missing

    def _emit_state(self):
        if self._problem:
            self.status_changed.emit(self._problem + " Pick another A.")
            self.mapping_changed.emit({})
            return

        if "A1" not in self._assignment:
            self.status_changed.emit(
                "Pick subsystem A: drag a line from A1 to a coupled qubit (A2), "
                "or click A1 then A2. Subsystem B is chosen automatically.")
            self.mapping_changed.emit({})
            return
        if "A2" not in self._assignment:
            self.status_changed.emit(
                f"A1 = qubit {self._assignment['A1']}. Click a coupled qubit for A2 "
                f"(or click A1 again to undo).")
            self.mapping_changed.emit({})
            return
        if len(self._assignment) != len(ROLES):
            # Only reachable via set_assignment() with a partial saved mapping.
            self.status_changed.emit("Incomplete assignment — pick subsystem A again.")
            self.mapping_changed.emit({})
            return

        missing = self._missing_pairs()
        if missing:
            pairs_txt = ", ".join(f"{r1}–{r2}" for r1, r2 in missing)
            self.status_changed.emit(f"Invalid: {pairs_txt} must be adjacent on this device.")
            self.mapping_changed.emit({})
        else:
            a = self._assignment
            self.status_changed.emit(
                f"Valid: A = {a['A1']}–{a['A2']}, B = {a['B1']}–{a['B2']} (chosen "
                f"automatically). Drag or click again to pick a different A.")
            self.mapping_changed.emit(dict(self._assignment))


# =========================================================================
#  DevicePanel — the widget Setup actually embeds: mode toggle + status line
# =========================================================================

class DevicePanel(QWidget):

    mapping_changed = Signal(dict)

    def __init__(self):
        super().__init__()
        self._backend_name = None
        self._had_selection = False

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
        self.reset_btn.setVisible(mode == "interactive")

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
        else:
            self.spec_label.hide()

        if self._had_selection and name != self._backend_name:
            self.status_label.setText(
                "Backend changed — qubit selection cleared.")
        self._backend_name = name

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

        self.topology_view.set_topology(num_qubits, edges, coords)
        self._had_selection = False
        self.mapping_changed.emit({})

    def current_mapping(self) -> dict:
        return self.topology_view.current_mapping()

    def set_assignment(self, mapping: dict):
        self.topology_view.set_assignment(mapping)

    def _note_selection_made(self, mapping: dict):
        self._had_selection = bool(mapping)
