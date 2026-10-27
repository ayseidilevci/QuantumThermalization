"""LaTeX-looking math for QLabel rich text.

Qt has no math typesetting, so `$...$` (inline) and `$$...$$` (display)
spans are rendered with matplotlib's mathtext in Computer Modern — the
LaTeX font — to PNGs in a temp-dir cache, and swapped for <img> tags that
QLabel's rich text loads from disk. Only mathtext's LaTeX subset is
supported (no \\begin{...} environments), see
https://matplotlib.org/stable/users/explain/text/mathtext.html.
"""

import hashlib
import math
import re
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import MathTextParser
from PySide6.QtGui import QFont, QFontInfo, QGuiApplication, QImage

from frontend.colors import Colors

_CACHE_DIR = Path(tempfile.gettempdir()) / "tfd_gui_math"

# Computer Modern has a smaller x-height than the UI's sans-serif font, so
# at the same nominal size inline math looks too small next to the text.
_INLINE_SIZE = 1.12
_DISPLAY_SIZE = 1.3

# Ratio of the UI font's x-height to its pixel size; used to line the math
# baseline up with the text baseline (see _render).
_X_HEIGHT = 0.52

# Fixed line height of prose, as a multiple of the font's pixel size, so
# lines with inline math aren't visibly taller than lines without.
_LINE_HEIGHT = 1.75

_DISPLAY = re.compile(r"\$\$(.+?)\$\$", re.S)
_INLINE = re.compile(r"\$(.+?)\$([.,;?!)]*)", re.S)
_parser = MathTextParser("agg")


def font_px(point_size: float) -> int:
    """Pixel size Qt uses for a font of `point_size`, to match math to text."""
    font = QFont()
    font.setPointSizeF(point_size)
    return QFontInfo(font).pixelSize()


def _pixel_ratio() -> int:
    screen = QGuiApplication.primaryScreen()
    return max(1, math.ceil(screen.devicePixelRatio())) if screen else 2


@lru_cache(maxsize=None)
def _render(tex: str, px: float, text_px: float, color: str) -> tuple[str, int, int]:
    """Render one formula; returns (png path, logical width, logical height).

    Rendered at the screen's pixel ratio n and saved as name@nx.png, which
    Qt's rich text picks up automatically, so it's drawn 1:1 — Qt draws
    scaled images unsmoothed, which drops thin strokes like minus signs.

    The image is cropped to the ink, then padded vertically so that, with
    Qt's `vertical-align: middle` (which centres an image xHeight/4 above
    the text baseline), the formula's baseline lands on the text baseline.
    """
    n = _pixel_ratio()
    key = hashlib.sha1(f"{tex}|{px}|{text_px}|{color}|{n}".encode()).hexdigest()[:16]
    path = _CACHE_DIR / f"{key}.png"
    file = path if n == 1 else _CACHE_DIR / f"{key}@{n}x.png"

    parsed = _parser.parse(
        f"${tex}$", dpi=72 * n,
        prop=FontProperties(size=px, math_fontfamily="cm"))
    alpha = np.asarray(parsed.image, dtype=np.uint8)
    baseline = alpha.shape[0] - int(round(parsed.depth))
    ink = np.nonzero(alpha.max(axis=1))[0]
    top = int(ink.min()) if ink.size else 0
    bottom = max(int(ink.max()) + 1 if ink.size else baseline, baseline)
    alpha = alpha[top:bottom]
    ascent, depth = baseline - top, bottom - baseline

    # Want ascent == depth + half_x, then whole logical pixels.
    half_x = int(round(_X_HEIGHT * text_px * n / 2))
    pad_top = max(0, depth + half_x - ascent)
    pad_bottom = max(0, ascent - depth - half_x)
    extra = -(alpha.shape[0] + pad_top + pad_bottom) % n
    pad_top += extra // 2
    pad_bottom += extra - extra // 2
    pad_x = n + (-alpha.shape[1] % n)
    alpha = np.pad(alpha, ((pad_top, pad_bottom), (n, pad_x)))

    h, w = alpha.shape
    rgba = np.empty((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = [int(c * 255) for c in to_rgb(color)]
    rgba[..., 3] = alpha

    if not file.exists():
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBA8888).save(str(file))
    return path.as_posix(), w // n, h // n


def _img(tex: str, px: float, text_px: float, color: str,
         max_width: int | None = None) -> str:
    src, w, h = _render(tex, px, text_px, color)
    if max_width and w > max_width:
        # Typeset smaller rather than let Qt scale the image (see _render).
        src, w, h = _render(tex, px * max_width / w, text_px, color)
    return f'<img src="{src}" width="{w}" height="{h}" style="vertical-align: middle">'


def tex_html(text: str, px: int, color: str = Colors.TEXT, max_width: int | None = None) -> str:
    """Replace the $...$ / $$...$$ spans in `text` (Qt rich text) with
    rendered images, for text drawn at `px` pixels (see font_px). Display
    formulas wider than `max_width` pixels are typeset smaller to fit."""

    def inline(match):
        # Qt ignores white-space: nowrap on spans, so trailing punctuation is
        # typeset inside the formula (braced, so mathtext doesn't space it as an
        # operator) to keep it from wrapping onto a line of its own.
        punctuation = "".join(f"{{{c}}}" for c in match.group(2))
        return _img(match.group(1).strip() + punctuation, px * _INLINE_SIZE, px, color)

    # Every prose run and every display formula gets its own block: Qt's
    # HTML parser otherwise lets prose that starts with an <img> after a
    # display formula's </p> spill into that centred paragraph.
    line_height = round(px * _LINE_HEIGHT)
    blocks = []
    for i, part in enumerate(_DISPLAY.split(text)):
        if i % 2:
            blocks.append(f'<p align="center" style="margin-top: 8px; margin-bottom: 8px">'
                          f'{_img(part.strip(), px * _DISPLAY_SIZE, px, color, max_width)}</p>')
        elif part.strip():
            blocks.append(f'<p style="margin: 0; line-height: {line_height}px; '
                          f'-qt-line-height-type: fixed">{_INLINE.sub(inline, part)}</p>')
    return "".join(blocks)
