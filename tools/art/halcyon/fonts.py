"""SoFont sheets built from the Halcyon 5x7 font.

SoFont format, as read by src/gfxengine/sofont.cpp: the top pixel row holds
magenta marker pixels, one at the start of each glyph cell, starting with
'!' (33). Glyph pixels live below the marker row. The bottom-left pixel must
be the transparent background.
"""

import numpy as np
from PIL import Image

from .fontdata import G, CHARS
from .palette import rgb

MARK = (255, 0, 255, 255)


def bitmap(ch):
    return np.array([[c == "#" for c in row] for row in G[ch]], dtype=bool)


def scale2x(b):
    h, w = b.shape
    o = np.zeros((h * 2, w * 2), dtype=bool)
    p = np.pad(b, 1)
    for y in range(h):
        for x in range(w):
            P = p[y + 1, x + 1]
            A, B, C, D = p[y, x + 1], p[y + 1, x + 2], p[y + 1, x], \
                p[y + 2, x + 1]
            e0 = e1 = e2 = e3 = P
            if C == A and C != D and A != B:
                e0 = A
            if A == B and A != C and B != D:
                e1 = B
            if D == C and D != B and C != A:
                e2 = C
            if B == D and B != A and D != C:
                e3 = D
            o[2 * y, 2 * x] = e0
            o[2 * y, 2 * x + 1] = e1
            o[2 * y + 1, 2 * x] = e2
            o[2 * y + 1, 2 * x + 1] = e3
    return o


def _dilate(m):
    o = m.copy()
    o[1:, :] |= m[:-1, :]
    o[:-1, :] |= m[1:, :]
    o[:, 1:] |= m[:, :-1]
    o[:, :-1] |= m[:, 1:]
    return o


def styled(b, grad, outline=None, shadow=None, pad=1):
    """Paint bitmap b with a vertical gradient, then outline and shadow.
    Returns an RGBA array with 'pad' pixels of margin on each side."""
    h, w = b.shape
    H, W = h + 2 * pad + 1, w + 2 * pad + 1
    ink = np.zeros((H, W), dtype=bool)
    ink[pad:pad + h, pad:pad + w] = b
    out = np.zeros((H, W, 4), dtype=np.uint8)
    if shadow is not None:
        sh = np.zeros_like(ink)
        sh[1:, 1:] = ink[:-1, :-1]
        if outline is not None:
            sh = _dilate(sh)
        out[sh] = shadow + (255,)
    if outline is not None:
        ol = _dilate(ink)
        out[ol] = outline + (255,)
    for y in range(H):
        t = (y - pad) / max(h - 1, 1)
        k = int(np.clip(round(t * (len(grad) - 1)), 0, len(grad) - 1))
        row = ink[y]
        out[y, row] = grad[k] + (255,)
    return out


def build(cells, height, gap=0):
    """Assemble glyph cells (RGBA arrays) into a SoFont sheet."""
    width = sum(c.shape[1] + 1 + gap for c in cells)
    sheet = np.zeros((height, width, 4), dtype=np.uint8)
    x = 0
    for c in cells:
        sheet[0, x] = MARK
        h = min(c.shape[0], height - 1)
        sheet[1:1 + h, x + 1:x + 1 + c.shape[1]] = c[:h]
        x += c.shape[1] + 1 + gap
    sheet[height - 1, 0] = (0, 0, 0, 0)
    return Image.fromarray(sheet, "RGBA")


def microfont():
    # Plain bright text with a one pixel drop shadow; 9 px tall sheet.
    grad = [rgb(7), rgb(7), rgb(6), rgb(6)]
    cells = []
    for ch in CHARS:
        cells.append(styled(bitmap(ch), grad, shadow=rgb(2), pad=0))
    return build(cells, 9)


def smallfont():
    # Outlined, gradient filled; 11 px tall sheet.
    grad = [rgb(8), rgb(7), rgb(7), rgb(6), rgb(6), rgb(5), rgb(5)]
    cells = [styled(bitmap(ch), grad, outline=rgb(1), pad=1)[:, :-1]
             for ch in CHARS]
    return build(cells, 11)


def bigfont():
    # Scale2x smoothed, 1990s arcade gradient: white hot top through gold
    # and orange to a red base, heavy outline and drop shadow.
    fire = [(255, 255, 238), (255, 246, 190), (255, 228, 120),
            (255, 204, 70), (255, 176, 40), (252, 146, 30), (240, 112, 24),
            (222, 80, 20), (196, 52, 18), (160, 30, 20)]
    grad = [fire[0], fire[1], fire[2], fire[2], fire[3], fire[3], fire[4],
            fire[5], fire[5], fire[6], fire[7], fire[8], fire[8], fire[9]]
    cells = []
    for ch in CHARS:
        b = scale2x(bitmap(ch))
        cells.append(styled(b, grad, outline=(40, 8, 12), shadow=rgb(1),
                            pad=1))
    return build(cells, 19)


def counterfont():
    # Digits only, 4x scaled with two Scale2x passes, amber to orange.
    grad = [rgb(20), rgb(20), rgb(19), rgb(19), rgb(19), rgb(18), rgb(18),
            rgb(17), rgb(17), rgb(16)]
    cells = []
    for ch in CHARS:
        if ch > "9":
            break
        if ch < "0":
            cells.append(np.zeros((1, 1, 4), dtype=np.uint8))
            continue
        b = scale2x(scale2x(bitmap(ch)))
        cells.append(styled(b, grad, outline=rgb(14), shadow=rgb(1), pad=1))
    return build(cells, 33)
