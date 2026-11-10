"""Hand-pixel style drawing helpers.

The SDF renderer in render.py is good at rounded, rotating sprites. The
base tiles and other static structures want something closer to how a
1990s arcade artist worked: hard bevels, panel seams, rivets and
hand-placed highlights at 1x. This module provides that:

- hue shifted colour ramps (shadows lean towards blue and violet,
  highlights towards warm yellow, saturation peaks in the mid tones), and
- a small RGBA canvas with mask based drawing operations.

Everything is deterministic.
"""

import colorsys
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Colour ramps
# ---------------------------------------------------------------------------

def _hue_towards(h, target, w):
    """Move hue h (degrees) towards target by fraction w, the short way
    round the colour wheel."""
    d = ((target - h + 180.0) % 360.0) - 180.0
    return (h + d * w) % 360.0


def hramp(hue, sat, n=9, v0=0.10, v1=1.0, shift=0.35, sat_top=0.35,
          gamma=0.9):
    """A hue shifted ramp of n RGB tuples, darkest first.

    hue, sat  key colour (degrees, 0..1)
    v0, v1    value of the darkest and brightest entries
    shift     how far shadows turn towards violet blue and highlights
              towards warm yellow
    sat_top   saturation multiplier at the brightest entry
    """
    out = []
    for i in range(n):
        t = i / float(n - 1)
        h = hue
        h = _hue_towards(h, 245.0, shift * (1 - t) ** 1.6)
        h = _hue_towards(h, 52.0, shift * 0.8 * t ** 2.2)
        s = sat * (0.85 + 0.45 * np.sin(np.pi * min(t * 1.15, 1.0)))
        s = s * (1 - (1 - sat_top) * t ** 3)
        s = float(np.clip(s, 0, 1))
        v = v0 + (v1 - v0) * t ** gamma
        r, g, b = colorsys.hsv_to_rgb(h / 360.0, s, float(np.clip(v, 0, 1)))
        out.append((int(round(r * 255)), int(round(g * 255)),
                    int(round(b * 255))))
    return out


def glow_ramp(hue, n=6, sat=1.0):
    """Emissive ramp: deep saturated colour up to a white hot centre."""
    out = []
    for i in range(n):
        t = i / float(n - 1)
        h = _hue_towards(hue, 55.0, 0.25 * t ** 2)
        s = sat * (1 - 0.85 * t ** 2.5)
        v = 0.35 + 0.65 * t ** 0.7
        r, g, b = colorsys.hsv_to_rgb(h / 360.0, s, v)
        out.append((int(round(r * 255)), int(round(g * 255)),
                    int(round(b * 255))))
    return out


def mix(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------

def shift(m, dx, dy):
    """Shift a boolean mask by (dx, dy) pixels, filling with False."""
    o = np.zeros_like(m)
    h, w = m.shape
    xs0, xs1 = max(0, -dx), min(w, w - dx)
    ys0, ys1 = max(0, -dy), min(h, h - dy)
    if xs1 > xs0 and ys1 > ys0:
        o[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx] = m[ys0:ys1, xs0:xs1]
    return o


def dilate(m, n=1, diag=False):
    for _ in range(n):
        o = m.copy()
        o |= shift(m, 1, 0) | shift(m, -1, 0) | shift(m, 0, 1) | \
            shift(m, 0, -1)
        if diag:
            o |= shift(m, 1, 1) | shift(m, -1, -1) | shift(m, 1, -1) | \
                shift(m, -1, 1)
        m = o
    return m


def erode(m, n=1):
    return ~dilate(~m, n)


def edges(m):
    """Directional edge masks of m: pixels of m whose neighbour on the
    given side is outside m."""
    return dict(
        top=m & ~shift(m, 0, 1),
        bottom=m & ~shift(m, 0, -1),
        left=m & ~shift(m, 1, 0),
        right=m & ~shift(m, -1, 0),
    )


def rect(shape, x0, y0, x1, y1):
    """Inclusive rectangle mask."""
    m = np.zeros(shape, dtype=bool)
    h, w = shape
    x0, x1 = max(int(x0), 0), min(int(x1), w - 1)
    y0, y1 = max(int(y0), 0), min(int(y1), h - 1)
    if x1 >= x0 and y1 >= y0:
        m[y0:y1 + 1, x0:x1 + 1] = True
    return m


def disc(shape, cx, cy, r):
    ys, xs = np.mgrid[0:shape[0], 0:shape[1]]
    return np.hypot(xs + 0.5 - cx, ys + 0.5 - cy) <= r


def octagon(shape, x0, y0, x1, y1, c):
    """Inclusive rectangle with corners cut by c pixels."""
    ys, xs = np.mgrid[0:shape[0], 0:shape[1]]
    m = (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
    m &= (xs - x0) + (ys - y0) >= c
    m &= (x1 - xs) + (ys - y0) >= c
    m &= (xs - x0) + (y1 - ys) >= c
    m &= (x1 - xs) + (y1 - ys) >= c
    return m


# ---------------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------------

class Canvas:
    """RGBA float canvas, values 0..255 for colour and 0..1 for alpha."""

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.rgb = np.zeros((h, w, 3))
        self.a = np.zeros((h, w))

    @property
    def shape(self):
        return (self.h, self.w)

    def fill(self, m, col, alpha=1.0):
        c = np.array(col, float)
        if alpha >= 1.0:
            self.rgb[m] = c
            self.a[m] = 1.0
        else:
            # Composite over what is already there
            a0 = self.a[m]
            na = alpha + a0 * (1 - alpha)
            safe = np.where(na > 0, na, 1)
            self.rgb[m] = (c * alpha + self.rgb[m] * (a0 * (1 - alpha))
                           [..., None]) / safe[..., None]
            self.a[m] = na

    def tint(self, m, col, t):
        """Blend existing opaque colour towards col by t (glows, grime)."""
        m = m & (self.a > 0)
        c = np.array(col, float)
        self.rgb[m] = self.rgb[m] + (c - self.rgb[m]) * t

    def add(self, m, col, t):
        """Additive light on existing pixels."""
        m = m & (self.a > 0)
        self.rgb[m] = np.clip(self.rgb[m] + np.array(col, float) * t, 0, 255)

    def px(self, x, y, col, alpha=1.0):
        if 0 <= x < self.w and 0 <= y < self.h:
            m = np.zeros(self.shape, dtype=bool)
            m[y, x] = True
            self.fill(m, col, alpha)

    def crop(self, x, y, w, h):
        c = Canvas(w, h)
        c.rgb = self.rgb[y:y + h, x:x + w].copy()
        c.a = self.a[y:y + h, x:x + w].copy()
        return c

    def image(self):
        out = np.zeros((self.h, self.w, 4), dtype=np.uint8)
        out[..., :3] = np.clip(np.round(self.rgb), 0, 255)
        out[..., 3] = np.clip(np.round(self.a * 255), 0, 255)
        out[out[..., 3] == 0] = 0
        return Image.fromarray(out, "RGBA")


def downscale(img, size):
    """Shrink a tile with premultiplied box filtering, then snap alpha so
    solid structure stays solid and only shadows remain translucent."""
    a = np.array(img).astype(float)
    al = a[..., 3:4] / 255.0
    pm = a.copy()
    pm[..., :3] *= al
    # Resize colour and alpha planes separately (Pillow would otherwise
    # premultiply again)
    planes = [Image.fromarray(pm[..., i].astype(np.uint8), "L")
              .resize((size, size), Image.BOX) for i in range(4)]
    r = np.stack([np.array(p).astype(float) for p in planes], axis=-1)
    alpha = r[..., 3] / 255.0
    safe = np.where(alpha > 0, alpha, 1)
    rgb = r[..., :3] / safe[..., None]
    out = np.zeros((size, size, 4), dtype=np.uint8)
    out[..., :3] = np.clip(np.round(rgb), 0, 255)
    solid = alpha >= 0.6
    out[..., 3] = np.where(solid, 255, np.round(alpha * 0.8 * 255))
    out[alpha < 0.08] = 0
    return Image.fromarray(out, "RGBA")
