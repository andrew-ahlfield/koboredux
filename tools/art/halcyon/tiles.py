"""Base tiles for the five Halcyon regions: armoured hull structures.

Kobo bases are mazes laid out on a two cell grid: junction cells sit on
the even positions, straight connecting cells between them, and the cells
in between are always empty. Drawn as thin corridors, such a maze reads as a
tangle of pipes. Halcyon instead draws every cell as a broad slab of hull
that fills its tile edge to edge wherever it connects, so a run of cells
reads as one continuous armoured deck and the empty cells read as the
trenches between hull blocks:

- Straight runs are low, darker wall walk decks: two armour plates per
  cell, a bulkhead joint where cells meet and a narrow sunken rail
  channel (the "pipe" is a groove in the hull, not the hull itself).
- Corners, three way and four way junctions carry towers: a bright roof
  high above the decks with a wall of lit windows dropping to deck level
  and a shadow on the deck, topped by fans, domes, stacks or dishes. Junction
  heavy clusters read as a fortress of towers joined by wall walks.
- Closed sides stop just short of the tile edge with chamfered corners;
  the south side shows a tall wall, so the structure has height.

Tile index layout (8 x 7 sheet), matching the engine's map.cpp:
  0         four-way junction, landing pad variant
  1-15      cells by connection mask (U=1 R=2 D=4 L=8);
            single-bit masks (1, 2, 4, 8) are the destructible cannons
  16-23     core, connected vertically (8 frame pulse)
  24-31     core, connected horizontally (8 frame pulse)
  32-35     armoured (indestructible) end caps U, R, D, L
  40-45     straight run variants (even vertical, odd horizontal)
  48-55     wreckage left after destruction (U, U, R, R, D, D, L, L)

The 16 px set is drawn in full detail. The 12 and 8 px sets for the
parallax layers are drawn separately as distant silhouettes (dark deck
masses, bright tower blocks with lit windows, no line detail) and box
filtered down.
"""

import math
import numpy as np
from PIL import Image

from .pix import (Canvas, hramp, glow_ramp, shift, dilate, erode,
                  edges, rect, disc, octagon, downscale)
from .render import sheet

U, R, D, L = 1, 2, 4, 8
T = 16          # tile size
P = 16          # padding around the tile on the work canvas
N = T + 2 * P   # work canvas size
WALL = 3        # height of the deck's south wall in pixels
X0, X1 = 1, 14  # deck columns on a closed side
Y0, Y1 = 1, 11  # deck rows on a closed side (the wall hangs below)

SHADOW = (6, 4, 18)
HAZARD_DARK = (24, 20, 26)


def _region(hue, sat, v0, v1, shift_, accent, light, core, stripe,
            trench_hue=None):
    metal = hramp(hue, sat, 10, v0=v0, v1=v1, shift=shift_)
    th = hue if trench_hue is None else trench_hue
    return dict(
        metal=metal,
        outline=hramp(th, min(1, sat + 0.25), 3, v0=0.03, v1=0.12,
                      shift=0.5)[1],
        accent=accent,
        light=light,
        core=core,
        stripe=stripe,
    )


# Five material identities. Ramps are 10 shades, darkest first.
REGIONS = [
    # 1: naval gunmetal, cyan lights, amber hazard paint, cyan reactor
    _region(212, 0.22, 0.10, 0.96, 0.30,
            accent=hramp(36, 0.85, 6, v0=0.30, v1=1.0, shift=0.2),
            light=glow_ramp(185),
            core=glow_ramp(190),
            stripe=hramp(44, 0.95, 4, v0=0.55, v1=1.0, shift=0.1)),
    # 2: desert fortress, sand armour, rust trim, green lights, fire core
    _region(40, 0.46, 0.09, 0.98, 0.22,
            accent=hramp(10, 0.75, 6, v0=0.22, v1=0.9, shift=0.25),
            light=glow_ramp(120),
            core=glow_ramp(14),
            stripe=hramp(4, 0.85, 4, v0=0.45, v1=0.95, shift=0.1)),
    # 3: jungle military, olive drab, yellow hazard, orange lights, toxic
    #    green reactor
    _region(84, 0.40, 0.08, 0.95, 0.30,
            accent=hramp(50, 0.9, 6, v0=0.28, v1=1.0, shift=0.15),
            light=glow_ramp(24),
            core=glow_ramp(95),
            stripe=hramp(52, 0.95, 4, v0=0.55, v1=1.0, shift=0.1)),
    # 4: alien crimson, plum armour, gold trim, hot pink lights, violet
    #    reactor
    _region(330, 0.48, 0.08, 0.92, 0.30,
            accent=hramp(44, 0.8, 6, v0=0.28, v1=1.0, shift=0.2),
            light=glow_ramp(312),
            core=glow_ramp(275),
            stripe=hramp(46, 0.85, 4, v0=0.55, v1=1.0, shift=0.1)),
    # 5: ice fortress, white ceramic, red markings, red lights, amber core
    _region(212, 0.20, 0.14, 1.00, 0.12,
            accent=hramp(356, 0.85, 6, v0=0.30, v1=0.95, shift=0.15),
            light=glow_ramp(354),
            core=glow_ramp(38),
            stripe=hramp(356, 0.9, 4, v0=0.50, v1=0.95, shift=0.1)),
]


# ---------------------------------------------------------------------------
# Geometry helpers (tile coordinates; the work canvas is offset by P)
# ---------------------------------------------------------------------------

def R_(x0, y0, x1, y1):
    return rect((N, N), x0 + P, y0 + P, x1 + P, y1 + P)


def OCT(x0, y0, x1, y1, c):
    return octagon((N, N), x0 + P, y0 + P, x1 + P, y1 + P, c)


def DISC(cx, cy, r):
    return disc((N, N), cx + P, cy + P, r)


def TILE():
    return R_(0, 0, T - 1, T - 1)


_YS, _XS = np.mgrid[0:N, 0:N]
XS = _XS - P
YS = _YS - P


def deck_mask(mask, chamfer=3, inset=0):
    """Top surface of a hull cell. Connected sides run far into the
    padding so neighbouring cells join seamlessly; closed sides stop at
    the deck margin, and a corner between two closed sides is chamfered."""
    x0, x1 = X0 + inset, X1 - inset
    y0, y1 = Y0 + inset, Y1 - inset
    m = R_(x0, y0, x1, y1)
    if mask & U:
        m |= R_(x0, -P, x1, y1)
    if mask & D:
        m |= R_(x0, y0, x1, T + P)
    if mask & L:
        m |= R_(-P, y0, x1, y1)
    if mask & R:
        m |= R_(x0, y0, T + P, y1)
    if chamfer:
        c = chamfer
        if not mask & (U | L):
            m &= ~(((XS - x0) + (YS - y0)) < c)
        if not mask & (U | R):
            m &= ~(((x1 - XS) + (YS - y0)) < c)
        if not mask & (D | L):
            m &= ~(((XS - x0) + (y1 - YS)) < c)
        if not mask & (D | R):
            m &= ~(((x1 - XS) + (y1 - YS)) < c)
    return m


# ---------------------------------------------------------------------------
# Painting
# ---------------------------------------------------------------------------

class Tile:
    def __init__(self, rg):
        self.rg = rg
        self.m = rg["metal"]
        self.c = Canvas(N, N)
        self.solid = np.zeros((N, N), dtype=bool)

    # Shading primitives -----------------------------------------------

    def block(self, top, height, base=5, cast=(1, 1), outline=True,
              ribs=True, ramp=None, bevel=2):
        """Extrude a top surface mask: shadow, south wall, bevelled top
        and outline."""
        m0 = self.m if ramp is None else ramp
        m = lambda i: m0[max(0, min(i, len(m0) - 1))]  # noqa: E731
        wall = np.zeros_like(top)
        for k in range(1, height + 1):
            wall |= shift(top, 0, k)
        wall &= ~top
        body = top | wall
        sh = shift(body, cast[0], cast[1]) & ~body
        if cast != (0, 0):
            self.c.fill(sh & ~self.solid, SHADOW, 0.55)
            self.c.tint(sh & self.solid, SHADOW, 0.5)
        if outline:
            ol = dilate(body) & ~body & ~self.solid
            self.c.fill(ol, self.rg["outline"])
        # Wall: lit lip, body, dark foot, vertical ribs
        self.c.fill(wall, m(2))
        lip = wall & shift(top, 0, 1)
        self.c.fill(lip, m(base - 1))
        foot = wall & ~shift(wall, 0, -1)
        self.c.fill(foot & ~lip, m(1))
        if ribs and height >= 2:
            rib = wall & ~lip & ~foot & (XS % 4 == 0)
            self.c.fill(rib, m(1))
            ribl = wall & ~lip & ~foot & (XS % 4 == 3)
            self.c.fill(ribl, m(3))
        # Top: two pixel bevel, bright top left, dark bottom right
        e = edges(top)
        self.c.fill(top, m(base))
        if bevel >= 2:
            inner = top & ~(e["top"] | e["left"] | e["bottom"] |
                            e["right"])
            ei = edges(inner)
            self.c.fill(ei["top"] | ei["left"], m(base + 1))
            self.c.fill(ei["bottom"] | ei["right"], m(base - 1))
        self.c.fill(e["right"], m(base - 2))
        self.c.fill(e["bottom"], m(base - 2))
        self.c.fill(e["left"], m(base + 2))
        self.c.fill(e["top"], m(base + 3))
        self.c.fill(e["top"] & e["left"], m(base + 4))
        self.solid |= body
        return body

    def plate(self, mk, base=5, inset=False):
        """A plate on an existing surface: engraved (inset) or raised
        outline, light from the top left."""
        e = edges(mk)
        self.c.fill(mk, self.m[base])
        hi, lo = (self.m[base - 2], self.m[min(base + 2, 9)]) if inset \
            else (self.m[min(base + 2, 9)], self.m[base - 2])
        self.c.fill(e["top"] | e["left"], hi)
        self.c.fill(e["bottom"] | e["right"], lo)

    def recess(self, mk, depth=3):
        """A sunken area: floor darker, the north and west inner walls in
        shadow, the south and east lips catching the light."""
        m = self.m
        self.c.fill(mk, m[depth])
        e = edges(mk)
        self.c.fill(e["top"], m[1])
        self.c.fill(e["left"], m[1])
        self.c.fill(shift(e["top"], 0, 1) & mk & ~e["left"], m[2])
        self.c.fill(e["right"] & ~e["top"], m[6])
        self.c.fill(e["bottom"] & ~e["left"], m[6])

    def rivet(self, x, y):
        self.c.px(x + P, y + P, self.m[9])
        self.c.px(x + P + 1, y + P + 1, self.m[2])

    def vent(self, x0, y0, x1, y1, vertical=False):
        """Grille: dark slots with lit lips."""
        self.c.fill(R_(x0, y0, x1, y1), self.m[1])
        if vertical:
            for x in range(x0, x1 + 1, 2):
                self.c.fill(R_(x, y0, x, y1), self.m[4])
        else:
            for y in range(y0 + 1, y1 + 1, 2):
                self.c.fill(R_(x0, y, x1, y), self.m[4])
        self.c.fill(R_(x0, y1 + 1, x1, y1 + 1), self.m[7])

    def hazard(self, mk, phase=0, period=4):
        """Diagonal hazard stripes inside mask."""
        band = ((XS + YS + phase) // (period // 2)) % 2 == 0
        st = self.rg["stripe"]
        self.c.fill(mk & band, st[2])
        self.c.fill(mk & ~band, HAZARD_DARK)
        e = edges(mk)
        self.c.fill(e["top"] & band, st[3])
        self.c.fill(e["bottom"] & band, st[1])

    def light(self, x, y, ramp=None, size=1, level=1.0, halo=True):
        """Emissive light with a soft halo on the surrounding metal."""
        r = self.rg["light"] if ramp is None else ramp
        n = len(r)
        if halo:
            hm = DISC(x + size / 2.0, y + size / 2.0, size + 1.6)
            self.c.tint(hm & self.solid, r[n - 3], 0.35 * level)
        core = R_(x, y, x + size - 1, y + size - 1)
        k = int(round((n - 2) * level))
        self.c.fill(core, r[max(k, 1)])
        self.c.px(x + P, y + P, r[min(k + 1, n - 1)])

    def dome(self, cx, cy, rad, ramp=None, spec=True):
        """Shaded hemisphere, lit from the top left."""
        m = self.m if ramp is None else ramp
        dx = (XS + 0.5 - cx) / rad
        dy = (YS + 0.5 - cy) / rad
        d2 = dx * dx + dy * dy
        mk = d2 <= 1.0
        nz = np.sqrt(np.clip(1 - d2, 0, 1))
        Lx, Ly, Lz = -0.5, -0.62, 0.6
        ln = math.sqrt(Lx * Lx + Ly * Ly + Lz * Lz)
        lit = np.clip((dx * Lx + dy * Ly + nz * Lz) / ln, 0, 1)
        n = len(m)
        k = np.clip(np.round(0.5 + lit * (n - 2)), 0, n - 1).astype(int)
        for i in range(n):
            self.c.fill(mk & (k == i), m[i])
        if spec:
            self.c.px(int(cx - rad * 0.4) + P, int(cy - rad * 0.45) + P,
                      m[n - 1])
        return mk

    def image(self):
        return self.c.crop(P, P, T, T).image()


# ---------------------------------------------------------------------------
# Hull decks
# ---------------------------------------------------------------------------

def hull(t, mask, base=5, chamfer=3):
    """Paint the hull slab of a cell: a flat armoured roof with a hard
    lit rim on the north and west, a dark rim on the south and east, a
    tall south wall and a dark outline. A second, inner coaming line
    follows the closed sides so the roof reads as heavy plate laid on a
    frame. Returns the top surface mask and the flat roof inside the
    rim."""
    m = t.m
    top = deck_mask(mask, chamfer=chamfer)
    t.block(top, WALL, base=base, cast=(0, 0), bevel=1)
    flat = erode(top, 1)
    return top, flat


def _groove(t, mask, keep):
    """The service conduit, sunk into the deck as a narrow groove along
    every connected side: shadowed west and north lips, lit east and
    south lips."""
    m = t.m
    g = np.zeros((N, N), dtype=bool)
    if mask & U:
        g |= R_(7, -P, 8, 6)
    if mask & D:
        g |= R_(7, 5, 8, T + P)
    if mask & L:
        g |= R_(-P, 5, 8, 6)
    if mask & R:
        g |= R_(7, 5, T + P, 6)
    g &= keep
    t.c.fill(g, m[2])
    t.c.fill(shift(g, 1, 0) & ~g & keep & dilate(g), m[6])
    t.c.fill(shift(g, 0, 1) & ~g & keep & dilate(g), m[6])
    t.c.fill(g & ~shift(g, 0, 1) & ~shift(g, 1, 0), m[3])
    return g


def _joint(t, mask, flat):
    """Bulkhead seam where two cells meet, across the flat deck only, so
    a straight run reads as one long segmented hull."""
    m = t.m
    if mask & U:
        t.c.fill(flat & (YS == 0), m[3])
        t.c.fill(flat & (YS == 1), m[7])
    if mask & L:
        t.c.fill(flat & (XS == 0), m[3])
        t.c.fill(flat & (XS == 1), m[7])


def deck(mask, rg, variant=0, seed=0):
    t = Tile(rg)
    nconn = bin(mask).count("1")
    straight = mask in (U | D, L | R)
    top, flat = hull(t, mask, base=3 if straight else 4)
    if straight:
        _straight_details(t, mask, flat, variant)
    else:
        _junction(t, mask, flat, nconn, variant, seed)
    return t.image()


def _straight_details(t, mask, flat, variant):
    m = t.m
    vert = mask == (U | D)
    if variant == 0:
        # Plain run: two armour plates per cell with a bulkhead joint, a
        # narrow sunken rail channel and rivets
        _joint(t, mask, flat)
        if vert:
            g = R_(7, -P, 8, T + P) & flat
            t.c.fill(g & (XS == 7), m[1])
            t.c.fill(g & (XS == 8), m[2])
            t.c.fill(flat & (YS == 8) & ((XS < 7) | (XS > 8)), m[2])
            t.c.fill(flat & (YS == 9) & ((XS < 7) | (XS > 8)), m[5])
            for (x, y) in ((3, 3), (11, 11)):
                t.rivet(x, y)
        else:
            g = R_(-P, 5, T + P, 6) & flat
            t.c.fill(g & (YS == 5), m[1])
            t.c.fill(g & (YS == 6), m[2])
            t.c.fill(flat & (XS == 8) & ((YS < 5) | (YS > 6)), m[2])
            t.c.fill(flat & (XS == 9) & ((YS < 5) | (YS > 6)), m[5])
            for (x, y) in ((3, 2), (12, 8)):
                t.rivet(x, y)
    elif variant == 1:
        # Bulkhead: a raised armour rib across the hull with hazard
        # chevrons on its face
        if vert:
            _groove(t, mask, ~R_(-P, 4, T + P, 10))
            t.block(R_(1, 5, 14, 7), 2, base=6, cast=(0, 1), ribs=False)
            t.hazard(R_(4, 6, 11, 6), phase=0)
            t.rivet(2, 5)
            t.rivet(12, 5)
        else:
            _groove(t, mask, ~R_(5, -P, 10, T + P))
            t.block(R_(6, 1, 8, 9), 2, base=6, cast=(1, 0), ribs=False)
            t.hazard(R_(7, 3, 7, 7), phase=1)
        _joint(t, mask, flat)
    elif variant == 2:
        # Flush intake grille across the deck, a running light on the rim
        if vert:
            _groove(t, mask, ~R_(-P, 3, T + P, 11))
            t.recess(R_(4, 4, 11, 10), depth=2)
            for y in range(5, 10, 2):
                t.c.fill(R_(5, y, 10, y), m[5])
            t.light(1, 7, size=1, level=0.9)
        else:
            _groove(t, mask, ~R_(3, -P, 12, T + P))
            t.recess(R_(4, 3, 11, 8), depth=2)
            for x in range(5, 11, 2):
                t.c.fill(R_(x, 4, x, 7), m[5])
            t.light(13, 1, size=1, level=0.9)
        _joint(t, mask, flat)
    else:
        # Access hatch: inset armour plate with a handle and two lights
        if vert:
            _groove(t, mask, ~R_(-P, 3, T + P, 12))
            t.plate(R_(4, 3, 11, 12), base=4, inset=True)
            t.c.fill(R_(6, 7, 9, 8), m[2])
            t.c.fill(R_(6, 7, 9, 7), m[7])
            t.light(5, 4, size=1, level=0.8, halo=False)
            t.light(10, 11, size=1, level=0.5, halo=False)
        else:
            _groove(t, mask, ~R_(3, -P, 12, T + P))
            t.plate(R_(3, 3, 12, 8), base=4, inset=True)
            t.c.fill(R_(6, 5, 9, 6), m[2])
            t.c.fill(R_(6, 5, 9, 5), m[7])
            t.light(4, 4, size=1, level=0.8, halo=False)
            t.light(11, 7, size=1, level=0.5, halo=False)
        _joint(t, mask, flat)


# ---------------------------------------------------------------------------
# Junction superstructures
# ---------------------------------------------------------------------------

def _fan(t, cx, cy, r, blades=5, phase=0.0):
    """Recessed exhaust fan: dark well, hub and angled blades."""
    m = t.m
    well = DISC(cx, cy, r)
    t.c.fill(dilate(well) & ~well, m[7])
    t.c.fill(well, m[1])
    ang = np.arctan2(YS + 0.5 - cy, XS + 0.5 - cx)
    rr = np.hypot(XS + 0.5 - cx, YS + 0.5 - cy)
    bl = (((ang + phase + rr * 0.35) * blades / (2 * math.pi)) % 1.0) < 0.45
    t.c.fill(well & bl & (rr > 1.0), m[3])
    t.c.fill(well & bl & (rr > 1.0) & (YS + 0.5 < cy), m[4])
    t.c.fill(DISC(cx, cy, 1.2), m[6])
    e = edges(well)
    t.c.fill(e["top"] | e["left"], m[0])


def _junction(t, mask, flat, nconn, variant, seed):
    m = t.m
    acc = t.rg["accent"]
    if variant == 1:
        # Four-way landing pad: flush hazard ring around a lit pad
        pad = OCT(2, 1, 13, 11, 3)
        inner = OCT(4, 3, 11, 9, 2)
        _groove(t, mask, ~pad)
        t.hazard(pad & ~inner, phase=2)
        t.recess(inner, depth=3)
        t.c.fill(R_(6, 4, 6, 8) | R_(9, 4, 9, 8) | R_(7, 6, 8, 6),
                 acc[4])
        for (lx, ly) in ((1, 6), (14, 6), (7, 0), (8, 12)):
            t.light(lx, ly, size=1, level=0.8, halo=False)
        return

    _groove(t, mask, ~R_(0, 0, 15, 12))
    _tower(t, mask, nconn, (mask * 7 + seed * 3) % 5)


TOWER_TOP = (0, 7)    # tower roof rows; its wall runs down to the deck edge


def _tower(t, mask, nconn, kind):
    """Junction cells carry a tall block: the roof sits high above the
    decks that meet it and a four pixel south wall with lit windows drops
    down to deck level, so a maze reads as a fortress of towers joined by
    wall walks rather than a tangle of pipes."""
    m = t.m
    acc = t.rg["accent"]
    lt = t.rg["light"]
    y0, y1 = TOWER_TOP
    x0 = 1 if not mask & L else 0
    x1 = 14 if not mask & R else 15
    c = 2 if nconn < 4 else 1
    roof = OCT(x0, y0, x1, y1, c) & R_(x0, y0, x1, y1)
    wallh = 4
    t.block(roof, wallh, base=6, cast=(2, 2))
    # Lit windows along the wall
    for x in range(x0 + 2, x1 - 1, 3):
        if (x + kind) % 2:
            t.c.px(x + P, y1 + 2 + P, lt[3])
        else:
            t.c.px(x + P, y1 + 2 + P, m[1])
    # Roof details by kind
    if kind == 0:
        # A big exhaust fan beside a roof grille
        _fan(t, 5.0, 3.6, 3.0, phase=mask)
        t.vent(10, 1, 13, 5, vertical=True)
    elif kind == 1:
        # Command dome on a collar with a beacon
        t.c.fill(DISC(7.5, 3.8, 3.6) & roof, m[2])
        t.dome(7.5, 3.5, 3.0, ramp=m[2:])
        t.light(2, 2, size=1, level=1.0)
        t.rivet(12, 2)
    elif kind == 2:
        # Roof grille and an exhaust stack
        t.vent(2, 2, 8, 5, vertical=True)
        st = R_(10, 1, 13, 4)
        t.block(st, 2, base=7, cast=(1, 1), ribs=False, bevel=1)
        t.c.fill(R_(11, 2, 12, 3), m[0])
        t.light(3, 6, size=1, level=0.8, halo=False)
    elif kind == 3:
        # Stepped second storey with a painted band
        up = R_(4, 1, 11, 3)
        t.block(up, 2, base=7, cast=(1, 1), ribs=False)
        t.c.fill(R_(x0 + 1, 6, x1 - 1, 6) & roof, acc[3])
        t.light(12, 2, size=1, level=1.0)
    else:
        # Radar mast: a dish on a turntable
        t.c.fill(DISC(7.5, 3.5, 3.4) & roof, acc[1])
        t.c.fill(DISC(7.5, 3.5, 2.6), m[7])
        t.c.fill(DISC(7.0, 3.0, 1.4), m[9])
        t.c.fill(DISC(8.5, 4.5, 1.2) & DISC(7.5, 3.5, 2.6), m[4])
        t.light(13, 5, size=1, level=1.0)


# ---------------------------------------------------------------------------
# Cannons, armoured caps, cores and wreckage
# ---------------------------------------------------------------------------

_AWAY = {U: (0, 1), D: (0, -1), L: (1, 0), R: (-1, 0)}


def cannon(mask, rg):
    """Destructible gun turret on a fat armoured pedestal. 'mask' is the
    side the hull leaves through; the barrels point the other way."""
    t = Tile(rg)
    top, flat = hull(t, mask, chamfer=4)
    _groove(t, mask, ~OCT(1, 0, 14, 12, 3))
    _joint(t, mask, flat)
    m = t.m
    acc = rg["accent"]
    ax, ay = _AWAY[mask]
    hcx = 7.5 - ax * 1.0
    hcy = 6.0 - ay * 1.0
    # Turret ring set into the slab, painted
    t.c.fill(DISC(hcx, hcy, 6.2) & t.solid, rg["outline"])
    t.c.fill(DISC(hcx, hcy, 5.6) & t.solid, acc[1])
    t.c.fill(DISC(hcx, hcy, 5.6) & ~DISC(hcx, hcy, 4.6) & t.solid &
             ((XS + YS) % 3 == 0), acc[3])
    # Twin barrels with muzzle brakes, reaching the tile edge
    if ay:
        y0, y1 = (int(hcy), 15) if ay > 0 else (0, int(hcy))
        tip = (14, 15) if ay > 0 else (0, 1)
        bars = [R_(4, y0, 6, y1), R_(9, y0, 11, y1)]
        brakes = [R_(3, tip[0], 7, tip[1]), R_(8, tip[0], 12, tip[1])]
    else:
        x0, x1 = (int(hcx), 15) if ax > 0 else (0, int(hcx))
        tip = (14, 15) if ax > 0 else (0, 1)
        bars = [R_(x0, 3, x1, 5), R_(x0, 8, x1, 10)]
        brakes = [R_(tip[0], 2, tip[1], 6), R_(tip[0], 7, tip[1], 11)]
    gun = np.zeros((N, N), dtype=bool)
    for b in bars + brakes:
        gun |= b
    t.c.fill(dilate(gun) & ~gun & TILE(), rg["outline"])
    for b in bars:
        e = edges(b)
        t.c.fill(b, m[5])
        t.c.fill(e["top"] | e["left"], m[8])
        t.c.fill(e["bottom"] | e["right"], m[2])
    for b in brakes:
        e = edges(b)
        t.c.fill(b, m[6])
        t.c.fill(e["top"] | e["left"], m[9])
        t.c.fill(e["bottom"] | e["right"], m[3])
    # Gun house: armoured dome with a glowing sensor eye
    house = DISC(hcx, hcy, 4.0)
    t.c.fill(dilate(house) & ~house, rg["outline"])
    t.dome(hcx, hcy, 4.0, ramp=m[1:])
    t.c.fill(DISC(hcx, hcy, 4.0) & ~DISC(hcx, hcy, 3.2) &
             (YS + 0.5 > hcy + 1.5), m[2])
    lx = int(round(hcx - 1 + ax * 2))
    ly = int(round(hcy - 1 + ay * 2))
    t.light(lx, ly, size=2, level=1.0)
    return t.image()


def hard_cap(mask, rg):
    """Indestructible armoured end cap: a heavy bunker with hazard
    chevrons and no lights."""
    t = Tile(rg)
    top, flat = hull(t, mask, base=4, chamfer=4)
    _groove(t, mask, ~OCT(1, 0, 14, 12, 3))
    _joint(t, mask, flat)
    t.hazard(OCT(2, 1, 13, 11, 4) & ~OCT(4, 3, 11, 9, 3), phase=mask)
    cap = OCT(4, 2, 11, 7, 2)
    t.block(cap, 2, base=6, cast=(1, 1), ribs=False)
    m = t.m
    t.c.fill(R_(5, 4, 10, 4), m[3])
    t.c.fill(R_(5, 5, 10, 5), m[8])
    for (x, y) in ((5, 3), (10, 3), (5, 6), (10, 6)):
        t.c.px(x + P, y + P, m[9])
    return t.image()


def core(frame, vertical, rg):
    t = Tile(rg)
    mask = (U | D) if vertical else (L | R)
    # The core slab fills the whole cell so it meets any neighbour
    top, flat = hull(t, 15, chamfer=0)
    m = t.m
    cr = rg["core"]
    pulse = 0.5 + 0.5 * math.cos(2 * math.pi * frame / 8)
    cx, cy = 8.0, 6.5
    # Hazard corners
    for (x0, y0) in ((0, 0), (13, 0), (0, 9), (13, 9)):
        hz = R_(x0, y0, x0 + 2, y0 + 2) & top
        t.hazard(hz, phase=frame // 4)
    # Energy feed glowing in the trench
    tr = _groove(t, mask, ~DISC(cx, cy, 6.5))
    t.c.fill(tr & TILE(), cr[2 + int(round(pulse * 2))])
    # Housing ring with clamps
    t.c.fill(DISC(cx, cy, 6.9), rg["outline"])
    t.c.fill(DISC(cx, cy, 6.4), m[3])
    e = edges(DISC(cx, cy, 6.4))
    t.c.fill(e["top"] | e["left"], m[7])
    t.c.fill(e["bottom"] | e["right"], m[1])
    for k in range(4):
        a = math.pi / 4 + k * math.pi / 2
        bx = cx + math.cos(a) * 5.6 - 0.5
        by = cy + math.sin(a) * 5.6 - 0.5
        t.c.fill(R_(int(bx), int(by), int(bx) + 1, int(by) + 1), m[8])
        t.c.px(int(bx) + 1 + P, int(by) + 1 + P, m[2])
    # Emissive core: brightness follows the pulse
    rad = 4.9
    inner = DISC(cx, cy, rad)
    t.c.fill(DISC(cx, cy, rad + 0.6) & ~inner, cr[1])
    d = np.hypot(XS + 0.5 - cx, YS + 0.5 - cy) / rad
    v = np.clip(1 - d, 0, 1) * (0.8 + 0.4 * pulse) + 0.2 + 0.15 * pulse
    k = np.clip(np.round(v * (len(cr) - 1)), 0, len(cr) - 1)
    for i in range(len(cr)):
        t.c.fill(inner & (k == i), cr[i])
    # Rotating shutter blades over the glow
    a0 = 2 * math.pi * frame / 8 / 3
    for b in range(3):
        a = a0 + b * 2 * math.pi / 3
        for rr in (2.8, 3.6, 4.4):
            x = cx + math.cos(a) * rr - 0.5
            y = cy + math.sin(a) * rr - 0.5
            t.c.px(int(round(x)) + P, int(round(y)) + P, m[2])
    halo = DISC(cx, cy, 8.5) & ~DISC(cx, cy, 6.9) & t.solid
    t.c.tint(halo, cr[3], 0.15 + 0.2 * pulse)
    on = frame % 4 < 2
    t.light(0, 5, size=1, level=1.0 if on else 0.3, halo=False)
    t.light(15, 5, size=1, level=0.3 if on else 1.0, halo=False)
    return t.image()


def wreck(side, variant, rg):
    """Wrecked hull stub still attached on 'side', with scorched ground,
    torn plating and glowing embers."""
    rng = np.random.RandomState(side * 13 + variant * 5 + 1)
    t = Tile(rg)
    ln = 5 + variant * 3
    full = deck_mask(side, chamfer=0)
    if side == U:
        cut = R_(-P, ln, T + P, T + P)
    elif side == D:
        cut = R_(-P, -P, T + P, T - 1 - ln - WALL)
    elif side == L:
        cut = R_(ln, -P, T + P, T + P)
    else:
        cut = R_(-P, -P, T - 1 - ln, T + P)
    stub = full & ~cut
    for _ in range(5):
        if side in (U, D):
            bx = rng.uniform(1, 15)
            by = ln if side == U else T - 1 - ln - WALL
        else:
            by = rng.uniform(1, 11)
            bx = ln if side == L else T - 1 - ln
        stub &= ~DISC(bx, by, rng.uniform(1.2, 2.8))
    sc = np.zeros((N, N), dtype=bool)
    for _ in range(5):
        sc |= DISC(rng.uniform(2, 14), rng.uniform(2, 14),
                   rng.uniform(2.0, 4.0))
    sc &= TILE()
    t.c.fill(sc, SHADOW, 0.35)
    t.c.fill(erode(sc), SHADOW, 0.3)
    t.block(stub, 2, base=4, cast=(1, 1))
    _groove(t, side, stub)
    m = t.m
    torn = stub & ~erode(stub, 2) & dilate(cut)
    t.c.fill(torn, m[0])
    emb = torn & (rng.uniform(size=(N, N)) < 0.25)
    t.c.fill(emb, glow_ramp(24)[3])
    for _ in range(6):
        x = int(rng.uniform(1, 14))
        y = int(rng.uniform(1, 14))
        if t.solid[y + P, x + P]:
            continue
        t.c.px(x + P, y + P, m[int(rng.randint(4, 8))])
        t.c.px(x + P + 1, y + P, m[2])
        t.c.px(x + P, y + P + 1, rg["outline"])
    return t.image()


# ---------------------------------------------------------------------------
# Distant silhouettes for the parallax layers
# ---------------------------------------------------------------------------

def far_tile(kind, arg, rg, seed=0):
    """A cell as seen from far away: one broad shaded mass with a lit
    north west rim, a dark south wall, a darker roof block on junctions
    and the odd light. No seams, trenches or rivets."""
    t = Tile(rg)
    m = rg["metal"]
    if kind == "core":
        mask = (U | D) if arg[0] else (L | R)
        top = deck_mask(mask, chamfer=0) | R_(0, 0, 15, Y1)
    elif kind == "wreck":
        side, variant = arg
        top = deck_mask(side, chamfer=0)
        ln = 5 + variant * 3
        if side == U:
            top &= R_(-P, -P, T + P, ln)
        elif side == D:
            top &= R_(-P, T - 1 - ln - WALL, T + P, T + P)
        elif side == L:
            top &= R_(-P, -P, ln, T + P)
        else:
            top &= R_(T - 1 - ln, -P, T + P, T + P)
    else:
        top = deck_mask(arg, chamfer=1 if kind == "deck" else 3)
    wall = np.zeros_like(top)
    for k in range(1, WALL + 1):
        wall |= shift(top, 0, k)
    wall &= ~top
    t.c.fill(wall, m[1])
    t.c.fill(top, m[2])
    e = edges(top)
    t.c.fill(e["top"] | e["left"], m[4])
    t.c.fill(e["bottom"] | e["right"], m[1])
    t.solid |= top | wall
    lt = rg["light"]
    if kind == "deck" and arg not in (U | D, L | R):
        # Towers: the bright masses, with a row of lit windows
        x0 = 1 if not arg & L else 0
        x1 = 14 if not arg & R else 15
        y0 = 0 if arg & U else 1
        roof = R_(x0, y0, x1, 8)
        wl = R_(x0, 9, x1, 11 + (0 if arg & D else WALL))
        t.c.fill(wl, m[1])
        t.c.fill(roof, m[6])
        t.c.fill(edges(roof)["top"] | edges(roof)["left"], m[8])
        t.c.fill(edges(roof)["bottom"], m[4])
        for x in range(x0 + 2, x1 - 1, 4):
            if (x // 4 + seed) % 3:
                t.c.fill(R_(x, 10, x + 1, 11), lt[5])
        if seed % 2:
            t.c.fill(R_(6, 2, 9, 4), lt[4])
    elif kind == "deck":
        if seed % 4 == 0:
            t.c.fill(R_(7, 5, 8, 6), lt[3])
    elif kind == "cannon":
        t.c.fill(DISC(7.5, 6.0, 4.5), m[6])
        t.c.fill(DISC(7.5, 6.0, 4.5) & ~DISC(7.0, 5.5, 3.6), m[2])
        t.c.fill(R_(7, 5, 8, 6), lt[4])
    elif kind == "cap":
        blk = OCT(3, 1, 12, 7, 2)
        t.c.fill(shift(blk, 0, 3) & ~blk, m[1])
        t.c.fill(blk, m[5])
    elif kind == "core":
        cr = rg["core"]
        pulse = 0.5 + 0.5 * math.cos(2 * math.pi * arg[1] / 8)
        t.c.fill(DISC(8.0, 6.5, 6.2), m[2])
        t.c.fill(DISC(8.0, 6.5, 4.6), cr[2 + int(round(pulse))])
        t.c.fill(DISC(8.0, 6.5, 2.6), cr[4 + int(round(pulse))])
    return t.image()


def _far_down(img, size):
    """Box filter a distant silhouette down and drop the half covered
    rim pixels so the mass keeps a clean edge."""
    return downscale(img, size)


# ---------------------------------------------------------------------------
# Sheets
# ---------------------------------------------------------------------------

def _specs():
    """(index, kind, arg) for every tile slot the engine uses."""
    out = [(0, "deck", 15)]
    for msk in range(1, 16):
        out.append((msk, "cannon" if msk in (U, R, D, L) else "deck", msk))
    for f in range(8):
        out.append((16 + f, "core", (True, f)))
        out.append((24 + f, "core", (False, f)))
    for k, msk in enumerate((U, R, D, L)):
        out.append((32 + k, "cap", msk))
    for k in range(3):
        out.append((40 + 2 * k, "deck", U | D))
        out.append((41 + 2 * k, "deck", L | R))
    for k, msk in enumerate((U, R, D, L)):
        for v in range(2):
            out.append((48 + 2 * k + v, "wreck", (msk, v)))
    return out


def tileset16(region):
    rg = REGIONS[region]
    blank = Image.new("RGBA", (T, T), (0, 0, 0, 0))
    tiles = [blank] * 56
    tiles[0] = deck(15, rg, variant=1, seed=region)
    for msk in range(1, 16):
        if msk in (U, R, D, L):
            tiles[msk] = cannon(msk, rg)
        else:
            tiles[msk] = deck(msk, rg, seed=region)
    for f in range(8):
        tiles[16 + f] = core(f, True, rg)
        tiles[24 + f] = core(f, False, rg)
    for k, msk in enumerate((U, R, D, L)):
        tiles[32 + k] = hard_cap(msk, rg)
    for k in range(3):
        tiles[40 + 2 * k] = deck(U | D, rg, variant=k + 1)
        tiles[41 + 2 * k] = deck(L | R, rg, variant=k + 1)
    for k, msk in enumerate((U, R, D, L)):
        for v in range(2):
            tiles[48 + 2 * k + v] = wreck(msk, v, rg)
    return tiles


def tileset_far(region, size):
    rg = REGIONS[region]
    blank = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    tiles = [blank] * 56
    for (i, kind, arg) in _specs():
        tiles[i] = _far_down(far_tile(kind, arg, rg, seed=region + i), size)
    return tiles


_CACHE = {}


def tileset(region, size=16):
    if size == T:
        if region not in _CACHE:
            _CACHE[region] = tileset16(region)
        return sheet(_CACHE[region], 8)
    return sheet(tileset_far(region, size), 8)
