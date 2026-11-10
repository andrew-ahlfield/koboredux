"""Base tiles for the five Halcyon regions: armoured installations.

Kobo bases are networks of single tile corridors. Each tile is drawn the
way a 1990s arcade ground set would be: a slab of armour plate seen from
above with a visible south wall, a drop shadow on the ground to the lower
right, engraved panel seams, rivets, vents, hazard striping and emissive
lights. Hubs carry a second storey (bunkers, domes, sensor masts),
cannons are domed twin barrel turrets, cores are glowing reactors.

Tile index layout (8 x 7 sheet), matching the engine's map.cpp:
  0         four-way junction, landing pad variant
  1-15      corridor pieces by connection mask (U=1 R=2 D=4 L=8);
            single-bit masks (1, 2, 4, 8) are the destructible cannons
  16-23     core, connected vertically (8 frame pulse)
  24-31     core, connected horizontally (8 frame pulse)
  32-35     armoured (indestructible) end caps U, R, D, L
  40-45     straight corridor variants (even vertical, odd horizontal)
  48-55     wreckage left after destruction (U, U, R, R, D, D, L, L)

Every tile is drawn at 16 px. The 12 and 8 px sets used for the parallax
layers are box filtered from it, which also softens them like distance.
"""

import math
import numpy as np
from PIL import Image

from .pix import (Canvas, hramp, glow_ramp, mix, shift, dilate, erode,
                  edges, rect, disc, octagon, downscale)
from .render import sheet

U, R, D, L = 1, 2, 4, 8
T = 16          # tile size
P = 16          # padding around the tile on the work canvas
N = T + 2 * P   # work canvas size
WALL = 3        # height of the base level south wall in pixels

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
    _region(58, 0.30, 0.09, 0.95, 0.35,
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


ARM_V = (2, 13)   # vertical corridor top surface columns
ARM_H = (2, 10)   # horizontal corridor top surface rows (wall below)


def arm_masks(mask):
    """Top surface masks of the corridor arms. Arms run far into the
    padding so neighbouring tiles join seamlessly."""
    arms = {}
    if mask & U:
        arms[U] = R_(ARM_V[0], -P, ARM_V[1], 8)
    if mask & D:
        arms[D] = R_(ARM_V[0], 6, ARM_V[1], T + P)
    if mask & L:
        arms[L] = R_(-P, ARM_H[0], 8, ARM_H[1])
    if mask & R:
        arms[R] = R_(7, ARM_H[0], T + P, ARM_H[1])
    return arms


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

    def block(self, top, height, base=5, cast=(2, 2), outline=True,
              ribs=True, ramp=None):
        """Extrude a top surface mask: drop shadow, south wall, bevelled
        top and outline."""
        m = self.m if ramp is None else ramp
        wall = np.zeros_like(top)
        for k in range(1, height + 1):
            wall |= shift(top, 0, k)
        wall &= ~top
        body = top | wall
        # Drop shadow: on the ground it is translucent, on structure it
        # darkens what is already there.
        sh = shift(body, cast[0], cast[1]) & ~body
        ground = sh & ~self.solid
        self.c.fill(ground, SHADOW, 0.55)
        self.c.tint(sh & self.solid, SHADOW, 0.45)
        # Outline (drawn before the body so the body overwrites it)
        if outline:
            ol = dilate(body) & ~body & ~self.solid
            self.c.fill(ol, self.rg["outline"])
        # Wall: lip row, body, dark foot, vertical ribs
        self.c.fill(wall, m[2])
        lip = wall & shift(top, 0, 1)
        self.c.fill(lip, m[4])
        foot = wall & ~shift(wall, 0, -1)
        self.c.fill(foot & ~lip, m[1])
        if ribs:
            xs = np.arange(N)[None, :].repeat(N, 0)
            rib = wall & ~lip & ~foot & ((xs - P) % 4 == 0)
            self.c.fill(rib, m[1])
            ribl = wall & ~lip & ~foot & ((xs - P) % 4 == 3)
            self.c.fill(ribl, m[3])
        # Top surface with bevelled edges
        e = edges(top)
        self.c.fill(top, m[base])
        self.c.fill(e["right"], m[base - 2])
        self.c.fill(e["bottom"], m[base - 1])
        self.c.fill(e["left"], m[base + 2])
        self.c.fill(e["top"], m[base + 3])
        self.c.fill(e["top"] & e["left"], m[min(base + 4, 9)])
        self.solid |= body
        return body

    def plate(self, mk, base=5, inset=False):
        """A plate on an existing surface: engraved (inset) or raised
        outline, light from the top left."""
        e = edges(mk)
        self.c.fill(mk, self.m[base])
        hi, lo = (self.m[base - 2], self.m[base + 2]) if inset else \
            (self.m[base + 2], self.m[base - 2])
        self.c.fill(e["top"] | e["left"], hi)
        self.c.fill(e["bottom"] | e["right"], lo)

    def rivet(self, x, y):
        self.c.px(x + P, y + P, self.m[9])
        self.c.px(x + P + 1, y + P + 1, self.m[2])

    def seam_h(self, y, x0, x1):
        self.c.fill(R_(x0, y, x1, y) & self.solid, self.m[2])
        self.c.fill(R_(x0, y + 1, x1, y + 1) & self.solid, self.m[7])

    def seam_v(self, x, y0, y1):
        self.c.fill(R_(x, y0, x, y1) & self.solid, self.m[2])
        self.c.fill(R_(x + 1, y0, x + 1, y1) & self.solid, self.m[7])

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
        ys, xs = np.mgrid[0:N, 0:N]
        band = ((xs + ys + phase) // (period // 2)) % 2 == 0
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
        ys, xs = np.mgrid[0:N, 0:N]
        dx = (xs + 0.5 - (cx + P)) / rad
        dy = (ys + 0.5 - (cy + P)) / rad
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
# Corridors
# ---------------------------------------------------------------------------

def _arm_details(t, mask, arms, keep):
    """Trench, armour plates, seams and rivets along each arm, clipped to
    'keep' (the arm area outside the hub)."""
    m = t.m
    for side, am in arms.items():
        clip = am & keep
        if side in (U, D):
            # Centre conduit: shadowed left wall, ribbed floor, lit right
            t.c.fill(R_(6, -P, 6, T + P) & clip, m[2])
            t.c.fill(R_(7, -P, 8, T + P) & clip, m[3])
            ys = np.arange(N)[:, None].repeat(N, 1)
            rib = R_(7, -P, 8, T + P) & clip & ((ys - P) % 2 == 0)
            t.c.fill(rib, m[5])
            t.c.fill(R_(9, -P, 9, T + P) & clip, m[7])
            # Seam across at the tile boundary
            sm = R_(3, 0, 12, 0) & clip
            t.c.fill(sm, m[2])
            t.c.fill(shift(sm, 0, 1), m[7])
            for x in (4, 11):
                if clip[P + 3, P + x]:
                    t.rivet(x, 3)
                if clip[P + 12, P + x]:
                    t.rivet(x, 12)
        else:
            t.c.fill(R_(-P, 5, T + P, 5) & clip, m[2])
            t.c.fill(R_(-P, 6, T + P, 6) & clip, m[3])
            xs = np.arange(N)[None, :].repeat(N, 0)
            rib = R_(-P, 6, T + P, 6) & clip & ((xs - P) % 2 == 0)
            t.c.fill(rib, m[5])
            t.c.fill(R_(-P, 7, T + P, 7) & clip, m[7])
            sm = R_(0, 3, 0, 9) & clip
            t.c.fill(sm, m[2])
            t.c.fill(shift(sm, 1, 0), m[7])
            for y in (3, 9):
                for x in (3, 12):
                    if clip[P + y, P + x]:
                        t.rivet(x, y if y == 3 else 8)


def _hub_mask():
    return OCT(1, 1, 14, 10, 2)


def corridor(mask, rg, variant=0, seed=0):
    t = Tile(rg)
    arms = arm_masks(mask)
    nconn = bin(mask).count("1")
    straight = mask in (U | D, L | R)
    top = np.zeros((N, N), dtype=bool)
    for am in arms.values():
        top |= am
    hub = None
    if not straight:
        hub = _hub_mask()
        top |= hub
    t.block(top, WALL)
    keep = np.ones((N, N), dtype=bool)
    if hub is not None:
        keep &= ~dilate(hub)
    _arm_details(t, mask, arms, keep)
    if hub is not None:
        t.plate(hub, base=5)
        _superstructure(t, mask, nconn, variant, seed)
    elif variant:
        _corridor_variant(t, mask == (U | D), variant)
    return t.image()


def _corridor_variant(t, vert, variant):
    m = t.m
    if variant == 1:
        # Hazard striped blast door band with a warning light
        if vert:
            band = R_(3, 5, 12, 10)
            t.hazard(band, phase=0)
            t.c.fill(R_(3, 4, 12, 4), m[2])
            t.c.fill(R_(3, 11, 12, 11), m[7])
        else:
            band = R_(5, 3, 10, 9)
            t.hazard(band, phase=1)
            t.c.fill(R_(4, 3, 4, 9), m[2])
            t.c.fill(R_(11, 3, 11, 9), m[7])
    elif variant == 2:
        # Vent block and a pair of running lights
        if vert:
            t.plate(R_(3, 3, 12, 12), base=4, inset=True)
            t.vent(5, 5, 10, 9)
            t.light(4, 11, size=1)
            t.light(11, 11, size=1, level=0.6)
        else:
            t.plate(R_(3, 3, 12, 9), base=4, inset=True)
            t.vent(5, 4, 10, 7, vertical=True)
            t.light(3, 4, size=1)
            t.light(12, 4, size=1, level=0.6)
    elif variant == 3:
        # Coolant pipe bundle with painted clamps and a status light
        acc = t.rg["accent"]
        if vert:
            for x in (3, 11):
                pipe = R_(x, 0, x + 1, 15)
                t.c.fill(pipe, m[6])
                t.c.fill(R_(x, 0, x, 15), m[8])
                t.c.fill(R_(x + 1, 0, x + 1, 15), m[3])
            for y in (4, 11):
                cl = R_(2, y, 13, y + 1)
                t.c.fill(cl, acc[3])
                t.c.fill(R_(2, y, 13, y), acc[5])
                t.c.fill(R_(2, y + 1, 13, y + 1), acc[1])
            t.light(7, 7, size=2, level=0.9)
        else:
            for y in (3, 8):
                pipe = R_(0, y, 15, y + 1)
                t.c.fill(pipe, m[6])
                t.c.fill(R_(0, y, 15, y), m[8])
                t.c.fill(R_(0, y + 1, 15, y + 1), m[3])
            for x in (3, 11):
                cl = R_(x, 2, x + 1, 10)
                t.c.fill(cl, acc[3])
                t.c.fill(R_(x, 2, x, 10), acc[5])
                t.c.fill(R_(x + 1, 2, x + 1, 10), acc[1])
            t.light(7, 5, size=2, level=0.9)


# ---------------------------------------------------------------------------
# Hub superstructures
# ---------------------------------------------------------------------------

def _superstructure(t, mask, nconn, variant, seed):
    m = t.m
    kind = (mask * 7 + variant * 3 + seed) % 5
    if variant == 1:
        kind = 5
    if kind == 0:
        # Bunker with a vent grille and twin lights
        top = R_(4, 2, 11, 5)
        t.block(top, 2, base=6, cast=(1, 1))
        t.vent(5, 3, 10, 4, vertical=True)
        t.light(4, 8, size=1)
        t.light(11, 8, size=1, level=0.5)
    elif kind == 1:
        # Sensor dome on a collar
        t.c.fill(DISC(7.5, 5.5, 4.4) & t.solid, m[1])
        t.dome(7.5, 5.2, 3.6, ramp=m[2:])
        t.light(7, 4, ramp=t.rg["light"], size=1)
        t.rivet(2, 8)
        t.rivet(12, 8)
    elif kind == 2:
        # Raised plate with a hazard rim and a heavy hatch
        top = R_(3, 2, 12, 6)
        t.block(top, 2, base=5, cast=(1, 1), ribs=False)
        t.hazard(R_(3, 2, 12, 2) | R_(3, 6, 12, 6))
        t.c.fill(R_(6, 3, 9, 5), m[2])
        t.c.fill(R_(6, 3, 9, 3), m[1])
        t.c.fill(R_(7, 4, 8, 4), m[6])
        t.light(13, 3, size=1, level=0.8)
    elif kind == 3:
        # Equipment tower: stacked block, mast light, side vents
        t.block(R_(5, 1, 10, 4), 3, base=6, cast=(1, 1))
        t.c.fill(R_(6, 2, 9, 2), m[8])
        t.vent(3, 7, 4, 8, vertical=True)
        t.vent(11, 7, 12, 8, vertical=True)
        t.light(7, 0, size=2, level=1.0)
    elif kind == 4:
        # Gun emplacement ring (inactive) with accent paint
        acc = t.rg["accent"]
        ring = DISC(7.5, 5.5, 4.4) & ~DISC(7.5, 5.5, 2.6)
        t.c.fill(DISC(7.5, 5.5, 4.4), m[2])
        t.c.fill(ring, acc[2])
        e = edges(DISC(7.5, 5.5, 4.4))
        t.c.fill(e["top"] | e["left"], acc[4])
        t.c.fill(e["bottom"] | e["right"], acc[0])
        t.c.fill(DISC(7.5, 5.5, 2.6), m[1])
        t.light(7, 5, size=1)
    else:
        # Four-way landing pad: hazard ring around a lit pad
        pad = OCT(2, 1, 13, 10, 3)
        t.hazard(pad & ~OCT(4, 3, 11, 8, 2), phase=2)
        t.plate(OCT(4, 3, 11, 8, 2), base=3, inset=True)
        t.c.fill(R_(6, 4, 6, 7) | R_(9, 4, 9, 7) | R_(7, 5, 8, 6),
                 t.rg["accent"][4])
        for (lx, ly) in ((1, 5), (14, 5), (7, 0), (8, 11)):
            t.light(lx, ly, size=1, level=0.8, halo=False)


# ---------------------------------------------------------------------------
# Cannons, armoured caps, cores and wreckage
# ---------------------------------------------------------------------------

_AWAY = {U: (0, 1), D: (0, -1), L: (1, 0), R: (-1, 0)}


def cannon(mask, rg):
    """Destructible gun turret. 'mask' is the side the corridor leaves
    through; the barrels point the other way."""
    t = Tile(rg)
    arms = arm_masks(mask)
    top = OCT(1, 1, 14, 11, 3)
    for am in arms.values():
        top |= am
    t.block(top, WALL)
    _arm_details(t, mask, arms, ~dilate(OCT(1, 1, 14, 11, 3)))
    t.plate(OCT(1, 1, 14, 11, 3), base=5)
    for (x, y) in ((3, 3), (12, 3), (3, 9), (12, 9)):
        t.rivet(x, y)
    m = t.m
    acc = rg["accent"]
    ax, ay = _AWAY[mask]
    # Gun house offset away from the barrels so they read clearly
    if ay:
        hx0, hy0 = 4, (1 if ay > 0 else 4)
    else:
        hx0, hy0 = (2 if ax > 0 else 6), 2
    house = OCT(hx0, hy0, hx0 + 7, hy0 + 7, 2)
    hcx, hcy = hx0 + 4.0, hy0 + 4.0
    # Turret ring set into the slab
    t.c.fill(DISC(hcx, hcy, 5.8) & t.solid, rg["outline"])
    t.c.fill(DISC(hcx, hcy, 5.2) & t.solid, acc[1])
    t.c.fill(DISC(hcx, hcy, 5.2) & ~DISC(hcx, hcy, 4.3) & t.solid &
             (np.add.outer(np.arange(N), np.arange(N)) % 3 == 0), acc[3])
    # Twin barrels with muzzle brakes, reaching the tile edge
    if ay:
        y0, y1 = (int(hcy), 15) if ay > 0 else (0, int(hcy))
        tip = (14, 15) if ay > 0 else (0, 1)
        bars = [R_(4, y0, 6, y1), R_(9, y0, 11, y1)]
        brakes = [R_(3, tip[0], 7, tip[1]), R_(8, tip[0], 12, tip[1])]
    else:
        x0, x1 = (int(hcx), 15) if ax > 0 else (0, int(hcx))
        tip = (14, 15) if ax > 0 else (0, 1)
        bars = [R_(x0, 2, x1, 4), R_(x0, 7, x1, 9)]
        brakes = [R_(tip[0], 1, tip[1], 5), R_(tip[0], 6, tip[1], 10)]
    gun = np.zeros((N, N), dtype=bool)
    for b in bars + brakes:
        gun |= b
    t.c.fill(dilate(gun) & ~gun & ~gun, rg["outline"])
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
    # Gun house: bevelled block with the sensor dome on top
    t.c.fill(dilate(house) & ~house, rg["outline"])
    t.plate(house, base=6)
    t.c.fill(edges(house)["bottom"], m[2])
    t.dome(hcx, hcy, 2.6, ramp=m[2:])
    # Glowing sensor eye facing the barrels
    lr = rg["light"]
    lx = int(round(hcx - 1 + ax * 2))
    ly = int(round(hcy - 1 + ay * 2))
    t.light(lx, ly, size=2, level=1.0)
    return t.image()


def hard_cap(mask, rg):
    """Indestructible armoured end cap: no lights, heavy plate, hazard
    chevrons."""
    t = Tile(rg)
    arms = arm_masks(mask)
    top = OCT(1, 1, 14, 11, 4)
    for am in arms.values():
        top |= am
    t.block(top, WALL, base=4)
    _arm_details(t, mask, arms, ~dilate(OCT(1, 1, 14, 11, 4)))
    t.plate(OCT(1, 1, 14, 11, 4), base=4)
    t.hazard(OCT(2, 2, 13, 10, 3) & ~OCT(4, 4, 11, 8, 2), phase=mask)
    cap = OCT(4, 3, 11, 8, 2)
    t.block(cap, 1, base=6, cast=(1, 1), ribs=False)
    m = t.m
    # Cross brace
    for k in range(5):
        t.c.px(5 + k + P, 4 + k // 1 + P if k < 4 else 7 + P, m[3])
    t.c.fill(R_(5, 5, 10, 5), m[3])
    t.c.fill(R_(5, 6, 10, 6), m[8])
    for (x, y) in ((5, 4), (10, 4), (5, 7), (10, 7)):
        t.c.px(x + P, y + P, m[9])
    return t.image()


def core(frame, vertical, rg):
    t = Tile(rg)
    mask = (U | D) if vertical else (L | R)
    arms = arm_masks(mask)
    slab = OCT(0, 0, 15, 12, 4)
    top = slab.copy()
    for am in arms.values():
        top |= am
    t.block(top, WALL)
    t.plate(slab, base=5)
    m = t.m
    cr = rg["core"]
    pulse = 0.5 + 0.5 * math.cos(2 * math.pi * frame / 8)
    cx, cy = 8.0, 6.5
    # Hazard corners
    for (x0, y0) in ((1, 1), (12, 1), (1, 9), (12, 9)):
        hz = R_(x0, y0, x0 + 2, y0 + 2) & slab
        t.hazard(hz, phase=frame // 4)
    # Energy feed glowing in the corridor conduits
    feed = R_(7, -P, 8, T + P) if vertical else R_(-P, 6, T + P, 6)
    t.c.fill(feed & t.solid & ~slab, cr[2 + int(round(pulse * 2))])
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
    ys, xs = np.mgrid[0:N, 0:N]
    d = np.hypot(xs + 0.5 - (cx + P), ys + 0.5 - (cy + P)) / rad
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
    # Glow spill onto the housing
    halo = DISC(cx, cy, 8.5) & ~DISC(cx, cy, 6.9) & t.solid
    t.c.tint(halo, cr[3], 0.15 + 0.2 * pulse)
    # Blinking status lights on the slab
    on = frame % 4 < 2
    t.light(1, 5, size=1, level=1.0 if on else 0.3, halo=False)
    t.light(14, 5, size=1, level=0.3 if on else 1.0, halo=False)
    return t.image()


def wreck(side, variant, rg):
    """Wrecked corridor stub still attached on 'side', with scorched
    ground, torn plating and glowing embers."""
    rng = np.random.RandomState(side * 13 + variant * 5 + 1)
    t = Tile(rg)
    ln = 6 + variant * 2
    full = arm_masks(side)[side]
    if side == U:
        cut = R_(-P, ln, T + P, T + P)
    elif side == D:
        cut = R_(-P, -P, T + P, T - 1 - ln)
    elif side == L:
        cut = R_(ln, -P, T + P, T + P)
    else:
        cut = R_(-P, -P, T - 1 - ln, T + P)
    stub = full & ~cut
    # Jagged torn end
    ys, xs = np.mgrid[0:N, 0:N]
    for _ in range(4):
        if side in (U, D):
            bx = rng.uniform(2, 14)
            by = ln if side == U else T - 1 - ln
        else:
            by = rng.uniform(2, 11)
            bx = ln if side == L else T - 1 - ln
        stub &= ~DISC(bx, by, rng.uniform(1.2, 2.4))
    # Scorch marks on the ground
    sc = np.zeros((N, N), dtype=bool)
    for _ in range(4):
        sc |= DISC(rng.uniform(3, 13), rng.uniform(3, 13),
                   rng.uniform(1.8, 3.6))
    sc &= TILE()
    t.c.fill(sc, SHADOW, 0.35)
    t.c.fill(erode(sc), SHADOW, 0.25)
    t.block(stub, 2, base=4)
    _arm_details(t, side, {side: stub}, stub)
    m = t.m
    # Torn edge: exposed dark interior and embers
    torn = stub & ~erode(stub, 2) & dilate(cut)
    t.c.fill(torn, m[0])
    acc = rg["light"]
    emb = torn & (rng.uniform(size=(N, N)) < 0.25)
    t.c.fill(emb, glow_ramp(24)[3])
    # Debris chunks
    for _ in range(5):
        x = int(rng.uniform(1, 14))
        y = int(rng.uniform(1, 14))
        if t.solid[y + P, x + P]:
            continue
        t.c.px(x + P, y + P, m[int(rng.randint(4, 8))])
        t.c.px(x + P + 1, y + P, m[2])
        t.c.px(x + P, y + P + 1, rg["outline"])
    del acc
    return t.image()


# ---------------------------------------------------------------------------
# Sheets
# ---------------------------------------------------------------------------

def tileset16(region):
    rg = REGIONS[region]
    blank = Image.new("RGBA", (T, T), (0, 0, 0, 0))
    tiles = [blank] * 56
    tiles[0] = corridor(15, rg, variant=1, seed=region)
    for msk in range(1, 16):
        if msk in (U, R, D, L):
            tiles[msk] = cannon(msk, rg)
        else:
            tiles[msk] = corridor(msk, rg, seed=region)
    for f in range(8):
        tiles[16 + f] = core(f, True, rg)
        tiles[24 + f] = core(f, False, rg)
    for k, msk in enumerate((U, R, D, L)):
        tiles[32 + k] = hard_cap(msk, rg)
    for k in range(3):
        tiles[40 + 2 * k] = corridor(U | D, rg, variant=k + 1)
        tiles[41 + 2 * k] = corridor(L | R, rg, variant=k + 1)
    for k, msk in enumerate((U, R, D, L)):
        for v in range(2):
            tiles[48 + 2 * k + v] = wreck(msk, v, rg)
    return tiles


_CACHE = {}


def tileset(region, size=16):
    if region not in _CACHE:
        _CACHE[region] = tileset16(region)
    tiles = _CACHE[region]
    if size != T:
        tiles = [downscale(t, size) for t in tiles]
    return sheet(tiles, 8)
