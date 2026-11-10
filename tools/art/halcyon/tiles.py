"""Base (pipe network) tiles for the five Halcyon regions.

Tile index layout (8 x 7 sheet), matching the engine's map.cpp:
  0         four-way junction, hatch variant
  1-15      pipe pieces by connection mask (U=1 R=2 D=4 L=8);
            single-bit masks (1, 2, 4, 8) are the destructible cannon nodes
  16-23     core, connected vertically (8 frame pulse)
  24-31     core, connected horizontally (8 frame pulse)
  32-35     armoured (indestructible) end nodes U, R, D, L
  40-45     straight pipe variants (even vertical, odd horizontal)
  48-55     scrap stubs left after destruction (U, U, R, R, D, D, L, L)
"""

import math
import numpy as np
from PIL import Image

from .render import (Part, render, sheet, sd_circle, sd_box, sd_capsule,
                     sd_poly, sd_union, sd_sub, sd_ring, sd_rot)
from .palette import rgb, RAMPS

U, R, D, L = 1, 2, 4, 8
PW = 4.5  # pipe half width in 16 px tile units

REGIONS = [
    # pipe ramp, accent ramp, core glow ramp, node eye ramp
    dict(pipe="steel", accent="teal", core="teal", eye="fire"),
    dict(pipe="gold", accent="red", core="fire", eye="teal"),
    dict(pipe="green", accent="dark", core="green", eye="fire"),
    dict(pipe="violet", accent="steel", core="rose", eye="teal"),
    dict(pipe="ice", accent="steel", core="amber", eye="red"),
]


def _arm(mask):
    """Pipe arms from tile centre to each connected edge."""
    arms = []
    if mask & U:
        arms.append(sd_box(0, -6.0, PW, 6.5))
    if mask & D:
        arms.append(sd_box(0, 6.0, PW, 6.5))
    if mask & L:
        arms.append(sd_box(-6.0, 0, 6.5, PW))
    if mask & R:
        arms.append(sd_box(6.0, 0, 6.5, PW))
    return arms


def _bands(mask):
    """Coupling rings near tile edges so neighbouring tiles read as
    separate segments."""
    b = []
    if mask & U:
        b.append(sd_box(0, -6.6, PW + 0.6, 0.7))
    if mask & D:
        b.append(sd_box(0, 6.6, PW + 0.6, 0.7))
    if mask & L:
        b.append(sd_box(-6.6, 0, 0.7, PW + 0.6))
    if mask & R:
        b.append(sd_box(6.6, 0, 0.7, PW + 0.6))
    return b


def _pipe_axis_parts(mask, rg):
    """Pipes are cylinders: give each arm its own dome profile along the
    right axis by rendering vertical and horizontal arms separately."""
    parts = []
    v = []
    h = []
    if mask & U:
        v.append(sd_box(0, -6.0, PW, 6.5))
    if mask & D:
        v.append(sd_box(0, 6.0, PW, 6.5))
    if mask & L:
        h.append(sd_box(-6.0, 0, 6.5, PW))
    if mask & R:
        h.append(sd_box(6.0, 0, 6.5, PW))
    # The bevel profile of a box follows its nearest edge, so a long box
    # already shades as a cylinder across its width.
    if v:
        parts.append(Part(sd_union(*v), rg["pipe"], z=0, h=2.2, bevel=PW,
                          dome=True, spec=0.4))
    if h:
        parts.append(Part(sd_union(*h), rg["pipe"], z=0, h=2.2, bevel=PW,
                          dome=True, spec=0.4))
    b = _bands(mask)
    if b:
        parts.append(Part(sd_union(*b), "dark", z=0.6, h=1.8, bevel=0.9,
                          dome=True))
    return parts


def pipe_tile(mask, rg, variant=0):
    parts = _pipe_axis_parts(mask, rg)
    nconn = bin(mask).count("1")
    straight = mask in (U | D, L | R)
    if nconn >= 3 or (nconn == 2 and not straight):
        # Junction hub
        parts.append(Part(sd_box(0, 0, 5.2, 5.2, 1.6), rg["pipe"], z=0.4,
                          h=2.4, bevel=2.6, dome=True, spec=0.5))
        parts.append(Part(sd_circle(0, 0, 2.0), rg["accent"], z=2.6, h=0.6,
                          bevel=1.0, dome=True))
        if nconn == 4 and variant == 1:
            # Hatch cross on the alternative four-way tile
            parts.append(Part(sd_union(sd_box(0, 0, 3.6, 0.5),
                                       sd_box(0, 0, 0.5, 3.6)),
                              "dark", z=2.8, h=0.3, bevel=0.4))
    elif straight and variant:
        vert = mask == (U | D)

        def o(x, y):
            return (x, y) if vert else (y, x)

        if variant == 1:
            # Vent grille
            for k in (-2, 0, 2):
                cx, cy = o(0, k)
                w, hh = o(2.6, 0.45)
                parts.append(Part(sd_box(cx, cy, w, hh), "dark", z=2.0,
                                  h=0.2, bevel=0.4))
        elif variant == 2:
            # Status light
            cx, cy = o(0, 0)
            parts.append(Part(sd_circle(cx, cy, 1.9), "dark", z=1.8, h=0.5,
                              bevel=0.7))
            parts.append(Part(sd_circle(cx, cy, 1.2), rg["eye"], glow=True,
                              bevel=1.2))
        elif variant == 3:
            # Valve wheel
            cx, cy = o(0, 0)
            parts.append(Part(sd_ring(cx, cy, 2.6, 0.7), rg["accent"],
                              z=2.4, h=0.6, bevel=0.6, dome=True))
            parts.append(Part(sd_circle(cx, cy, 0.9), "dark", z=2.4, h=0.4,
                              bevel=0.5))
    return parts


def node_tile(mask, rg, hard=False):
    """End node. 'mask' is the side the pipe leaves through. Destructible
    nodes are gun turrets; hard nodes are armoured caps."""
    parts = _pipe_axis_parts(mask, rg)
    ang = {U: math.pi, R: -math.pi / 2, D: 0.0, L: math.pi / 2}[mask]
    if hard:
        oct_ = sd_poly([(math.cos(a) * 7.2, math.sin(a) * 7.2)
                        for a in [math.pi / 8 + k * math.pi / 4
                                  for k in range(8)]])
        parts.append(Part(oct_, "dark", z=0.4, h=3.0, bevel=3.0, dome=False,
                          spec=0.5, bright=0.1))
        parts.append(Part(sd_box(0, 0, 3.2, 3.2, 0.8), rg["pipe"], z=3.2,
                          h=0.8, bevel=1.2, spec=0.6))
        for k in range(4):
            a = math.pi / 4 + k * math.pi / 2
            parts.append(Part(sd_circle(math.cos(a) * 5.0, math.sin(a) * 5.0,
                                        0.8), "steel", z=3.2, h=0.4,
                              bevel=0.7, dome=True))
    else:
        barrel = sd_rot(sd_box(0, -5.8, 1.3, 2.4, 0.4), ang)
        parts.append(Part(sd_circle(0, 0, 6.6), rg["pipe"], z=0.4, h=2.6,
                          bevel=3.0, dome=True, spec=0.5, bright=-0.05))
        parts.append(Part(barrel, "dark", z=2.6, h=1.0, bevel=1.0,
                          dome=True))
        parts.append(Part(sd_circle(0, 0, 3.4), "dark", z=2.8, h=1.0,
                          bevel=1.2, dome=True))
        parts.append(Part(sd_circle(0, 0, 2.0), rg["eye"], glow=True,
                          bevel=2.0, bright=0.1))
    return parts


def core_tile(frame, vertical, rg):
    mask = (U | D) if vertical else (L | R)
    parts = _pipe_axis_parts(mask, rg)
    pulse = 0.5 + 0.5 * math.sin(2 * math.pi * frame / 8)
    shell = sd_box(0, 0, 7.6, 7.6, 3.0)
    parts.append(Part(shell, rg["pipe"], z=0.5, h=3.2, bevel=3.4, dome=True,
                      spec=0.5, bright=-0.1))
    # Cooling fins
    for k in range(4):
        a = math.pi / 4 + k * math.pi / 2
        parts.append(Part(sd_rot(sd_box(0, -6.2, 1.0, 1.4, 0.3), a), "dark",
                          z=3.0, h=0.6, bevel=0.6))
    parts.append(Part(sd_circle(0, 0, 4.8), "dark", z=3.0, h=0.8, bevel=1.2))
    parts.append(Part(sd_circle(0, 0, 3.6 + 0.6 * pulse), rg["core"],
                      glow=True, bevel=3.0, bright=-0.25 + 0.4 * pulse))
    # Spinning spokes over the glow
    a0 = 2 * math.pi * frame / 8 / 3
    spokes = sd_union(*[sd_rot(sd_box(0, -2.6, 0.45, 1.6), a0 + k * 2 *
                               math.pi / 3) for k in range(3)])
    parts.append(Part(spokes, "dark", z=3.6, h=0.2, bevel=0.3))
    return parts


def scrap_tile(side, variant, rg):
    """Broken stub still attached on 'side'."""
    rng = np.random.RandomState(side * 7 + variant)
    ln = 4.5 + variant * 1.5
    if side == U:
        base = sd_box(0, -8 + ln / 2, PW - 0.6, ln / 2)
    elif side == D:
        base = sd_box(0, 8 - ln / 2, PW - 0.6, ln / 2)
    elif side == L:
        base = sd_box(-8 + ln / 2, 0, ln / 2, PW - 0.6)
    else:
        base = sd_box(8 - ln / 2, 0, ln / 2, PW - 0.6)
    bites = []
    for k in range(3):
        bx = rng.uniform(-3, 3)
        by = rng.uniform(-3, 3)
        if side in (U, D):
            by = (-8 + ln) if side == U else (8 - ln)
            by += rng.uniform(-1.2, 1.2)
        else:
            bx = (-8 + ln) if side == L else (8 - ln)
            bx += rng.uniform(-1.2, 1.2)
        bites.append(sd_circle(bx, by, rng.uniform(1.0, 1.8)))
    stub = sd_sub(base, sd_union(*bites))
    debris = sd_union(*[sd_circle(rng.uniform(-5, 5), rng.uniform(-5, 5),
                                  rng.uniform(0.6, 1.0)) for _ in range(3)])
    return [
        Part(debris, "rock", z=0, h=0.5, bevel=0.6, dome=True),
        Part(stub, "dark", z=0, h=1.6, bevel=PW - 0.6, dome=True, spec=0.2),
    ]


def tileset(region, size=16):
    rg = REGIONS[region]
    blank = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    tiles = [blank] * 56
    scale = size / 16.0
    outline = size >= 12

    def r(parts):
        return render(parts, size, size, scale=scale, outline=outline)

    tiles = list(tiles)
    tiles[0] = r(pipe_tile(15, rg, variant=1))
    for m in range(1, 16):
        if m in (U, R, D, L):
            tiles[m] = r(node_tile(m, rg))
        else:
            tiles[m] = r(pipe_tile(m, rg))
    for f in range(8):
        tiles[16 + f] = r(core_tile(f, True, rg))
        tiles[24 + f] = r(core_tile(f, False, rg))
    for k, m in enumerate((U, R, D, L)):
        tiles[32 + k] = r(node_tile(m, rg, hard=True))
    for k in range(3):
        tiles[40 + 2 * k] = r(pipe_tile(U | D, rg, variant=k + 1))
        tiles[41 + 2 * k] = r(pipe_tile(L | R, rg, variant=k + 1))
    for k, m in enumerate((U, R, D, L)):
        for v in range(2):
            tiles[48 + 2 * k + v] = r(scrap_tile(m, v, rg))
    return sheet(tiles, 8)
