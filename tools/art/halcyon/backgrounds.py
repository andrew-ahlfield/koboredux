"""Seamless backdrops: deep space and per-region planet surfaces.

The engine tiles every backdrop in both directions and uses the planet image
as a wrapping sphere texture, so everything here is generated with periodic
noise and wrapped distances.
"""

import math
import numpy as np
from PIL import Image

from . import noise
from .palette import rgb
from .pix import hramp, glow_ramp
from .tiles import REGIONS as TILE_REGIONS


def _quant(vals, ramp, dither=0.0):
    """Map [0, 1] values to a colour ramp with optional ordered dither."""
    h, w = vals.shape
    bayer = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9],
                      [15, 7, 13, 5]]) / 16.0 - 0.47
    by = np.tile(bayer, (h // 4 + 1, w // 4 + 1))[:h, :w]
    n = len(ramp)
    k = np.clip(np.round(vals * (n - 1) + by * dither), 0, n - 1)
    k = k.astype(int)
    pal = np.array(ramp, dtype=np.uint8)
    return pal[k]


def space(w=512, h=1024, seed=1):
    ys, xs = np.mgrid[0:h, 0:w].astype(float)
    P = 3  # noise period in cells across the width
    PP = (P, P * h // w)
    u = xs / w * P
    v = ys / w * P  # same scale both axes; height is 2 periods tall
    neb = noise.fbm2(u, v, seed=seed, octaves=6, period=PP)
    warp = noise.fbm2(u, v, seed=seed + 9, octaves=4, period=PP)
    neb2 = noise.fbm2(u + warp * 1.5 * P, v + warp * 1.5 * P, seed=seed + 4,
                      octaves=6, period=PP)
    # Two coloured gas layers
    a = np.clip((neb - 0.42) * 2.8, 0, 1) ** 1.4
    b = np.clip((neb2 - 0.47) * 3.0, 0, 1) ** 1.5
    base = np.array(rgb(1), float)
    teal = np.array(rgb(10), float)
    teal2 = np.array(rgb(11), float)
    vio = np.array(rgb(21), float)
    vio2 = np.array(rgb(22), float)
    img = np.zeros((h, w, 3))
    img[:] = base
    img += (teal - base) * a[..., None] * 0.9
    img += (teal2 - teal) * np.clip(a - 0.6, 0, 1)[..., None] * 1.5
    img += (vio - base) * b[..., None] * 0.9
    img += (vio2 - vio) * np.clip(b - 0.55, 0, 1)[..., None] * 1.6
    # Dark dust lanes
    dust = np.clip((noise.fbm2(u * 2, v * 2, seed=seed + 21, octaves=5,
                               period=(PP[0] * 2, PP[1] * 2)) - 0.55) * 4,
                   0, 1)
    img *= (1 - 0.7 * dust)[..., None]
    # Posterise each channel to a small set of levels for a pixel art look,
    # with ordered dither between levels
    bayer = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9],
                      [15, 7, 13, 5]]) / 16.0 - 0.5
    by = np.tile(bayer, (h // 4, w // 4))
    lv = 10.0
    img = np.floor(img / 255.0 * lv + 0.5 + by[..., None] * 0.9) / lv * 255
    img = np.clip(img, 0, 255).astype(np.uint8)
    # Stars
    rng = np.random.RandomState(seed + 100)
    star_cols = [rgb(6), rgb(7), rgb(8), rgb(13), rgb(19), rgb(24)]
    for _ in range(900):
        x, y = rng.randint(0, w), rng.randint(0, h)
        c = star_cols[rng.randint(0, len(star_cols))]
        f = rng.uniform(0.3, 1.0)
        img[y, x] = (np.array(c) * f).astype(np.uint8)
    for _ in range(40):
        x, y = rng.randint(0, w), rng.randint(0, h)
        c = star_cols[rng.randint(0, len(star_cols))]
        img[y, x] = rgb(8)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            img[(y + dy) % h, (x + dx) % w] = c
    # A handful of big flared stars
    for _ in range(4):
        x, y = rng.randint(0, w), rng.randint(0, h)
        for k in range(-5, 6):
            f = 1 - abs(k) / 6.0
            for (xx, yy) in (((x + k) % w, y), (x, (y + k) % h)):
                old = img[yy, xx].astype(float)
                img[yy, xx] = np.clip(old + np.array(rgb(7)) * f, 0,
                                      255).astype(np.uint8)
        img[y, x] = rgb(8)
    return Image.fromarray(img, "RGB")


# Per region ground materials. 'a' is the main ground ramp, 'b' the
# secondary material (rock, vegetation, water, crystal, ice) mixed in by a
# biome map, 'glow' an emissive ramp for cracks and lights.
GROUNDS = [
    # 1: grey blue lunar rock with dark basalt seas
    dict(a=hramp(218, 0.22, 12, v0=0.07, v1=0.82, shift=0.3),
         b=hramp(232, 0.30, 10, v0=0.05, v1=0.55, shift=0.3),
         glow=glow_ramp(185), feature="craters"),
    # 2: ochre desert with rust red mesas and dune ripples
    dict(a=hramp(34, 0.52, 12, v0=0.08, v1=0.95, shift=0.35),
         b=hramp(12, 0.55, 10, v0=0.07, v1=0.72, shift=0.35),
         glow=glow_ramp(120), feature="dunes"),
    # 3: jungle canopy cut by teal rivers
    dict(a=hramp(110, 0.60, 12, v0=0.04, v1=0.66, shift=0.35),
         b=hramp(178, 0.60, 10, v0=0.06, v1=0.62, shift=0.3),
         glow=glow_ramp(24), feature="jungle"),
    # 4: violet alien crust with glowing magma fissures
    dict(a=hramp(262, 0.42, 12, v0=0.04, v1=0.62, shift=0.3),
         b=hramp(338, 0.55, 10, v0=0.06, v1=0.60, shift=0.3),
         glow=glow_ramp(330), feature="fissures"),
    # 5: ice sheet with deep blue crevasses
    dict(a=hramp(205, 0.30, 12, v0=0.12, v1=0.82, shift=0.25),
         b=hramp(214, 0.55, 10, v0=0.10, v1=0.70, shift=0.25),
         glow=glow_ramp(190), feature="ice"),
]


def _wrapd(xs, ys, cx, cy, size):
    dx = (xs - cx + size / 2) % size - size / 2
    dy = (ys - cy + size / 2) % size - size / 2
    return dx, dy


def terrain(size, region, seed=0, P=4, crater_count=60, liquid=False,
            outposts=0):
    """Seamless, lit, multi material terrain texture."""
    g = GROUNDS[region]
    ys, xs = np.mgrid[0:size, 0:size].astype(float)
    u = xs / size * P
    v = ys / size * P
    hgt = noise.fbm2(u, v, seed=seed + region * 31, octaves=7, period=P)
    ridge = 1 - np.abs(noise.fbm2(u * 2, v * 2, seed=seed + 5, octaves=5,
                                  period=P * 2) * 2 - 1)
    hgt = hgt * 0.7 + ridge * 0.3
    biome = noise.fbm2(u + 3.1, v, seed=seed + 77, octaves=5, period=P)
    feat = g["feature"]
    glow = np.zeros((size, size))
    if feat == "dunes":
        warp = noise.fbm2(u, v, seed=seed + 9, octaves=3, period=P)
        k = 2 * math.pi * P * 4
        rip = np.sin((xs / size) * k * 0.35 + (ys / size) * k + warp * 9)
        hgt = hgt + 0.02 * rip * (biome < 0.55)
    elif feat == "jungle":
        bumps = noise.fbm2(u * 8, v * 8, seed=seed + 3, octaves=3,
                           period=P * 8)
        hgt = hgt + 0.10 * bumps * (biome < 0.58)
    elif feat == "fissures":
        cr = np.abs(noise.fbm2(u * 2, v * 2, seed=seed + 41, octaves=5,
                               period=P * 2) * 2 - 1)
        glow = np.clip(1 - cr / 0.035, 0, 1)
        hgt = hgt - 0.08 * glow
    elif feat == "ice":
        cr = np.abs(noise.fbm2(u * 2, v * 2, seed=seed + 43, octaves=4,
                               period=P * 2) * 2 - 1)
        crev = np.clip(1 - cr / 0.03, 0, 1)
        hgt = hgt - 0.10 * crev
        glow = crev * 0.35
    rng = np.random.RandomState(seed + region * 7 + size)
    if feat in ("craters", "fissures", "ice", "dunes"):
        n_cr = crater_count if feat == "craters" else crater_count // 3
        for _ in range(n_cr):
            cx, cy = rng.uniform(0, size), rng.uniform(0, size)
            r = rng.uniform(3, size / 14.0) * (rng.uniform(0.3, 1.0) ** 2)
            r = max(r, 2.5)
            dx, dy = _wrapd(xs, ys, cx, cy, size)
            d = np.hypot(dx, dy) / r
            bowl = np.where(d < 1, (d * d - 1) * 0.18, 0)
            rim = np.where((d >= 0.85) & (d < 1.5),
                           0.07 * np.cos((d - 1.0) / 0.65 * math.pi / 2) **
                           2, 0)
            hgt = hgt + bowl + rim
    # Lighting from the wrapped gradient, light from the top left
    gx = (np.roll(hgt, -1, axis=1) - np.roll(hgt, 1, axis=1)) * size / 7.0
    gy = (np.roll(hgt, -1, axis=0) - np.roll(hgt, 1, axis=0)) * size / 7.0
    nl = np.sqrt(gx * gx + gy * gy + 1)
    Lv = np.array([-0.6, -0.65, 0.5])
    Lv = Lv / np.linalg.norm(Lv)
    lit = np.clip((-gx * Lv[0] - gy * Lv[1] + Lv[2]) / nl, 0, 1)
    hn = (hgt - hgt.min()) / (hgt.max() - hgt.min() + 1e-9)
    val = 0.6 * lit + 0.4 * hn
    val = (val - np.percentile(val, 1)) / (np.percentile(val, 99.5) -
                                          np.percentile(val, 1) + 1e-9)
    val = np.clip(val, 0, 1)
    # Material split with a dithered border
    bayer = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9],
                      [15, 7, 13, 5]]) / 16.0 - 0.47
    by = np.tile(bayer, (size // 4 + 1, size // 4 + 1))[:size, :size]
    if feat == "jungle" or liquid:
        second = (hn + by * 0.03) < 0.36
        valb = 0.25 + 0.5 * hn / 0.36 + 0.25 * lit
    else:
        second = (biome + by * 0.04) > 0.58
        valb = val
    ca = _quant(val, g["a"], dither=0.9)
    cb = _quant(np.clip(valb, 0, 1), g["b"], dither=0.9)
    img = np.where(second[..., None], cb, ca).astype(float)
    if feat == "jungle" or liquid:
        # Shore line foam and lit wave ticks on the water
        shore = second & ~np.roll(second, 1, 0) | \
            second & ~np.roll(second, 1, 1)
        img[shore] = g["b"][-1]
    # Emissive cracks
    if glow.any():
        gr = g["glow"]
        k = np.clip(np.round(glow * (len(gr) - 1) + by * 0.6), 0,
                    len(gr) - 1).astype(int)
        on = glow > 0.12
        img[on] = np.array(gr, float)[k[on]]
    # Outposts: small bevelled pads with lights, tying the ground to the
    # bases built on it
    metal = TILE_REGIONS[region]["metal"]
    light = TILE_REGIONS[region]["light"]
    for _ in range(outposts):
        w = int(rng.randint(4, 12)) * 2
        h = int(rng.randint(3, 8)) * 2
        x0 = int(rng.randint(0, size))
        y0 = int(rng.randint(0, size))
        for yy in range(-1, h + 3):
            for xx in range(-1, w + 3):
                X = (x0 + xx) % size
                Y = (y0 + yy) % size
                if xx in (-1, w + 2) or yy in (-1, h + 2):
                    img[Y, X] = img[Y, X] * 0.45
                elif xx >= w or yy >= h:
                    img[Y, X] = metal[1]
                elif yy == 0 or xx == 0:
                    img[Y, X] = metal[6]
                elif yy == h - 1 or xx == w - 1:
                    img[Y, X] = metal[2]
                elif (xx % 6 == 0) or (yy % 6 == 0):
                    img[Y, X] = metal[3]
                else:
                    img[Y, X] = metal[4]
        for k in range(2):
            lx = (x0 + 2 + k * (w - 5)) % size
            ly = (y0 + 2) % size
            img[ly, lx] = light[5]
            img[ly, (lx + 1) % size] = light[3]
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB")


def planet(region):
    return terrain(512, region, seed=11, P=4, crater_count=80,
                   liquid=(region == 2))


def ground_l8(region):
    return terrain(384, region, seed=23, P=3, crater_count=40,
                   liquid=(region == 2), outposts=6)


def ground_l10(region):
    return terrain(768, region, seed=37, P=5, crater_count=120,
                   liquid=(region == 2), outposts=18)
