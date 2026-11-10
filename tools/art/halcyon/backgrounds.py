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


TERRAIN = [
    # Region 1: blue grey moon
    [rgb(i) for i in (1, 2, 3, 4, 5, 6, 7)],
    # Region 2: ochre desert world
    [rgb(i) for i in (14, 25, 26, 27, 28, 19, 20)],
    # Region 3: jungle world
    [rgb(i) for i in (9, 29, 10, 30, 11, 31, 13)],
    # Region 4: violet crystal world
    [rgb(i) for i in (14, 21, 15, 22, 23, 24, 8)],
    # Region 5: ice world
    [rgb(i) for i in (2, 3, 5, 6, 7, 13, 8)],
]


def terrain(size, region, seed=0, P=4, crater_count=60, liquid=False):
    """Seamless lit terrain texture, quantised to the region ramp."""
    ys, xs = np.mgrid[0:size, 0:size].astype(float)
    u = xs / size * P
    v = ys / size * P
    hgt = noise.fbm2(u, v, seed=seed + region * 31, octaves=7, period=P)
    ridge = 1 - np.abs(noise.fbm2(u * 2, v * 2, seed=seed + 5, octaves=5,
                                  period=P * 2) * 2 - 1)
    hgt = hgt * 0.75 + ridge * 0.25
    rng = np.random.RandomState(seed + region * 7 + size)
    for _ in range(crater_count):
        cx, cy = rng.uniform(0, size), rng.uniform(0, size)
        r = rng.uniform(3, size / 14.0) * (rng.uniform(0.3, 1.0) ** 2)
        r = max(r, 2.5)
        dx = (xs - cx + size / 2) % size - size / 2
        dy = (ys - cy + size / 2) % size - size / 2
        d = np.hypot(dx, dy) / r
        bowl = np.where(d < 1, (d * d - 1) * 0.18, 0)
        rim = np.where((d >= 0.85) & (d < 1.5),
                       0.07 * np.cos((d - 1.0) / 0.65 * math.pi / 2) ** 2, 0)
        hgt = hgt + bowl + rim
    # Lighting from the wrapped gradient
    gx = (np.roll(hgt, -1, axis=1) - np.roll(hgt, 1, axis=1)) * size / 8.0
    gy = (np.roll(hgt, -1, axis=0) - np.roll(hgt, 1, axis=0)) * size / 8.0
    nl = np.sqrt(gx * gx + gy * gy + 1)
    L = np.array([-0.6, -0.6, 0.55])
    L = L / np.linalg.norm(L)
    lit = np.clip((-gx * L[0] - gy * L[1] + L[2]) / nl, 0, 1)
    hn = (hgt - hgt.min()) / (hgt.max() - hgt.min() + 1e-9)
    val = 0.55 * lit + 0.45 * hn
    if liquid:
        sea = hn < 0.38
        val = np.where(sea, 0.12 + 0.1 * hn, val)
    val = (val - val.min()) / (val.max() - val.min() + 1e-9)
    return Image.fromarray(_quant(val, TERRAIN[region], dither=0.8), "RGB")


def planet(region):
    return terrain(512, region, seed=11, P=4, crater_count=80,
                   liquid=(region == 2))


def ground_l8(region):
    return terrain(384, region, seed=23, P=3, crater_count=40,
                   liquid=(region == 2))


def ground_l10(region):
    return terrain(768, region, seed=37, P=5, crater_count=120,
                   liquid=(region == 2))
