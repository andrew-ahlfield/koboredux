"""Deterministic value noise, optionally tileable, for procedural textures."""

import numpy as np


def _hash(ix, iy, seed):
    ix = np.asarray(ix, dtype=np.int64)
    iy = np.asarray(iy, dtype=np.int64)
    h = (ix * 374761393 + iy * 668265263 + seed * 1442695041) & 0xffffffff
    h = ((h ^ (h >> 13)) * 1274126177) & 0xffffffff
    return ((h ^ (h >> 16)) & 0xffff) / 65535.0


def value2(x, y, seed=0, period=None):
    """Smooth value noise in [0, 1]. If period is given, tiles every
    'period' units on both axes."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = x - x0
    fy = y - y0
    x1 = x0 + 1
    y1 = y0 + 1
    if period:
        px, py = period if isinstance(period, tuple) else (period, period)
        x0 %= px
        x1 %= px
        y0 %= py
        y1 %= py
    sx = fx * fx * (3 - 2 * fx)
    sy = fy * fy * (3 - 2 * fy)
    a = _hash(x0, y0, seed)
    b = _hash(x1, y0, seed)
    c = _hash(x0, y1, seed)
    d = _hash(x1, y1, seed)
    return (a + (b - a) * sx) * (1 - sy) + (c + (d - c) * sx) * sy


def fbm2(x, y, seed=0, octaves=4, period=None, gain=0.5):
    """Fractal sum, roughly in [0, 1]. With 'period', x and y must be given
    in units where the texture repeats every 'period' units."""
    total = 0.0
    amp = 1.0
    norm = 0.0
    f = 1.0
    for o in range(octaves):
        if isinstance(period, tuple):
            p = (int(period[0] * f), int(period[1] * f))
        else:
            p = int(period * f) if period else None
        total = total + amp * value2(np.asarray(x) * f, np.asarray(y) * f,
                                     seed + o * 17, p)
        norm += amp
        amp *= gain
        f *= 2.0
    return total / norm
