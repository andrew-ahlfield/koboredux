"""Screen effects: interference noise and grid transition tiles."""

import numpy as np
from PIL import Image

from .palette import rgb
from .render import sheet


def noise_lines(ramp, seed, frames=16, width=256):
    """Each frame is one 256 px line of coloured static."""
    rng = np.random.RandomState(seed)
    img = np.zeros((frames, width, 4), dtype=np.uint8)
    n = len(ramp)
    for f in range(frames):
        level = (f + 1) / frames
        v = rng.random_sample(width)
        # Streaks: runs of similar value read better than pure static
        v = np.convolve(v, np.ones(3) / 3, mode="same")
        for x in range(width):
            if v[x] < 1 - level * 0.9:
                continue
            k = int(np.clip(v[x] * n, 0, n - 1))
            img[f, x, :3] = ramp[k]
            img[f, x, 3] = 255
    return Image.fromarray(img, "RGBA")


def noise():
    return noise_lines([rgb(i) for i in (2, 3, 10, 11, 4, 12, 7)], 5)


def hitnoise():
    return noise_lines([rgb(i) for i in (14, 15, 16, 22, 17, 23, 24)], 9)


def grid_tiles(size, levels, edge, fill, light, dark):
    """Transition tiles: frame 0 fully covers the cell, the last frame is
    almost gone. The engine picks the level from the transition progress."""
    frames = []
    for i in range(levels):
        t = 1 - i / float(levels)
        s = max(1, int(round(size * t)))
        img = np.zeros((size, size, 4), dtype=np.uint8)
        o = (size - s) // 2
        img[o:o + s, o:o + s] = (*fill, 255)
        if s >= 4:
            img[o, o:o + s] = (*light, 255)
            img[o:o + s, o] = (*light, 255)
            img[o + s - 1, o:o + s] = (*dark, 255)
            img[o:o + s, o + s - 1] = (*dark, 255)
        if s >= 10:
            # Inset detail: a small engraved square
            q = s // 4
            img[o + q, o + q:o + s - q] = (*dark, 255)
            img[o + q:o + s - q, o + q] = (*dark, 255)
            img[o + s - q - 1, o + q:o + s - q] = (*edge, 255)
            img[o + q:o + s - q, o + s - q - 1] = (*edge, 255)
        frames.append(Image.fromarray(img, "RGBA"))
    return frames


def gridtfx_small():
    return sheet(grid_tiles(16, 16, rgb(4), rgb(2), rgb(11), rgb(1)), 4)


def gridtfx_fullscreen():
    return sheet(grid_tiles(40, 32, rgb(4), rgb(2), rgb(5), rgb(1)), 8)
