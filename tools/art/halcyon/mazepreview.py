"""Preview base tiles assembled into engine style mazes.

Usage (from tools/art):
    python3 -m halcyon.mazepreview [-s scale] out.png [region ...]

Builds a few bases the way the engine's map.cpp does (a core on a two
cell grid, a random spanning maze grown from it, connection masks from the
neighbours, some end nodes made indestructible), then draws them over the
region's ground with the 16 px tiles, and a second strip with the 12 and
8 px parallax tiles dimmed as the theme dims them. Regions are 1 to 5;
the default is all five.
"""

import sys
import numpy as np
from PIL import Image

from . import tiles, backgrounds

U, R, D, L = 1, 2, 4, 8
CORE, HARD = 16, 32


def maze(w, h, rng, bases):
    """Return a w x h grid of cell masks (0 empty)."""
    g = np.zeros((h, w), dtype=int)
    occ = np.zeros((h, w), dtype=bool)
    for (cx, cy, dx, dy) in bases:
        # Core: horizontal or vertical
        sites = []
        if rng.rand() < 0.5:
            occ[cy, cx] = True
            g[cy, cx] = CORE
            sites += [(cx - 1, cy), (cx + 1, cy)]
        else:
            occ[cy, cx] = True
            g[cy, cx] = CORE
            sites += [(cx, cy - 1), (cx, cy + 1)]
        for (x, y) in sites:
            occ[y, x] = True
        while sites:
            i = rng.randint(len(sites))
            sites[i], sites[-1] = sites[-1], sites[i]
            vx, vy = sites.pop()
            dirs = []
            for (ddx, ddy) in ((2, 0), (0, 2), (-2, 0), (0, -2)):
                x, y = vx + ddx, vy + ddy
                if abs(x - cx) > dx or abs(y - cy) > dy:
                    continue
                if x < 0 or y < 0 or x >= w or y >= h or occ[y, x]:
                    continue
                dirs.append((ddx, ddy))
            if not dirs:
                continue
            ddx, ddy = dirs[rng.randint(len(dirs))]
            occ[vy + ddy // 2, vx + ddx // 2] = True
            occ[vy + ddy, vx + ddx] = True
            sites.append((vx + ddx, vy + ddy))
            sites.append((vx, vy))
    out = np.zeros((h, w), dtype=int)
    for y in range(h):
        for x in range(w):
            if not occ[y, x]:
                continue
            p = g[y, x] & CORE
            if y > 0 and occ[y - 1, x]:
                p |= U
            if x < w - 1 and occ[y, x + 1]:
                p |= R
            if y < h - 1 and occ[y + 1, x]:
                p |= D
            if x > 0 and occ[y, x - 1]:
                p |= L
            if p in (U, R, D, L) and rng.rand() < 0.3:
                p |= HARD
            out[y, x] = p | 64
    return out


def tile_index(p, rng):
    if p & CORE:
        return 16 if p & (U | D) else 24
    m = p & 15
    if p & HARD:
        return {U: 32, R: 33, D: 34, L: 35}[m]
    if m == 5:
        return [40, 42, 44, 5, 5, 5, 5, 5][rng.randint(8)]
    if m == 10:
        return [41, 43, 45, 10, 10, 10, 10, 10][rng.randint(8)]
    if m == 15:
        return 0 if rng.rand() < 0.5 else 15
    return m


def compose(region, grid, size, ground, dim=1.0, seed=1):
    rng = np.random.RandomState(seed)
    sh = tiles.tileset(region, size)
    h, w = grid.shape
    img = ground.copy().resize((w * size, h * size))
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    for y in range(h):
        for x in range(w):
            if not grid[y, x]:
                continue
            i = tile_index(grid[y, x], rng)
            t = sh.crop(((i % 8) * size, (i // 8) * size,
                         (i % 8 + 1) * size, (i // 8 + 1) * size))
            layer.alpha_composite(t, (x * size, y * size))
    if dim < 1.0:
        a = np.array(layer).astype(float)
        a[..., :3] *= dim
        layer = Image.fromarray(a.astype(np.uint8), "RGBA")
    img.alpha_composite(layer)
    return img


def preview(region, seed=3):
    rng = np.random.RandomState(seed)
    w, h = 30, 22
    bases = [(7, 6, 6, 5), (21, 7, 7, 6), (11, 16, 9, 4), (25, 17, 4, 4)]
    grid = maze(w, h, rng, bases)
    gr = backgrounds.ground_l8(region).convert("RGBA")
    tiled = Image.new("RGBA", (w * 16, h * 16))
    for yy in range(0, h * 16, gr.height):
        for xx in range(0, w * 16, gr.width):
            tiled.paste(gr, (xx, yy))
    near = compose(region, grid, 16, tiled)
    gsmall = Image.new("RGBA", (w * 12, h * 12), (8, 10, 22, 255))
    far12 = compose(region, grid, 12, gsmall, dim=0.6)
    far8 = compose(region, grid, 8, gsmall.crop((0, 0, w * 8, h * 8)),
                   dim=0.38)
    o = Image.new("RGBA", (near.width + far12.width + 8,
                           max(near.height, far12.height + far8.height + 8)),
                  (0, 0, 0, 255))
    o.paste(near, (0, 0))
    o.paste(far12, (near.width + 8, 0))
    o.paste(far8, (near.width + 8, far12.height + 8))
    return o


def main():
    args = sys.argv[1:]
    s = 2
    if args and args[0] == "-s":
        s = int(args[1])
        args = args[2:]
    out = args[0]
    regions = [int(a) - 1 for a in args[1:]] or list(range(5))
    ims = [preview(r) for r in regions]
    W = max(i.width for i in ims)
    H = sum(i.height + 8 for i in ims)
    o = Image.new("RGBA", (W, H), (0, 0, 0, 255))
    y = 0
    for i in ims:
        o.paste(i, (0, y))
        y += i.height + 8
    o = o.resize((o.width * s, o.height * s), Image.NEAREST)
    o.save(out)


if __name__ == "__main__":
    main()
