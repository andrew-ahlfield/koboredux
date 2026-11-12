"""Tiny shaded pixel art renderer.

Objects are described as an ordered list of parts. Each part is a 2D signed
distance shape in object space (pixels, origin at the sprite centre, nose
pointing up along -y) plus a bevel profile, a base height and a material
ramp. The renderer rotates the sample grid into object space, builds a height
field by painting parts in order, derives normals from the height field,
lights it with a fixed top-left light (so lighting stays put while the
object rotates, the way a hand-pixelled rotation set would), quantises each
pixel to its material ramp and finally adds a selective dark outline.

Everything is deterministic: same input, same pixels.
"""

import math
import numpy as np
from PIL import Image

from .palette import RAMPS, OUTLINE

SS = 4  # supersampling factor per axis

LIGHT = np.array([-0.55, -0.7, 0.9])
LIGHT = LIGHT / np.linalg.norm(LIGHT)


# ---------------------------------------------------------------------------
# Signed distance helpers (vectorised; x, y are numpy arrays)
# ---------------------------------------------------------------------------

def sd_circle(cx, cy, r):
    return lambda x, y: np.hypot(x - cx, y - cy) - r


def sd_ellipse(cx, cy, rx, ry):
    # Cheap approximation, good enough at sprite scale.
    def f(x, y):
        k = np.hypot((x - cx) / rx, (y - cy) / ry)
        return (k - 1.0) * min(rx, ry)
    return f


def sd_box(cx, cy, hw, hh, r=0.0):
    def f(x, y):
        qx = np.abs(x - cx) - hw + r
        qy = np.abs(y - cy) - hh + r
        out = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))
        inn = np.minimum(np.maximum(qx, qy), 0)
        return out + inn - r
    return f


def sd_capsule(ax, ay, bx, by, r):
    def f(x, y):
        pax, pay = x - ax, y - ay
        bax, bay = bx - ax, by - ay
        h = np.clip((pax * bax + pay * bay) / (bax * bax + bay * bay), 0, 1)
        return np.hypot(pax - bax * h, pay - bay * h) - r
    return f


def sd_poly(pts):
    pts = [(float(a), float(b)) for a, b in pts]

    def f(x, y):
        d = np.full(x.shape, 1e9)
        s = np.ones(x.shape)
        n = len(pts)
        for i in range(n):
            ax, ay = pts[i]
            bx, by = pts[(i + 1) % n]
            ex, ey = bx - ax, by - ay
            wx, wy = x - ax, y - ay
            h = np.clip((wx * ex + wy * ey) / (ex * ex + ey * ey), 0, 1)
            dx, dy = wx - ex * h, wy - ey * h
            d = np.minimum(d, dx * dx + dy * dy)
            c1 = y >= ay
            c2 = y < by
            c3 = ex * wy > ey * wx
            flip = (c1 & c2 & c3) | (~c1 & ~c2 & ~c3)
            s = np.where(flip, -s, s)
        return s * np.sqrt(d)
    return f


def sd_union(*fs):
    def f(x, y):
        d = fs[0](x, y)
        for g in fs[1:]:
            d = np.minimum(d, g(x, y))
        return d
    return f


def sd_sub(a, b):
    return lambda x, y: np.maximum(a(x, y), -b(x, y))


def sd_inter(a, b):
    return lambda x, y: np.maximum(a(x, y), b(x, y))


def sd_ring(cx, cy, r, w):
    return lambda x, y: np.abs(np.hypot(x - cx, y - cy) - r) - w


def mirror_x(f):
    return lambda x, y: f(np.abs(x), y)


def sd_rot(f, ang, cx=0.0, cy=0.0):
    """Rotate shape f by ang radians (clockwise on screen) around cx, cy."""
    c, s = math.cos(ang), math.sin(ang)

    def g(x, y):
        dx, dy = x - cx, y - cy
        return f(cx + dx * c + dy * s, cy - dx * s + dy * c)
    return g


# ---------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------

class Part:
    """One shaded shape.

    sdf      signed distance function in object space
    mat      material ramp name, or a list of RGB tuples
    z        base height
    h        bevel height
    bevel    bevel width in pixels
    dome     use a rounded profile instead of a flat chamfer
    glow     emissive: brightness from bevel profile, no lighting
    spec     specular strength
    bright   intensity offset (to push a part lighter or darker)
    hfunc    optional extra height function added inside the part
    facet    optional object space normal (nx, ny, nz) for a flat facet:
             the whole part is lit as one plane (nx mirrored for x < 0, so
             a mirror_x pair gets mirrored facets) instead of from the
             height field, so a thin strip such as a leading edge shades
             as one solid tone at every rotation instead of breaking into
             a checker of alternating ramp steps
    """

    def __init__(self, sdf, mat, z=0.0, h=1.0, bevel=1.5, dome=False,
                 glow=False, spec=0.35, bright=0.0, hfunc=None, cut=False,
                 facet=None):
        self.sdf = sdf
        self.mat = RAMPS[mat] if isinstance(mat, str) else mat
        self.z = z
        self.h = h
        self.bevel = bevel
        self.dome = dome
        self.glow = glow
        self.spec = spec
        self.bright = bright
        self.hfunc = hfunc
        self.cut = cut  # paints a groove instead of a raised part
        self.facet = facet


def _grid(w, h, ang, ox=0.0, oy=0.0, scale=1.0):
    ys, xs = np.mgrid[0:h * SS, 0:w * SS]
    sx = (xs + 0.5) / SS - w / 2.0 - ox
    sy = (ys + 0.5) / SS - h / 2.0 - oy
    sx = sx / scale
    sy = sy / scale
    c, s = math.cos(ang), math.sin(ang)
    # Screen = R(ang) * object  =>  object = R(-ang) * screen
    x = sx * c + sy * s
    y = -sx * s + sy * c
    return x, y


def render(parts, w, h, ang=0.0, outline=True, dither=False, ox=0.0, oy=0.0,
           scale=1.0, light=None, coverage=0.5, outline_color=None,
           shade_bias=0.0, cast_shadow=0.0):
    """Render parts to an RGBA PIL image of w x h pixels."""
    L = LIGHT if light is None else light
    x, y = _grid(w, h, ang, ox, oy, scale)
    H = np.full(x.shape, -1.0)
    M = np.full(x.shape, -1, dtype=np.int32)
    inside_any = np.zeros(x.shape, dtype=bool)
    for i, p in enumerate(parts):
        d = p.sdf(x, y) * scale
        ins = d < 0
        if not ins.any():
            continue
        t = np.clip(-d / max(p.bevel, 1e-3), 0, 1)
        if p.dome:
            prof = np.sqrt(np.clip(1 - (1 - t) ** 2, 0, 1))
        else:
            prof = t
        hh = p.z + p.h * prof * (-1 if p.cut else 1)
        if p.hfunc is not None:
            hh = hh + p.hfunc(x, y)
        H = np.where(ins, hh, H)
        M = np.where(ins, i, M)
        inside_any |= ins

    # Normals from the height field (in supersampled pixels)
    Hf = np.where(inside_any, H, np.nan)
    Hfill = np.where(inside_any, H, 0.0)
    gy, gx = np.gradient(Hfill)
    gx = np.nan_to_num(gx) * SS
    gy = np.nan_to_num(gy) * SS
    # Edge pixels: gradients against the background are meaningless
    edge = np.zeros_like(inside_any)
    edge[1:, :] |= ~inside_any[:-1, :]
    edge[:-1, :] |= ~inside_any[1:, :]
    edge[:, 1:] |= ~inside_any[:, :-1]
    edge[:, :-1] |= ~inside_any[:, 1:]
    gx = np.where(edge & inside_any, 0, gx)
    gy = np.where(edge & inside_any, 0, gy)
    gx = np.clip(gx, -3, 3)
    gy = np.clip(gy, -3, 3)
    nz = np.ones_like(gx)
    nl = np.sqrt(gx * gx + gy * gy + nz)
    nx, ny, nz = -gx / nl, -gy / nl, nz / nl
    diff = np.clip(nx * L[0] + ny * L[1] + nz * L[2], 0, 1)
    # Specular (viewer straight above)
    rz = 2 * diff * nz - L[2]
    spec_term = np.clip(rz, 0, 1) ** 12

    I = np.zeros(x.shape)
    for i, p in enumerate(parts):
        sel = M == i
        if not sel.any():
            continue
        if p.glow:
            d = p.sdf(x, y) * scale
            t = np.clip(-d / max(p.bevel, 1e-3), 0, 1)
            val = 0.35 + 0.65 * t + p.bright
        elif p.facet is not None:
            fx, fy, fz = p.facet
            fx = np.where(x < 0, -fx, fx)
            fn = math.sqrt(p.facet[0] ** 2 + fy * fy + fz * fz)
            ca, sa = math.cos(ang), math.sin(ang)
            snx = (fx * ca - fy * sa) / fn
            sny = (fx * sa + fy * ca) / fn
            fd = np.clip(snx * L[0] + sny * L[1] + fz / fn * L[2], 0, 1)
            val = 0.12 + 0.95 * fd + p.bright + shade_bias
        else:
            val = 0.12 + 0.95 * diff + p.spec * spec_term + p.bright \
                + shade_bias
        I = np.where(sel, val, I)

    # Downsample
    def ds(a):
        return a.reshape(h, SS, w, SS).mean(axis=(1, 3))

    cov = ds(inside_any.astype(float))
    # Material: pick the most common material among covered subsamples
    Mr = M.reshape(h, SS, w, SS).transpose(0, 2, 1, 3).reshape(h, w, SS * SS)
    Ir = I.reshape(h, SS, w, SS).transpose(0, 2, 1, 3).reshape(h, w, SS * SS)
    out = np.zeros((h, w, 4), dtype=np.uint8)
    bayer = np.array([[0, 2], [3, 1]]) / 4.0 - 0.375
    matmap = np.full((h, w), -1, dtype=np.int32)
    for py in range(h):
        for px in range(w):
            if cov[py, px] < coverage:
                continue
            ms = Mr[py, px]
            ms = ms[ms >= 0]
            vals, counts = np.unique(ms, return_counts=True)
            # Prefer topmost part when counts are close
            best = vals[np.argmax(counts + vals * 0.01)]
            iv = Ir[py, px][Mr[py, px] == best].mean()
            ramp = parts[best].mat
            n = len(ramp)
            f = iv * (n - 1)
            if dither and not parts[best].glow:
                f += bayer[py & 1, px & 1] * 0.5
            k = int(np.clip(round(f), 0, n - 1))
            out[py, px, :3] = ramp[k]
            out[py, px, 3] = 255
            matmap[py, px] = best

    if outline:
        oc = OUTLINE if outline_color is None else outline_color
        a = out[:, :, 3] > 0
        o = np.zeros_like(a)
        o[1:, :] |= a[:-1, :]
        o[:-1, :] |= a[1:, :]
        o[:, 1:] |= a[:, :-1]
        o[:, :-1] |= a[:, 1:]
        o &= ~a
        out[o, :3] = oc
        out[o, 3] = 255
    return Image.fromarray(out, "RGBA")


def sheet(frames, cols):
    fw, fh = frames[0].size
    rows = (len(frames) + cols - 1) // cols
    im = Image.new("RGBA", (fw * cols, fh * rows), (0, 0, 0, 0))
    for i, f in enumerate(frames):
        im.paste(f, ((i % cols) * fw, (i // cols) * fh))
    return im


def rotations(build, n, w, h, **kw):
    """Render n clockwise rotations starting with the nose pointing up."""
    parts = build()
    return [render(parts, w, h, ang=2 * math.pi * i / n, **kw)
            for i in range(n)]


def glow_dot(w, h, cx, cy, r, ramp, falloff=1.0):
    """Soft emissive blob quantised to a ramp (no outline)."""
    ys, xs = np.mgrid[0:h, 0:w]
    d = np.hypot(xs + 0.5 - cx, ys + 0.5 - cy) / max(r, 1e-3)
    v = np.clip(1 - d, 0, 1) ** falloff
    out = np.zeros((h, w, 4), dtype=np.uint8)
    n = len(ramp)
    for yy in range(h):
        for xx in range(w):
            if v[yy, xx] <= 0.02:
                continue
            k = int(np.clip(v[yy, xx] * n, 0, n - 1))
            out[yy, xx, :3] = ramp[k]
            out[yy, xx, 3] = 255
    return Image.fromarray(out, "RGBA")
