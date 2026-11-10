"""Ships, enemies and projectiles for the Halcyon theme."""

import math
import numpy as np
from PIL import Image

from .render import (Part, render, sheet, rotations, glow_dot, sd_circle,
                     sd_ellipse, sd_box, sd_capsule, sd_poly, sd_union,
                     sd_sub, sd_inter, sd_ring, mirror_x, sd_rot)
from .palette import RAMPS, rgb
from . import noise


# ---------------------------------------------------------------------------
# Player: "Halcyon" interceptor. A chunky armoured gunship: a fat fuselage
# with a gun at each end (Kobo fires nose and tail), canards, broad
# cranked wings with red armoured leading edges, twin engine pods with
# glowing exhausts, and a bubble canopy with a glint.
# ---------------------------------------------------------------------------

def player_parts():
    body = sd_poly([(0, -10.4), (2.4, -7.6), (3.3, -2.0), (3.8, 2.6),
                    (5.4, 5.4), (5.4, 9.4), (-5.4, 9.4), (-5.4, 5.4),
                    (-3.8, 2.6), (-3.3, -2.0), (-2.4, -7.6)])
    wing = mirror_x(sd_poly([(2.5, -1.0), (9.0, 3.6), (11.0, 6.2),
                             (11.0, 8.4), (5.0, 8.4), (2.5, 6.5)]))
    edge = mirror_x(sd_poly([(2.8, -0.9), (9.2, 3.5), (11.0, 6.0),
                             (11.0, 7.1), (8.8, 5.0), (2.8, 1.1)]))
    canard = mirror_x(sd_poly([(2.2, -7.0), (5.6, -4.6), (5.6, -3.2),
                               (2.6, -4.0)]))
    nozzle = mirror_x(sd_box(2.9, 9.4, 1.6, 0.9, 0.3))
    flame = mirror_x(sd_circle(2.9, 10.6, 1.5))
    guns = sd_union(sd_capsule(0, -11.6, 0, -8.0, 0.9),
                    sd_capsule(0, 8.0, 0, 11.6, 0.9))
    canopy = sd_ellipse(0, -3.4, 1.9, 3.4)
    glint = sd_circle(-0.7, -4.8, 0.7)
    intake = mirror_x(sd_poly([(3.8, 2.8), (5.0, 5.0), (3.4, 5.0)]))
    lights = mirror_x(sd_circle(10.3, 7.6, 0.8))
    return [
        Part(guns, "dark", z=0.0, h=0.8, bevel=0.9, dome=True, spec=0.6),
        Part(wing, "gunship", z=0.2, h=1.2, bevel=2.2, spec=0.45,
             bright=-0.06),
        Part(edge, "red", z=1.2, h=0.4, bevel=0.8, spec=0.6),
        Part(lights, "teal", glow=True, bevel=0.8, bright=0.3),
        Part(canard, "gunship", z=0.4, h=0.8, bevel=1.2, spec=0.4),
        Part(flame, "hotfire", glow=True, bevel=1.5, bright=0.15),
        Part(body, "hull", z=1.0, h=2.4, bevel=3.4, dome=True, spec=0.7,
             bright=-0.1),
        Part(intake, "dark", z=2.2, h=0.2, bevel=0.4),
        Part(nozzle, "dark", z=1.2, h=0.6, bevel=0.6, spec=0.5),
        Part(canopy, "glass", z=2.8, h=1.4, bevel=1.6, dome=True, spec=1.0),
        Part(glint, "white", glow=True, bevel=0.7, bright=0.4),
    ]


def player():
    return sheet(rotations(player_parts, 64, 24, 24, shade_bias=-0.06), 8)


# ---------------------------------------------------------------------------
# Enemy 1: dart missile (16x16, 16 directions)
# ---------------------------------------------------------------------------

def dart_parts(body_mat, fin_mat, glow_mat="hotfire", fat=1.0):
    body = sd_capsule(0, -4.6, 0, 3.8, 2.0 * fat)
    nose = sd_poly([(0, -7.6), (1.9 * fat, -4.6), (-1.9 * fat, -4.6)])
    fins = mirror_x(sd_poly([(1.0, 0.6), (5.2, 4.6), (5.2, 6.6),
                             (1.0, 4.8)]))
    flame = sd_circle(0, 6.4, 1.6)
    band = sd_box(0, -1.6, 2.2 * fat, 0.7)
    glint = sd_circle(-0.7, -3.2, 0.5)
    return [
        Part(fins, fin_mat, z=0.0, h=0.8, bevel=1.2, spec=0.4),
        Part(flame, glow_mat, glow=True, bevel=1.6, bright=0.1),
        Part(body, body_mat, z=0.4, h=1.8, bevel=2.2, dome=True, spec=0.8),
        Part(band, fin_mat, z=1.9, h=0.3, bevel=0.6),
        Part(nose, fin_mat, z=0.8, h=1.2, bevel=1.6, dome=True, spec=0.6),
        Part(glint, "white", glow=True, bevel=0.5, bright=0.3),
    ]


def missile1():
    return sheet(rotations(lambda: dart_parts("ice", "red"), 16, 16, 16), 4)


def missile2():
    # Enemy 3: heavy torpedo, crimson, armour rings and an amber seeker
    def b():
        body = sd_capsule(0, -4.2, 0, 3.8, 3.0)
        cap = sd_circle(0, -4.6, 1.9)
        fins = mirror_x(sd_poly([(2.0, 0.5), (5.8, 4.0), (5.6, 6.6),
                                 (2.0, 4.4)]))
        flame = sd_union(mirror_x(sd_circle(1.2, 6.4, 1.2)))
        ribs = sd_union(sd_box(0, -1.0, 3.1, 0.5), sd_box(0, 1.8, 3.1, 0.5))
        glint = sd_circle(-0.8, -5.3, 0.6)
        return [
            Part(fins, "dark", h=0.8, bevel=1.1),
            Part(flame, "hotfire", glow=True, bevel=1.4, bright=0.1),
            Part(body, "red", z=0.4, h=2.0, bevel=2.8, dome=True, spec=0.7),
            Part(ribs, "wine_dark", z=2.0, h=0.3, bevel=0.5),
            Part(cap, "amber", z=2.0, h=1.0, bevel=1.6, dome=True,
                 spec=0.9),
            Part(glint, "white", glow=True, bevel=0.6, bright=0.3),
        ]
    return sheet(rotations(b, 16, 16, 16), 4)


def missile3():
    # Enemy 4: seeker with a forked bow and a glowing eye, teal
    def b():
        body = sd_capsule(0, -3.6, 0, 3.4, 2.3)
        fork = mirror_x(sd_capsule(2.0, -6.8, 1.6, -1.6, 1.2))
        fins = mirror_x(sd_poly([(1.4, 0.4), (5.6, 3.6), (5.6, 5.8),
                                 (1.4, 4.4)]))
        flame = sd_circle(0, 5.8, 1.4)
        eye = sd_circle(0, -2.2, 1.2)
        return [
            Part(flame, "plasma", glow=True, bevel=1.4, bright=0.2),
            Part(fins, "dark", h=0.7, bevel=1.1, spec=0.4),
            Part(fork, "teal", z=0.4, h=1.0, bevel=1.0, dome=True,
                 spec=0.6),
            Part(body, "teal", z=0.6, h=1.6, bevel=2.2, dome=True, spec=0.7,
                 bright=-0.1),
            Part(eye, "hotfire", glow=True, bevel=1.2, bright=0.25),
        ]
    return sheet(rotations(b, 16, 16, 16), 4)


# ---------------------------------------------------------------------------
# Enemy 2: interceptor (24x24, 32 directions). A heavy violet body with
# forward swept amber claw wings, twin engines and a red sensor eye.
# ---------------------------------------------------------------------------

def fighter():
    def b():
        body = sd_poly([(0, -9.8), (3.0, -5.5), (3.6, 3.0), (2.6, 8.4),
                        (-2.6, 8.4), (-3.6, 3.0), (-3.0, -5.5)])
        wing = mirror_x(sd_poly([(2.4, 2.0), (8.6, -5.4), (11.2, -4.6),
                                 (11.0, 0.4), (6.0, 7.6), (2.4, 7.6)]))
        claw = mirror_x(sd_poly([(9.0, -5.0), (10.2, -10.0), (11.4, -9.4),
                                 (11.2, -4.0)]))
        edge = mirror_x(sd_poly([(8.8, -5.0), (11.0, -4.4), (10.9, -2.6),
                                 (3.6, 3.8), (3.2, 2.0)]))
        cockpit = sd_ellipse(0, -3.0, 1.7, 2.8)
        eye = sd_circle(0, -7.2, 1.0)
        engine = mirror_x(sd_capsule(1.8, 3.0, 1.8, 8.2, 1.4))
        flame = mirror_x(sd_circle(1.8, 9.2, 1.2))
        glint = sd_circle(-0.6, -4.0, 0.6)
        return [
            Part(flame, "hotfire", glow=True, bevel=1.2, bright=0.1),
            Part(wing, "amber", z=0.2, h=1.0, bevel=1.8, spec=0.5,
                 bright=-0.12),
            Part(edge, "red", z=1.2, h=0.3, bevel=0.6, spec=0.6),
            Part(claw, "dark", z=0.6, h=1.0, bevel=1.0, dome=True,
                 spec=0.6),
            Part(engine, "dark", z=0.8, h=1.2, bevel=1.4, dome=True,
                 spec=0.6),
            Part(body, "violet", z=1.0, h=2.0, bevel=2.8, dome=True,
                 spec=0.7),
            Part(cockpit, "glass", z=2.6, h=1.0, bevel=1.4, dome=True,
                 spec=1.0),
            Part(eye, "hotfire", glow=True, bevel=1.0, bright=0.2),
            Part(glint, "white", glow=True, bevel=0.6, bright=0.3),
        ]
    return sheet(rotations(b, 32, 24, 24), 8)


# ---------------------------------------------------------------------------
# Enemies 5-7: "Crescent" drones (24x24, 16 directions). A heavy armoured
# crescent with spiked horns, plate seams and a glowing reactor eye.
# ---------------------------------------------------------------------------

def crescent(mat, core_mat):
    def b():
        outer = sd_circle(0, 1.0, 10.0)
        inner = sd_circle(0, -5.2, 8.0)
        horn = sd_sub(outer, inner)
        seams = sd_inter(horn, sd_union(
            sd_rot(sd_box(0, 7.0, 0.35, 4.0), 0.9),
            sd_rot(sd_box(0, 7.0, 0.35, 4.0), -0.9)))
        spikes = mirror_x(sd_poly([(6.4, -2.0), (9.6, -6.6), (8.8, -1.0)]))
        core = sd_circle(0, 3.0, 3.6)
        coreglow = sd_circle(0, 3.0, 2.2)
        tips = mirror_x(sd_circle(7.6, -1.4, 1.4))
        glint = sd_circle(-0.9, 2.0, 0.6)
        return [
            Part(spikes, "dark", z=0.0, h=0.8, bevel=1.0, spec=0.6),
            Part(horn, mat, z=0.2, h=2.0, bevel=3.0, dome=True, spec=0.7),
            Part(seams, "dark", z=1.0, h=0.2, bevel=0.3, cut=True),
            Part(tips, "steel", z=1.6, h=0.8, bevel=1.2, dome=True,
                 spec=0.8),
            Part(core, "steel", z=1.8, h=1.2, bevel=1.8, dome=True,
                 spec=0.6),
            Part(coreglow, core_mat, glow=True, bevel=2.0, bright=0.15),
            Part(glint, "white", glow=True, bevel=0.6, bright=0.3),
        ]
    return sheet(rotations(b, 16, 24, 24), 4)


# ---------------------------------------------------------------------------
# M1-M4: mothership (48x48, 32 frame spin cycle)
# ---------------------------------------------------------------------------

def bigship():
    frames = []
    for i in range(32):
        a = 2 * math.pi * i / 32
        disc = sd_circle(0, 0, 21.5)
        rim = sd_ring(0, 0, 18.5, 1.6)
        pods = []
        for k in range(6):
            ang = a + k * math.pi / 3
            pods.append(sd_circle(16.0 * math.sin(ang), -16.0 * math.cos(ang),
                                  3.6))
        pod = sd_union(*pods)
        lights = []
        for k in range(6):
            ang = a + k * math.pi / 3 + math.pi / 6
            lights.append(sd_circle(19.0 * math.sin(ang),
                                    -19.0 * math.cos(ang), 1.2))
        light = sd_union(*lights)
        panels = []
        for k in range(3):
            ang = -a * 0.5 + k * 2 * math.pi / 3
            panels.append(sd_rot(sd_box(0, -10.0, 2.2, 3.5, 0.8), ang))
        panel = sd_union(*panels)
        seams = []
        for k in range(6):
            ang = a + k * math.pi / 3
            seams.append(sd_rot(sd_box(0, -14.0, 0.45, 6.0), ang))
        seam = sd_inter(sd_union(*seams), sd_circle(0, 0, 21.0))
        dome = sd_circle(0, 0, 8.5)
        core = sd_circle(0, 0, 4.0)
        parts = [
            Part(disc, "gunship", z=0.0, h=5.0, bevel=14.0, dome=True,
                 spec=0.6, bright=-0.12),
            Part(rim, "dark", z=4.0, h=0.5, bevel=1.0),
            Part(seam, "dark", z=3.0, h=0.4, bevel=0.4, cut=True),
            Part(panel, "violet", z=4.6, h=0.5, bevel=1.2),
            Part(pod, "steel", z=4.4, h=2.0, bevel=3.0, dome=True, spec=0.8,
                 bright=-0.05),
            Part(light, "fire", glow=True, bevel=1.2,
                 bright=0.25 * math.sin(a * 6)),
            Part(dome, "violet", z=5.0, h=4.0, bevel=7.0, dome=True,
                 spec=0.7),
            Part(core, "rose", glow=True, bevel=4.0,
                 bright=0.15 * math.sin(a * 4)),
        ]
        frames.append(render(parts, 48, 48, ang=0.0))
    return sheet(frames, 8)


# ---------------------------------------------------------------------------
# Asteroids (32x32, 32 frame rotation)
# ---------------------------------------------------------------------------

def rock():
    rng = np.random.RandomState(7)
    lobes = [(rng.uniform(0, 2 * math.pi), rng.uniform(0.6, 1.4),
              rng.uniform(0.05, 0.14)) for _ in range(6)]
    craters = [(rng.uniform(-8, 8), rng.uniform(-8, 8), rng.uniform(1.4, 3.2))
               for _ in range(7)]

    def radius(th):
        r = 11.5
        for ph, fr, amp in lobes:
            r = r * (1 + amp * np.sin(th * round(fr * 3) + ph))
        return r

    def shape(x, y):
        th = np.arctan2(y, x)
        return np.hypot(x, y) - radius(th)

    def bumps(x, y):
        h = 1.6 * noise.fbm2(x * 0.25, y * 0.25, seed=3, octaves=3)
        for cx, cy, r in craters:
            d = np.hypot(x - cx, y - cy) / r
            h = h - np.where(d < 1, 2.2 * (1 - d * d), 0) \
                + np.where((d >= 1) & (d < 1.4), 0.9 * (1 - (d - 1) / 0.4), 0)
        return h

    parts = [Part(shape, "rock", z=0, h=9.0, bevel=11.0, dome=True, spec=0.1,
                  bright=-0.12,
                  hfunc=bumps)]
    frames = [render(parts, 32, 32, ang=-2 * math.pi * i / 32)
              for i in range(32)]
    return sheet(frames, 8)


def rockexpl():
    """12 frames, 64x64: shattering rock, dust ring and embers."""
    rng = np.random.RandomState(11)
    chunks = [(rng.uniform(0, 2 * math.pi), rng.uniform(0.5, 1.0),
               rng.uniform(1.2, 2.6), rng.uniform(-3, 3)) for _ in range(14)]
    blast = _fireball_frames(RAMPS["hotfire"], 64, 12, 13, debris=10,
                             sparks=10, scale=0.75)
    frames = []
    for f in range(12):
        t = (f + 1) / 12.0
        img = np.zeros((64, 64, 4), dtype=np.uint8)
        ys, xs = np.mgrid[0:64, 0:64]
        dx, dy = xs + 0.5 - 32, ys + 0.5 - 32
        d = np.hypot(dx, dy)
        # Dust ring
        R = 6 + 24 * (1 - (1 - t) ** 2)
        width = 3 + 5 * t
        n = noise.fbm2(xs * 0.15 + f * 0.7, ys * 0.15, seed=5, octaves=3)
        ring = np.clip(1 - np.abs(d - R) / width, 0, 1) * (0.6 + 0.6 * n)
        ring *= (1 - t) ** 0.8 * 1.3
        dust = RAMPS["rock"]
        fire = RAMPS["fire"]
        # The fireball comes from _fireball_frames; only the dust ring
        # is drawn here
        core = np.zeros_like(d)
        for yy in range(64):
            for xx in range(64):
                c = core[yy, xx]
                r = ring[yy, xx]
                if c > 0.12:
                    k = int(np.clip(c * len(fire) * 1.3, 0, len(fire) - 1))
                    img[yy, xx, :3] = fire[k]
                    img[yy, xx, 3] = 255
                elif r > 0.18:
                    k = int(np.clip(r * len(dust), 0, len(dust) - 1))
                    if (xx + yy + f) % 2 == 0 or r > 0.45:
                        img[yy, xx, :3] = dust[k]
                        img[yy, xx, 3] = 255
        im = Image.fromarray(img, "RGBA")
        im.alpha_composite(blast[f])
        # Tumbling chunks
        for ang, sp, sz, spin in chunks:
            if t > 0.85 * sp + 0.15:
                continue
            dist = 4 + 26 * t * sp
            cx = 32 + dist * math.cos(ang)
            cy = 32 + dist * math.sin(ang)
            s = sz * (1 - 0.5 * t)
            parts = [Part(sd_poly([(-s, -s * 0.6), (s * 0.7, -s),
                                   (s, s * 0.5), (-s * 0.4, s)]),
                          "rock", h=1.0, bevel=s, dome=True)]
            n = int(s * 2 + 4)
            ch = render(parts, n, n, ang=spin * t * 3)
            im.alpha_composite(ch, (int(cx - n / 2), int(cy - n / 2)))
        frames.append(im)
    return sheet(frames, 4)


# ---------------------------------------------------------------------------
# Enemy bullets (6x6) and their impacts (16x16, 6 frames)
# ---------------------------------------------------------------------------

def _bullet_frames(ramp, n, spikes):
    """Fat glowing orb filling the 6x6 cell: dark rim, saturated body,
    white hot core and a highlight that runs round the rim."""
    frames = []
    ys, xs = np.mgrid[0:6, 0:6]
    dx, dy = xs + 0.5 - 3, ys + 0.5 - 3
    d = np.hypot(dx, dy)
    th = np.arctan2(dy, dx)
    for i in range(n):
        ph = 2 * math.pi * i / n
        r = 2.95 + (0.35 * np.cos(th * spikes + ph) if spikes else 0)
        v = np.clip(1 - d / r, 0, 1)
        v = v * (0.9 + 0.12 * math.sin(ph * 2))
        v = v + 0.25 * np.clip(np.cos(th - ph), 0, 1) * (d > 1.2)
        img = np.zeros((6, 6, 4), dtype=np.uint8)
        inside = d < r
        k = np.clip(np.round(0.5 + v * (len(ramp) - 1) * 1.15), 0,
                    len(ramp) - 1).astype(int)
        for j in range(len(ramp)):
            sel = inside & (k == j)
            img[sel, :3] = ramp[j]
            img[sel, 3] = 255
        frames.append(Image.fromarray(img, "RGBA"))
    return frames


def bullets():
    return (sheet(_bullet_frames(RAMPS["plasma"], 14, 4), 7),
            sheet(_bullet_frames(RAMPS["toxic"], 7, 0), 7),
            sheet(_bullet_frames(RAMPS["hotfire"][1:], 14, 3), 7))


def _burst_frames(ramp, size, n, seed, rays=6, ring=True):
    rng = np.random.RandomState(seed)
    sparks = [(rng.uniform(0, 2 * math.pi), rng.uniform(0.6, 1.0))
              for _ in range(rays)]
    frames = []
    c = size / 2.0
    for f in range(n):
        t = f / float(max(n - 1, 1))
        img = np.zeros((size, size, 4), dtype=np.uint8)
        ys, xs = np.mgrid[0:size, 0:size]
        dx, dy = xs + 0.5 - c, ys + 0.5 - c
        d = np.hypot(dx, dy)
        flash = np.clip(1 - d / (c * (0.55 - 0.45 * t) + 0.5), 0, 1) \
            * (1 - t) ** 1.2
        rr = c * (0.3 + 0.65 * t)
        rv = np.clip(1 - np.abs(d - rr) / 1.0, 0, 1) * (1 - t) if ring \
            else 0 * d
        v = np.maximum(flash * 1.4, rv * 0.8)
        for ang, sp in sparks:
            sx = c + math.cos(ang) * c * (0.25 + 0.7 * t * sp)
            sy = c + math.sin(ang) * c * (0.25 + 0.7 * t * sp)
            ds_ = np.hypot(xs + 0.5 - sx, ys + 0.5 - sy)
            v = np.maximum(v, np.clip(1 - ds_ / 1.1, 0, 1) * (1 - t * 0.9))
        for yy in range(size):
            for xx in range(size):
                if v[yy, xx] <= 0.12:
                    continue
                k = int(np.clip(v[yy, xx] * len(ramp), 0, len(ramp) - 1))
                img[yy, xx, :3] = ramp[k]
                img[yy, xx, 3] = 255
        frames.append(Image.fromarray(img, "RGBA"))
    return frames


def _fireball_frames(ramp, size, n, seed, smoke=True, debris=6,
                     sparks=6, scale=1.0):
    """Fat 1990s style blast: a lumpy fireball with a white hot core that
    cools through the ramp, rolls into dark smoke puffs at its rim, and
    throws glowing sparks and tumbling debris."""
    rng = np.random.RandomState(seed)
    puffs = [(rng.uniform(0, 2 * math.pi), rng.uniform(0.25, 0.75),
              rng.uniform(0.35, 0.6)) for _ in range(7)]
    flying = [(rng.uniform(0, 2 * math.pi), rng.uniform(0.7, 1.15),
               rng.randint(0, 2)) for _ in range(debris + sparks)]
    smk = RAMPS["smoke"]
    frames = []
    c = size / 2.0
    ys, xs = np.mgrid[0:size, 0:size]
    for f in range(n):
        t = f / float(max(n - 1, 1))
        grow = 1 - (1 - min(1.0, t * 1.6)) ** 2
        R = c * scale * (0.25 + 0.6 * grow)
        dx, dy = xs + 0.5 - c, ys + 0.5 - c
        lump = noise.fbm2(xs * 0.35 + seed, ys * 0.35 + f * 0.6,
                          seed=seed, octaves=3)
        d = np.hypot(dx, dy) / max(R, 0.5) - (lump - 0.5) * 0.7
        heat = np.clip(1 - d, 0, 1) ** 0.8 * (1.25 - 1.1 * t)
        fire = heat > 0.08
        smoke_v = np.zeros_like(heat)
        if smoke:
            for ang, dist, rr in puffs:
                pr = c * scale * rr * (0.4 + 0.6 * grow)
                px = c + math.cos(ang) * c * scale * dist * grow
                py = c + math.sin(ang) * c * scale * dist * grow - t * 2
                pd = np.hypot(xs + 0.5 - px, ys + 0.5 - py) / max(pr, 0.5)
                smoke_v = np.maximum(smoke_v, np.clip(1 - pd - (lump - 0.5)
                                                      * 0.6, 0, 1))
            smoke_v *= np.clip(1.6 * t + 0.1, 0, 1) * (1.2 - 0.9 * t)
        img = np.zeros((size, size, 4), dtype=np.uint8)
        bay = (((xs & 1) ^ (ys & 1)) * 0.5)
        sm = (smoke_v > 0.18 + bay * 0.2) & ~(heat > 0.25)
        k = np.clip((smoke_v * 1.4 * (len(smk) - 1)) * (1 - 0.5 * t), 0,
                    len(smk) - 1).astype(int)
        for i in range(len(smk)):
            img[sm & (k == i), :3] = smk[i]
        img[sm, 3] = 255
        k = np.clip(heat * (len(ramp) - 1) * 1.15, 0, len(ramp) - 1)
        k = k.astype(int)
        for i in range(len(ramp)):
            sel = fire & (k == i)
            img[sel, :3] = ramp[i]
            img[sel, 3] = 255
        if smoke and t > 0.4:
            # Cooling edges of the fireball roll over into smoke
            cool = fire & (heat < 0.15 + 0.5 * (t - 0.4)) & (bay < 0.5)
            img[cool, :3] = smk[1 + int(lump.mean() > 0.5)]
        # Sparks (hot dots with trails) and debris (dark chunks)
        for j, (ang, sp, kind) in enumerate(flying):
            if t > 0.95:
                continue
            dist = c * scale * (0.2 + 0.85 * t * sp)
            x = int(c + math.cos(ang) * dist)
            y = int(c + math.sin(ang) * dist + t * t * 3)
            if not (0 <= x < size - 1 and 0 <= y < size - 1):
                continue
            if j < sparks:
                col = ramp[-1] if t < 0.5 else ramp[len(ramp) // 2]
                img[y, x, :3] = col
                img[y, x, 3] = 255
                tx = int(c + math.cos(ang) * (dist - 1.5))
                ty = int(c + math.sin(ang) * (dist - 1.5))
                if 0 <= tx < size and 0 <= ty < size and \
                        img[ty, tx, 3] == 0:
                    img[ty, tx, :3] = ramp[len(ramp) // 2 - 1]
                    img[ty, tx, 3] = 255
            else:
                img[y, x, :3] = RAMPS["steel"][5]
                img[y, x + kind, :3] = RAMPS["steel"][2]
                img[y + 1, x, :3] = RAMPS["dark"][1]
                img[y, x, 3] = img[y, x + kind, 3] = img[y + 1, x, 3] = 255
        frames.append(Image.fromarray(img, "RGBA"))
    return frames


def bullet_explosions():
    blue = RAMPS["dark"][2:] + RAMPS["ice"][2:]
    green = RAMPS["green"][1:]
    red = RAMPS["red"][1:] + [rgb(19), rgb(20)]
    return (sheet(_fireball_frames(RAMPS["plasma"], 16, 6, 1, smoke=False,
                                   debris=0, sparks=5), 6),
            sheet(_fireball_frames(RAMPS["toxic"], 16, 6, 2, smoke=False,
                                   debris=0, sparks=5), 6),
            sheet(_fireball_frames(RAMPS["hotfire"][1:], 16, 6, 3,
                                   smoke=False, debris=0, sparks=5), 6))


# ---------------------------------------------------------------------------
# Ring (16x16, 16 frames): spinning plasma ring mine
# ---------------------------------------------------------------------------

def ring():
    frames = []
    ramp = RAMPS["violet"][1:] + [rgb(8)]
    for i in range(16):
        ph = 2 * math.pi * i / 16
        img = np.zeros((16, 16, 4), dtype=np.uint8)
        ys, xs = np.mgrid[0:16, 0:16]
        dx, dy = xs + 0.5 - 8, ys + 0.5 - 8
        d = np.hypot(dx, dy)
        th = np.arctan2(dy, dx)
        R = 5.2 + 0.5 * math.sin(ph * 2)
        band = np.clip(1 - np.abs(d - R) / 2.0, 0, 1)
        swirl = 0.55 + 0.45 * np.cos(th * 3 - ph * 3)
        v = band * (0.45 + 0.75 * swirl)
        core = np.clip(1 - d / 2.0, 0, 1) * (0.4 + 0.3 * math.sin(ph * 4))
        v = np.maximum(v, core)
        for yy in range(16):
            for xx in range(16):
                if v[yy, xx] <= 0.15:
                    continue
                k = int(np.clip(v[yy, xx] * len(ramp), 0, len(ramp) - 1))
                img[yy, xx, :3] = ramp[k]
                img[yy, xx, 3] = 255
        frames.append(Image.fromarray(img, "RGBA"))
    return sheet(frames, 4)


def ringexpl():
    ramp = RAMPS["violet"][1:] + [rgb(8)]
    a = _fireball_frames(RAMPS["magenta"], 20, 8, 21, smoke=False,
                         debris=2, sparks=8)
    b = _fireball_frames(RAMPS["plasma"], 20, 8, 22, smoke=False,
                         debris=2, sparks=6)
    return sheet(a + b, 8)


# ---------------------------------------------------------------------------
# Bombs (12x12). 0-15 bomb 1 idle, 16-31 bomb 2 idle, 32-37 armed flash.
# ---------------------------------------------------------------------------

def _mine(i, n, mat, light_mat, spikes, armed=False):
    a = 2 * math.pi * i / n
    shell = sd_circle(0, 0, 3.6)
    sp = []
    for k in range(spikes):
        ang = a + k * 2 * math.pi / spikes
        sp.append(sd_capsule(0, 0, 5.0 * math.sin(ang), -5.0 * math.cos(ang),
                             0.7))
    spike = sd_union(*sp)
    eye = sd_circle(0, 0, 1.5)
    pulse = 0.5 + 0.5 * math.sin(a * (1 if not armed else 3))
    return render([
        Part(spike, "dark", z=0, h=0.6, bevel=0.8, dome=True),
        Part(shell, mat, z=0.4, h=1.4, bevel=2.4, dome=True, spec=0.5),
        Part(eye, light_mat, glow=True, bevel=1.5,
             bright=-0.3 + 0.5 * pulse if not armed else 0.3),
    ], 12, 12)


def bombs():
    frames = [_mine(i, 16, "steel", "green", 4) for i in range(16)]
    frames += [_mine(i, 16, "gold", "amber", 6) for i in range(16)]
    for i in range(6):
        img = _mine(i, 6, "red" if i % 2 == 0 else "fire", "white", 4,
                    armed=True)
        frames.append(img)
    while len(frames) < 48:
        frames.append(Image.new("RGBA", (12, 12), (0, 0, 0, 0)))
    return sheet(frames, 16)


def bombdeto():
    ramp = RAMPS["fire"] + [rgb(8)]
    return sheet(_fireball_frames(RAMPS["hotfire"], 20, 8, 31, debris=4,
                                  sparks=8), 8)


# ---------------------------------------------------------------------------
# Player bolts (8x8, 128 frames). See KOBO_myship::bolt_frame().
#   0-31   muzzle flash per direction (8 dirs x 4)
#   32-63  bolt in flight per axis (4 axes x 8: anim 0-3, 4 = fade)
#   80-111 bolt impact sparks
# ---------------------------------------------------------------------------

def _bolt_img(ang, length, width, ramp, bright=1.0):
    img = np.zeros((8, 8, 4), dtype=np.uint8)
    ys, xs = np.mgrid[0:8, 0:8]
    dx, dy = xs + 0.5 - 4, ys + 0.5 - 4
    c, s = math.cos(ang), math.sin(ang)
    along = dx * s - dy * c
    across = dx * c + dy * s
    v = np.clip(1 - np.abs(across) / width, 0, 1) \
        * np.clip(1 - np.abs(along) / length, 0, 1) ** 0.5 * bright
    for yy in range(8):
        for xx in range(8):
            if v[yy, xx] <= 0.1:
                continue
            k = int(np.clip(v[yy, xx] * len(ramp) * 1.2, 0, len(ramp) - 1))
            img[yy, xx, :3] = ramp[k]
            img[yy, xx, 3] = 255
    return Image.fromarray(img, "RGBA")


def bolts():
    ramp = RAMPS["hotfire"][2:] + [rgb(8)]
    frames = [Image.new("RGBA", (8, 8), (0, 0, 0, 0)) for _ in range(128)]
    for d in range(8):
        ang = d * math.pi / 4
        for k in range(4):
            # Muzzle: a short hot blob growing into a bolt
            frames[d * 4 + k] = _bolt_img(ang, 2.2 + k * 0.6,
                                          3.0 - k * 0.35, ramp)
    for ax in range(4):
        ang = ax * math.pi / 4
        for k in range(4):
            frames[32 + ax * 8 + k] = _bolt_img(ang, 4.0 + (k % 2) * 0.3,
                                                2.0 + 0.3 * (k == 1),
                                                ramp, 1.05 - 0.05 * k)
        frames[32 + ax * 8 + 4] = _bolt_img(ang, 3.2, 1.5, ramp[:4], 0.85)
        for k in range(5, 8):
            frames[32 + ax * 8 + k] = _bolt_img(ang, 3.8, 1.8, ramp, 0.95)
    sparks = _fireball_frames(ramp, 8, 32, 41, smoke=False, debris=0,
                              sparks=4, scale=1.1)
    for i in range(32):
        # Smooth decay so any 8 frame window starting at 80 + 4k reads
        # as a flash fading out.
        frames[80 + i] = sparks[min(31, i * 2 // 2)]
    return sheet(frames, 16)


# ---------------------------------------------------------------------------
# Shield (32x32, 8 frames): hexagonal energy lattice
# ---------------------------------------------------------------------------

def shield():
    """Energy bubble: bright rim, sweeping highlight arc and a rotating
    hexagonal node lattice on the bubble surface."""
    frames = []
    ramp = [rgb(10), rgb(11), rgb(12), rgb(13), rgb(8)]
    # Hex lattice nodes on the bubble (precomputed in bubble space)
    nodes = []
    for j in range(-4, 5):
        for i in range(-4, 5):
            u = (i + 0.5 * (j & 1)) * 0.42
            v = j * 0.36
            if u * u + v * v < 0.95:
                nodes.append((u, v))
    for f in range(8):
        ph = 2 * math.pi * f / 8
        img = np.zeros((32, 32, 4), dtype=np.uint8)
        ys, xs = np.mgrid[0:32, 0:32]
        dx, dy = xs + 0.5 - 16, ys + 0.5 - 16
        d = np.hypot(dx, dy)
        th = np.arctan2(dy, dx)
        rim = np.clip(1 - np.abs(d - 13.8) / 1.3, 0, 1)
        sweep = np.clip(np.cos(th - ph), 0, 1) ** 3
        val = rim * (0.5 + 0.5 * sweep)
        # Lattice nodes, slowly rotating on the sphere
        for (u, v) in nodes:
            uu = u + (f / 8.0) * 0.42
            uu = (uu + 1.05) % 2.1 - 1.05
            if uu * uu + v * v >= 0.9:
                continue
            z = math.sqrt(max(0.0, 1 - uu * uu - v * v))
            px, py = 16 + uu * 13.0, 16 + v * 13.0
            nd = np.hypot(xs + 0.5 - px, ys + 0.5 - py)
            val = np.maximum(val, np.clip(1 - nd / 0.9, 0, 1)
                             * (0.35 + 0.4 * z))
        for yy in range(32):
            for xx in range(32):
                if val[yy, xx] <= 0.25:
                    continue
                k = int(np.clip(val[yy, xx] * len(ramp), 0, len(ramp) - 1))
                img[yy, xx, :3] = ramp[k]
                img[yy, xx, 3] = 255
        frames.append(Image.fromarray(img, "RGBA"))
    return sheet(frames, 8)


def crosshair():
    img = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    px = img.load()
    c1, c2 = rgb(12) + (255,), rgb(1) + (255,)
    for i in range(16):
        for j in range(16):
            dx, dy = i - 7.5, j - 7.5
            r = math.hypot(dx, dy)
            gap = abs(dx) < 1 or abs(dy) < 1
            if 5.0 <= r <= 6.2 and not gap:
                px[i, j] = c1
            elif (abs(dx) < 0.6 and 1.5 < abs(dy) < 4.5) or \
                    (abs(dy) < 0.6 and 1.5 < abs(dx) < 4.5):
                px[i, j] = c1
    # Shadow outline
    a = np.array(img)
    m = a[:, :, 3] > 0
    o = np.zeros_like(m)
    o[1:, :] |= m[:-1, :]
    o[:-1, :] |= m[1:, :]
    o[:, 1:] |= m[:, :-1]
    o[:, :-1] |= m[:, 1:]
    o &= ~m
    a[o] = c2
    return Image.fromarray(a, "RGBA")
