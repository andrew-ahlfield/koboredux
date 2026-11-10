"""Dashboard frame, indicator LEDs, weapon slots, weapon icons and logo."""

import math
import numpy as np
from PIL import Image
from scipy import ndimage

from .render import (Part, render, sheet, sd_box, sd_circle, sd_poly,
                     sd_union, sd_sub, sd_capsule, glow_dot)
from .palette import rgb, RAMPS
from . import fonts, sprites
from .fontdata import G


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mask_sdf(mask, scale, cx, cy):
    """Turn a boolean mask into an SDF usable as a Part shape. 'scale' is
    object units per mask pixel, (cx, cy) the object position of the mask
    centre."""
    pad = 8
    mask = np.pad(mask, pad)
    inside = ndimage.distance_transform_edt(mask)
    outside = ndimage.distance_transform_edt(~mask)
    sdf = (outside - inside) * scale
    h, w = mask.shape

    def f(x, y):
        mx = (x - cx) / scale + w / 2.0 - 0.5
        my = (y - cy) / scale + h / 2.0 - 0.5
        return ndimage.map_coordinates(sdf, [my, mx], order=1,
                                       mode="constant", cval=1e3)
    return f


def text_mask(text, upscale_passes=0, spacing=1):
    cols = []
    for ch in text:
        if ch == " ":
            cols.append(np.zeros((7, 3), dtype=bool))
            continue
        b = fonts.bitmap(ch)
        cols.append(b)
        cols.append(np.zeros((7, spacing), dtype=bool))
    m = np.concatenate(cols[:-1], axis=1)
    for _ in range(upscale_passes):
        m = fonts.scale2x(m)
    return m


def draw_text(img, x, y, text, color, shadow=None):
    """Blit 5x7 text into an RGBA numpy image."""
    m = text_mask(text)
    h, w = m.shape
    if shadow is not None:
        sub = img[y + 1:y + 1 + h, x + 1:x + 1 + w]
        sub[m[:sub.shape[0], :sub.shape[1]]] = shadow + (255,)
    sub = img[y:y + h, x:x + w]
    sub[m[:sub.shape[0], :sub.shape[1]]] = color + (255,)
    return w


def text_width(text):
    return text_mask(text).shape[1]


# ---------------------------------------------------------------------------
# Dashboard (640 x 360). Window rectangles follow the layout in main.theme.
# ---------------------------------------------------------------------------

MAIN = (132, 4, 352, 352)
CONSOLE = (8, 52, 104, 256)
RADAR = (504, 52, 128, 256)
CHARGE = (118, 80, 12, 200)
HEALTH = (486, 80, 12, 200)


def _rect_sdf(x, y, w, h, r=0.0, grow=0.0):
    # Rectangles in screen pixels; the dashboard renders with its centre at
    # (320, 180).
    return sd_box(x + w / 2.0 - 320, y + h / 2.0 - 180, w / 2.0 + grow,
                  h / 2.0 + grow, r)


def _chamfer(x0, y0, x1, y1, c):
    pts = [(x0 + c, y0), (x1 - c, y0), (x1, y0 + c), (x1, y1 - c),
           (x1 - c, y1), (x0 + c, y1), (x0, y1 - c), (x0, y0 + c)]
    return sd_poly([(px - 320, py - 180) for px, py in pts])


def dashboard():
    left = _chamfer(2, 30, 136, 330, 14)
    right = _chamfer(480, 30, 638, 330, 14)
    main_outer = _rect_sdf(126, 0, 364, 360, 8)
    main_inner = _rect_sdf(*MAIN, r=2, grow=1)
    main_frame = sd_sub(main_outer, main_inner)
    # Struts tying the wings to the main frame
    struts = sd_union(_rect_sdf(110, 18, 30, 8, 2), _rect_sdf(110, 334, 30, 8, 2),
                      _rect_sdf(476, 18, 30, 8, 2), _rect_sdf(476, 334, 30, 8, 2))
    bezels = sd_union(
        sd_sub(_rect_sdf(*CONSOLE, r=4, grow=5), _rect_sdf(*CONSOLE, grow=0)),
        sd_sub(_rect_sdf(*RADAR, r=4, grow=5), _rect_sdf(*RADAR, grow=0)),
        sd_sub(_rect_sdf(*CHARGE, r=3, grow=3), _rect_sdf(*CHARGE)),
        sd_sub(_rect_sdf(*HEALTH, r=3, grow=3), _rect_sdf(*HEALTH)))
    glowline = sd_union(
        sd_sub(_rect_sdf(*CONSOLE, grow=1.2), _rect_sdf(*CONSOLE, grow=0.2)),
        sd_sub(_rect_sdf(*RADAR, grow=1.2), _rect_sdf(*RADAR, grow=0.2)))
    rivets = []
    for (x, y) in [(14, 40), (124, 40), (14, 320), (124, 320),
                   (492, 40), (626, 40), (492, 320), (626, 320),
                   (136, 10), (480, 10), (136, 350), (480, 350)]:
        rivets.append(sd_circle(x - 320, y - 180, 2.2))
    rivet = sd_union(*rivets)
    # Vent slots on the wing plates
    slots = sd_union(*([_rect_sdf(16 + k * 7, 313, 3, 10, 1.4)
                        for k in range(13)] +
                       [_rect_sdf(504 + k * 7, 313, 3, 10, 1.4)
                        for k in range(18)]))
    parts = [
        Part(left, "steel", z=0, h=4.0, bevel=6.0, spec=0.3, bright=-0.22),
        Part(right, "steel", z=0, h=4.0, bevel=6.0, spec=0.3, bright=-0.22),
        Part(struts, "dark", z=1.0, h=2.0, bevel=2.0, dome=True),
        Part(main_frame, "steel", z=0.5, h=3.0, bevel=3.0, dome=True,
             spec=0.5, bright=-0.12),
        Part(bezels, "steel", z=2.5, h=2.0, bevel=2.5, dome=True, spec=0.6,
             bright=-0.05),
        Part(glowline, "amber", glow=True, bevel=1.0, bright=-0.1),
        Part(slots, "dark", z=1.0, h=1.0, bevel=1.0, cut=True),
        Part(rivet, "steel", z=3.0, h=1.0, bevel=1.6, dome=True, spec=0.8),
    ]
    img = np.array(render(parts, 640, 360, outline=True))
    # Hazard striped warning bands on the wing plates, 1990s cabinet style
    st = RAMPS["amber"]
    for (x0, y0, x1, y1) in ((18, 37, 120, 44), (496, 37, 622, 44)):
        for yy in range(y0, y1 + 1):
            for xx in range(x0, x1 + 1):
                on = ((xx + yy) // 3) % 2 == 0
                c = st[5] if on else RAMPS["dark"][0]
                if on and yy == y0:
                    c = st[7]
                elif on and yy == y1:
                    c = st[3]
                img[yy, xx] = (*c, 255)
        img[y0 - 1, x0 - 1:x1 + 2] = (*RAMPS["dark"][0], 255)
        img[y1 + 1, x0 - 1:x1 + 2] = (*RAMPS["steel"][6], 255)
    # Opaque black where the game draws its windows
    for (x, y, w, h) in (MAIN, CONSOLE, RADAR, CHARGE, HEALTH):
        img[y:y + h, x:x + w] = (0, 0, 0, 255)
    # Logo plates
    draw_text(img, 20, 336, "KOBO REDUX", rgb(17), shadow=rgb(14))
    draw_text(img, 548, 336, "HALCYON", rgb(12), shadow=rgb(9))
    # Background is opaque black so the dashboard covers the whole screen
    out = np.zeros_like(img)
    out[..., 3] = 255
    a = img[..., 3:4] / 255.0
    out[..., :3] = (img[..., :3] * a).astype(np.uint8)
    return Image.fromarray(out, "RGBA").convert("RGB")


# ---------------------------------------------------------------------------
# LEDs
# ---------------------------------------------------------------------------

def _led(w, h, color, level, horizontal=True):
    """One LED cell: rounded lit segment with highlight, or dark when
    level == 0."""
    img = np.zeros((h, w, 4), dtype=np.uint8)
    off = np.array(rgb(2), float)
    c = np.array(color, float)
    base = off + (c - off) * level
    hi = np.minimum(base + (255 - base) * 0.45 * level, 255)
    if horizontal:
        img[:, 1:w - 1, :3] = base
        img[:, 1:w - 1, 3] = 255
        img[0, 2:w - 2, :3] = hi
        img[:, 0] = (*rgb(1), 255)
        img[:, w - 1] = (*rgb(1), 255)
    else:
        img[1:h - 1, :, :3] = base
        img[1:h - 1, :, 3] = 255
        img[2:h - 2, 0, :3] = hi
        img[0, :] = (*rgb(1), 255)
        img[h - 1, :] = (*rgb(1), 255)
    return img


def proxy_leds(horizontal):
    w, h = (8, 2) if horizontal else (2, 8)
    frames = [_led(w, h, rgb(2), 0, horizontal)]
    for col in (rgb(17), rgb(31), rgb(6), rgb(19)):
        for k in range(3):
            frames.append(_led(w, h, col, 0.45 + 0.275 * k, horizontal))
    ims = [Image.fromarray(f, "RGBA") for f in frames]
    return sheet(ims, 13 if horizontal else 1)


def _bar(col, level):
    """12x4 bar graph segment."""
    img = np.zeros((4, 12, 4), dtype=np.uint8)
    off = np.array(rgb(2), float)
    c = np.array(col, float)
    base = off + (c - off) * level
    hi = np.minimum(base + (255 - base) * 0.5 * level, 255)
    img[0:3, 1:11, :3] = base
    img[0:3, 1:11, 3] = 255
    img[0, 2:10, :3] = hi
    img[0:3, 0] = (*rgb(1), 255)
    img[0:3, 11] = (*rgb(1), 255)
    img[3, :] = (*rgb(0), 255)
    return Image.fromarray(img, "RGBA")


def _grad(col_a, col_b, t):
    a = np.array(col_a, float)
    b = np.array(col_b, float)
    return tuple((a + (b - a) * t).astype(int))


def health_leds():
    frames = [_bar(rgb(2), 0)]
    for col in (rgb(17), rgb(19), rgb(31)):
        for k in range(4):
            frames.append(_bar(col, 0.3 + 0.7 * (k + 1) / 4))
    for k in range(4):  # green to bright blue (overcharge)
        frames.append(_bar(_grad(rgb(31), rgb(6), (k + 1) / 4), 1.0))
    for k in range(4):  # bright blue timer
        frames.append(_bar(_grad(rgb(5), rgb(13), (k + 1) / 4), 0.6 + 0.1 * k))
    return sheet(frames, 1)


def charge_leds():
    frames = [_bar(rgb(2), 0)]
    for col in (rgb(16), rgb(18), rgb(12)):
        for k in range(4):
            frames.append(_bar(col, 0.3 + 0.7 * (k + 1) / 4))
    return sheet(frames, 1)


# ---------------------------------------------------------------------------
# Weapon slots (52x52, index slot + state * 8 + fire * 4) and icons (40x40)
# ---------------------------------------------------------------------------

SLOT_LABELS = ["NOSE", "TAIL", "BLAST", "BLOSSOM"]
SLOT_COLORS = [rgb(12), rgb(12), rgb(18), rgb(24)]


def weaponslots():
    frames = []
    for state in range(3):
        for fire in range(2):
            for slot in range(4):
                img = np.zeros((52, 52, 4), dtype=np.uint8)
                accent = SLOT_COLORS[slot]
                if state == 0:
                    frame_c, label_c = rgb(2), rgb(3)
                elif state == 1:
                    frame_c, label_c = rgb(3), rgb(4)
                else:
                    frame_c = rgb(4) if not fire else accent
                    label_c = accent if not fire else rgb(8)
                # Frame
                img[0:52, 0:52] = (*frame_c, 255)
                img[1:51, 1:51] = (*rgb(1), 255)
                img[9:51, 2:50] = (*rgb(0), 255)
                # Label bar
                bar_c = frame_c if state else rgb(1)
                img[1:9, 1:51] = (*bar_c, 255) if state == 2 and fire \
                    else (*rgb(1), 255)
                tw = text_width(SLOT_LABELS[slot])
                draw_text(img, (52 - tw) // 2, 1, SLOT_LABELS[slot],
                          label_c if not (state == 2 and fire) else rgb(1))
                if state == 0:
                    # Hatched "not installed"
                    for yy in range(10, 50):
                        for xx in range(3, 49):
                            if (xx + yy) % 6 == 0:
                                img[yy, xx] = (*rgb(2), 255)
                # Corner ticks
                for (cx, cy) in ((2, 10), (49, 10), (2, 49), (49, 49)):
                    img[cy, cx] = (*frame_c, 255)
                frames.append(Image.fromarray(img, "RGBA"))
    return sheet(frames, 4)


def _icon(slot, avail):
    base = Image.new("RGBA", (40, 40), (0, 0, 0, 0))
    parts = sprites.player_parts()
    if not avail:
        for p in parts:
            p.mat = RAMPS["dark"]
            p.glow = False
    ship = render(parts, 40, 40, scale=1.15)
    base.alpha_composite(ship, (0, 0))
    hot = [rgb(17), rgb(18), rgb(19), rgb(20), rgb(8)]
    dim = [rgb(2), rgb(3), rgb(4)]
    ramp = hot if avail else dim

    def bolt(x, y, l=3):
        for k in range(l):
            g = glow_dot(6, 6, 3, 3, 2.6 - k * 0.5, ramp)
            base.alpha_composite(g, (x - 3, y - 3 + k * 2))

    if slot == 0:
        bolt(20, 3)
    elif slot == 1:
        bolt(20, 34)
    elif slot == 2:
        orb = glow_dot(14, 14, 7, 7, 7, ramp if avail else dim, 0.8)
        base.alpha_composite(orb, (13, 0))
    else:
        for k in range(8):
            a = k * math.pi / 4
            x = int(20 + 16 * math.sin(a))
            y = int(20 - 16 * math.cos(a))
            g = glow_dot(6, 6, 3, 3, 2.8, ramp)
            base.alpha_composite(g, (x - 3, y - 3))
    return base


def weapons():
    frames = [_icon(s, False) for s in range(4)] + \
        [_icon(s, True) for s in range(4)]
    return sheet(frames, 4)


# ---------------------------------------------------------------------------
# Title logo (320 x 128, 2 frames: full and dimmed)
# ---------------------------------------------------------------------------

def logo():
    kobo = text_mask("KOBO", upscale_passes=3, spacing=1)   # 8x scaled
    redux = text_mask("REDUX", upscale_passes=2, spacing=1)  # 4x scaled
    # Heavier strokes than the plain font
    kobo = ndimage.binary_dilation(kobo, iterations=3)
    redux = ndimage.binary_dilation(redux, iterations=1)
    frames = []
    for dim in (0.0, -0.35):
        s1 = 1.15  # object px per mask px for KOBO
        k_sdf = mask_sdf(kobo, s1, 0, -22)
        r_sdf = mask_sdf(redux, 1.0, 0, 42)
        bar = sd_box(0, 23, 150, 1.6, 1.0)
        parts = [
            Part(bar, "teal", glow=True, bevel=1.6, bright=dim),
            Part(k_sdf, "hull", z=0, h=7.0, bevel=7.0, dome=True, spec=0.8,
                 bright=dim - 0.05),
            Part(r_sdf, "fire", z=0, h=3.0, bevel=3.0, dome=True, spec=0.5,
                 bright=dim + 0.05),
        ]
        frames.append(render(parts, 320, 128))
    return sheet(frames, 1)
