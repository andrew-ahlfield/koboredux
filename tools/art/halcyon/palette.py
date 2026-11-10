"""Halcyon palette and material ramps.

The 64 entry master palette is written out as halcyon.gpl and is what the
theme's realtime effects (fire, stars, radar, menu focus effect) index
into: 0 to 31 are the base colours, 32 to 63 the effect ramps (hot fire,
smoke, plasma, toxic and magenta glows). Sprites are shaded with the
longer, hue shifted material ramps below: shadows lean towards violet
blue, highlights towards warm yellow, in the manner of 1990s arcade
pixel art.
"""

from .pix import hramp

MASTER = [
    "000000",  # 0  black
    "0b0e1f",  # 1  void navy (outlines)
    "1a1f3d",  # 2  deep slate
    "2b3463",  # 3  slate
    "3f4f8a",  # 4  steel blue
    "5a74b8",  # 5  cornflower steel
    "86a3dc",  # 6  pale steel
    "c4d6f2",  # 7  ice
    "ffffff",  # 8  white
    "0d2b2e",  # 9  abyss teal
    "145e5a",  # 10 deep teal
    "1f9c8a",  # 11 teal
    "43d6b0",  # 12 aqua
    "a6f5d8",  # 13 mint glow
    "2a0f1e",  # 14 wine black
    "6b1a2e",  # 15 wine
    "b02a35",  # 16 crimson
    "e8503a",  # 17 vermilion
    "ff8a3d",  # 18 orange
    "ffc65c",  # 19 amber
    "fff1a8",  # 20 pale gold
    "3b1d52",  # 21 deep violet
    "6e2f8f",  # 22 violet
    "a84fc4",  # 23 orchid
    "e58ae0",  # 24 pink glow
    "2e2a26",  # 25 basalt
    "57483b",  # 26 umber
    "8a6e52",  # 27 tan
    "bf9a72",  # 28 sand
    "1e4a1c",  # 29 moss
    "3e8a2e",  # 30 leaf
    "8bd04a",  # 31 lime
    # Effect ramps
    "280609", "600c0e", "a01810", "d63812",   # 32-35 fire, dark to hot
    "f6701c", "ffaa32", "ffde6e", "fffadc",   # 36-39
    "161214", "282222", "3e3632", "584e46",   # 40-43 warm smoke
    "766a5e", "968a7c",                       # 44-45
    "080e30", "102878", "1e5ac8", "46a0ff",   # 46-49 plasma blue
    "96d7ff", "e6f8ff",                       # 50-51
    "0a1e08", "18500c", "3c9614", "82dc28",   # 52-55 toxic green
    "c8ff6e", "f5ffd7",                       # 56-57
    "280628", "640e60", "b41eaa", "f050dc",   # 58-61 magenta
    "ffa0f0", "ffe6fc",                       # 62-63
]


def rgb(i):
    h = MASTER[i]
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def ramp(*idx):
    return [rgb(i) for i in idx]


# Material ramps, darkest to brightest.
RAMPS = {
    "steel":  hramp(218, 0.26, 9, v0=0.08, v1=0.97, shift=0.3),
    "hull":   hramp(212, 0.13, 9, v0=0.16, v1=1.00, shift=0.25),
    "dark":   hramp(232, 0.32, 6, v0=0.05, v1=0.46, shift=0.2),
    "gunship": hramp(216, 0.34, 9, v0=0.09, v1=1.00, shift=0.3),
    "teal":   hramp(174, 0.72, 8, v0=0.08, v1=0.96, shift=0.25),
    "red":    hramp(356, 0.82, 8, v0=0.10, v1=1.00, shift=0.25),
    "fire":   ramp(33, 34, 35, 36, 37, 38, 39),
    "amber":  hramp(34, 0.88, 8, v0=0.14, v1=1.00, shift=0.25),
    "gold":   hramp(44, 0.72, 8, v0=0.10, v1=1.00, shift=0.3),
    "violet": hramp(280, 0.62, 8, v0=0.08, v1=0.96, shift=0.25),
    "rose":   hramp(334, 0.66, 8, v0=0.10, v1=1.00, shift=0.25),
    "rock":   hramp(26, 0.30, 9, v0=0.06, v1=0.82, shift=0.35),
    "green":  hramp(108, 0.66, 8, v0=0.06, v1=0.96, shift=0.3),
    "glass":  hramp(194, 0.72, 7, v0=0.06, v1=1.00, shift=0.3),
    "ice":    hramp(204, 0.34, 7, v0=0.30, v1=1.00, shift=0.2),
    "white":  ramp(6, 7, 8),
    "wine_dark": ramp(1, 14, 15),
    "smoke":  ramp(40, 41, 42, 43, 44, 45),
    "plasma": ramp(46, 47, 48, 49, 50, 51),
    "toxic":  ramp(52, 53, 54, 55, 56, 57),
    "magenta": ramp(58, 59, 60, 61, 62, 63),
    "hotfire": ramp(32, 33, 34, 35, 36, 37, 38, 39),
}

OUTLINE = rgb(1)


def write_gpl(path):
    with open(path, "w") as f:
        # The engine's .gpl reader expects exactly four header lines:
        # magic, name, columns and a single comment line.
        f.write("GIMP Palette\nName: Halcyon\nColumns: 8\n")
        f.write("# Halcyon master palette, Copyright 2026 Andrew Ahlfield, "
                "CC BY-SA 4.0\n")
        for i, h in enumerate(MASTER):
            r, g, b = rgb(i)
            f.write("%3d %3d %3d\t%d\n" % (r, g, b, i))
