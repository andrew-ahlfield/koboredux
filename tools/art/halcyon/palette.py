"""Halcyon palette and material ramps.

All colours in the Halcyon theme come from this file. The 32 entry master
palette is written out as halcyon.gpl and is what the theme's realtime
effects (fire, stars, radar, menu focus effect) index into. Sprites use the
ramps below, which are drawn from the same master palette.
"""

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
]


def rgb(i):
    h = MASTER[i]
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def ramp(*idx):
    return [rgb(i) for i in idx]


# Material ramps, darkest to brightest.
RAMPS = {
    "steel":  ramp(1, 2, 3, 4, 5, 6, 7),
    "hull":   ramp(2, 3, 4, 5, 6, 7, 8),
    "dark":   ramp(1, 2, 3, 4, 5),
    "teal":   ramp(9, 10, 11, 12, 13),
    "red":    ramp(14, 15, 16, 17, 18),
    "fire":   ramp(15, 16, 17, 18, 19, 20),
    "amber":  ramp(26, 27, 18, 19, 20),
    "gold":   ramp(25, 26, 27, 19, 20),
    "violet": ramp(14, 21, 22, 23, 24),
    "rose":   ramp(14, 15, 16, 24, 8),
    "rock":   ramp(1, 25, 26, 27, 28),
    "green":  ramp(9, 29, 30, 31, 13),
    "glass":  ramp(1, 3, 4, 11, 12, 13),
    "ice":    ramp(3, 5, 6, 7, 8),
    "white":  ramp(6, 7, 8),
    "wine_dark": ramp(1, 14, 15),
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
