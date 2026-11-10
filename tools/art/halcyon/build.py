"""Generate every image of the Halcyon graphics theme.

Usage (from tools/art):  python3 -m halcyon.build [output_dir]

The default output is data/gfx/halcyon in the repository. The theme
description files (header.theme, main.theme) and LICENSE are hand written
and live next to the generated images; this script only writes images and
the palette.
"""

import os
import sys
import time

from PIL import Image

from . import sprites, tiles, backgrounds, hud, fonts, fx, palette

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DEFAULT_OUT = os.path.join(REPO, "data", "gfx", "halcyon")


def save(img, out, rel, colors=None):
    path = os.path.join(out, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if colors:
        # Few-colour images compress far better as paletted PNGs
        img = img.convert("RGB").quantize(colors=colors, method=2,
                                          dither=Image.Dither.NONE)
    img.save(path, optimize=True)
    print("  %-34s %dx%d" % (rel, img.width, img.height))


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT
    t0 = time.time()
    print("Writing Halcyon theme images to", out)
    os.makedirs(out, exist_ok=True)
    palette.write_gpl(os.path.join(out, "halcyon.gpl"))

    # Sprites
    s = "sprites/"
    save(sprites.player(), out, s + "player.png")
    save(sprites.crescent("green", "green"), out, s + "crescent-green.png")
    save(sprites.crescent("violet", "rose"), out, s + "crescent-violet.png")
    save(sprites.crescent("rose", "amber"), out, s + "crescent-rose.png")
    save(sprites.fighter(), out, s + "fighter.png")
    save(sprites.missile1(), out, s + "missile1.png")
    save(sprites.missile2(), out, s + "missile2.png")
    save(sprites.missile3(), out, s + "missile3.png")
    save(sprites.bolts(), out, s + "bolt.png")
    save(sprites.rock(), out, s + "rock.png")
    save(sprites.rockexpl(), out, s + "rockexpl.png")
    blue, green, red = sprites.bullets()
    save(blue, out, s + "bullet-blue.png")
    save(green, out, s + "bullet-green.png")
    save(red, out, s + "bullet-red.png")
    blue, green, red = sprites.bullet_explosions()
    save(blue, out, s + "bulletexpl-blue.png")
    save(green, out, s + "bulletexpl-green.png")
    save(red, out, s + "bulletexpl-red.png")
    save(sprites.ring(), out, s + "ring.png")
    save(sprites.ringexpl(), out, s + "ringexpl.png")
    save(sprites.bombs(), out, s + "bomb.png")
    save(sprites.bombdeto(), out, s + "bombdeto.png")
    save(sprites.bigship(), out, s + "mothership.png")

    # Regions
    for r in range(5):
        d = "region-%d/" % (r + 1)
        save(tiles.tileset(r, 16), out, d + "tiles.png")
        save(tiles.tileset(r, 12), out, d + "tiles-small.png")
        save(tiles.tileset(r, 8), out, d + "tiles-tiny.png")
        save(backgrounds.planet(r), out, d + "planet.png", colors=16)
        save(backgrounds.ground_l8(r), out, d + "ground-l8.png", colors=16)
        save(backgrounds.ground_l10(r), out, d + "ground-l10.png",
             colors=16)
    save(backgrounds.space(), out, "space.png", colors=128)

    # Dashboard
    d = "dashboard/"
    save(hud.dashboard(), out, d + "dashboard.png", colors=64)
    save(sprites.crosshair(), out, d + "crosshair.png")
    save(hud.proxy_leds(True), out, d + "leds-h.png")
    save(hud.proxy_leds(False), out, d + "leds-v.png")
    save(hud.health_leds(), out, d + "shieldleds.png")
    save(hud.charge_leds(), out, d + "chargeleds.png")
    save(hud.weaponslots(), out, d + "weaponslots.png")
    save(hud.weapons(), out, d + "weapons.png")
    save(hud.logo(), out, d + "logo.png")

    # Fonts
    d = "fonts/"
    save(fonts.microfont(), out, d + "microfont.png")
    save(fonts.smallfont(), out, d + "smallfont.png")
    save(fonts.bigfont(), out, d + "bigfont.png")
    save(fonts.counterfont(), out, d + "counterfont.png")

    # Effects
    d = "fx/"
    save(fx.noise(), out, d + "noise.png")
    save(fx.hitnoise(), out, d + "hitnoise.png")
    save(sprites.shield(), out, d + "shield.png")
    save(fx.gridtfx_small(), out, d + "gridtfx-tiles.png")
    save(fx.gridtfx_fullscreen(), out, d + "fs-gridtfx-tiles.png")

    print("Done in %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
