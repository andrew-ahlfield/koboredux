"""Scaled preview sheet: python3 -m halcyon.preview [-sN] out.png in.png..."""

import sys
from PIL import Image


def preview(out, files, scale=4, bg=(12, 14, 30)):
    ims = [Image.open(f).convert("RGBA") for f in files]
    W = max(i.width for i in ims) * scale + 8
    H = sum(i.height * scale + 8 for i in ims)
    o = Image.new("RGBA", (W, H), bg + (255,))
    y = 0
    for i in ims:
        o.alpha_composite(i.resize((i.width * scale, i.height * scale),
                                   Image.NEAREST), (4, y + 4))
        y += i.height * scale + 8
    o.save(out)


if __name__ == "__main__":
    s = 4
    args = sys.argv[1:]
    if args[0].startswith("-s"):
        s = int(args[0][2:])
        args = args[1:]
    preview(args[0], args[1:], s)
