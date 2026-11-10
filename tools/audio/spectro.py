"""Draw waveform envelopes and spectrograms of WAV files into one PNG.

Usage: python3 spectro.py out.png a.wav [b.wav ...]

Used to eyeball sound effects and music rendered with a2render: envelope
shape (attack, decay, clipping) on top, log-frequency spectrogram below.
"""

import sys
import wave

import numpy as np
from PIL import Image, ImageDraw


def load(path):
    with wave.open(path) as w:
        n = w.getnframes()
        d = np.frombuffer(w.readframes(n), dtype=np.int16) / 32768.0
        return d, w.getframerate()


def panel(path, width=900, eh=60, sh=140):
    x, rate = load(path)
    img = Image.new("RGB", (width, eh + sh + 16), (10, 12, 24))
    dr = ImageDraw.Draw(img)
    dr.text((4, 2), "%s  %.2fs" % (path.split("/")[-1], len(x) / rate),
            fill=(200, 210, 240))
    # Envelope
    n = max(1, len(x) // width)
    for i in range(width):
        seg = x[i * n:(i + 1) * n]
        if not len(seg):
            break
        hi, lo = seg.max(), seg.min()
        y0 = 16 + eh // 2 - int(hi * eh / 2)
        y1 = 16 + eh // 2 - int(lo * eh / 2)
        col = (255, 80, 60) if max(hi, -lo) > 0.99 else (90, 200, 170)
        dr.line([(i, y0), (i, y1)], fill=col)
    # Spectrogram, log frequency 40 Hz to 16 kHz
    win = 1024
    hop = max(1, (len(x) - win) // width)
    spec = np.zeros((sh, width))
    freqs = np.fft.rfftfreq(win, 1.0 / rate)
    bins = np.geomspace(40, 16000, sh + 1)
    idx = np.searchsorted(freqs, bins)
    w_ = np.hanning(win)
    for i in range(width):
        seg = x[i * hop:i * hop + win]
        if len(seg) < win:
            break
        m = np.abs(np.fft.rfft(seg * w_))
        for k in range(sh):
            a, b = idx[k], max(idx[k + 1], idx[k] + 1)
            spec[sh - 1 - k, i] = m[a:b].max()
    db = 20 * np.log10(spec + 1e-6)
    v = np.clip((db + 70) / 70, 0, 1)
    rgb = np.stack([v ** 0.7 * 255, v ** 1.5 * 230, (1 - v) * 90 + v * 120],
                   axis=-1).astype(np.uint8)
    img.paste(Image.fromarray(rgb, "RGB"), (0, eh + 16))
    return img


def main():
    out = sys.argv[1]
    panels = [panel(p) for p in sys.argv[2:]]
    h = sum(p.height + 4 for p in panels)
    o = Image.new("RGB", (panels[0].width, h), (0, 0, 0))
    y = 0
    for p in panels:
        o.paste(p, (0, y))
        y += p.height + 4
    o.save(out)


if __name__ == "__main__":
    main()
