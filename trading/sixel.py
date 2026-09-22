"""Print a PNG inside the terminal with the Sixel protocol.

Windows Terminal (1.22 and later) and most modern terminals draw Sixel images
in place. The image is resized, reduced to a small palette and encoded six
pixel rows at a time.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def encode(path: Path, width: int = 1100, colors: int = 255) -> str:
    from PIL import Image

    img = Image.open(path).convert("RGB")
    if img.width != width:
        img = img.resize((width, max(1, round(img.height * width / img.width))), Image.LANCZOS)
    q = img.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    pixels = np.asarray(q, dtype=np.int16)
    palette = np.asarray(q.getpalette()[: 3 * colors], dtype=np.float64).reshape(-1, 3)
    h, w = pixels.shape
    out = ['\x1bP0;1;0q"1;1;%d;%d' % (w, h)]
    for i, (r, g, b) in enumerate(palette):
        out.append("#%d;2;%d;%d;%d" % (i, round(r * 100 / 255), round(g * 100 / 255), round(b * 100 / 255)))
    weights = (1 << np.arange(6)).reshape(6, 1)
    for top in range(0, h, 6):
        band = pixels[top:top + 6]
        if band.shape[0] < 6:
            band = np.vstack([band, np.full((6 - band.shape[0], w), -1, dtype=np.int16)])
        first = True
        for c in np.unique(band[band >= 0]):
            bits = ((band == c) * weights).sum(axis=0) + 63
            # run-length: "!n" + char for repeated sixels
            change = np.flatnonzero(np.diff(bits)) + 1
            starts = np.concatenate(([0], change))
            ends = np.concatenate((change, [w]))
            parts = []
            for s, e in zip(starts, ends):
                ch = chr(int(bits[s]))
                n = e - s
                parts.append(("!%d%s" % (n, ch)) if n > 3 else ch * n)
            out.append(("" if first else "$") + "#%d" % c + "".join(parts))
            first = False
        out.append("-")
    out.append("\x1b\\")
    return "".join(out)
