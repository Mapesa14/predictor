"""Generate every icon the web manifest and the native shells need.

The project had one SVG favicon, which is enough for a browser tab and nothing
else: an installable PWA wants 192 and 512 PNGs, Android wants a *maskable*
variant because launchers crop icons to whatever shape the phone uses, and the
store listing wants 1024.

Drawn rather than rasterised, so the maskable version can be laid out for its
own safe zone instead of being a scaled copy that loses its corners. The design
is the existing favicon's: dark ground, blue ring, dashed green inner ring, the
1X2 wordmark.

    python web/tools/icons.py
"""
from __future__ import annotations

import os
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.dirname(HERE)

GROUND = (13, 17, 23, 255)      # #0d1117
BLUE = (47, 129, 247, 255)      # #2f81f7
GREEN = (126, 231, 135, 255)    # #7ee787
INK = (230, 237, 243, 255)      # #e6edf3

# Any bold sans will do; the wordmark is three characters.
FONTS = [
    r"C:\Windows\Fonts\segoeuib.ttf",
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\calibrib.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
]


def _font(size: int):
    for p in FONTS:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
    return ImageFont.load_default()


def draw(size: int, maskable: bool = False) -> Image.Image:
    """One icon.

    `maskable` fills the whole canvas and shrinks the artwork into the centre
    66%, which is what survives a circular crop. A plain icon keeps its rounded
    square, because nothing is going to cut it.
    """
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    if maskable:
        d.rectangle([0, 0, size, size], fill=GROUND)
        scale = 0.66          # Android's safe zone is the middle 80% diameter
    else:
        d.rounded_rectangle([0, 0, size - 1, size - 1],
                            radius=int(size * 0.1875), fill=GROUND)
        scale = 0.92

    c = size / 2
    outer = size * scale * 0.34
    # Wider than the favicon's, and the wordmark below is smaller, so the text
    # sits inside the dashed ring instead of cutting through it. At 64px that
    # overlap read as texture; at 512 it reads as a mistake.
    inner = size * scale * 0.235
    wide = max(1, int(size * scale * 0.062))
    thin = max(1, int(size * scale * 0.042))

    d.ellipse([c - outer, c - outer, c + outer, c + outer],
              outline=BLUE, width=wide)

    # A dashed ring, drawn as arcs: PIL has no dash pattern.
    step, gap = 30, 11
    for a in range(0, 360, step):
        d.arc([c - inner, c - inner, c + inner, c + inner],
              start=a, end=a + step - gap, fill=GREEN, width=thin)

    text = "1X2"
    f = _font(max(8, int(size * scale * 0.215)))
    box = d.textbbox((0, 0), text, font=f)
    d.text((c - (box[2] - box[0]) / 2 - box[0],
            c - (box[3] - box[1]) / 2 - box[1]), text, font=f, fill=INK)
    return img


def splash(w: int = 2732, h: int = 2732) -> Image.Image:
    """A plain centred mark on the app's own ground - no text to translate."""
    img = Image.new("RGBA", (w, h), GROUND)
    mark = draw(int(min(w, h) * 0.22), maskable=False)
    img.alpha_composite(mark, ((w - mark.width) // 2, (h - mark.height) // 2))
    return img


def main() -> int:
    icons = os.path.join(WEB, "public", "icons")
    res = os.path.join(WEB, "resources")
    os.makedirs(icons, exist_ok=True)
    os.makedirs(res, exist_ok=True)

    made = []
    for n in (192, 512):
        p = os.path.join(icons, "icon-%d.png" % n)
        draw(n).save(p)
        made.append(p)
        p = os.path.join(icons, "maskable-%d.png" % n)
        draw(n, maskable=True).save(p)
        made.append(p)

    # apple-touch-icon has no mask and no transparency: iOS rounds it itself.
    p = os.path.join(icons, "apple-touch-icon.png")
    draw(180).convert("RGB").save(p)
    made.append(p)

    # What @capacitor/assets consumes to fill in every native density.
    p = os.path.join(res, "icon.png")
    draw(1024).convert("RGB").save(p)
    made.append(p)
    p = os.path.join(res, "icon-foreground.png")
    draw(1024, maskable=True).save(p)
    made.append(p)
    p = os.path.join(res, "icon-background.png")
    Image.new("RGB", (1024, 1024), GROUND[:3]).save(p)
    made.append(p)
    p = os.path.join(res, "splash.png")
    splash().convert("RGB").save(p)
    made.append(p)
    p = os.path.join(res, "splash-dark.png")
    splash().convert("RGB").save(p)
    made.append(p)

    for f in made:
        print("%8d B  %s" % (os.path.getsize(f), os.path.relpath(f, WEB)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
