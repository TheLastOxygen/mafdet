"""
Builds the MAFDET identity assets (SVG) from the shapes of the visual identity
v2 (claude.ai/design, "MAFDET Identity v2"), kept in mafdet/identity.py.

    python assets/build_assets.py           # SVG files
    python assets/build_assets.py --ico     # also mafdet/ui/mafdet.ico (needs Microsoft Edge)
"""
from __future__ import annotations

import base64
import os
import shutil
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from mafdet.identity import AMBER, INK, PAPER, SHAPES, WORDMARK, WORDMARK_SIZE  # noqa: E402


def n(x: float) -> str:
    return f"{x:.3f}".rstrip("0").rstrip(".")


def shape(name: str, fg: str, mono: bool = False, x: float = 0, y: float = 0, size: float = 100) -> str:
    """One shape of the 100-unit grid, scaled to `size` and placed at (x, y)."""
    paths = []
    for d, accent, open_ in SHAPES[name]:
        if accent and mono and open_:
            continue
        fill = fg if not accent or mono else AMBER
        paths.append(f'<path fill-rule="evenodd" fill="{fill}" d="{d}"/>')
    return f'<g transform="translate({n(x)} {n(y)}) scale({n(size / 100)})">{"".join(paths)}</g>'


def wordmark(fg: str, x: float = 0, y: float = 0, height: float = 100) -> str:
    """The wordmark, always monochrome, with its cap height scaled to `height`."""
    paths = "".join(f'<path fill="{fg}" transform="translate({tx} 0)" d="{d}"/>' for tx, d in WORDMARK)
    return f'<g transform="translate({n(x)} {n(y)}) scale({n(height / 100)})">{paths}</g>'


def svg(width: float, height: float, body: str, title: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{n(width)}" height="{n(height)}" '
            f'viewBox="0 0 {n(width)} {n(height)}" role="img" aria-label="{title}">'
            f"<title>{title}</title>{body}</svg>\n")


def tile(name: str, title: str, box: float, radius: float, mark: float) -> str:
    """The shape in paper and amber on an ink tile with rounded corners: it reads on light and dark pages."""
    offset = (box - mark) / 2
    body = f'<rect width="{n(box)}" height="{n(box)}" rx="{n(radius)}" fill="{INK}"/>' + \
        shape(name, PAPER, x=offset, y=offset, size=mark)
    return svg(box, box, body, title)


def build() -> dict[str, str]:
    files = {}
    for suffix, fg in (("", INK), ("-dark", PAPER)):
        files[f"mafdet-logo{suffix}.svg"] = svg(100, 100, shape("mask", fg), "MAFDET")
        w, h = WORDMARK_SIZE
        files[f"mafdet-wordmark{suffix}.svg"] = svg(w, h, wordmark(fg), "MAFDET")
        # Horizontal lockup: wordmark 0.4× the symbol height, gap 1/3 of the symbol.
        files[f"mafdet-lockup{suffix}.svg"] = svg(
            100 + 100 / 3 + 280, 100, shape("mask", fg) + wordmark(fg, x=100 + 100 / 3, y=30, height=40), "MAFDET")
        # Vertical lockup: wordmark 1.3× the symbol width.
        wm_h = 130 / 7
        gap = 22
        files[f"mafdet-lockup-vertical{suffix}.svg"] = svg(
            130, 100 + gap + wm_h, shape("mask", fg, x=15) + wordmark(fg, y=100 + gap, height=wm_h), "MAFDET")
    files["mafdet-logo-mono.svg"] = svg(100, 100, shape("mask", "#000000", mono=True), "MAFDET")
    # App icon and avatar: the mask at 68% of an ink tile with 22% corners. Favicon: 13 of 16 px.
    files["mafdet-icon.svg"] = tile("mask", "MAFDET", 200, 44, 136)
    files["favicon.svg"] = tile("mask", "MAFDET", 16, 3, 13)
    for name in ("awaken", "watch", "strike", "hunt", "race", "judge"):
        files[f"icons/{name}.svg"] = tile(name, f"mafdet {name}", 100, 22, 68)
    return files


# --------------------------------------------------------------------------
# Window icon (.ico): the app icon rendered by Edge, packed as PNG entries
# --------------------------------------------------------------------------

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
ICO_SIZES = (16, 24, 32, 48, 64, 128, 256)


def render_png(svg_text: str, size: int, workdir: str) -> bytes:
    page = os.path.join(workdir, f"icon{size}.html")
    png = os.path.join(workdir, f"icon{size}.png")
    data = base64.b64encode(svg_text.encode("utf-8")).decode("ascii")
    with open(page, "w", encoding="utf-8") as f:
        f.write(f'<html><body style="margin:0;background:transparent">'
                f'<img src="data:image/svg+xml;base64,{data}" '
                f'width="{size}" height="{size}" style="display:block"></body></html>')
    subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                    "--default-background-color=00000000", f"--window-size={size},{size}",
                    f"--screenshot={png}", f"file:///{page}"], check=True, capture_output=True)
    with open(png, "rb") as f:
        return f.read()


def build_ico(path: str):
    """An .ico with one PNG per size (supported since Windows Vista)."""
    # Small sizes: the favicon geometry (mark at 13/16), which keeps the cuts readable.
    small, large = build()["favicon.svg"], build()["mafdet-icon.svg"]
    workdir = tempfile.mkdtemp()
    try:
        images = [render_png(small if s <= 32 else large, s, workdir) for s in ICO_SIZES]
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    header = struct.pack("<HHH", 0, 1, len(images))
    offset = 6 + 16 * len(images)
    entries, data = b"", b""
    for size, png in zip(ICO_SIZES, images):
        entries += struct.pack("<BBBBHHII", size % 256, size % 256, 0, 0, 1, 32, len(png), offset + len(data))
        data += png
    with open(path, "wb") as f:
        f.write(header + entries + data)


def main():
    for rel, content in build().items():
        path = os.path.join(HERE, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        print(f"assets/{rel}")
    if "--ico" in sys.argv:
        ico = os.path.join(os.path.dirname(HERE), "mafdet", "ui", "mafdet.ico")
        build_ico(ico)
        print("mafdet/ui/mafdet.ico")


if __name__ == "__main__":
    main()
