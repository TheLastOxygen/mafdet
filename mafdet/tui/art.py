"""
The identity in the terminal: the shapes of identity.py drawn with half-block characters.

Each character cell holds two pixels, one above the other: '▀' with the upper
pixel as foreground colour and the lower one as background. The polygons are
filled with the even-odd rule, like the SVG files, so the eyes of the mask are
cut out of its mass and drawn in amber on their own.
"""
from __future__ import annotations

import re

from rich.style import Style
from rich.text import Text

from ..identity import AMBER, SHAPES
from .fx import TEXT as PAPER   # the amber phosphor's text colour: the mask is drawn in it

# The wordmark in block characters, as in the visual identity's terminal mockup (and cli.py).
WORDMARK = ["█▀▄▀█ ▄▀█ █▀▀ █▀▄ █▀▀ ▀█▀",
            "█ ▀ █ █▀█ █▀  █▄▀ ██▄  █"]


def _polygons(d: str) -> list[list[tuple[float, float]]]:
    """'M x y L x y ... Z M ...' (straight segments only) into lists of points."""
    polys = []
    for sub in re.findall(r"M[^M]*", d):
        nums = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", sub)]
        polys.append(list(zip(nums[0::2], nums[1::2])))
    return polys


def _inside(x: float, y: float, polys) -> bool:
    """Even-odd rule over all sub-paths: a hole is a sub-path inside another."""
    hit = False
    for poly in polys:
        n = len(poly)
        for i in range(n):
            (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
            if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
                hit = not hit
    return hit


def raster(name: str, width: int, eyes: bool = True, ink: str = PAPER, accent: str = AMBER) -> list[list[str | None]]:
    """The shape as a grid of colours (None = empty), `width` pixels wide and as many tall."""
    layers = [(_polygons(d), is_accent) for d, is_accent, _ in SHAPES[name]]
    grid = []
    for row in range(width):
        line = []
        for col in range(width):
            x, y = (col + .5) * 100 / width, (row + .5) * 100 / width
            colour = None
            for polys, is_accent in layers:
                if _inside(x, y, polys):
                    colour = (accent if eyes else None) if is_accent else ink
            line.append(colour)
        grid.append(line)
    return grid


def halfblocks(grid: list[list[str | None]], background: str | None = None) -> Text:
    """Two pixel rows per text line."""
    out = Text()
    for r in range(0, len(grid), 2):
        top, bottom = grid[r], grid[r + 1] if r + 1 < len(grid) else [None] * len(grid[r])
        for a, b in zip(top, bottom):
            if a is None and b is None:
                out.append(" ", Style(bgcolor=background) if background else None)
            elif a == b:
                out.append("█", Style(color=a))
            elif a is None:
                out.append("▄", Style(color=b, bgcolor=background))
            elif b is None:
                out.append("▀", Style(color=a, bgcolor=background))
            else:
                out.append("▀", Style(color=a, bgcolor=b))
        out.append("\n")
    out.rstrip()
    return out


def mark(name: str = "mask", width: int = 24, eyes: bool = True, ink: str = PAPER) -> Text:
    return halfblocks(raster(name, width, eyes, ink))


def wordmark(colour: str = PAPER) -> Text:
    return Text("\n".join(WORDMARK), style=Style(color=colour, bold=True))
