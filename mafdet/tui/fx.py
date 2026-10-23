"""
The terminal app's look: an amber phosphor screen, and the effects that go with it.

The palette is a monochrome CRT ramp built around the brand's amber (identity.AMBER), from the
brightest phosphor down to the faint glow of a border; alarms keep a red of their own. The effects
(text that decrypts into place, falling glyphs, a glitch) all have a still version, used when the
app's Settings → Cinematic effects is off.
"""
from __future__ import annotations

import random
import time

from rich.style import Style
from rich.text import Text
from textual.widget import Widget
from textual.widgets import Static

from ..identity import AMBER

BLACK = "#050302"
HOT = "#FFC46B"                 # the brightest phosphor: live state, cursor, OK
TEXT = "#F2CB8E"                # body text
SOFT = "#D9AE6E"                # secondary body text
DIM = "#A8742F"                 # labels, hints
FAINT = "#7A5424"               # the least important text
LINE = "#4A3312"                # borders
DEEP = "#2E200C"                # dividers, table cursor
PANEL = "#1A1108"
SURFACE = "#0F0A05"
RAIL = "#080503"
ALARM = "#F0603A"

# Glyphs for the rain and the decrypting text: ASCII only, every one a single cell wide in any terminal.
GLYPHS = "01#$%&*+=<>/\\|{}[]:;~^ABCDEFabcdef0123456789"


def effects_on(app) -> bool:
    return getattr(app, "effects", True)


def scramble(target: str, shown: int, rng: random.Random) -> str:
    """`target` with its first `shown` characters in place and the rest of its letters and digits scrambled."""
    return "".join(c if i < shown or not c.isalnum() else rng.choice(GLYPHS) for i, c in enumerate(target))


def glitch(lines: list[str], rng: random.Random, style: Style | str = "") -> Text:
    """A frame of interference: a line or two slipped sideways, a few glyphs in the wrong place."""
    out = []
    for line in lines:
        chars = list(line)
        for _ in range(rng.randint(0, 2)):
            if chars:
                chars[rng.randrange(len(chars))] = rng.choice(GLYPHS)
        shift = rng.choice((0, 0, 0, 1, -1, 2))
        text = "".join(chars)
        out.append(" " * shift + text if shift >= 0 else text[-shift:])
    return Text("\n".join(out), style=style)


class Decrypt(Static):
    """Text that resolves from random glyphs into place, left to right."""

    def __init__(self, text: str = "", style: Style | str = "", duration: float = 0.45, **kwargs):
        super().__init__("", **kwargs)
        self.target, self.look, self.duration = text, style, duration
        self._timer = None

    def on_mount(self):
        self.play(self.target)

    def play(self, text: str, style: Style | str | None = None):
        self.target = text
        if style is not None:
            self.look = style
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        if not effects_on(self.app):
            self.update(Text(text, style=self.look))
            return
        self.t0, self.rng = time.perf_counter(), random.Random()
        self._frame()
        self._timer = self.set_interval(1 / 30, self._frame)

    def _frame(self):
        progress = min(1.0, (time.perf_counter() - self.t0) / self.duration)
        shown = int(len(self.target) * progress)
        text = Text(scramble(self.target, shown, self.rng), style=self.look)
        if progress < 1:
            text.stylize(Style(color=DIM), shown)
        self.update(text)
        if progress >= 1 and self._timer is not None:
            self._timer.stop()
            self._timer = None


class Rain(Widget):
    """Glyphs falling down the screen, in columns, the head of each drop brightest."""

    DEFAULT_CSS = "Rain { width: 100%; height: 100%; }"

    def on_mount(self):
        self.rng = random.Random()
        self.drops: dict[int, list[float]] = {}         # column → [head row, speed, length]
        self.cells: dict[tuple[int, int], str] = {}
        if effects_on(self.app):
            self.set_interval(1 / 16, self.step)

    def step(self):
        w, h = self.size.width, self.size.height
        seeding = not self.drops and not self.cells     # the first frame: drops already all the way down
        for col in range(0, w, 2):                      # every other column: the classic look, half the work
            drop = self.drops.get(col)
            if seeding and self.rng.random() < 0.35:
                length = self.rng.randint(4, max(5, h // 2))
                self.drops[col] = [self.rng.uniform(0, h + length), self.rng.uniform(0.5, 1.4), length]
            elif drop is None:
                if self.rng.random() < 0.04:
                    self.drops[col] = [0.0, self.rng.uniform(0.5, 1.4), self.rng.randint(4, max(5, h // 2))]
            else:
                drop[0] += drop[1]
                if drop[0] - drop[2] > h:
                    del self.drops[col]
        if self.rng.random() < 0.5 and self.cells:     # a glyph here and there changes as it falls
            self.cells.pop(self.rng.choice(list(self.cells)))
        self.refresh()

    def render(self) -> Text:
        w, h = self.size.width, self.size.height
        grid = [[(" ", None)] * w for _ in range(h)]
        for col, (head, _, length) in self.drops.items():
            if col >= w:
                continue
            top = int(head)
            for k in range(int(length)):
                row = top - k
                if 0 <= row < h:
                    glyph = self.cells.setdefault((col, row), self.rng.choice(GLYPHS))
                    colour = HOT if k == 0 else AMBER if k < 3 else DIM if k < length / 2 else FAINT
                    grid[row][col] = (glyph, colour)
        out = Text()
        for r, row in enumerate(grid):
            run, colour = "", None
            for glyph, c in row:
                if c != colour and run:
                    out.append(run, style=colour)
                    run = ""
                run, colour = run + glyph, c
            out.append(run, style=colour)
            if r < h - 1:
                out.append("\n")
        return out
