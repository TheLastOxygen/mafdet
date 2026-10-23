"""
The launcher: `mafdet` with no command, in a terminal. Which interface to open, the window or
the terminal app, with an option to remember the choice (then `mafdet` opens it directly;
`mafdet --ask` shows the launcher again).
"""
from __future__ import annotations

from rich.style import Style
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Center, Horizontal, Middle, Vertical
from textual.widgets import Checkbox, Static

from .. import __version__
from ..identity import AMBER
from . import art
from .fx import Rain
from .app import FAINT, LINE, MUTED, PAPER, THEME

CHOICES = [
    ("gui", "WINDOW", "The full app in its own window: chat, models, hardware and settings, in English, Italian, "
                      "French or German.", "W"),
    ("tui", "TERMINAL", "The same app right here, in this terminal. Keyboard first: F1 to F4 move between views.",
     "T"),
]


class Choice(Static):
    DEFAULT_CSS = """
    Choice { width: 46; height: 9; border: round #4A3312; padding: 1 2; margin: 0 1; background: #0F0A05; }
    Choice.on { border: round #D4892A; background: #1A1108; }
    Choice:hover { border: round #A8742F; }
    """

    def __init__(self, key: str, title: str, body: str, letter: str):
        super().__init__()
        self.key, self.title, self.body, self.letter = key, title, body, letter

    def render(self) -> Text:
        on = self.has_class("on")
        t = Text()
        t.append("◆ " if on else "◇ ", style=AMBER if on else FAINT)
        t.append(self.title, style=Style(color=PAPER, bold=True))
        t.append(f"   {self.letter}\n\n", style=FAINT)
        t.append(self.body, style=PAPER if on else MUTED)
        return t

    def on_click(self):
        self.app.pick(self.key)


class Launcher(App):
    TITLE = "MAFDET"
    CSS = """
    Screen { align: center middle; background: $background; }
    Screen { layers: rain front; }
    Rain { layer: rain; dock: top; }
    #box { layer: front; width: 104; height: auto; background: #050302; padding: 1 2; }
    #mark, #word, #tag, #hint { width: auto; }
    #word { margin-top: 1; }
    #tag { margin: 1 0 1 0; }
    #choices { width: auto; height: auto; margin-top: 1; }
    #remember { margin-top: 1; background: transparent; border: none; }
    #hint { margin-top: 1; }
    """
    BINDINGS = [
        Binding("left,shift+tab", "move(-1)", show=False),
        Binding("right,tab", "move(1)", show=False),
        Binding("enter", "confirm", show=False),
        Binding("w", "pick('gui')", show=False),
        Binding("t", "pick('tui')", show=False),
        Binding("r", "toggle_remember", show=False),
        Binding("escape,ctrl+q,q", "quit", show=False),
    ]

    def __init__(self):
        super().__init__()
        self.choice = "gui"
        self.result = None

    def compose(self) -> ComposeResult:
        yield Rain()
        with Vertical(id="box"):
            yield Center(Static(art.mark("mask", 24, eyes=False), id="mark"))
            yield Center(Static(art.wordmark(), id="word"))
            yield Center(Static(Text.assemble(("YOU TYPE. IT ALREADY KNOWS.", Style(color=FAINT)),
                                              (f"   v{__version__}", Style(color=LINE))), id="tag"))
            with Center(), Horizontal(id="choices"):
                for c in CHOICES:
                    yield Choice(*c)
            yield Center(Checkbox("Remember my choice", id="remember"))
            yield Center(Static(Text("← → choose · Enter open · R remember · Esc quit", style=FAINT), id="hint"))

    def on_mount(self):
        self.register_theme(THEME)
        self.theme = "mafdet"
        self.select(self.choice)
        # The keys belong to the launcher: a focused checkbox would take Enter for itself.
        self.query_one("#remember", Checkbox).can_focus = False
        for i, w in enumerate(self.query(Choice)):
            w.styles.opacity = 0
            self.set_timer(0.35 + i * 0.15, lambda w=w: w.styles.animate("opacity", 1.0, duration=0.4))
        self.set_timer(0.5, lambda: self.query_one("#mark", Static).update(art.mark("mask", 24)))
        self.set_interval(3.2, self.blink_eyes)

    def blink_eyes(self):
        mark = self.query_one("#mark", Static)
        mark.update(art.mark("mask", 24, eyes=False))
        self.set_timer(0.12, lambda: mark.update(art.mark("mask", 24)))

    def select(self, key: str):
        self.choice = key
        for w in self.query(Choice):
            w.set_class(w.key == key, "on")
            w.refresh()

    def action_move(self, step: int):
        keys = [c[0] for c in CHOICES]
        self.select(keys[(keys.index(self.choice) + step) % len(keys)])

    def action_pick(self, key: str):
        self.pick(key)

    def pick(self, key: str):
        self.select(key)
        self.action_confirm()

    def action_toggle_remember(self):
        box = self.query_one("#remember", Checkbox)
        box.value = not box.value

    def action_confirm(self):
        self.result = (self.choice, self.query_one("#remember", Checkbox).value)
        self.exit(self.result)


def choose() -> tuple[str, bool] | None:
    """Show the launcher; (interface, remember) or None if the user quit."""
    return Launcher().run()
