"""
mafdet tui: the whole app in the terminal (Textual), on the same ChatService as the window.

    mafdet tui

F1 Chat · F2 Models · F3 Hardware · F4 Settings · Ctrl+N new chat · Esc stop · Ctrl+Q quit.
"""
from __future__ import annotations

import time

from rich.style import Style
from rich.table import Table
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.css.query import NoMatches
from textual.containers import Center, Horizontal, Middle, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.theme import Theme
from textual.widgets import (Button, Collapsible, ContentSwitcher, DataTable, Footer, Input, Label, LoadingIndicator,
                             Markdown, ProgressBar, Select, Static, Switch)

from .. import __version__
from ..core.chat import ChatService
from ..identity import AMBER, INK, PAPER
from . import art, mathtext

MUTED, FAINT, GREEN, RUST, LINE = "#A39D92", "#7D776D", "#7FA36B", "#C46A3A", "#3A3733"

THEME = Theme(name="mafdet", primary=AMBER, secondary=PAPER, accent=AMBER, foreground="#E4E0D8",
              background=INK, surface="#121211", panel="#191816", success=GREEN, warning=AMBER, error=RUST,
              dark=True, variables={"footer-key-foreground": AMBER, "block-cursor-background": AMBER,
                                    "block-cursor-foreground": INK, "input-selection-background": "#D4892A55"})

PLACEMENT = {"gpu": ("All on GPU", GREEN), "partial": ("{p}% on GPU", AMBER), "cpu": ("On CPU", PAPER),
             "too_big": ("Too large", RUST)}
TIER = {"tiny": "Tiny and instant: simple chats and quick drafts.",
        "light": "Light and quick: everyday questions.",
        "balanced": "Balanced: solid reasoning at full speed on most GPUs.",
        "strong": "Strong: sharper reasoning, at its best with 8 GB of VRAM.",
        "very_strong": "Very strong: careful reasoning, at its best with 12 GB or more.",
        "moe": "Mixture of experts: big-model knowledge at small-model speed. Needs plenty of RAM.",
        "best": "The largest: top quality, for 24 GB cards."}
HINT = {"gpu": "Runs entirely on your GPU: the fastest setup.",
        "partial": "Part of it runs on the CPU: it works, but slower, and less reasoning fits before Enter.",
        "cpu": "Runs on the CPU.", "too_big": "Needs more memory than this machine has."}
ERRORS = {"llama_missing": "llama.cpp is not installed.", "port_busy": "The port is taken by another program.",
          "server_exited": "llama.cpp stopped while loading the model.",
          "server_timeout": "llama.cpp did not answer for 10 minutes.",
          "no_gpu": "No NVIDIA card found: set Run on to CPU in Settings.",
          "server_stopped": "llama.cpp stopped answering.", "load_failed": "The model could not be loaded."}
SPIN = "◆◇◆◇◆◇◆◈"


def gb(mb: float) -> str:
    return f"{mb / 1024:.1f} GB" if mb >= 1024 else f"{mb:.0f} MB"


def dur(s: float | None) -> str:
    if s is None:
        return "–"
    return f"{s:.1f} s" if s < 10 else f"{s:.0f} s" if s < 60 else f"{s / 60:.1f} min"


def bar(fraction: float, width: int = 20, colour: str = AMBER) -> Text:
    full = round(max(0.0, min(1.0, fraction)) * width)
    return Text("▰" * full, style=colour) + Text("▱" * (width - full), style=LINE)


def model_name(service: ChatService, variant_id: str) -> str:
    for m in service.catalog()["models"]:
        if m["id"] == variant_id:
            return f"{m['name']} · {m['quant']}"
    return variant_id


# =====================================================================================
# Splash
# =====================================================================================

class Splash(Screen):
    """The mask, first without eyes, then with them; the wordmark; a scanning line."""

    DEFAULT_CSS = """
    Splash { align: center middle; background: $background; }
    #splash-box { width: auto; height: auto; align: center middle; }
    #splash-mark, #splash-word, #splash-tag, #splash-scan { width: auto; content-align: center middle; }
    #splash-word { margin-top: 1; }
    #splash-tag { color: #7D776D; margin-top: 1; }
    #splash-scan { margin-top: 1; }
    """

    def compose(self) -> ComposeResult:
        with Center(), Middle(), Vertical(id="splash-box"):
            yield Center(Static(art.mark("mask", 32, eyes=False), id="splash-mark"))
            yield Center(Static(art.wordmark(), id="splash-word"))
            yield Center(Static(Text("YOU TYPE. IT ALREADY KNOWS.", style=Style(color=FAINT)), id="splash-tag"))
            yield Center(Static("", id="splash-scan"))

    def on_mount(self):
        self.frame = 0
        for w in ("#splash-word", "#splash-tag"):
            self.query_one(w).styles.opacity = 0
        self.set_timer(0.35, lambda: self.query_one("#splash-word").styles.animate("opacity", 1.0, duration=0.5))
        self.set_timer(0.6, lambda: self.query_one("#splash-mark", Static).update(art.mark("mask", 32)))
        self.set_timer(0.8, lambda: self.query_one("#splash-tag").styles.animate("opacity", 1.0, duration=0.5))
        self.set_interval(0.06, self.scan)
        self.set_timer(1.9, self.leave)

    def scan(self):
        self.frame += 1
        width, pos = 30, self.frame % 40
        line = Text()
        for i in range(width):
            line.append("━", style=AMBER if pos - 8 <= i <= pos else LINE)
        self.query_one("#splash-scan", Static).update(line)

    def leave(self):
        self.styles.animate("opacity", 0.0, duration=0.35, on_complete=self.dismiss)


# =====================================================================================
# Chat
# =====================================================================================

class UserMessage(Vertical):
    DEFAULT_CSS = """
    UserMessage { height: auto; margin: 1 0 0 0; }
    UserMessage .who { color: #A39D92; }
    UserMessage .body { background: $surface; border: round #3A3733; padding: 0 1; color: #EEEAE2; }
    """

    def __init__(self, text: str):
        super().__init__()
        self.text = text

    def compose(self) -> ComposeResult:
        yield Label("YOU", classes="who")
        yield Static(self.text, classes="body", markup=False)


class BotMessage(Vertical):
    DEFAULT_CSS = """
    BotMessage { height: auto; margin: 1 0 0 0; }
    BotMessage .who { color: #D4892A; }
    BotMessage Collapsible { border: none; padding: 0; margin: 0; background: transparent; }
    BotMessage Collapsible > Contents { padding: 0 0 0 2; }
    BotMessage .reasoning { color: #A39D92; }
    BotMessage Markdown { margin: 0; padding: 0; background: transparent; }
    BotMessage .meta { color: #7D776D; }
    """

    def compose(self) -> ComposeResult:
        yield Label("◆ MAFDET", classes="who")
        with Collapsible(title="Thinking", collapsed=True, id="think"):
            yield Static("", classes="reasoning", markup=False)
        yield Markdown("▍")
        yield Static("", classes="meta")

    def on_mount(self):
        self.query_one("#think").display = False


class ChatView(Vertical):
    DEFAULT_CSS = """
    ChatView { height: 1fr; }
    #thread { height: 1fr; padding: 0 2; scrollbar-size-vertical: 1; }
    #empty { height: auto; align: center middle; padding: 2 0; }
    #empty Static { width: auto; text-align: center; }
    #empty .title { color: #EEEAE2; text-style: bold; margin-top: 1; }
    #empty .body { color: #C9C4BA; max-width: 70; }
    #empty Button { margin-top: 1; }
    #empty LoadingIndicator { height: 1; color: #D4892A; margin-top: 1; }
    #status { height: 1; padding: 0 2; }
    #input { margin: 0 1 0 1; border: tall #3A3733; background: $surface; }
    #input:focus { border: tall #A39D92; }
    #input.ahead { border: tall #D4892A; }
    """

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="thread"):
            yield Vertical(id="empty")
        yield Input(placeholder="Message MAFDET   ·   Enter send · Esc stop · Ctrl+N new chat", id="input")
        yield Static("", id="status")

    def on_mount(self):
        self.shown = ""
        self.generating = False
        self.spin = 0
        self.model_seen = None

    @property
    def service(self) -> ChatService:
        return self.app.service

    # ---- empty states ---------------------------------------------------------------

    def show_empty(self, kind: str, *widgets):
        if self.shown == kind:
            return
        self.shown = kind
        box = self.query_one("#empty", Vertical)
        box.remove_children()
        box.mount_all([Center(w) for w in widgets])
        box.styles.opacity = 0
        box.styles.animate("opacity", 1.0, duration=0.4)

    def render_empty(self, s: dict):
        has_messages = bool(self.query("UserMessage"))
        self.query_one("#empty").display = not has_messages
        if has_messages:
            return
        state = s["state"]
        if state == "ready":
            self.show_empty("hero", Static(art.mark("mask", 24)), Static(art.wordmark()),
                            Static("You type. It already knows.", classes="title"),
                            Static("MAFDET reads your message while you write it, and starts thinking about it "
                                   "before you press Enter.", classes="body"))
        elif state == "no-runtime":
            self.show_empty("no-runtime", Static(art.mark("awaken", 16)), Static("One step left: the runtime", classes="title"),
                            Static("MAFDET runs models with llama.cpp. Install it once: it takes about a minute.",
                                   classes="body"), Button("Install llama.cpp", variant="primary", id="install-runtime"))
        elif state == "no-model":
            self.show_empty(f"no-model-{s.get('model_id')}", Static(art.mark("race", 16)),
                            Static("Choose a model to start", classes="title"),
                            Static("Open Models (F2): every model says how fast it runs on this machine.", classes="body"),
                            Button("Open the catalog", variant="primary", id="open-models"))
        elif state == "loading":
            self.show_empty(f"loading-{s.get('model_id')}", Static(art.mark("mask", 20)), Static("Waking up", classes="title"),
                            Static(f"Starting llama.cpp and loading {model_name(self.service, s.get('model_id', ''))}.",
                                   classes="body"), LoadingIndicator())
        elif state == "error":
            self.show_empty(f"error-{s.get('error')}", Static(art.mark("judge", 16)),
                            Static("The model did not start", classes="title"),
                            Static(ERRORS.get(s.get("error_code", ""), "The model could not be loaded."), classes="body"),
                            Static(Text(s.get("error", ""), style=RUST), classes="body"),
                            Button("Try again", variant="primary", id="retry"))

    @on(Button.Pressed, "#install-runtime")
    def install_runtime(self):
        self.service.install_runtime()
        self.app.show("hardware")

    @on(Button.Pressed, "#open-models")
    def open_models(self):
        self.app.show("models")

    @on(Button.Pressed, "#retry")
    def retry(self):
        self.app.load_model(self.service.status().get("model_id", ""))

    # ---- status line ------------------------------------------------------------------

    def render_status(self, s: dict):
        self.spin = (self.spin + 1) % len(SPIN)
        line = Text()
        field = self.query_one("#input", Input)
        field.remove_class("ahead")
        state = s["state"]
        if state == "loading":
            line.append(f"{SPIN[self.spin]} WAKING UP · starting llama.cpp and loading the model", style=AMBER)
        elif state == "error":
            line.append(ERRORS.get(s.get("error_code", ""), "The model could not be loaded."), style=RUST)
        elif state == "no-runtime":
            line.append("llama.cpp is not installed · F3 Hardware", style=MUTED)
        elif state != "ready":
            line.append("No model loaded · F2 Models", style=MUTED)
        elif self.generating or s.get("generating"):
            line.append(f"{SPIN[self.spin]} ANSWERING · Esc to stop", style=AMBER)
        elif not s.get("has_text"):
            line.append("◇ WATCHING · start typing, MAFDET reads as you write", style=MUTED)
        else:
            read, total = s.get("read", 0), s.get("message_tokens", 0)
            thought = f" · {s['thinking_ahead']:,} reasoning tokens ahead" if s.get("thinking_ahead") else ""
            if s.get("ahead_writing"):
                field.add_class("ahead")
                what = (f"GATHERING AHEAD · {s['thinking_ahead']:,} tokens of material" if s.get("gatherer")
                        else f"THINKING AHEAD · {s['thinking_ahead']:,} reasoning tokens" if s.get("think") and not s.get("ahead")
                        else f"ANSWERING AHEAD · {s.get('ahead', 0):,} answer tokens")
                line.append(f"{SPIN[self.spin]} {what}", style=AMBER)
            elif s.get("busy"):
                line.append(f"◇ READING · {read:,} of {total:,} tokens{thought}", style=MUTED)
            elif s.get("answer_ready"):
                line.append("◆ ANSWER READY ON ENTER · written ahead in full", style=AMBER)
            elif s.get("think") and s.get("thinking_ahead"):
                line.append(f"◆ PAUSED · {s['thinking_ahead']:,} reasoning tokens ahead, more after Enter", style=PAPER)
            else:
                line.append(f"◆ READY ON ENTER · {read:,} of {total:,} tokens read{thought}", style=PAPER)
            line.append("  ")
            line.append_text(bar(read / total if total else 0, 12, AMBER if not s.get("pending") else MUTED))
        if state == "ready":
            pad = max(2, self.size.width - len(line) - 24)
            line.append(" " * pad)
            line.append(f"context {s.get('context', 0):,} / {s.get('ctx', 0):,}", style=FAINT)
        self.query_one("#status", Static).update(line)

    def on_status(self, s: dict):
        if s["state"] == "ready" and self.model_seen is not None and s.get("model_id") != self.model_seen:
            self.clear_thread()                 # a different model is a different session
        if s["state"] == "ready":
            self.model_seen = s.get("model_id")
        self.render_empty(s)
        self.render_status(s)

    # ---- conversation -------------------------------------------------------------------

    def clear_thread(self):
        for w in self.query("UserMessage, BotMessage"):
            w.remove()
        self.shown = ""

    @on(Input.Changed, "#input")
    def draft(self, event: Input.Changed):
        self.service.draft(event.value)

    @on(Input.Submitted, "#input")
    def submit(self, event: Input.Submitted):
        text = event.value
        if not text.strip() or self.generating or self.service.status()["state"] != "ready":
            return
        event.input.value = ""
        self.service.draft("")
        thread = self.query_one("#thread", VerticalScroll)
        self.query_one("#empty").display = False
        thread.mount(UserMessage(text))
        bot = BotMessage()
        thread.mount(bot)
        thread.scroll_end(animate=False)
        self.generating = True
        self.stream(text, bot)

    @work(thread=True, exclusive=True)
    def stream(self, text: str, bot: BotMessage):
        answer, reasoning, last, done = "", "", 0.0, None
        try:
            for ev in self.service.send(text):
                if ev["type"] == "token":
                    answer += ev["text"]
                elif ev["type"] == "think":
                    reasoning += ev["text"]
                elif ev["type"] == "done":
                    done = ev
                elif ev["type"] == "error":
                    answer += f"\n\n*{ev.get('message', 'error')}*"
                now = time.perf_counter()
                if now - last > 0.08:          # about 12 redraws a second: smooth, and cheap for a terminal
                    last = now
                    self.app.call_from_thread(self.paint, bot, answer, reasoning, None)
        finally:
            self.app.call_from_thread(self.paint, bot, answer, reasoning, done)

    async def paint(self, bot: BotMessage, answer: str, reasoning: str, done: dict | None):
        think = bot.query_one("#think", Collapsible)
        if reasoning:
            think.display = True
            bot.query_one(".reasoning", Static).update(reasoning)
            think.title = ("Reasoning · {:,} tokens, {:,} before Enter".format(done["think_tokens"], done["think_ahead"])
                           if done and done.get("think_tokens") else "Reasoning" if done or answer else "Thinking …")
        await bot.query_one(Markdown).update(mathtext.render(answer) + ("" if done is not None else " ▍"))
        if done is not None:
            self.generating = False
            meta = [f"◆ first word {done['answer_ms']:,.0f} ms" if done.get("think_tokens") else
                    f"◆ first token {done.get('ttft_ms', 0):,.0f} ms"]
            if done.get("message_tokens"):
                meta.append(f"{done['read']:,} of {done['message_tokens']:,} tokens read while you typed")
            if done.get("think_tokens"):
                meta.append(f"{done['think_ahead']:,} of {done['think_tokens']:,} reasoning tokens before Enter")
            meta.append(f"{done['tokens']:,} tokens · {done['tps']} tok/s")
            bot.query_one(".meta", Static).update(Text(" / ".join(meta), style=FAINT))
        elif not answer and not reasoning:
            pass
        thread = self.query_one("#thread", VerticalScroll)
        if thread.max_scroll_y - thread.scroll_y < 4:
            thread.scroll_end(animate=False)

    def stop(self):
        self.service.stop()

    def new_chat(self):
        self.service.stop()
        self.service.new_chat()
        self.clear_thread()
        self.query_one("#input", Input).focus()


# =====================================================================================
# Models
# =====================================================================================

class ModelsView(Vertical):
    """The catalog as one table: a row per model, its quantization switched in place (Q).

    Everything is said once: the table has the numbers, the line under it what the numbers do
    not say (the family, what the model is good at, where it runs), the footer the keys.
    """

    DEFAULT_CSS = """
    ModelsView { height: 1fr; padding: 0 2; }
    #models-head { height: 2; margin: 1 0 1 0; }
    #models-table { height: 1fr; }
    #models-line { height: 2; margin-top: 1; }
    """
    BINDINGS = [
        Binding("enter", "use", "Use"),
        Binding("d", "download", "Download"),
        Binding("m", "measure", "Measure"),
        Binding("x", "cancel", "Cancel"),
        Binding("q", "quant(1)", "Quant"),
        Binding("tab", "family(1)", "Family"),
        Binding("shift+tab", "family(-1)", show=False),
        Binding("f", "only('fits')", "Fits GPU"),
        Binding("i", "only('installed')", "Downloaded"),
    ]
    COLUMNS = [("", "rec"), ("Model", "name"), ("Quant", "quant"), ("Runs", "where"), ("Writes", "tps"),
               ("Thinks after Enter", "think"), ("300-token answer", "answer"), ("Size", "size"), ("", "state")]

    def compose(self) -> ComposeResult:
        yield Static("", id="models-head")
        yield DataTable(cursor_type="row", zebra_stripes=True, id="models-table")
        yield Static("", id="models-line")

    def on_mount(self):
        self.family = "all"
        self.only = ""                        # "", "fits" or "installed"
        self.picked: dict[str, str] = {}      # model id → quantization
        self.selected = ""                    # model id under the cursor
        table = self.query_one(DataTable)
        for label, key in self.COLUMNS:
            table.add_column(label, key=key)
        self.refresh_table()

    @property
    def service(self) -> ChatService:
        return self.app.service

    # ---- data ----------------------------------------------------------------------

    def models(self, cat: dict) -> list[list[dict]]:
        """The variants of each model shown, in catalog order, the recommended one first."""
        groups: dict[str, list[dict]] = {}
        for e in cat["models"]:
            groups.setdefault(e["model_id"], []).append(e)
        out = [vs for vs in groups.values() if self.family in ("all", vs[0]["family"])]
        if self.only == "fits":
            out = [vs for vs in out if any(v["placement"] == "gpu" for v in vs)]
        elif self.only == "installed":
            out = [vs for vs in out if any(v["installed"] for v in vs)]
        best = self.best(cat).split(":")[0]
        return sorted(out, key=lambda vs: vs[0]["model_id"] != best)

    def best(self, cat: dict) -> str:
        return (cat["recommended"] if self.family == "all" else cat["family_best"].get(self.family)) or ""

    def shown(self, vs: list[dict]) -> dict:
        """The quantization picked, else the one in use, else a downloaded one, else the default."""
        loaded = self.service.status().get("model_id")
        mid = vs[0]["model_id"]
        return (next((v for v in vs if v["quant"] == self.picked.get(mid)), None)
                or next((v for v in vs if v["id"] == loaded), None)
                or next((v for v in vs if v["installed"] and v["default"]), None)
                or next((v for v in vs if v["default"]), vs[0]))

    def task(self, kind: str, target: str) -> dict | None:
        return next((t for t in self.service.jobs()
                     if t["kind"] == kind and t["target"] == target and t["state"] == "running"), None)

    # ---- cells ---------------------------------------------------------------------

    def state_cell(self, e: dict) -> Text:
        s = self.service.status()
        dl, bench = self.task("download", e["id"]), self.task("bench", e["id"])
        if dl:
            speed = f" {dl['speed_bps'] / 2**20:.0f} MB/s" if dl["speed_bps"] else ""
            return Text(f"↓ {dl['progress']:.0%}{speed}", style=AMBER)
        if bench:
            return Text(f"{SPIN[int(time.time() * 4) % len(SPIN)]} measuring", style=AMBER)
        if s.get("model_id") == e["id"] and s["state"] in ("ready", "loading"):
            return Text("● in use" if s["state"] == "ready" else "◌ loading", style=GREEN)
        return Text("✓ on disk", style=FAINT) if e["installed"] else Text("")

    def cells(self, vs: list[dict], best: str) -> list:
        e = self.shown(vs)
        label, colour = PLACEMENT[e["placement"]]
        tps = Text(f"{e['decode_tps']:.0f}" if e["decode_tps"] >= 20 else f"{e['decode_tps']:.1f}")
        tps.append(" tok/s", style=FAINT)
        tps.append(" ✓" if e["measured"] else "", style=GREEN)
        quant = Text(e["quant"])
        if len(vs) > 1:
            quant.append(f" +{len(vs) - 1}", style=FAINT)
        think = (Text(f"≈ {dur(e['with_mafdet_s'])}", style=AMBER) + Text(f" ({dur(e['without_s'])})", style=FAINT)
                 if e["reason_ahead"] else Text("–", style=FAINT))
        return [Text("★", style=AMBER) if e["model_id"] == best.split(":")[0] else Text(""),
                Text(e["name"], style=Style(bold=True)), quant,
                Text(label.format(p=round(e["gpu_share"] * 100)), style=colour), tps, think,
                Text(f"≈ {dur(e['answer_s'])}"), Text(gb(e["bytes"] / 2**20), style=MUTED), self.state_cell(e)]

    # ---- drawing ---------------------------------------------------------------------

    def refresh_table(self):
        cat = self.service.catalog()
        groups = self.models(cat)
        best = self.best(cat)

        hw = self.service.hardware()
        g = hw.get("gpu")
        head = Text()
        head.append("MODELS  ", style=Style(color=PAPER, bold=True))
        machine = f"{g['name'].replace('NVIDIA GeForce ', '')} {g['vram_mb'] / 1024:.0f} GB" if g else "CPU only"
        head.append(f"{machine} · {hw['ram_mb'] / 1024:.0f} GB RAM", style=MUTED)
        if cat.get("vram_others_mb", 0) > 1024:
            head.append(f" · {gb(cat['vram_others_mb'])} of VRAM in use elsewhere", style=AMBER)
        head.append("     ★ best for you  ✓ measured  (…) without MAFDET  ⏎ use\n", style=FAINT)
        fams = [("all", "All")] + [(f["id"], f["name"]) for f in cat["families"]]
        for fid, name in fams:
            on = fid == self.family
            head.append(f" {name} ", style=Style(color=INK, bgcolor=PAPER, bold=True) if on else Style(color=MUTED))
            head.append(" ")
        if self.only:
            head.append(f"  ▣ {'fits my GPU' if self.only == 'fits' else 'downloaded'} only", style=AMBER)
        self.query_one("#models-head", Static).update(head)

        table = self.query_one(DataTable)
        keep = self.selected
        table.clear()
        for vs in groups:
            table.add_row(*self.cells(vs, best), key=vs[0]["model_id"])
        ids = [vs[0]["model_id"] for vs in groups]
        if ids:
            row = ids.index(keep) if keep in ids else 0
            table.move_cursor(row=row)
            self.selected = ids[row]
        else:
            self.selected = ""
        self.describe()

    def update_states(self):
        """Progress only: the state column and the line under the table."""
        table = self.query_one(DataTable)
        cat = self.service.catalog()
        groups = {vs[0]["model_id"]: vs for vs in self.models(cat)}
        for mid, vs in groups.items():
            try:
                table.update_cell(mid, "state", self.state_cell(self.shown(vs)))
            except Exception:
                pass
        self.describe()

    def current(self) -> tuple[list[dict], dict] | None:
        cat = self.service.catalog()
        vs = [e for e in cat["models"] if e["model_id"] == self.selected]
        return (vs, self.shown(vs)) if vs else None

    def describe(self):
        cur = self.current()
        line = Text()
        if cur is None:
            line.append("No model matches: Tab for another family, F or I to show everything.", style=MUTED)
        else:
            _, e = cur
            line.append(f"{e['name']} · {e['vendor']} · {e['license']}", style=Style(color=PAPER, bold=True))
            line.append(f"   {TIER[e['tier']]}\n", style=PAPER)
            line.append(HINT[e["placement"]], style=MUTED)
            line.append("   Reasons while you type." if e["reason_ahead"] else
                        "   Reads while you type; writes the answer ahead at your pauses.", style=FAINT)
        self.query_one("#models-line", Static).update(line)

    # ---- events and keys ---------------------------------------------------------------

    @on(DataTable.RowHighlighted)
    def highlight(self, event: DataTable.RowHighlighted):
        if event.row_key is not None and event.row_key.value:
            self.selected = event.row_key.value
            self.describe()

    @on(DataTable.RowSelected)
    def chosen(self, event: DataTable.RowSelected):
        self.action_use()

    def action_quant(self, step: int):
        cur = self.current()
        if cur is None or len(cur[0]) < 2:
            return
        vs, e = cur
        quants = [v["quant"] for v in vs]
        self.picked[self.selected] = quants[(quants.index(e["quant"]) + step) % len(quants)]
        best = self.best(self.service.catalog())
        for (_, key), value in zip(self.COLUMNS, self.cells(vs, best)):
            self.query_one(DataTable).update_cell(self.selected, key, value)
        self.describe()

    def action_family(self, step: int):
        ids = ["all"] + [f["id"] for f in self.service.catalog()["families"]]
        self.family = ids[(ids.index(self.family) + step) % len(ids)]
        self.refresh_table()

    def action_only(self, what: str):
        self.only = "" if self.only == what else what
        self.refresh_table()

    def action_use(self):
        cur = self.current()
        if cur is None:
            return
        e = cur[1]
        if e["installed"]:
            self.app.load_model(e["id"])
            self.app.show("chat")
        else:
            self.app.pending_use.add(e["id"])          # download, then load
            self.service.download(e["id"])
            self.update_states()

    def action_download(self):
        cur = self.current()
        if cur and not cur[1]["installed"]:
            self.service.download(cur[1]["id"])
            self.update_states()

    def action_measure(self):
        cur = self.current()
        if cur and cur[1]["installed"]:
            self.service.benchmark(cur[1]["id"])
            self.update_states()

    def action_cancel(self):
        cur = self.current()
        task = cur and self.task("download", cur[1]["id"])
        if task:
            self.service.cancel(task["id"])


# =====================================================================================
# Hardware
# =====================================================================================

class HardwareView(VerticalScroll):
    DEFAULT_CSS = """
    HardwareView { height: 1fr; padding: 1 2; }
    #hw-text { height: auto; }
    #rt-progress { margin: 1 0; }
    #rt-install { margin-top: 1; }
    """

    def compose(self) -> ComposeResult:
        yield Static("", id="hw-text")
        yield ProgressBar(total=100, show_eta=False, id="rt-progress")
        yield Button("Install llama.cpp", variant="primary", id="rt-install")

    def on_mount(self):
        self.refresh_view()

    def refresh_view(self, fresh: bool = False):
        service: ChatService = self.app.service
        hw, rt = service.hardware(fresh), service.runtime()
        t = Table.grid(padding=(0, 3))
        t.add_column(style=MUTED, no_wrap=True)
        t.add_column(style=PAPER)
        g = hw.get("gpu")
        if g:
            used = g["vram_used_mb"] / g["vram_mb"] if g["vram_mb"] else 0
            t.add_row("GRAPHICS CARD", Text(g["name"], style=Style(bold=True)))
            t.add_row("", Text(f"Driver {g['driver']}", style=MUTED))
            t.add_row("VRAM", bar(used, 24, GREEN) + Text(f"  {gb(g['vram_mb'])} · {gb(g['vram_used_mb'])} in use"))
            t.add_row("BANDWIDTH", f"{g['bandwidth_gbs']:.0f} GB/s" + ("" if g["bandwidth_known"] else " (estimated)"))
        else:
            t.add_row("GRAPHICS CARD", Text("No NVIDIA card found: models run on the CPU.", style=MUTED))
        t.add_row("", "")
        t.add_row("PROCESSOR", Text(hw["cpu"], style=Style(bold=True)))
        t.add_row("", f"{hw['cores']} cores · {hw['threads']} threads")
        t.add_row("MEMORY", bar(1 - hw["ram_free_mb"] / hw["ram_mb"], 24, GREEN) +
                  Text(f"  {gb(hw['ram_mb'])} · {gb(hw['ram_free_mb'])} free"))
        t.add_row("BANDWIDTH", f"{hw['ram_bandwidth_gbs']:.0f} GB/s (estimated)")
        t.add_row("SYSTEM", f"{hw['os']} · {hw['arch']}")
        t.add_row("", "")
        t.add_row("RUNTIME", Text(f"llama.cpp {rt['build'] or '?'}" if rt["installed"] else "Not installed",
                                  style=Style(bold=True, color=PAPER if rt["installed"] else RUST)))
        t.add_row("", f"MAFDET is tested with {rt['tested_build']}. Build for this machine: {rt['recommended']}")
        if rt["installed"]:
            t.add_row("", Text(("Installed by MAFDET in " if rt["managed"] else "Your own llama.cpp, found at ") + rt["path"],
                               style=FAINT))
        title = Text("HARDWARE\n", style=Style(color=PAPER, bold=True))
        title.append("What MAFDET found on this machine, and the runtime that runs the models.  R refresh\n", style=MUTED)
        self.query_one("#hw-text", Static).update(_stack(title, t))
        self.query_one("#rt-install", Button).label = "Reinstall" if rt["installed"] else "Install llama.cpp"
        self.update_progress()

    def update_progress(self):
        task = next((t for t in self.app.service.jobs() if t["kind"] == "llama" and t["state"] == "running"), None)
        bar_ = self.query_one("#rt-progress", ProgressBar)
        bar_.display = task is not None
        self.query_one("#rt-install").display = task is None
        if task:
            bar_.update(total=100, progress=task["progress"] * 100)

    @on(Button.Pressed, "#rt-install")
    def install(self):
        self.app.service.install_runtime()
        self.update_progress()


def _stack(*renderables):
    grid = Table.grid()
    grid.add_column()
    for r in renderables:
        grid.add_row(r)
    return grid


# =====================================================================================
# Settings
# =====================================================================================

class SettingsView(VerticalScroll):
    DEFAULT_CSS = """
    SettingsView { height: 1fr; padding: 1 2; }
    SettingsView .section { color: #EEEAE2; text-style: bold; margin-top: 1; border-bottom: heavy #EEEAE2; }
    SettingsView .row { height: auto; padding: 1 0 0 0; border-bottom: solid #2A2825; }
    SettingsView .text { width: 1fr; height: auto; }
    SettingsView Select { width: 24; }
    SettingsView Switch { width: auto; }
    """

    def compose(self) -> ComposeResult:
        st = self.app.service.settings
        yield Static(Text.assemble(("SETTINGS\n", Style(color=PAPER, bold=True)),
                                   ("Saved on this computer, kept from one start to the next. The window (mafdet gui) "
                                    "also has a language switch: this terminal app is in English.", MUTED)))
        yield Static("REASONING", classes="section")
        yield self.row("Reason while I type", "The model starts thinking about your message before you send it. "
                       "Off: no reasoning, it answers right away.", Switch(value=st["think"], id="think"))
        yield self.row("Reasoning kept across edits", "The longest reasoning carried over while you keep typing.",
                       self.select("think_ahead", [512, 1024, 2048, 4096], st["think_ahead"], "{:,} tokens"))
        yield self.row("Answer written ahead", "Answer tokens written during a pause, while the message does not change.",
                       self.select("ahead", [0, 128, 256, 512, 1024], st["ahead"], "{:,} tokens"))
        yield self.row("Longest answer", "As long as the context allows, or a limit.",
                       self.select("max_tokens", [0, 1024, 2048, 4096, 8192], st["max_tokens"], "{:,} tokens", "No limit"))
        yield Static("MODEL AND MEMORY", classes="section")
        yield self.row("Context", "How much of the conversation the model keeps in memory. More takes more VRAM.",
                       self.select("ctx", [4096, 8192, 16384, 32768], st["ctx"], "{:,}"))
        yield self.row("Run on", "Auto uses the NVIDIA card when there is one.",
                       Select([("Auto", "auto"), ("GPU", "cuda"), ("CPU", "cpu")], value=st["device"], allow_blank=False,
                              id="device"))
        yield Static("EXPERIMENTAL", classes="section")
        yield self.row("Recycle drafts β", "Reuses the answers written for earlier versions of the message to write "
                       "faster (+4% to +20% measured).", Switch(value=st["recycle"], id="recycle"))
        small = [m for m in self.app.service.catalog()["models"] if m["installed"] and m["params_b"] <= 1.7]
        yield self.row("Gatherer β", "A small model gathers material while you type and the main model reads it. "
                       "Measured on the RTX 2060: no faster than reasoning ahead. For research.",
                       Select([("Off", "")] + [(f"{m['name']} · {m['quant']}", m["id"]) for m in small],
                              value=st["gatherer"] if st["gatherer"] in [m["id"] for m in small] else "",
                              allow_blank=False, id="gatherer"))
        yield self.row("Guess the ending β", "The model also guesses how you will finish the message and reasons on "
                       "the guess. Measured: the exact message guessed 1 time in 3, wait 16% shorter on average. Only with "
                       "models that fit in VRAM.",
                       Switch(value=st["branches"], id="branches"))
        yield Static(Text.assemble(("\nMAFDET ", Style(color=PAPER, bold=True)), (f"v{__version__} · Apache-2.0 · ", MUTED),
                                   ("github.com/TheLastOxygen/mafdet", AMBER)))

    @staticmethod
    def select(key, values, current, fmt, zero=None):
        options = [(zero if zero and v == 0 else fmt.format(v), v) for v in values]
        if current not in values:
            options.append((fmt.format(current), current))
        return Select(options, value=current, allow_blank=False, id=key)

    @staticmethod
    def row(name: str, hint: str, control) -> Horizontal:
        return Horizontal(Static(Text.assemble((name + "\n", Style(color=PAPER, bold=True)), (hint, MUTED)), classes="text"),
                          control, classes="row")

    def save(self, key: str, value):
        service: ChatService = self.app.service
        reload = service.session is not None and service.settings.get(key) != value
        service.update_settings({key: value})
        self.app.notify("Settings saved" + (" · the model is reloading" if reload else ""), timeout=3)

    @on(Switch.Changed)
    def switched(self, event: Switch.Changed):
        self.save(event.switch.id, event.value)

    @on(Select.Changed)
    def selected(self, event: Select.Changed):
        if event.value is not Select.BLANK and self.app.service.settings.get(event.select.id) != event.value:
            self.save(event.select.id, event.value)


# =====================================================================================
# First start
# =====================================================================================

class Setup(ModalScreen):
    """Welcome, the machine, llama.cpp, a first model, ready."""

    DEFAULT_CSS = """
    Setup { align: center middle; background: #0A0909 80%; }
    #setup { width: 84; height: 30; border: round #3A3733; background: $background; padding: 1 3; }
    #setup-steps { height: 1; }
    #setup-body { height: 1fr; margin-top: 1; }
    #setup-progress { margin: 1 0; }
    #setup-actions { height: 3; align-horizontal: right; }
    #setup-actions Button { margin-left: 1; }
    #setup-skip { dock: left; }
    """
    STEPS = ["Welcome", "Your machine", "Runtime", "Model", "Ready"]

    def compose(self) -> ComposeResult:
        with Vertical(id="setup"):
            yield Static("", id="setup-steps")
            yield Static("", id="setup-body")
            yield ProgressBar(total=100, show_eta=False, id="setup-progress")
            with Horizontal(id="setup-actions"):
                yield Button("Skip setup", id="setup-skip")
                yield Button("Back", id="setup-back")
                yield Button("Install llama.cpp", variant="warning", id="setup-install")
                yield Button("Download and use", variant="warning", id="setup-get")
                yield Button("Continue", variant="primary", id="setup-next")

    def on_mount(self):
        self.step = 0
        self.revealed = 0
        self.set_interval(0.3, self.tick)
        self.show()

    @property
    def service(self) -> ChatService:
        return self.app.service

    def show(self):
        self.revealed = 0
        steps = Text()
        for i, name in enumerate(self.STEPS):
            steps.append(f" {name.upper()} ", style=Style(color=INK, bgcolor=AMBER, bold=True) if i == self.step
                         else Style(color=MUTED if i < self.step else FAINT))
            steps.append("─" if i < len(self.STEPS) - 1 else "", style=LINE)
        self.query_one("#setup-steps", Static).update(steps)
        body = self.query_one("#setup-body", Static)
        body.styles.opacity = 0
        body.styles.animate("opacity", 1.0, duration=0.35)
        self.render_step()

    def render_step(self):
        name = self.STEPS[self.step]
        body = self.query_one("#setup-body", Static)
        rt = self.service.runtime() if name == "Runtime" else None
        cat = self.service.catalog() if name == "Model" else None
        installing = any(t["kind"] == "llama" and t["state"] == "running" for t in self.service.jobs())
        best = cat["recommended"] if cat else None
        downloading = best and any(t["kind"] == "download" and t["target"] == best and t["state"] == "running"
                                   for t in self.service.jobs())
        if name == "Welcome":
            body.update(_stack(art.mark("mask", 16), Text("\nWelcome to MAFDET\n", style=Style(color=PAPER, bold=True)),
                               Text("Local AI that reads your message while you type it, so the answer is ready the "
                                    "moment you press Enter. Setting it up takes about two minutes.", style=PAPER)))
        elif name == "Your machine":
            hw = self.service.hardware()
            g = hw.get("gpu")
            rows = [("GRAPHICS CARD", g["name"] if g else "–"), ("VRAM", gb(g["vram_mb"]) if g else "–"),
                    ("PROCESSOR", hw["cpu"]), ("MEMORY", gb(hw["ram_mb"]))]
            t = Table.grid(padding=(0, 3))
            t.add_column(style=MUTED)
            t.add_column(style=Style(color=PAPER, bold=True))
            for i, (k, v) in enumerate(rows):
                t.add_row(k, v if i < self.revealed else Text(SPIN[(self.revealed + i) % len(SPIN)] + " scanning", style=AMBER))
            found = ""
            if self.revealed >= len(rows):
                found = (f"{g['name']} with {gb(g['vram_mb'])} of VRAM: models will run on the GPU." if g
                         else "No NVIDIA card: models will run on the CPU. Smaller models answer fastest.")
            body.update(_stack(Text("Looking at your machine\n", style=Style(color=PAPER, bold=True)), t,
                               Text("\n" + found, style=PAPER)))
        elif name == "Runtime":
            text = (f"llama.cpp {rt['build']} is ready." if rt["installed"] else
                    f"llama.cpp runs the models. MAFDET picks the build for this machine: {rt['recommended']}.")
            body.update(_stack(art.mark("awaken", 12), Text("\nInstalling the runtime\n", style=Style(color=PAPER, bold=True)),
                               Text(text, style=PAPER)))
        elif name == "Model":
            e = next((x for x in cat["models"] if x["id"] == best), None)
            t = Text("Your first model\n", style=Style(color=PAPER, bold=True))
            t.append("Chosen for your hardware. You can change it at any time in Models (F2).\n\n", style=MUTED)
            if e:
                t.append(f"★ {e['name']} {e['quant']}\n", style=Style(color=AMBER, bold=True))
                t.append(TIER[e["tier"]] + "\n", style=PAPER)
                t.append(f"{PLACEMENT[e['placement']][0].format(p=round(e['gpu_share'] * 100))} · "
                         f"{e['decode_tps']:.0f} tok/s · answer after Enter ≈ {dur(e['with_mafdet_s'])} "
                         f"({dur(e['without_s'])} without MAFDET) · {gb(e['bytes'] / 2**20)}\n", style=MUTED)
                t.append("Downloaded." if e["installed"] else "", style=GREEN)
            body.update(t)
        else:
            body.update(_stack(art.mark("mask", 16), Text("\nAll set\n", style=Style(color=PAPER, bold=True)),
                               Text("Start typing and watch the line under the message: MAFDET reads and thinks while "
                                    "you write.", style=PAPER)))
        model_e = next((x for x in cat["models"] if x["id"] == best), None) if cat else None
        self.query_one("#setup-back").display = 0 < self.step < 4
        self.query_one("#setup-skip").display = self.step < 4
        self.query_one("#setup-install").display = name == "Runtime" and not rt["installed"] and not installing
        self.query_one("#setup-get").display = bool(name == "Model" and model_e and not model_e["installed"] and not downloading)
        self.query_one("#setup-next", Button).label = "Start chatting" if self.step == 4 else "Continue"
        self.query_one("#setup-next").disabled = (name == "Runtime" and not rt["installed"]) or \
            (name == "Your machine" and self.revealed < 4)
        progress = self.query_one("#setup-progress", ProgressBar)
        task = next((t for t in self.service.jobs() if t["state"] == "running" and
                     ((name == "Runtime" and t["kind"] == "llama") or (name == "Model" and t["target"] == best))), None)
        progress.display = task is not None
        if task:
            progress.update(total=100, progress=task["progress"] * 100)

    def tick(self):
        if self.STEPS[self.step] == "Your machine" and self.revealed < 4:
            self.revealed += 1
        if self.STEPS[self.step] in ("Your machine", "Runtime", "Model"):
            self.render_step()

    @on(Button.Pressed, "#setup-next")
    def next_step(self):
        if self.step == 4:
            return self.finish()
        self.step += 1
        self.show()

    @on(Button.Pressed, "#setup-back")
    def back(self):
        self.step -= 1
        self.show()

    @on(Button.Pressed, "#setup-skip")
    def skip(self):
        self.finish()

    @on(Button.Pressed, "#setup-install")
    def install(self):
        self.service.install_runtime()
        self.render_step()

    @on(Button.Pressed, "#setup-get")
    def get(self):
        best = self.service.catalog()["recommended"]
        self.app.pending_use.add(best)
        self.service.download(best)
        self.render_step()

    def finish(self):
        self.service.update_settings({"first_run_done": True})
        st = self.service.settings
        installed = next((m for m in self.service.catalog()["models"] if m["id"] == st["model"] and m["installed"]), None)
        if installed and self.service.status()["state"] not in ("ready", "loading"):
            self.app.load_model(st["model"])
        self.dismiss()


# =====================================================================================
# The app
# =====================================================================================

class Rail(Vertical):
    DEFAULT_CSS = """
    Rail { width: 26; background: #0A0909; border-right: solid #2A2825; padding: 1 1; }
    #rail-logo { height: auto; margin-bottom: 1; }
    Rail Button { width: 100%; min-width: 0; border: none; height: 3; background: transparent; color: #A39D92;
                  content-align: left middle; text-style: bold; }
    Rail Button:hover { background: #191816; color: #EEEAE2; }
    Rail Button.on { background: #191816; color: #EEEAE2; border-left: outer #D4892A; }
    #rail-model { dock: bottom; height: auto; border: round #2A2825; padding: 0 1; color: #C9C4BA; }
    """

    def compose(self) -> ComposeResult:
        logo = art.mark("mask", 10)
        logo.append("\n")
        logo.append_text(Text("MAFDET", style=Style(color=PAPER, bold=True)))
        yield Static(logo, id="rail-logo")
        for key, label, view in (("F1", "CHAT", "chat"), ("F2", "MODELS", "models"), ("F3", "HARDWARE", "hardware"),
                                 ("F4", "SETTINGS", "settings")):
            yield Button(f"{key}  {label}", id=f"nav-{view}")
        yield Static("", id="rail-model")

    @on(Button.Pressed)
    def go(self, event: Button.Pressed):
        if event.button.id and event.button.id.startswith("nav-"):
            self.app.show(event.button.id[4:])


class MafdetTUI(App):
    TITLE = "MAFDET"
    CSS = """
    Screen { background: $background; }
    #main { height: 1fr; }
    #views { width: 1fr; }
    Button { text-style: bold; }
    Button.-primary, Button.-warning { color: #0E0D0C; }
    Button.-primary:hover, Button.-warning:hover { color: #0E0D0C; }
    DataTable > .datatable--cursor { background: #2A2825; color: #EEEAE2; text-style: bold; }
    DataTable:focus > .datatable--cursor { background: #3A3733; }
    DataTable > .datatable--header { background: #191816; color: #A39D92; }
    DataTable > .datatable--hover { background: #191816; }
    """
    BINDINGS = [
        Binding("f1", "show('chat')", "Chat"),
        Binding("f2", "show('models')", "Models"),
        Binding("f3", "show('hardware')", "Hardware"),
        Binding("f4", "show('settings')", "Settings"),
        Binding("ctrl+n", "new_chat", "New chat"),
        Binding("escape", "stop", "Stop", show=False),
        Binding("r", "refresh_hardware", show=False),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, service: ChatService | None = None):
        super().__init__()
        self.service = service or ChatService()
        self.pending_use: set[str] = set()
        self.seen_tasks: dict[str, str] = {}
        self.last_state = ""

    def compose(self) -> ComposeResult:
        with Horizontal(id="main"):
            yield Rail()
            with ContentSwitcher(initial="chat", id="views"):
                yield ChatView(id="chat")
                yield ModelsView(id="models")
                yield HardwareView(id="hardware")
                yield SettingsView(id="settings")
        yield Footer()

    def on_mount(self):
        self.register_theme(THEME)
        self.theme = "mafdet"
        self.service.autoload()
        self.show("chat")
        self.set_interval(0.2, self.poll)
        self.set_interval(0.5, self.poll_tasks)
        self.push_screen(Splash(), callback=self.after_splash)

    def after_splash(self, _=None):
        if not self.service.settings["first_run_done"]:
            self.push_screen(Setup())
        else:
            self.query_one("#input").focus()

    # ---- navigation ------------------------------------------------------------------

    def show(self, view: str):
        switcher = self.query_one("#views", ContentSwitcher)
        switcher.current = view
        for b in self.query("Rail Button"):
            b.set_class(b.id == f"nav-{view}", "on")
        widget = self.query_one(f"#{view}")
        widget.styles.opacity = 0
        widget.styles.animate("opacity", 1.0, duration=0.25)
        if view == "models":
            self.query_one(ModelsView).refresh_table()
            self.query_one(DataTable).focus()
        elif view == "hardware":
            self.query_one(HardwareView).refresh_view()
        elif view == "chat":
            self.query_one("#input").focus()

    def action_show(self, view: str):
        self.show(view)

    def action_new_chat(self):
        self.show("chat")
        self.query_one(ChatView).new_chat()

    def action_stop(self):
        self.query_one(ChatView).stop()

    def action_refresh_hardware(self):
        if self.query_one("#views", ContentSwitcher).current == "hardware":
            self.query_one(HardwareView).refresh_view(fresh=True)

    def load_model(self, variant_id: str):
        self.run_worker(lambda: self.service.load(variant_id), thread=True, exclusive=True, group="load")

    # ---- polling ---------------------------------------------------------------------

    def poll(self):
        # The timers keep firing while the app shuts down, after the views are gone: nothing to update then.
        try:
            self._poll()
        except NoMatches:
            pass

    def poll_tasks(self):
        try:
            self._poll_tasks()
        except NoMatches:
            pass

    def _poll(self):
        s = self.service.status()
        if s["state"] != self.last_state:
            if s["state"] == "ready" and self.last_state == "loading":
                self.notify(f"{s.get('model') or model_name(self.service, s.get('model_id', ''))} is ready", timeout=3)
            self.last_state = s["state"]
        self.query_one(ChatView).on_status(s)
        pill = Text()
        colour = {"ready": GREEN, "loading": AMBER, "error": RUST}.get(s["state"], FAINT)
        pill.append("◆ ", style=colour)
        pill.append(s.get("model") if s["state"] == "ready" else
                    f"Loading {model_name(self.service, s.get('model_id', ''))}" if s["state"] == "loading" else
                    "Model not loaded" if s["state"] == "error" else "No model loaded")
        self.query_one("#rail-model", Static).update(pill)

    def _poll_tasks(self):
        tasks = self.service.jobs()
        changed = False
        for t in tasks:
            before = self.seen_tasks.get(t["id"])
            self.seen_tasks[t["id"]] = t["state"]
            if t["state"] == "running" or before == "running":
                changed = True
            if before in (None, "running") and t["state"] != "running" and t.get("ended"):
                if before == "running" or time.time() - t["ended"] < 1:
                    self.task_ended(t)
        if changed:
            current = self.query_one("#views", ContentSwitcher).current
            if current == "models":
                self.query_one(ModelsView).update_states()
            if current == "hardware":
                self.query_one(HardwareView).update_progress()

    def task_ended(self, t: dict):
        name = model_name(self.service, t["target"])
        if t["state"] == "error":
            self.notify(f"{name}: {t['error']}", severity="error", timeout=6)
        elif t["state"] == "done":
            if t["kind"] == "download":
                self.notify(f"{name} downloaded", timeout=3)
                if t["target"] in self.pending_use:
                    self.pending_use.discard(t["target"])
                    self.load_model(t["target"])
            elif t["kind"] == "bench":
                self.notify(f"{name}: {t['result']['decode_tps']} tok/s measured", timeout=4)
            elif t["kind"] == "llama":
                self.notify(f"llama.cpp {t['result'].get('build', '')} installed", timeout=4)
                self.query_one(HardwareView).refresh_view()
        if self.query_one("#views", ContentSwitcher).current == "models":
            self.query_one(ModelsView).refresh_table()

    def on_unmount(self):
        self.service.close()


def run(args=None) -> int:
    MafdetTUI().run()
    return 0
