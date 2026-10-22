"""
Live demo: incremental prefill, seen step by step (llama.cpp backend).

Type a message. At every pause the model reads the new tokens into the KV
cache: the panels show in real time what it has read, what is waiting and
why. On Enter you see the real wait next to the one a computation from
scratch would have taken, then the answer streams in.

    python demo_live.py                  # GPU
    python demo_live.py --device cpu     # CPU: the differences become seconds
    python demo_live.py --auto           # the demo types an example by itself

Keys: Enter send · ←/→ Home End move the cursor (to edit in the middle)
      F2 paste the sample document · Ctrl+L clear · Esc quit
"""
from __future__ import annotations

import argparse
import collections
import http.client
import json
import os
import ctypes
import random
import sys
import threading
import time
from dataclasses import dataclass

# Intel's Fortran runtime (loaded by some numeric libraries) prints "forrtl: error (200)
# ... window-CLOSE event" when the window closes: harmless, switched off.
os.environ.setdefault("FOR_DISABLE_CONSOLE_CTRL_HANDLER", "1")

from rich.console import Console, Group  # noqa: E402
from rich.layout import Layout
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.segment import Segment, Segments
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from mafdet.cli import add_common_args  # noqa: E402
from mafdet.llama_backend import LlamaPrefill, setup_llama  # noqa: E402
from mafdet.text import Template, common_prefix_len, encode, read_sample, stable_ids  # noqa: E402

# The palette of the visual identity v2 (assets/build_assets.py): black, paper and a
# single amber accent, used for what is happening right now.
INK, PAPER, AMBER = "#0E0D0C", "#EEEAE2", "#D4892A"
RUST = "#C46A3A"                   # cache no longer valid, crop
SOFT, MUTED, LINE = "#C9C4BA", "#7D776D", "#3E3A34"
BASE = f"{PAPER} on {INK}"
TEXT = {"": SOFT, "bold": f"bold {PAPER}", "ok": PAPER, "warn": AMBER, "crop": RUST, "dim": MUTED}

# The answer in Markdown with the same colours, instead of rich's defaults.
MARKDOWN_THEME = Theme({
    "markdown.h1": f"bold {AMBER}", "markdown.h1.border": LINE,
    **{f"markdown.h{i}": f"bold {PAPER}" for i in range(2, 7)},
    "markdown.item.bullet": AMBER, "markdown.item.number": AMBER,
    "markdown.code": f"{AMBER} on #1A1917", "markdown.block_quote": MUTED, "markdown.hr": LINE,
    "markdown.link": f"underline {AMBER}", "markdown.link_url": MUTED,
})

# The wordmark in block characters, as in the identity guide's terminal.
WORDMARK = ["█▀▄▀█ ▄▀█ █▀▀ █▀▄ █▀▀ ▀█▀",
            "█ ▀ █ █▀█ █▀  █▄▀ ██▄  █ "]

# Token states and colours (two alternating shades, to see where a token ends)
STYLES = {
    "read": (f"{INK} on {PAPER}", f"{INK} on #C9C4BA"),
    "reading": (f"{INK} on {AMBER}", f"{INK} on #E8B06A"),
    "ready": (f"{PAPER} on #5A554D", f"{PAPER} on #6A655D"),
    "held": (f"{AMBER} on #2A2825", f"{AMBER} on #3A3733"),
}
LEGEND = [("read", "already read (in cache)"), ("reading", "being read now"),
          ("ready", "ready, waiting for a pause"), ("held", "held back")]


def plural(n: int, one: str, many: str) -> str:
    return one if n == 1 else many


# --------------------------------------------------------------------------
# Keyboard
# --------------------------------------------------------------------------

SPECIAL = {"K": "LEFT", "M": "RIGHT", "G": "HOME", "O": "END", "S": "DELETE", "<": "F2"}


class Keyboard:
    """Reads keys without blocking the program (Windows)."""

    def __init__(self):
        if os.name != "nt":
            raise SystemExit("Key reading is written for Windows: elsewhere use --auto.")
        import msvcrt
        self.m = msvcrt

    def read(self) -> list[str]:
        keys = []
        while self.m.kbhit():
            c = self.m.getwch()
            if c in ("\x00", "\xe0"):
                keys.append(SPECIAL.get(self.m.getwch(), ""))
                continue
            if c == "\r":
                # An Enter followed right away by more characters is a line break inside
                # pasted text, not sending the message.
                time.sleep(0.003)
                c = "\n" if self.m.kbhit() else "ENTER"
            keys.append(c)
        return keys

    def flush(self):
        while self.m.kbhit():
            self.m.getwch()

    def restart(self):
        pass


class AutoTyper:
    """Types an example by itself: pastes a document, types a question with a typo it
    corrects and an insertion shortly before the end, then sends. The text is in Italian,
    like the bundled sample document it asks about."""

    def __init__(self, seed: int = 1, cps: float = 7.0):
        rng = random.Random(seed)
        plan: list[tuple[float, str]] = []

        def type_(s):
            for ch in s:
                pause = 1.0 / cps * rng.lognormvariate(0, 0.35)
                if ch in ".?!,":
                    pause += 0.5
                plan.append((pause, ch))

        plan.append((0.0, "F2"))                       # paste the sample document
        type_("Leggi la specifica qui sopra e dimmi quali scenari di guasto non sono copreti")
        plan[1] = (2.5, plan[1][1])                    # time to skim the document
        plan += [(0.6, "\x08")] + [(0.08, "\x08")] * 3  # notices the typo
        type_("erti, in particolare cosa succede se il normalizzatore si blocca dopo aver salvato "
              "la chiave di idempotenza.")
        plan.append((1.2, "LEFT"))                     # goes back before the final full stop...
        type_(" ma prima di inoltrare l'ordine")       # ...and adds a piece
        plan.append((0.8, "END"))
        plan.append((1.5, "ENTER"))
        self.plan = collections.deque(plan)
        self.due = time.perf_counter() + 1.5

    def read(self) -> list[str]:
        keys = []
        while self.plan and time.perf_counter() >= self.due:
            keys.append(self.plan.popleft()[1])
            self.due = time.perf_counter() + (self.plan[0][0] if self.plan else 0)
        if not self.plan and not keys and time.perf_counter() >= self.due:
            keys.append("ESC")                         # the example is over: quit
            self.due = float("inf")
        return keys

    def flush(self):
        pass

    def restart(self):
        self.due = time.perf_counter() + 8             # time to read the answer before quitting


# --------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------

@dataclass
class Job:
    target: list[int]
    reason: str
    new: int                    # tokens to read (estimated when submitted)


class Demo:
    def __init__(self, args, tok, server, eng, ref, device: str, console: Console):
        self.args, self.tok, self.server, self.eng, self.ref = args, tok, server, eng, ref
        self.console = console
        self.tpl = Template.from_tokenizer(tok, thinking=False)
        self.tpl_first_token = encode(tok, self.tpl.prefix)[0]
        self.holdback, self.min_new = args.holdback, args.min_new
        self.debounce = args.debounce / 1000
        self.label = f"{os.path.basename(server.model_path)} · llama.cpp {'GPU' if device.endswith('cuda') else 'CPU'}"
        self.pieces: dict[int, str] = {}
        self.events: collections.deque = collections.deque(maxlen=40)
        self.cv = threading.Condition()
        self.job: Job | None = None
        self.pending: tuple[list[int], str] | None = None
        self.t_start = time.perf_counter()
        self.gpu: tuple[str, int, int] | None = None   # (P-state, core MHz, memory MHz)
        self.on_gpu = device.endswith("cuda")
        self.keep_warm = self.on_gpu and not args.let_gpu_sleep
        self.last_gpu_work = 0.0
        self.new_message()
        threading.Thread(target=self._worker, daemon=True).start()
        if self.on_gpu:
            threading.Thread(target=self._gpu_monitor, daemon=True).start()
        if self.keep_warm:
            threading.Thread(target=self._keep_warm, daemon=True).start()
        self.log("Ready. Type a message: the model will read it at every pause.", "bold")

    # ---- the message ---------------------------------------------------------
    def new_message(self):
        self.chars: list[str] = []
        self.cursor = 0
        self.last_key = 0.0
        self.mid_edit_logged = False
        self.phase = "typing"
        self.stage = ""
        self.result: dict = {}
        self.answer = ""
        self.totals = {"reads": 0, "computed": 0, "discarded": 0, "ms": 0.0}
        self.warm_pings = 0
        self.eng.reset()                  # the first read of a new message starts from scratch
        self._retokenize()

    def _retokenize(self):
        text = "".join(self.chars)
        self.full_ids = encode(self.tok, self.tpl.prefix + text)
        self.final_ids = encode(self.tok, self.tpl.prefix + text + self.tpl.suffix)
        self.target = stable_ids(self.full_ids, self.holdback)

    def log(self, msg: str, style: str = ""):
        self.events.append((time.perf_counter() - self.t_start, msg, style))

    # ---- keys ----------------------------------------------------------------
    def handle(self, keys: list[str]) -> str | None:
        changed = False
        for k in keys:
            if k in ("ESC", "\x1b", "\x03"):
                return "quit"
            if k == "ENTER":
                return "send" if "".join(self.chars).strip() else None
            at_end = self.cursor == len(self.chars)
            edited = True
            if k == "\x08":
                edited = self.cursor > 0
                if edited:
                    del self.chars[self.cursor - 1]
                    self.cursor -= 1
                    changed = True
            elif k == "DELETE":
                edited = self.cursor < len(self.chars)
                if edited:
                    del self.chars[self.cursor]
                    changed = True
            elif k in ("LEFT", "RIGHT", "HOME", "END"):
                edited = False
                self.cursor = {"LEFT": max(0, self.cursor - 1), "RIGHT": min(len(self.chars), self.cursor + 1),
                               "HOME": 0, "END": len(self.chars)}[k]
            elif k == "\x0c":                          # Ctrl+L
                self.chars, self.cursor, changed = [], 0, True
                self.log("Text cleared.")
            elif k == "F2":
                doc = read_sample("documento.txt").strip() + "\n\n"
                self.chars[self.cursor:self.cursor] = list(doc)
                self.cursor += len(doc)
                changed = True
                self.log(f"Pasted the sample document ({len(doc)} characters).", "bold")
            elif len(k) == 1 and (k == "\n" or k >= " "):
                self.chars.insert(self.cursor, k)
                self.cursor += 1
                changed = True
            else:
                continue
            if edited and not at_end and not self.mid_edit_logged:
                self.mid_edit_logged = True
                self.log(f"Edit in the middle of the text (character {self.cursor}): the cached tokens after this "
                         "point no longer hold, because every token depends on all the ones before it.", "crop")
            if at_end:
                self.mid_edit_logged = False
        if changed:
            self.last_key = time.perf_counter()
            self._retokenize()
        return None

    # ---- when to read: threshold and debounce --------------------------------
    def schedule(self):
        with self.cv:
            if self.job or self.pending:
                return
        target, done = self.target, self.eng.ids
        if not target or target == done:
            return
        n = common_prefix_len(done, target)
        new = len(target) - n
        if n == len(done) and new >= self.min_new:
            self.submit(target, f"{new} new tokens, threshold {self.min_new}")
        elif time.perf_counter() - self.last_key >= self.debounce:
            self.submit(target, f"pause of {self.args.debounce:.0f} ms")

    def submit(self, target: list[int], reason: str):
        new = len(target) - common_prefix_len(self.eng.cached_tokens, target)
        with self.cv:
            self.pending = (list(target), reason)
            self.job = Job(list(target), reason, new)
            self.cv.notify_all()

    def wait_idle(self):
        with self.cv:
            while self.job or self.pending:
                self.cv.wait()

    def _worker(self):
        while True:
            with self.cv:
                while self.pending is None:
                    self.cv.wait()
                target, reason = self.pending
                self.pending = None
            st = self.eng.stats
            before = (st.tokens_computed, st.tokens_discarded, st.forward_passes)
            old_len = len(self.eng.cached_tokens)
            t0 = time.perf_counter()
            self.eng.sync(target)
            ms = (time.perf_counter() - t0) * 1000
            st = self.eng.stats
            computed, discarded = st.tokens_computed - before[0], st.tokens_discarded - before[1]
            if st.forward_passes == before[2]:
                extra = old_len - len(target)
                what = ("the extra token stays in cache and is cut" if extra == 1 else
                        f"the {extra} extra tokens stay in cache and are cut")
                self.log(f"Text shortened: nothing to compute, {what} at the next read.", "dim")
            else:
                t = self.eng.last_timings
                msg = (f"Read ({reason}): {t['cache_n']} tokens reused from the cache, "
                       f"{t['prompt_n']} read in {t['prompt_ms']:.0f} ms")
                if discarded:
                    msg += f"; {discarded} {plural(discarded, 'token', 'tokens')} cut away (crop) because the text changed"
                self.log(msg, "crop" if discarded else "ok")
                self.totals["reads"] += 1
                self.totals["computed"] += computed
                self.totals["discarded"] += discarded
                self.totals["ms"] += ms
            self.last_gpu_work = time.perf_counter()
            with self.cv:
                self.job = None
                self.cv.notify_all()

    # ---- GPU: state and keeping it awake -----------------------------------------
    def _gpu_monitor(self):
        """Reads the GPU's P-state and clocks every half second, with NVML (the library behind nvidia-smi)."""
        try:
            nvml = ctypes.WinDLL("nvml.dll") if os.name == "nt" else ctypes.CDLL("libnvidia-ml.so.1")
            handle = ctypes.c_void_p()
            if nvml.nvmlInit_v2() or nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)):
                return
        except OSError:
            return                                      # no NVML: the panel shows "—"
        pstate, core, mem = ctypes.c_int(), ctypes.c_uint(), ctypes.c_uint()
        while True:
            nvml.nvmlDeviceGetPerformanceState(handle, ctypes.byref(pstate))
            nvml.nvmlDeviceGetClockInfo(handle, 0, ctypes.byref(core))      # 0 = graphics core
            nvml.nvmlDeviceGetClockInfo(handle, 2, ctypes.byref(mem))       # 2 = memory
            self.gpu = (f"P{pstate.value}", core.value, mem.value)
            time.sleep(0.5)

    def _keep_warm(self):
        """Keeps the GPU awake while the user types.

        After 1-2 seconds without work the driver lowers the clocks (down to
        300 MHz for the core and 405 MHz for the memory) and the next request
        runs up to 4 times slower. While the user types, the GPU will be needed
        soon: a 1-token request every ~0.3 s, on the comparison slot, keeps it
        ready without touching the message's cache. If other applications
        (browser, video) already keep the GPU awake, nothing changes.
        """
        conn = http.client.HTTPConnection("127.0.0.1", self.server.port, timeout=60)
        body = json.dumps({"prompt": [self.tpl_first_token], "n_predict": 0, "cache_prompt": True,
                           "id_slot": self.ref.slot})
        while True:
            time.sleep(0.05)
            now = time.perf_counter()
            typing = self.phase == "typing" and self.chars and now - self.last_key < 20
            if not typing or self.job or self.pending or now - self.last_gpu_work < 0.3:
                continue
            try:
                conn.request("POST", "/completion", body, {"Content-Type": "application/json"})
                conn.getresponse().read()
            except (http.client.HTTPException, OSError):
                conn = http.client.HTTPConnection("127.0.0.1", self.server.port, timeout=60)
                continue
            if self.warm_pings == 0:
                self.log("Keeping the GPU awake while you type: a 1-token request every ~0.3 s on another slot, "
                         "so the reads and Enter do not find the GPU at low clocks.", "dim")
            self.warm_pings += 1
            self.last_gpu_work = time.perf_counter()

    # ---- Enter ---------------------------------------------------------------
    def send(self):
        self.phase = "sending"
        t_enter = time.perf_counter()
        gpu = f" GPU {self.gpu[0]}, memory at {self.gpu[2]} MHz." if self.gpu else ""
        self.log("Enter pressed." + gpu, "bold")
        if self.job or self.pending:
            self.stage = "The model was still reading: waiting for it to finish..."
            self.log("The model was still reading: the wait starts here.", "warn")
        self.wait_idle()
        final = list(self.final_ids)
        reused = min(common_prefix_len(self.eng.cached_tokens, final), len(final) - 1)
        tail = len(final) - reused
        self.stage = (f"Computing the tail: {tail} tokens (end of the text + start of the answer). "
                      f"The other {reused} were read while you typed.")
        self.eng.sync(final, need_logits=True)
        ttft = (time.perf_counter() - t_enter) * 1000
        self.result = {"ttft": ttft, "tail": tail, "reused": reused, "total": len(final)}
        self.log(f"First token ready after {ttft:.0f} ms: only {tail} of {len(final)} tokens computed.", "ok")
        if not self.args.no_compare:
            self.stage = (f"For comparison, computing all {len(final)} tokens from scratch on a second slot "
                          "with an empty cache. Only the demo does this, to measure the difference...")
            self.ref.reset()
            t0 = time.perf_counter()
            self.ref.sync(final, need_logits=True)
            self.result["base"] = (time.perf_counter() - t0) * 1000
            self.log(f"From scratch, without incremental prefill, it would have taken {self.result['base']:.0f} ms.", "warn")
        self.phase = "answer"
        self.stage = "The model writes the answer (decode: one token at a time, using the KV cache)."
        self.stream(final)
        self.phase = "done"
        self.stage = (("Answer cut at the token limit. " if self.result.get("cut") else "Done. ")
                      + "Press a key for a new message, Esc to quit.")

    def stream(self, prompt: list[int]):
        # Without --max-tokens the answer can take all the context left in the slot.
        limit = self.args.max_tokens or max(16, self.args.ctx - len(prompt) - 8)
        conn = http.client.HTTPConnection("127.0.0.1", self.server.port, timeout=600)
        body = {"prompt": prompt, "id_slot": self.eng.slot, "cache_prompt": True, "stream": True,
                "n_predict": limit, "temperature": 0.7, "top_p": 0.8, "top_k": 20}
        conn.request("POST", "/completion", json.dumps(body), {"Content-Type": "application/json"})
        resp = conn.getresponse()
        t0, n = time.perf_counter(), 0
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            d = json.loads(line[6:])
            if d.get("content"):
                self.answer += d["content"]
                n += 1
                self.result["gen"] = (n, n / max(time.perf_counter() - t0, 1e-6))
            if d.get("stop"):
                t = d.get("timings") or {}
                if t.get("predicted_n"):
                    self.result["gen"] = (t["predicted_n"], t["predicted_per_second"])
                # "eos": the model finished by itself; "limit": cut at the token limit
                self.result["cut"] = d.get("stop_type") == "limit"
                break
        conn.close()
        n, speed = self.result.get("gen", (0, 0))
        if self.result.get("cut"):
            self.log(f"Answer cut after {n} tokens: the limit of {limit} tokens was reached "
                     f"({'--max-tokens' if self.args.max_tokens else 'no room left in the slot context'}).",
                     "warn")
        else:
            self.log(f"Answer complete: {n} tokens at {speed:.0f} tokens/s.", "ok")
        self.eng.reset()      # the slot's cache now holds the answer too

    # ---- main loop -------------------------------------------------------------
    def run(self, source):
        while True:
            keys = source.read()
            if self.phase == "typing":
                action = self.handle(keys) if keys else None
                if action == "quit":
                    return
                if action == "send":
                    self.send()
                    source.flush()        # keys pressed while the answer streamed do not count
                    source.restart()
                    continue
                self.schedule()
            elif self.phase == "done" and keys:
                if any(k in ("ESC", "\x1b", "\x03") for k in keys):
                    return
                self.new_message()
                self.log("New message.", "bold")
            time.sleep(0.01)

    # ---- drawing ---------------------------------------------------------------
    def piece(self, tid: int) -> str:
        if tid not in self.pieces:
            self.pieces[tid] = self.tok.decode([tid]).replace("\n", "↵")
        return self.pieces[tid]

    def counts(self) -> dict:
        full, cache, target = self.full_ids, self.eng.cached_tokens, self.target
        valid = common_prefix_len(cache, full)
        read_now = 0
        job = self.job
        if job:
            read_now = max(0, len(job.target) - common_prefix_len(cache, job.target))
        final = self.final_ids
        remaining = len(final) - min(common_prefix_len(cache, final), len(final) - 1)
        return {"full": len(full), "valid": valid, "invalid": max(0, len(cache) - valid),
                "pending": max(0, len(target) - common_prefix_len(cache, target)) - read_now,
                "held": len(full) - len(target), "reading": read_now, "remaining": remaining,
                "final": len(final)}

    def render(self) -> Layout:
        width = self.console.width
        layout = Layout()
        key = f"bold {PAPER}"
        head = Text.assemble((WORDMARK[0], key), "   ", ("Incremental prefill, live", AMBER),
                             f"  ·  {self.label}  ·  holdback {self.holdback}"
                             f" · debounce {self.args.debounce:.0f} ms · threshold {self.min_new} tokens\n",
                             (WORDMARK[1], key), "   ",
                             ("Enter", key), " send · ", ("← → Home End", key), " move the cursor · ",
                             ("F2", key), " paste a document · ", ("Ctrl+L", key), " clear · ",
                             ("Esc", key), " quit", style=MUTED, no_wrap=True, overflow="ellipsis")
        head = Padding(head, (0, 1), style=BASE, expand=True)
        if self.phase == "typing":
            layout.split_column(Layout(head, size=2), Layout(self._message(width), size=6),
                                Layout(self._tokens(width), size=10), Layout(self._now(), size=8),
                                Layout(name="bottom"))
            layout["bottom"].split_row(Layout(self._numbers(), ratio=2), Layout(self._events(), ratio=3))
        else:
            layout.split_column(Layout(head, size=2), Layout(self._message(width, lines=2), size=4),
                                Layout(self._result(width), size=9), Layout(name="bottom"))
            layout["bottom"].split_row(Layout(self._answer(width * 3 // 5 - 4, self.console.height - 19), ratio=3),
                                       Layout(self._events(), ratio=2))
        return layout

    @staticmethod
    def _panel(body, title: str, accent: bool = False) -> Panel:
        """A panel on the identity's black; amber only for what matters right now."""
        return Panel(body, title=Text(f" {title} ", style=f"bold {AMBER if accent else PAPER}"),
                     title_align="left", border_style=AMBER if accent else LINE, style=BASE)

    def _message(self, width: int, lines: int = 4) -> Panel:
        text = "".join(self.chars)
        title = f"Your message ({len(text)} characters)"
        if not text:
            return self._panel(Text("(type your message here)", style=MUTED), title)
        # The lines ending at the cursor are shown, counting line breaks and the
        # panel's width: the point where you type always stays visible.
        w = max(20, width - 6)
        before, after = text[:self.cursor], text[self.cursor:]
        rows = []
        for para in before[-w * lines * 2:].split("\n"):
            rows += [para[i:i + w] for i in range(0, len(para), w)] or [""]
        rows = rows[-lines:]
        rest = after.split("\n", 1)[0][:max(0, w - len(rows[-1]) - 1)]
        t = Text("\n".join(rows), style=PAPER)
        if rest:
            t.append(rest[0], style=f"{INK} on {AMBER}")
            t.append(rest[1:])
        else:
            t.append(" ", style=f"on {AMBER}")         # cursor at the end or before a line break
        return self._panel(t, title)

    def _tokens(self, width: int) -> Panel:
        full, cache, job = self.full_ids, self.eng.cached_tokens, self.job
        valid = common_prefix_len(cache, full)
        reading_end = len(job.target) if job else 0
        hold_start = len(self.target)
        budget, i, used = int((width - 6) * 7 * 0.85), len(full), 0
        while i > 0 and used + len(self.piece(full[i - 1])) + 1 < budget:
            i -= 1
            used += len(self.piece(full[i])) + 1
        t = Text("… " if i else "", style=MUTED)
        for j in range(i, len(full)):
            if j < valid:
                state = "read"
            elif j < reading_end:
                state = "reading"
            elif j >= hold_start:
                state = "held"
            else:
                state = "ready"
            t.append(self.piece(full[j]), style=STYLES[state][j % 2])
            t.append(" ")
        legend = Text()
        for state, label in LEGEND:
            legend.append(" abc ", style=STYLES[state][0])        # a sample token
            legend.append(f" {label}   ", style=MUTED)
        return self._panel(Group(t, legend), f"What the model sees: {len(full)} tokens (template + text)")

    def _now(self) -> Panel:
        c = self.counts()
        lines = Text(style=SOFT)
        if self.job:
            lines.append(f"◆ The model is reading {c['reading']} tokens ({self.job.reason}). ", style=f"bold {AMBER}")
            lines.append("This is the prefill: for every token it computes keys and values and adds them to the "
                         "KV cache.\n")
        elif not self.chars:
            lines.append("◆ Type a message. ", style=TEXT["bold"])
            lines.append("The model will read it piece by piece while you type, instead of waiting for Enter.\n")
        elif c["invalid"] and c["valid"] < len(self.target):
            lines.append(f"◆ You edited the text: {c['invalid']} cached tokens no longer hold. ",
                         style=f"bold {RUST}")
            lines.append("At the next pause the cache is cut there (crop) and read again from that point.\n")
        elif c["pending"] > 0:
            lines.append(f"◆ You are typing: {c['pending']} {plural(c['pending'], 'token', 'tokens')} ready to read. ",
                         style=TEXT["bold"])
            lines.append(f"The model waits for a pause of {self.args.debounce:.0f} ms or {self.min_new} new tokens: "
                         "every read has a fixed cost, better to group them.\n")
        else:
            lines.append(f"◆ The model has caught up: it has read {c['valid']} tokens. ", style=TEXT["bold"])
            lines.append("It waits for you to type more or press Enter.\n")
        if self.chars:
            lines.append(f"◆ Last {c['held']} tokens held back: ", style=AMBER)
            lines.append("the word you are typing can still change how it is split into tokens.\n")
            lines.append(f"◆ If you press Enter now, {c['remaining']} of {c['final']} tokens are left to compute",
                         style=TEXT["bold"])
            lines.append(" (the rest is already in cache)." if c["remaining"] < c["final"] else ".")
            if self.gpu and self.gpu[0] not in ("P0", "P1", "P2"):
                lines.append(f"\n◆ GPU idle ({self.gpu[0]}, memory at {self.gpu[2]} MHz): ", style=AMBER)
                lines.append("the next read will be slower until the driver raises the clocks again.")
        return self._panel(lines, "Now", accent=True)

    def _numbers(self) -> Panel:
        c, tot = self.counts(), self.totals
        tb = Table.grid(padding=(0, 2))
        tb.add_column(style=MUTED, no_wrap=True)
        tb.add_column(style=PAPER, justify="right", no_wrap=True)
        for label, val in [("Tokens (with template)", c["full"]),
                           ("Read and valid", c["valid"]),
                           ("In cache, no longer valid", c["invalid"]),
                           ("Ready, waiting", max(0, c["pending"])),
                           ("Held back", c["held"]),
                           ("Reads done", tot["reads"]),
                           ("Computed / discarded", f"{tot['computed']} / {tot['discarded']}"),
                           ("Reading time", f"{tot['ms']:.0f} ms"),
                           ("Left on Enter", f"{c['remaining']} tokens")]:
            tb.add_row(label, str(val))
        body = Group(tb, Text(""), self._gpu_text()) if self.on_gpu else tb
        return self._panel(body, "Numbers")

    def _gpu_text(self) -> Text:
        if self.gpu is None:
            return Text("GPU: state not available", style=MUTED)
        state, core, mem = self.gpu
        awake = state in ("P0", "P1", "P2")
        t = Text("GPU ", style=MUTED)
        t.append("awake" if awake else "idle", style=TEXT["bold"] if awake else f"bold {AMBER}")
        t.append(f": {state}, core {core} MHz, memory {mem} MHz", style=PAPER if awake else AMBER)
        if self.keep_warm:
            t.append(f"\nkept awake with {self.warm_pings} minimal requests", style=MUTED)
        return t

    def _events(self) -> Panel:
        t = Text()
        # The most recent first: if the panel is short, the oldest are the ones left out.
        for ts, msg, style in reversed(list(self.events)[-14:]):
            t.append(f"{ts:6.1f}s  ", style=MUTED)
            t.append(msg + "\n", style=TEXT[style])
        return self._panel(t, "Log, step by step (most recent first)")

    def _result(self, width: int) -> Panel:
        r = self.result
        t = Text(style=SOFT)
        t.append("◆ ", style=AMBER)
        t.append(self.stage + "\n\n", style=TEXT["bold"])
        if "ttft" in r:
            bar_w = max(10, width - 60)
            base = r.get("base", r["ttft"])
            scale = bar_w / max(base, r["ttft"])
            t.append(f"{'with incremental prefill':<27}", style=AMBER)
            t.append("█" * max(1, round(r["ttft"] * scale)), style=AMBER)
            t.append(f" {r['ttft']:.0f} ms  ({r['tail']} tokens computed on Enter)\n")
            if "base" in r:
                t.append(f"{'without (all from scratch)':<27}", style=MUTED)
                t.append("█" * max(1, round(base * scale)), style="#5A554D")
                t.append(f" {base:.0f} ms  ({r['total']} tokens)\n")
                t.append(f"\nWhile you typed the model had already read {r['reused']} of {r['total']} tokens: "
                         f"on Enter you waited {100 * r['ttft'] / base:.0f}% of the time.", style=TEXT["bold"])
        return self._panel(t, "Enter: wait before the first token", accent=True)

    def _answer(self, width: int, lines: int) -> Panel:
        gen = self.result.get("gen")
        title = "Answer" + (f" · {gen[0]} tokens, {gen[1]:.0f} tokens/s" if gen else "")
        if self.result.get("cut"):
            title += " · CUT at the token limit"
        # The answer is Markdown: it is laid out and its end is shown, because lines
        # that do not fit in the panel would be cut at the bottom.
        opts = self.console.options.update(width=max(20, width))
        # Laying out the tail is enough: it is the only visible part, and redoing the whole
        # text on every frame would get slow with answers thousands of tokens long.
        rendered = self.console.render_lines(Markdown(self.answer[-6000:]), opts, pad=False)[-max(3, lines):]
        segs = []
        for line in rendered:
            segs += line + [Segment.line()]
        return self._panel(Segments(segs), title)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--holdback", type=int, default=3)
    p.add_argument("--debounce", type=float, default=250.0, help="ms")
    p.add_argument("--min-new", type=int, default=16, help="new tokens at the end that trigger a read")
    p.add_argument("--max-tokens", type=int, default=0,
                   help="longest answer (0 = as long as there is room in the slot context)")
    p.add_argument("--no-compare", action="store_true",
                   help="do not measure the computation from scratch on Enter (on CPU with long texts it takes "
                        "tens of seconds)")
    p.add_argument("--auto", action="store_true", help="the demo types an example by itself")
    p.add_argument("--let-gpu-sleep", action="store_true",
                   help="do not keep the GPU awake while you type: to see how much slower it gets when it idles")
    # Two slots (the message and the comparison from scratch): 4096 tokens each are enough,
    # and the 4B's KV cache stays ~1.2 GB, so everything fits in the 2060's 6 GB.
    p.set_defaults(ctx=4096)
    return p


def main():
    # No crash when the output goes to a pipe or a cp1252 file.
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors="replace")
    args = build_parser().parse_args()
    args.backend = "llama"

    console = Console(theme=MARKDOWN_THEME)
    console.print("[bold]Starting llama-server...[/bold] (the first time can take a minute)")
    tok, server, device, _ = setup_llama(args, n_slots=2)  # slot 0: your message; slot 1: the comparison
    engines = [LlamaPrefill(server, slot=server.take_slot()) for _ in range(2)]
    demo = Demo(args, tok, server, engines[0], engines[1], device, console)
    source = AutoTyper() if args.auto else Keyboard()
    with Live(get_renderable=demo.render, console=console, screen=True, refresh_per_second=12):
        try:
            demo.run(source)
        except KeyboardInterrupt:
            pass
    server.close()


if __name__ == "__main__":
    main()
