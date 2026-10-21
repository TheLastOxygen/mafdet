"""
Demo interattiva: il prefill incrementale visto dal vivo (backend llama.cpp).

Scrivi un messaggio. A ogni pausa il modello legge i token nuovi e li mette
nella KV cache: i pannelli mostrano in tempo reale cosa ha già letto, cosa
aspetta e perché. All'invio vedi l'attesa vera accanto a quella che avresti
avuto calcolando tutto da zero, poi la risposta in streaming.

    python demo_live.py                  # GPU
    python demo_live.py --device cpu     # CPU: le differenze diventano di secondi
    python demo_live.py --auto           # la demo scrive da sola un esempio

Tasti: Invio invia · ←/→ Home Fine spostano il cursore (per correggere a metà)
       F2 incolla il documento di esempio · Ctrl+L cancella · Esc esce
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

# PyTorch (importato da common) carica il runtime Fortran di Intel, che alla chiusura
# della finestra stampa "forrtl: error (200) ... window-CLOSE event": è innocuo, lo si spegne.
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
from mafdet.engine import (Template, common_prefix_len, encode, make_engine, read_sample, setup,  # noqa: E402
                           stable_ids)

# Palette dell'identità visiva v2 (assets/build_assets.py): nero, carta e un solo
# accento ambra, usato per ciò che sta succedendo adesso.
INK, PAPER, AMBER = "#0E0D0C", "#EEEAE2", "#D4892A"
RUST = "#C46A3A"                   # cache non più valida, crop
SOFT, MUTED, LINE = "#C9C4BA", "#7D776D", "#3E3A34"
BASE = f"{PAPER} on {INK}"
TEXT = {"": SOFT, "bold": f"bold {PAPER}", "ok": PAPER, "warn": AMBER, "crop": RUST, "dim": MUTED}

# La risposta in Markdown con gli stessi colori, invece di quelli predefiniti di rich.
MARKDOWN_THEME = Theme({
    "markdown.h1": f"bold {AMBER}", "markdown.h1.border": LINE,
    **{f"markdown.h{i}": f"bold {PAPER}" for i in range(2, 7)},
    "markdown.item.bullet": AMBER, "markdown.item.number": AMBER,
    "markdown.code": f"{AMBER} on #1A1917", "markdown.block_quote": MUTED, "markdown.hr": LINE,
    "markdown.link": f"underline {AMBER}", "markdown.link_url": MUTED,
})

# Il wordmark a caratteri pieni, come nel terminale della guida d'identità.
WORDMARK = ["█▀▄▀█ ▄▀█ █▀▀ █▀▄ █▀▀ ▀█▀",
            "█ ▀ █ █▀█ █▀  █▄▀ ██▄  █ "]

# Stati dei token e colori (due tonalità alternate per vedere dove finisce un token)
STYLES = {
    "letto": (f"{INK} on {PAPER}", f"{INK} on #C9C4BA"),
    "lettura": (f"{INK} on {AMBER}", f"{INK} on #E8B06A"),
    "pronto": (f"{PAPER} on #5A554D", f"{PAPER} on #6A655D"),
    "trattenuto": (f"{AMBER} on #2A2825", f"{AMBER} on #3A3733"),
}
LEGEND = [("letto", "già letti (in cache)"), ("lettura", "in lettura adesso"),
          ("pronto", "pronti, aspettano una pausa"), ("trattenuto", "trattenuti (holdback)")]


def plural(n: int, one: str, many: str) -> str:
    return one if n == 1 else many


# --------------------------------------------------------------------------
# Tastiera
# --------------------------------------------------------------------------

SPECIAL = {"K": "LEFT", "M": "RIGHT", "G": "HOME", "O": "END", "S": "DELETE", "<": "F2"}


class Keyboard:
    """Legge i tasti senza bloccare il programma (Windows)."""

    def __init__(self):
        if os.name != "nt":
            raise SystemExit("La lettura dei tasti è scritta per Windows: altrove usa --auto.")
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
                # Un Invio seguito subito da altri caratteri è un a capo dentro un
                # testo incollato, non l'invio del messaggio.
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
    """Scrive da sola un esempio: incolla un documento, scrive una domanda con un refuso
    corretto e un'aggiunta poco prima della fine, poi invia."""

    def __init__(self, seed: int = 1, cps: float = 7.0):
        rng = random.Random(seed)
        plan: list[tuple[float, str]] = []

        def type_(s):
            for ch in s:
                pause = 1.0 / cps * rng.lognormvariate(0, 0.35)
                if ch in ".?!,":
                    pause += 0.5
                plan.append((pause, ch))

        plan.append((0.0, "F2"))                       # incolla il documento di esempio
        type_("Leggi la specifica qui sopra e dimmi quali scenari di guasto non sono copreti")
        plan[1] = (2.5, plan[1][1])                    # il tempo di scorrere il documento
        plan += [(0.6, "\x08")] + [(0.08, "\x08")] * 3  # si accorge del refuso
        type_("erti, in particolare cosa succede se il normalizzatore si blocca dopo aver salvato "
              "la chiave di idempotenza.")
        plan.append((1.2, "LEFT"))                     # torna prima del punto finale...
        type_(" ma prima di inoltrare l'ordine")       # ...e aggiunge un pezzo
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
            keys.append("ESC")                         # finito l'esempio, si esce
            self.due = float("inf")
        return keys

    def flush(self):
        pass

    def restart(self):
        self.due = time.perf_counter() + 8             # lascia leggere la risposta prima di uscire


# --------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------

@dataclass
class Job:
    target: list[int]
    reason: str
    new: int                    # token da leggere (stima al momento dell'invio)


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
        self.gpu: tuple[str, int, int] | None = None   # (P-state, MHz core, MHz memoria)
        self.on_gpu = device.endswith("cuda")
        self.keep_warm = self.on_gpu and not args.gpu_libera
        self.last_gpu_work = 0.0
        self.new_message()
        threading.Thread(target=self._worker, daemon=True).start()
        if self.on_gpu:
            threading.Thread(target=self._gpu_monitor, daemon=True).start()
        if self.keep_warm:
            threading.Thread(target=self._keep_warm, daemon=True).start()
        self.log("Pronto. Scrivi un messaggio: il modello lo leggerà a ogni pausa.", "bold")

    # ---- stato del messaggio ---------------------------------------------
    def new_message(self):
        self.chars: list[str] = []
        self.cursor = 0
        self.last_key = 0.0
        self.mid_edit_logged = False
        self.phase = "typing"
        self.stage = ""
        self.result: dict = {}
        self.answer = ""
        self.totals = {"letture": 0, "calcolati": 0, "scartati": 0, "ms": 0.0}
        self.warm_pings = 0
        self.eng.reset()                  # la prima lettura del nuovo messaggio parte da zero
        self._retokenize()

    def _retokenize(self):
        text = "".join(self.chars)
        self.full_ids = encode(self.tok, self.tpl.prefix + text)
        self.final_ids = encode(self.tok, self.tpl.prefix + text + self.tpl.suffix)
        self.target = stable_ids(self.full_ids, self.holdback)

    def log(self, msg: str, style: str = ""):
        self.events.append((time.perf_counter() - self.t_start, msg, style))

    # ---- tastiera ------------------------------------------------------------
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
                self.log("Testo cancellato.")
            elif k == "F2":
                doc = read_sample("documento.txt").strip() + "\n\n"
                self.chars[self.cursor:self.cursor] = list(doc)
                self.cursor += len(doc)
                changed = True
                self.log(f"Incollato il documento di esempio ({len(doc)} caratteri).", "bold")
            elif len(k) == 1 and (k == "\n" or k >= " "):
                self.chars.insert(self.cursor, k)
                self.cursor += 1
                changed = True
            else:
                continue
            if edited and not at_end and not self.mid_edit_logged:
                self.mid_edit_logged = True
                self.log(f"Modifica a metà testo (carattere {self.cursor}): i token in cache dopo questo punto "
                         "non valgono più, perché ogni token dipende da tutti quelli prima.", "crop")
            if at_end:
                self.mid_edit_logged = False
        if changed:
            self.last_key = time.perf_counter()
            self._retokenize()
        return None

    # ---- quando leggere: soglia e debounce ---------------------------------
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
            self.submit(target, f"{new} token nuovi, soglia {self.min_new}")
        elif time.perf_counter() - self.last_key >= self.debounce:
            self.submit(target, f"pausa di {self.args.debounce:.0f} ms")

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
                what = ("il token in più resta in cache e verrà tagliato" if extra == 1 else
                        f"i {extra} token in più restano in cache e verranno tagliati")
                self.log(f"Testo accorciato: niente da calcolare, {what} alla prossima lettura.", "dim")
            else:
                t = self.eng.last_timings
                msg = (f"Lettura ({reason}): {t['cache_n']} token riusati dalla cache, "
                       f"{t['prompt_n']} {plural(t['prompt_n'], 'letto', 'letti')} in {t['prompt_ms']:.0f} ms")
                if discarded:
                    msg += f"; {discarded} token tagliati via (crop) perché il testo era cambiato"
                self.log(msg, "crop" if discarded else "ok")
                self.totals["letture"] += 1
                self.totals["calcolati"] += computed
                self.totals["scartati"] += discarded
                self.totals["ms"] += ms
            self.last_gpu_work = time.perf_counter()
            with self.cv:
                self.job = None
                self.cv.notify_all()

    # ---- GPU: stato e risveglio ------------------------------------------------
    def _gpu_monitor(self):
        """Legge P-state e frequenze della GPU ogni mezzo secondo, con NVML (la libreria di nvidia-smi)."""
        try:
            nvml = ctypes.WinDLL("nvml.dll") if os.name == "nt" else ctypes.CDLL("libnvidia-ml.so.1")
            handle = ctypes.c_void_p()
            if nvml.nvmlInit_v2() or nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)):
                return
        except OSError:
            return                                      # niente NVML: il pannello mostra "—"
        pstate, core, mem = ctypes.c_int(), ctypes.c_uint(), ctypes.c_uint()
        while True:
            nvml.nvmlDeviceGetPerformanceState(handle, ctypes.byref(pstate))
            nvml.nvmlDeviceGetClockInfo(handle, 0, ctypes.byref(core))      # 0 = core grafico
            nvml.nvmlDeviceGetClockInfo(handle, 2, ctypes.byref(mem))       # 2 = memoria
            self.gpu = (f"P{pstate.value}", core.value, mem.value)
            time.sleep(0.5)

    def _keep_warm(self):
        """Tiene la GPU sveglia mentre l'utente scrive.

        Dopo 1-2 secondi senza lavoro il driver abbassa le frequenze (fino a
        300 MHz il core e 405 MHz la memoria) e la richiesta successiva va
        fino a 4 volte più lenta. Mentre si scrive il sistema sa che la GPU
        servirà a breve: una richiesta di 1 token ogni ~0,3 s, sullo slot del
        confronto, la tiene pronta senza toccare la cache del messaggio.
        Se altre applicazioni (browser, video) tengono già sveglia la GPU,
        non cambia nulla.
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
                self.log("Tengo sveglia la GPU mentre scrivi: una richiesta di 1 token ogni ~0,3 s su un altro "
                         "slot, così le letture e l'invio non trovano la GPU a basse frequenze.", "dim")
            self.warm_pings += 1
            self.last_gpu_work = time.perf_counter()

    # ---- invio ---------------------------------------------------------------
    def send(self):
        self.phase = "sending"
        t_enter = time.perf_counter()
        gpu = f" GPU {self.gpu[0]}, memoria a {self.gpu[2]} MHz." if self.gpu else ""
        self.log("Invio premuto." + gpu, "bold")
        if self.job or self.pending:
            self.stage = "Il modello stava ancora leggendo: aspetto che finisca..."
            self.log("Il modello stava ancora leggendo: l'attesa parte da qui.", "warn")
        self.wait_idle()
        final = list(self.final_ids)
        reused = min(common_prefix_len(self.eng.cached_tokens, final), len(final) - 1)
        tail = len(final) - reused
        self.stage = (f"Calcolo la coda: {tail} token (fine del testo + inizio della risposta). "
                      f"Gli altri {reused} li ha già letti mentre scrivevi.")
        self.eng.sync(final, need_logits=True)
        ttft = (time.perf_counter() - t_enter) * 1000
        self.result = {"ttft": ttft, "tail": tail, "reused": reused, "total": len(final)}
        self.log(f"Primo token pronto dopo {ttft:.0f} ms: calcolati solo {tail} token su {len(final)}.", "ok")
        if not self.args.no_confronto:
            self.stage = (f"Per confronto calcolo da zero tutti i {len(final)} token, su un secondo slot "
                          "con cache vuota. Succede solo nella demo, per misurare la differenza...")
            self.ref.reset()
            t0 = time.perf_counter()
            self.ref.sync(final, need_logits=True)
            self.result["base"] = (time.perf_counter() - t0) * 1000
            self.log(f"Da zero, senza prefill incrementale, sarebbero serviti {self.result['base']:.0f} ms.", "warn")
        self.phase = "answer"
        self.stage = "Il modello scrive la risposta (decode: un token alla volta, usando la KV cache)."
        self.stream(final)
        self.phase = "done"
        self.stage = (("Risposta interrotta al limite di token. " if self.result.get("cut") else "Fatto. ")
                      + "Premi un tasto per un nuovo messaggio, Esc per uscire.")

    def stream(self, prompt: list[int]):
        # Senza --max-tokens la risposta può occupare tutto il contesto rimasto nello slot.
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
                # "eos": il modello ha finito da solo; "limit": troncata al limite di token
                self.result["cut"] = d.get("stop_type") == "limit"
                break
        conn.close()
        n, speed = self.result.get("gen", (0, 0))
        if self.result.get("cut"):
            self.log(f"Risposta interrotta dopo {n} token: raggiunto il limite di {limit} token "
                     f"({'--max-tokens' if self.args.max_tokens else 'spazio finito nel contesto dello slot'}).",
                     "warn")
        else:
            self.log(f"Risposta completa: {n} token a {speed:.0f} token/s.", "ok")
        self.eng.reset()      # la cache dello slot ora contiene anche la risposta

    # ---- ciclo principale ------------------------------------------------------
    def run(self, source):
        while True:
            keys = source.read()
            if self.phase == "typing":
                action = self.handle(keys) if keys else None
                if action == "quit":
                    return
                if action == "send":
                    self.send()
                    source.flush()        # i tasti premuti mentre scorreva la risposta non contano
                    source.restart()
                    continue
                self.schedule()
            elif self.phase == "done" and keys:
                if any(k in ("ESC", "\x1b", "\x03") for k in keys):
                    return
                self.new_message()
                self.log("Nuovo messaggio.", "bold")
            time.sleep(0.01)

    # ---- disegno -------------------------------------------------------------
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
        head = Text.assemble((WORDMARK[0], key), "   ", ("Prefill incrementale dal vivo", AMBER),
                             f"  ·  {self.label}  ·  holdback {self.holdback}"
                             f" · debounce {self.args.debounce:.0f} ms · soglia {self.min_new} token\n",
                             (WORDMARK[1], key), "   ",
                             ("Invio", key), " invia · ", ("← → Home Fine", key), " sposta il cursore · ",
                             ("F2", key), " incolla un documento · ", ("Ctrl+L", key), " cancella · ",
                             ("Esc", key), " esci", style=MUTED, no_wrap=True, overflow="ellipsis")
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
        """Pannello sul fondo nero dell'identità; l'ambra solo per quello che conta adesso."""
        return Panel(body, title=Text(f" {title} ", style=f"bold {AMBER if accent else PAPER}"),
                     title_align="left", border_style=AMBER if accent else LINE, style=BASE)

    def _message(self, width: int, lines: int = 4) -> Panel:
        text = "".join(self.chars)
        title = f"Il tuo messaggio ({len(text)} caratteri)"
        if not text:
            return self._panel(Text("(scrivi qui il tuo messaggio)", style=MUTED), title)
        # Si mostrano le righe che finiscono al cursore, contando gli a capo e la
        # larghezza del pannello: così il punto in cui scrivi resta sempre visibile.
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
            t.append(" ", style=f"on {AMBER}")         # cursore in fondo o prima di un a capo
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
                state = "letto"
            elif j < reading_end:
                state = "lettura"
            elif j >= hold_start:
                state = "trattenuto"
            else:
                state = "pronto"
            t.append(self.piece(full[j]), style=STYLES[state][j % 2])
            t.append(" ")
        legend = Text()
        for state, label in LEGEND:
            legend.append(" abc ", style=STYLES[state][0])        # un token di esempio
            legend.append(f" {label}   ", style=MUTED)
        return self._panel(Group(t, legend), f"Cosa vede il modello: {len(full)} token (template + testo)")

    def _now(self) -> Panel:
        c = self.counts()
        lines = Text(style=SOFT)
        if self.job:
            lines.append(f"◆ Il modello sta leggendo {c['reading']} token ({self.job.reason}). ", style=f"bold {AMBER}")
            lines.append("È il prefill: per ogni token calcola chiavi e valori e li aggiunge alla KV cache.\n")
        elif not self.chars:
            lines.append("◆ Scrivi un messaggio. ", style=TEXT["bold"])
            lines.append("Il modello lo leggerà a pezzi mentre scrivi, invece di aspettare l'invio.\n")
        elif c["invalid"] and c["valid"] < len(self.target):
            lines.append(f"◆ Hai modificato il testo: {c['invalid']} token in cache non valgono più. ",
                         style=f"bold {RUST}")
            lines.append("Alla prossima pausa la cache viene tagliata lì (crop) e si rilegge da quel punto.\n")
        elif c["pending"] > 0:
            lines.append(f"◆ Stai scrivendo: {c['pending']} token {plural(c['pending'], 'pronto', 'pronti')} "
                         "da leggere. ", style=TEXT["bold"])
            lines.append(f"Il modello aspetta una pausa di {self.args.debounce:.0f} ms o {self.min_new} token "
                         "nuovi: ogni lettura ha un costo fisso, meglio raggrupparle.\n")
        else:
            lines.append(f"◆ Il modello è in pari: ha già letto {c['valid']} token. ", style=TEXT["bold"])
            lines.append("Aspetta che tu scriva ancora o che premi Invio.\n")
        if self.chars:
            lines.append(f"◆ Ultimi {c['held']} token trattenuti: ", style=AMBER)
            lines.append("la parola che stai scrivendo può ancora cambiare il modo in cui viene spezzata in token.\n")
            lines.append(f"◆ Se premi Invio adesso restano da calcolare {c['remaining']} token su {c['final']}",
                         style=TEXT["bold"])
            lines.append(" (il resto è già in cache)." if c["remaining"] < c["final"] else ".")
            if self.gpu and self.gpu[0] not in ("P0", "P1", "P2"):
                lines.append(f"\n◆ GPU a riposo ({self.gpu[0]}, memoria a {self.gpu[2]} MHz): ", style=AMBER)
                lines.append("la prossima lettura sarà più lenta finché il driver non rialza le frequenze.")
        return self._panel(lines, "Adesso", accent=True)

    def _numbers(self) -> Panel:
        c, tot = self.counts(), self.totals
        tb = Table.grid(padding=(0, 2))
        tb.add_column(style=MUTED, no_wrap=True)
        tb.add_column(style=PAPER, justify="right", no_wrap=True)
        for label, val in [("Token (con template)", c["full"]),
                           ("Letti e validi", c["valid"]),
                           ("In cache, non più validi", c["invalid"]),
                           ("Pronti, aspettano", max(0, c["pending"])),
                           ("Trattenuti (holdback)", c["held"]),
                           ("Letture eseguite", tot["letture"]),
                           ("Calcolati / scartati", f"{tot['calcolati']} / {tot['scartati']}"),
                           ("Tempo di lettura", f"{tot['ms']:.0f} ms"),
                           ("All'invio resterebbero", f"{c['remaining']} token")]:
            tb.add_row(label, str(val))
        body = Group(tb, Text(""), self._gpu_text()) if self.on_gpu else tb
        return self._panel(body, "Numeri")

    def _gpu_text(self) -> Text:
        if self.gpu is None:
            return Text("GPU: stato non disponibile", style=MUTED)
        state, core, mem = self.gpu
        awake = state in ("P0", "P1", "P2")
        t = Text("GPU ", style=MUTED)
        t.append("sveglia" if awake else "a riposo", style=TEXT["bold"] if awake else f"bold {AMBER}")
        t.append(f": {state}, core {core} MHz, memoria {mem} MHz", style=PAPER if awake else AMBER)
        if self.keep_warm:
            t.append(f"\ntenuta sveglia con {self.warm_pings} richieste minime", style=MUTED)
        return t

    def _events(self) -> Panel:
        t = Text()
        # I più recenti in alto: se il pannello è basso, si perdono i più vecchi.
        for ts, msg, style in reversed(list(self.events)[-14:]):
            t.append(f"{ts:6.1f}s  ", style=MUTED)
            t.append(msg + "\n", style=TEXT[style])
        return self._panel(t, "Registro, passo passo (più recenti in alto)")

    def _result(self, width: int) -> Panel:
        r = self.result
        t = Text(style=SOFT)
        t.append("◆ ", style=AMBER)
        t.append(self.stage + "\n\n", style=TEXT["bold"])
        if "ttft" in r:
            bar_w = max(10, width - 60)
            base = r.get("base", r["ttft"])
            scale = bar_w / max(base, r["ttft"])
            t.append(f"{'con prefill incrementale':<27}", style=AMBER)
            t.append("█" * max(1, round(r["ttft"] * scale)), style=AMBER)
            t.append(f" {r['ttft']:.0f} ms  ({r['tail']} token calcolati all'invio)\n")
            if "base" in r:
                t.append(f"{'senza (tutto da zero)':<27}", style=MUTED)
                t.append("█" * max(1, round(base * scale)), style="#5A554D")
                t.append(f" {base:.0f} ms  ({r['total']} token)\n")
                t.append(f"\nMentre scrivevi il modello aveva già letto {r['reused']} token su {r['total']}: "
                         f"all'invio hai aspettato il {100 * r['ttft'] / base:.0f}% del tempo.", style=TEXT["bold"])
        return self._panel(t, "Invio: attesa prima del primo token", accent=True)

    def _answer(self, width: int, lines: int) -> Panel:
        gen = self.result.get("gen")
        title = "Risposta" + (f" · {gen[0]} token, {gen[1]:.0f} token/s" if gen else "")
        if self.result.get("cut"):
            title += " · INTERROTTA al limite di token"
        # La risposta è in Markdown: la si impagina e se ne mostra la fine, perché
        # le righe che non stanno nel pannello verrebbero tagliate in basso.
        opts = self.console.options.update(width=max(20, width))
        # Basta impaginare la coda: è l'unica parte visibile, e rifare tutto il testo a
        # ogni fotogramma diventerebbe lento con risposte di migliaia di token.
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
    p.add_argument("--min-new", type=int, default=16, help="token nuovi in coda che fanno scattare una lettura")
    p.add_argument("--max-tokens", type=int, default=0,
                   help="lunghezza massima della risposta (0 = finché c'è spazio nel contesto dello slot)")
    p.add_argument("--no-confronto", action="store_true",
                   help="non misurare il calcolo da zero all'invio (su CPU con testi lunghi richiede decine di secondi)")
    p.add_argument("--auto", action="store_true", help="la demo scrive da sola un esempio")
    p.add_argument("--gpu-libera", action="store_true",
                   help="non tenere sveglia la GPU mentre scrivi: per vedere quanto rallenta quando si addormenta")
    # Due slot (messaggio e confronto da zero): 4096 token ciascuno bastano, e la KV
    # cache del 4B resta ~1,2 GB, così tutto sta nei 6 GB della 2060.
    p.set_defaults(ctx=4096)
    return p


def main():
    # Niente crash se l'output finisce in una pipe o in un file con codifica cp1252.
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors="replace")
    args = build_parser().parse_args()
    args.backend = "llama"

    console = Console(theme=MARKDOWN_THEME)
    console.print("[bold]Avvio llama-server...[/bold] (la prima volta può volerci un minuto)")
    tok, server, device, _ = setup(args, n_slots=2)        # slot 0: il tuo messaggio; slot 1: il confronto da zero
    demo = Demo(args, tok, server, make_engine(server, device), make_engine(server, device), device, console)
    source = AutoTyper() if args.auto else Keyboard()
    with Live(get_renderable=demo.render, console=console, screen=True, refresh_per_second=12):
        try:
            demo.run(source)
        except KeyboardInterrupt:
            pass
    server.close()


if __name__ == "__main__":
    main()
