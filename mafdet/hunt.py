"""
mafdet hunt: simulate a person typing and measure how much incremental
prefill shortens the wait after sending.

The simulation uses "virtual time": typing is not replayed in real time, but
every prefill really runs on the model and its measured duration is placed on
the typing timeline. If a prefill lasts longer than the pause after it, the
delay piles up and, if it lasts until the user sends, it is added to the wait.
That is how a real system with the model working in the background behaves.

Scenarios:
  clean   the user types the message from start to finish, without mistakes
  messy   typos fixed with backspace, a word deleted and retyped, an insertion
          in the middle of the text, no guarantees about punctuation
  paste   the user pastes a long document, then types a question

Examples:
    mafdet hunt --scenario paste
    mafdet hunt --scenario messy --holdback 0 1 3 6 --debounce 100 250 500
"""
import csv
import os
import random
import statistics
from dataclasses import asdict

from .engine import (Template, common_prefix_len, compare_logits, device_label, encode, make_engine,
                     model_label, read_sample, setup, stable_ids, timed, to_host)

LETTERS = "abcdefghilmnoprstuvz"


def sample(name: str) -> str:
    return read_sample(name).strip()


# --------------------------------------------------------------------------
# Typing events
# --------------------------------------------------------------------------

class Editor:
    def __init__(self):
        self.chars: list[str] = []
        self.cursor = 0

    def insert(self, s: str):
        self.chars[self.cursor:self.cursor] = list(s)
        self.cursor += len(s)

    def backspace(self, n: int = 1):
        n = min(n, self.cursor)
        del self.chars[self.cursor - n:self.cursor]
        self.cursor -= n

    @property
    def text(self) -> str:
        return "".join(self.chars)


class Typist:
    """Produces the sequence (time in ms, text after the event)."""

    def __init__(self, cps: float, seed: int):
        self.rng = random.Random(seed)
        self.cps = cps
        self.t = 0.0
        self.ed = Editor()
        self.events: list[tuple[float, str]] = []

    def _gap(self) -> float:
        return 1000.0 / self.cps * self.rng.lognormvariate(0, 0.35)

    def _emit(self):
        self.events.append((self.t, self.ed.text))

    def pause(self, lo: float, hi: float):
        self.t += self.rng.uniform(lo, hi)

    def type(self, s: str, typo_p: float = 0.0):
        for c in s:
            if typo_p and c.isalpha() and self.rng.random() < typo_p:
                self.t += self._gap()
                self.ed.insert(self.rng.choice(LETTERS))   # typo
                self._emit()
                self.pause(200, 450)                       # time to notice it
                self.ed.backspace()
                self._emit()
            self.t += self._gap()
            self.ed.insert(c)
            self._emit()
            if c == " " and self.rng.random() < 0.08:
                self.pause(600, 1500)                      # pause to think
            elif c in ".?!\n":
                self.pause(700, 1800)

    def delete_last_word_and_retype(self):
        text = self.ed.text[:self.ed.cursor].rstrip()
        cut = len(self.ed.text[:self.ed.cursor]) - (text.rfind(" ") + 1)
        word = self.ed.text[self.ed.cursor - cut:self.ed.cursor]
        self.pause(300, 700)
        for _ in range(cut):                               # backspace held down
            self.t += 60
            self.ed.backspace()
            self._emit()
        self.type(word)

    def move_cursor(self, pos: int):
        self.pause(500, 1200)                              # mouse click
        self.ed.cursor = pos

    def paste(self, s: str):
        self.pause(300, 600)
        self.ed.insert(s)
        self._emit()


def build_events(scenario: str, cps: float, seed: int, text: str | None, doc_repeat: int):
    ty = Typist(cps, seed)
    if scenario == "clean":
        ty.type(text or sample("messaggio_medio.txt"))
    elif scenario == "messy":
        msg = text or sample("messaggio_medio.txt")
        a, b = int(len(msg) * 0.4), int(len(msg) * 0.7)
        ty.type(msg[:a], typo_p=0.03)
        ty.delete_last_word_and_retype()
        ty.type(msg[a:b], typo_p=0.03)
        # go back and add something in the middle of the text already typed
        spaces = [i for i, c in enumerate(ty.ed.text) if c == " " and i < len(ty.ed.text) // 2]
        end = len(ty.ed.text)
        pos = ty.rng.choice(spaces) if spaces else end
        ty.move_cursor(pos)
        # Italian, like the bundled samples ("and in detail"): changing it would change the measurements.
        ty.type(" e in modo dettagliato", typo_p=0.03)
        ty.move_cursor(len(ty.ed.text))
        ty.type(msg[b:], typo_p=0.03)
    elif scenario == "paste":
        doc = "\n\n".join([sample("documento.txt")] * doc_repeat)
        ty.paste(doc + "\n\n")
        ty.pause(1500, 3000)                               # reads it again, scrolls, thinks
        ty.type(text or sample("domanda.txt"))
    else:
        raise ValueError(scenario)
    t_enter = ty.t + ty.rng.uniform(300, 800)
    return ty.events, t_enter


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------

def simulate(eng, tok, tpl: Template, events, t_enter,
             holdback: int, debounce: float, min_new: int, device: str):
    eng.reset()
    busy_until = 0.0
    busy_total = 0.0
    max_queue = 0.0

    def run_sync(fire_t, target):
        nonlocal busy_until, busy_total, max_queue
        if target == eng.ids:
            return
        start = max(fire_t, busy_until)
        max_queue = max(max_queue, start - fire_t)
        _, d = timed(device, eng.sync, target)
        busy_until = start + d
        busy_total += d

    for i, (t, text) in enumerate(events):
        next_t = events[i + 1][0] if i + 1 < len(events) else t_enter
        target = stable_ids(encode(tok, tpl.prefix + text), holdback)
        # threshold: enough new stable tokens at the end, and the model is idle.
        # It only applies when the cache is a prefix of the text: an edit in the
        # middle invalidates the whole tail, and recomputing it on every keystroke
        # would be a waste. Those edits wait for the debounce, i.e. a pause.
        n = common_prefix_len(eng.ids, target)
        if busy_until <= t and n == len(eng.ids) and len(target) - n >= min_new:
            run_sync(t, target)
        # debounce: the user stopped long enough
        if next_t - t >= debounce:
            run_sync(t + debounce, target)

    final_text = events[-1][1]
    final_ids = encode(tok, tpl.prefix + final_text + tpl.suffix)
    wait = max(0.0, busy_until - t_enter)
    logits, d_final = timed(device, eng.sync, final_ids, need_logits=True)
    stats = asdict(eng.stats)
    return dict(
        final_tokens=len(final_ids),
        typing_s=round(t_enter / 1000, 1),
        wait_ms=round(wait, 1),
        final_sync_ms=round(d_final, 1),
        ttft_inc_ms=round(wait + d_final, 1),
        gpu_busy_pct=round(100 * busy_total / max(t_enter, 1), 1),
        max_queue_ms=round(max_queue, 1),
        **stats,
    ), to_host(logits), final_ids


def baseline(eng, final_ids, device, repeats=3):
    times, logits = [], None
    for _ in range(repeats):
        eng.reset()
        logits, ms = timed(device, eng.sync, final_ids, need_logits=True)
        times.append(ms)
    return statistics.median(times), to_host(logits)


def run(args) -> int:
    tok, model, device, dtype = setup(args)
    tpl = Template.from_tokenizer(tok, thinking=False)
    eng = make_engine(model, device)
    custom = open(args.text, encoding="utf-8").read().strip() if args.text else None
    dev_name = device_label(device)
    dt_name = str(dtype).replace("torch.", "")

    for _ in range(2):                                     # warm-up
        eng.reset()
        eng.sync(encode(tok, tpl.prefix + sample("messaggio_breve.txt") + tpl.suffix), need_logits=True)

    os.makedirs(args.out, exist_ok=True)
    out_csv = os.path.join(args.out, "typing.csv")
    new_file = not os.path.exists(out_csv)
    rows = []

    for scenario in args.scenario:
        for cps in args.cps:
            events, t_enter = build_events(scenario, cps, args.seed, custom, args.doc_repeat)
            final_ids = encode(tok, tpl.prefix + events[-1][1] + tpl.suffix)
            base_ms, base_logits = baseline(eng, final_ids, device)
            print(f"\n=== {scenario} | {cps:g} chars/s | {len(events)} events | {len(final_ids)} final tokens "
                  f"| wait without incremental prefill: {base_ms:.1f} ms")
            print(f"{'holdback':>8} {'debounce':>8} {'wait ms':>10} {'saved':>9} {'syncs':>8} "
                  f"{'computed':>11} {'discarded':>9} {'busy':>8} {'check':>8}")
            for hb in args.holdback:
                for db in args.debounce:
                    res, inc_logits, _ = simulate(eng, tok, tpl, events, t_enter, hb, db, args.min_new, device)
                    cmp = compare_logits(inc_logits, base_logits)
                    row = dict(model=model_label(args), device=dev_name, dtype=dt_name, scenario=scenario, cps=cps,
                               holdback=hb, debounce_ms=db, min_new=args.min_new,
                               ttft_base_ms=round(base_ms, 1), **res,
                               reduction_pct=round(100 * (1 - res["ttft_inc_ms"] / base_ms), 1),
                               same_argmax=cmp["same_argmax"], max_abs_diff=round(cmp["max_abs_diff"], 5),
                               rel_diff=round(cmp["rel_diff"], 5), check=cmp["verdict"])
                    rows.append(row)
                    print(f"{hb:>8} {db:>8.0f} {res['ttft_inc_ms']:>10.1f} {row['reduction_pct']:>8.1f}% "
                          f"{res['syncs']:>8} {res['tokens_computed']:>11} {res['tokens_discarded']:>9} "
                          f"{res['gpu_busy_pct']:>7.1f}% {cmp['verdict']:>8}")

    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new_file:
            w.writeheader()
        w.writerows(rows)
    print(f"\nResults appended to {out_csv}")
    print("Columns: 'wait ms' = time from send to the logits of the first token; 'syncs' = how many\n"
          "times the cache was updated while typing; 'discarded' = tokens computed and then thrown\n"
          "away because of edits or unstable tokenization; 'busy' = share of the typing time the\n"
          "model was working; 'check' = comparison with a from-scratch computation.")
    return 0
