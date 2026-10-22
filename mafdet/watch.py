"""
The chat session: a conversation on one llama-server slot, its cache following the draft.

The interfaces (the window of mafdet gui, the terminal app of mafdet tui) send
the draft on every change, through core/chat.py, and the session keeps the
slot's KV cache aligned with it: common prefix, hold-back, threshold and debounce, the same
rules measured by `mafdet hunt`. On Enter only the tail is left to compute,
then the answer streams back.

The whole conversation stays in the cache: after an answer the next user turn
is appended to the tokens already there, so every turn starts where the
previous one ended instead of rereading the history.

    mafdet watch                    # the window, with the model and options of these commands
    mafdet strike --no-think
"""
from __future__ import annotations

import http.client
import json
import os
import re
import socket
import threading
import time


# Qwen3 in non-thinking and thinking mode: the sampling settings recommended by its authors.
# In thinking mode the presence penalty (which they suggest between 0 and 2) keeps the 4B
# out of loops that repeat a sentence until the token limit.
SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0}
THINK_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0, "presence_penalty": 1.5}

# Reasoning ahead (--think) is written by the model as its own thoughts, framed by a few
# sentences of ours: a list of material for the answer, one line per key term, name or
# number already typed. Measured on Qwen3-4B, frames compared on the same typing:
#   - none: the notes remark that the message is unfinished and, after every edit, copy
#     their last paragraph again;
#   - "work out what you can": 86% of the notes still about the user or the message;
#   - this list, with a line per new words: 41-54%;
#   - the list with an example line (from another message) and META_BIAS below: 17-26%, no
#     answer lost in a loop, the least reasoning left after Enter. The example ends up copied
#     into about one set of notes in nine.
SEED = ("<think>\nThe user is still typing. I will not wonder what they will ask, nor remark that the text is "
        "unfinished. While I wait, I collect material for the answer: for each key term, name or number already "
        "written, one line with what it is and the facts I know about it, or a partial result. For example, from "
        "\"Quanto pesa la Luna rispetto alla\" I would write \"- Luna (the Moon): mass 7.35e22 kg, radius "
        "1,737 km\".\n\n- ")
# When the user typed more: a new line of the same list, naming the new words (whole words, two or more).
BRIDGE = "\n- New words \"{new}\": "
BRIDGE_EDIT = "\n- The text was edited, so: "   # when nothing was added: an edit or a deletion
BRIDGE_NEW_CHARS = 300
BRIDGE_MIN_WORDS = 2
# On Enter: the message is complete now.
SENT = "\n\nThe user has now sent the complete message. Let me reread it and answer it, using the material above.\n\n"
# While reasoning ahead (and only then) these words are made less likely: thoughts about the
# user or the unfinished text are the waste the frame does not remove. The token ids come from
# the tokenizer of the model loaded, so this works with any model. 0 turns it off.
META_WORDS = ("user", "User", "incomplete", "unfinished", "typing", "typed", "intent", "intends", "asking",
              "wondering", "seems")
META_BIAS = -4.0

# --gatherer: the small model's llama-server, and how the main model is handed its material.
GATHERER_PORT = 8090
# A plain small model (not trained for it) gets the list frame without the example line: a 0.6B
# copies the example into its notes.
GATHER_SEED_FRAME = ("<think>\nThe user is still typing. I will not wonder what they will ask, nor remark that the "
                     "text is unfinished. While I wait, I collect material for the answer: for each key term, name "
                     "or number already written, one line with what it is and the facts I know about it, or a "
                     "partial result.\n\n- ")
# A gatherer trained with training/gatherer_colab.ipynb: exactly the format of its training examples.
# Each step names the words typed since the last one; it writes their lines and ends the step.
GATHER_SEED_TRAINED = ("<think>\nMaterial for the answer, gathered while the user types: for each key term, name or "
                       "number already written, one line with what it is and the facts I know about it, or a partial "
                       "result.\n\n")
GATHER_HEADER = "\nNew words: \"{new}\"\n"
# Its steps as long as in training (2 to 5 per message: about 5 words each on a word problem). Given a
# step every 2 words, it learned that so few words seldom deserve a line: one or two lines per problem.
GATHER_STEP_WORDS = 5
NOTES_INTRO = "<think>\nMaterial gathered while the user was typing, by a fast assistant (it may contain mistakes):\n"
THINK_MIN_WORDS = 3                 # no reasoning ahead on the first couple of words
AHEAD_CHUNK = 128                   # tokens per request written ahead; the next one carries on from there
THINK_CHUNK = 32                    # the same while reasoning: short, so it follows the text as it is typed
BRIDGE_EVERY = 64                   # new reasoning tokens before an edit earns a bridge


def turn_separator(tok) -> str:
    """What the chat template puts between an answer and the next user message.

    For Qwen3: '<|im_end|>\\n<|im_start|>user\\n'.
    """
    a, b, c = "<<<U1>>>", "<<<A1>>>", "<<<U2>>>"
    msgs = [{"role": "user", "content": a}, {"role": "assistant", "content": b}, {"role": "user", "content": c}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    return s.split(b, 1)[1].split(c, 1)[0]


class Stream:
    """One streamed generation on the slot, run in its own thread and read by whoever needs it.

    An answer written ahead (mafdet strike) is a Stream started at a pause,
    before Enter: it fills `tokens` while nobody shows them. On Enter, if the
    prompt is still the same, the page gets all of them at once and then
    follows the stream live. Sampling has no penalties, so each token depends
    only on the tokens before it: the answer written ahead is drawn from the
    same distribution as one started after Enter.
    """

    def __init__(self, session: "Session", prompt: list[int], limit: int, text: str = "", bias: list | None = None):
        self.session, self.prompt, self.limit, self.text, self.bias = session, prompt, limit, text, bias
        self.tokens: list[int] = []
        self.info: dict = {}                       # the last event: stop_type, timings, tokens_cached
        self.done = self.cancelled = False
        self.cond = threading.Condition()
        self._conn: http.client.HTTPConnection | None = None
        threading.Thread(target=self._run, daemon=True).start()

    @property
    def hit_limit(self) -> bool:
        return self.info.get("stop_type") == "limit"

    def _run(self):
        s = self.session
        body = {"prompt": self.prompt, "id_slot": s.eng.slot, "cache_prompt": True, "stream": True,
                "n_predict": self.limit, "return_tokens": True, **s.sampling}
        if self.bias:
            body["logit_bias"] = self.bias
        with s.slot_lock:
            try:
                if not self.cancelled:
                    self._conn = http.client.HTTPConnection("127.0.0.1", s.server.port, timeout=600)
                    self._conn.request("POST", "/completion", json.dumps(body), {"Content-Type": "application/json"})
                    for raw in self._conn.getresponse():
                        if self.cancelled:
                            break                  # closing the connection stops llama-server too
                        line = raw.decode("utf-8", "replace").strip()
                        if not line.startswith("data: "):
                            continue
                        d = json.loads(line[6:])
                        with self.cond:
                            self.tokens += d.get("tokens") or []
                            if d.get("stop"):
                                self.info = d
                            self.cond.notify_all()
                        if d.get("stop"):
                            break
            except (OSError, http.client.HTTPException, ValueError) as e:
                if not self.cancelled:
                    self.info = {"error": str(e)}
            finally:
                if self._conn is not None:
                    self._conn.close()
                    # The slot now holds the prompt and what was generated: the next request starts from there.
                    cached = self.prompt + self.tokens
                    s.eng.assume_cached(cached[:self.info.get("tokens_cached", len(cached))])
                with self.cond:
                    self.done = True
                    self.cond.notify_all()

    def cancel(self):
        """Stop now: shutting the socket down also wakes a read waiting for the next token."""
        self.cancelled = True
        conn = self._conn
        if conn is not None and conn.sock is not None:
            try:
                conn.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def wait(self):
        with self.cond:
            while not self.done:
                self.cond.wait()

    def follow(self, stop: threading.Event):
        """The tokens as they arrive: what is already there in one go, then as they come."""
        i = 0
        while True:
            with self.cond:
                while i == len(self.tokens) and not self.done and not stop.is_set():
                    self.cond.wait(0.1)
                new, done = self.tokens[i:], self.done
            i += len(new)
            if new:
                yield new
            if done or stop.is_set():
                return


class Detokenizer:
    """Text from a growing list of tokens, never cutting a character in two.

    The text streamed by the server is cut where each request ends: a letter
    whose bytes fall across two chunks (an accent, a typographic quote) comes
    out as two replacement characters. Decoding the tokens here, a short window
    at a time, holds such a character back until its last byte arrives.
    """

    def __init__(self, tok, lock: threading.Lock, skip: int):
        self.tok, self.lock, self.skip = tok, lock, skip
        self.ids: list[int] = []
        self.start = self.read = 0

    def push(self, tokens: list[int]) -> str:
        self.ids += [t for t in tokens if t != self.skip]
        with self.lock:
            head = self.tok.decode(self.ids[self.start:self.read])
            text = self.tok.decode(self.ids[self.start:])
        if text.endswith("�"):
            return ""                              # a character still incomplete: wait for its other bytes
        self.start, self.read = self.read, len(self.ids)
        return text[len(head):]


class Session:
    """One conversation on one llama-server slot, its cache following the draft."""

    def __init__(self, args, role: str = "main"):
        self.args, self.role = args, role
        self.state, self.error, self.label = "loading", "", ""
        self.holdback, self.min_new, self.debounce = args.holdback, args.min_new, args.debounce / 1000
        self.ahead = getattr(args, "ahead", 0)     # mafdet strike: tokens written at a pause, before Enter
        self.think = getattr(args, "think", False)  # mafdet strike: reason ahead, keeping the notes across edits
        self.think_cap = getattr(args, "think_ahead", 0)
        # --gatherer: a small model writes the notes (a second Session in the "gatherer" role, on
        # its own llama-server); this one, the main model, only reads them and answers.
        self.gatherer: Session | None = None
        self._g_notes: list[int] | None = None     # the gatherer's notes last read into this model
        if role == "gatherer":
            self.think, self.ahead = True, 1       # it reasons ahead, and stops where an answer would start
        elif getattr(args, "gatherer", ""):
            if not self.think:
                raise SystemExit("--gatherer hands its material to the main model's reasoning: drop --no-think.")
            self.ahead = 0                         # the gatherer reasons ahead in its place
        self.slot_lock = threading.Lock()          # one request at a time on the slot
        self.tok_lock = threading.Lock()           # the fast tokenizer is not safe across threads
        self.spec_lock = threading.Lock()          # starting an answer ahead vs. pressing Enter
        self.text, self.text_t = "", 0.0
        self.target: list[int] = []                # what the cache should hold now (draft minus hold-back)
        self.final_ids: list[int] = []             # the prompt if the user pressed Enter now
        self.final_text = ""
        self.history: list[int] = []               # the conversation so far, up to the start of the next user text
        self.spec: Stream | None = None            # the answer being written ahead for final_ids
        self._failed: list[int] = []               # a prompt whose answer ahead failed
        # Written ahead, in the prompt after the message. The reasoning (--think) survives edits:
        # notes. The answer only holds for the exact text it was written for: answer_ahead
        # (with --think it starts with </think>).
        self.notes: list[int] = []
        self.answer_ahead: list[int] = []
        self.notes_text = ""                       # the text notes and answer_ahead were last extended on
        self.notes_fresh = 0                       # notes tokens written since the last bridge
        self.bridge_text = ""                      # the text at the last bridge (or at the frame)
        self.last_bridge: list[int] = []
        self.fmt = "frame"                         # how the notes are framed; "trained": the trained gatherer's steps
        self.awaiting = False                      # trained gatherer: new words given, their lines not finished yet
        self.chunk_t = 0.0                         # when the last chunk written ahead started or ended
        self.busy = self.generating = False
        self.stop = threading.Event()
        self._loader = threading.Thread(target=self._load, daemon=True)
        self._loader.start()

    # ---- start-up ------------------------------------------------------------
    def _load(self):
        try:
            # Imported here: transformers takes seconds, and the window is already open.
            from .llama_backend import LlamaPrefill, setup_llama
            from .text import Template, common_prefix_len, encode, stable_ids
            self._prefix_len, self._encode, self._stable = common_prefix_len, encode, stable_ids
            if self.role == "main" and getattr(self.args, "gatherer", ""):
                # The gatherer first: it takes a fixed share of VRAM, then the main model fits in what is left.
                self.gatherer = Session(self.args, role="gatherer")
                while self.gatherer.state == "loading":
                    time.sleep(0.1)
                if self.gatherer.state == "error":
                    raise SystemExit(f"Gatherer: {self.gatherer.error}")
            args = self._gatherer_args() if self.role == "gatherer" else self.args
            self.tok, self.server, device, quant = setup_llama(args, n_slots=1)
            self.eng = LlamaPrefill(self.server, slot=self.server.take_slot())
            tpl = Template.from_tokenizer(self.tok, thinking=self.think)
            self.sampling = THINK_SAMPLING if self.think else SAMPLING
            self.start_ids = encode(self.tok, tpl.prefix)
            self.suffix = tpl.suffix
            self.between_ids = encode(self.tok, turn_separator(self.tok))
            self.end_id = self.between_ids[0]      # end of a turn: <|im_end|> for Qwen3
            self.think_end = self.tok.convert_tokens_to_ids("</think>")
            if self.role == "gatherer":
                # A gatherer trained with training/gatherer_colab.ipynb is told apart by its file name.
                trained = "mafdet-gatherer" in os.path.basename(self.server.model_path)
                self.fmt = "trained" if trained else "frame"
                seed = GATHER_SEED_TRAINED if trained else GATHER_SEED_FRAME
                if trained:
                    # Every line starts with "- " and names a word of the message: the penalty on tokens
                    # already used makes it end its steps early (measured: 1-2 lines per word problem, some
                    # invented, against 3-5 right ones without it).
                    self.sampling = {**THINK_SAMPLING, "presence_penalty": 0}
            else:
                seed = SEED
            self.seed_ids = encode(self.tok, seed)
            self.sent_ids = encode(self.tok, SENT)
            # Single-token spellings only: biasing the first piece of a longer word would hit other words too.
            meta = {t[0] for w in META_WORDS for v in (w, " " + w) for t in [encode(self.tok, v)] if len(t) == 1}
            self.meta_bias = [[t, META_BIAS] for t in sorted(meta)] if META_BIAS else []
            # How far the cache can run past the stable text after a read at a pause.
            self.tail_slack = len(encode(self.tok, tpl.suffix)) + self.holdback + 2
            self.history = list(self.start_ids)
            name = os.path.basename(self.server.model_path).removesuffix(".gguf").removesuffix(f"-{quant}")
            self.label = f"{name} · {quant} · {'GPU' if device == 'llama-cuda' else 'CPU'}"
            self._retarget()
            if self.state == "closed":             # closed while loading: close() shuts the servers down
                return
            self.state = "ready"
        except BaseException as e:                 # SystemExit included: it carries the message for the user
            if self.state != "closed":
                self.state, self.error = "error", str(e) or type(e).__name__
            return
        self._prefill_loop()

    def close(self):
        """Stop for good: no more reads, the answer in progress stops, both llama-servers are shut down.

        Safe while still loading: the server being started is closed as soon as it is up.
        """
        self.stop.set()
        loading = self.state == "loading"
        self.state = "closed"
        if loading:
            def later():
                self._loader.join()
                self._shutdown()
            threading.Thread(target=later, daemon=True).start()
        else:
            self._shutdown()

    def _shutdown(self):
        for s in (self, self.gatherer):
            server = getattr(s, "server", None) if s is not None else None
            if server is not None:
                server.close()

    # ---- draft and prefill -------------------------------------------------------
    def draft(self, text: str):
        self.text, self.text_t = text, time.perf_counter()
        if self.gatherer is not None:
            self.gatherer.draft(text)
        if self.state == "ready":
            self._retarget()

    # ---- gatherer --------------------------------------------------------------------
    def _gatherer_args(self):
        """The options of the gatherer's own llama-server: its model, device, port and a short context."""
        import argparse
        from . import DEFAULT_MODEL
        args = argparse.Namespace(**vars(self.args))
        args.gguf, args.model, args.device = self.args.gatherer, DEFAULT_MODEL, self.args.gatherer_device
        # A local file from the training notebook names its base model: mafdet-gatherer-Qwen3-0.6B-Q8_0.gguf.
        base = re.match(r"mafdet-gatherer-(.+)-[^-]+\.gguf$", os.path.basename(args.gguf))
        if base and os.path.exists(args.gguf):
            args.model = f"Qwen/{base.group(1)}"
        args.ctx, args.port, args.recycle, args.draft = 4096, GATHERER_PORT, False, ""
        # On the CPU it shares the cores with the main model's layers there: a few threads are enough.
        args.llama_args = "-t 4" if args.device == "cpu" else ""
        return args

    def material(self) -> str:
        """The gatherer's notes as text, without the frame (and without a bridge left empty)."""
        notes = self.notes
        if len(notes) <= len(self.seed_ids):
            return ""
        if self.last_bridge and notes[-len(self.last_bridge):] == self.last_bridge:
            notes = notes[:-len(self.last_bridge)]
        with self.tok_lock:
            text = self.tok.decode(notes[len(self.seed_ids):])
        if self.fmt == "trained":                  # its lines only, not the steps' headers, each once
            return "\n".join(dict.fromkeys(l for l in text.splitlines() if l.startswith("- ")))
        # A line the small model repeats (often, after every few words) is read once; blank lines stay.
        seen = set()
        lines = [l for l in ("- " + text.strip()).splitlines()
                 if not l.strip() or not (l in seen or seen.add(l))]
        return "\n".join(lines)

    def take_notes(self, text: str) -> str:
        """On Enter: stop gathering and hand over the material written so far for `text`."""
        with self.spec_lock:
            self.generating = True
            spec, self.spec = self.spec, None
        if spec is not None:
            spec.cancel()
            if not spec.info.get("error"):
                self._absorb(spec, changed=spec.text.rstrip() != text.rstrip(), bridge_after=10 ** 9)
        return self.material()

    def next_turn(self):
        self.notes, self.answer_ahead, self.notes_fresh, self.notes_text = [], [], 0, ""
        self.generating = False
        if self.state == "ready":
            self._retarget()

    def _read_material(self, sent: bool = False):
        """The gatherer's material, in this model's tokens, after the message: inside its own reasoning."""
        material = self.gatherer.material() if self.gatherer.state == "ready" else ""
        with self.tok_lock:
            self.notes = self._encode(self.tok, NOTES_INTRO + material) if material else []
        if sent and self.notes:
            self.notes = self.notes + self.sent_ids

    def _prompt(self, text: str) -> list[int]:
        """The prompt on Enter: history, message, start of the answer, then what was written ahead."""
        # Trailing spaces and newlines are dropped: one typed after a pause must not throw away the answer written ahead.
        with self.tok_lock:
            ids = self.history + self._encode(self.tok, text.rstrip() + self.suffix) + self.notes
        if self.answer_ahead and text.rstrip() == self.notes_text.rstrip():
            ids += self.answer_ahead
        return ids

    def _retarget(self):
        with self.tok_lock:
            msg = self._encode(self.tok, self.text)
        self.target = self._stable(self.history + msg, self.holdback)
        self.final_text = self.text                # the text final_ids was built from
        self.final_ids = self._prompt(self.text)

    def _start_chunk(self, prompt: list[int], text: str):
        """A chunk written ahead on `prompt`, built from `text` (call it holding spec_lock)."""
        room = self.args.ctx - len(prompt) - 8
        if room < 32:
            return
        if self.think and not self.answer_ahead:
            chunk = min(THINK_CHUNK, self.think_cap - len(self.notes)) if not self._paused() else THINK_CHUNK
        else:
            chunk = min(AHEAD_CHUNK, self.ahead - len(self.answer_ahead))
        bias = self.meta_bias if self.think and not self.answer_ahead else None
        self.spec = Stream(self, prompt, max(1, min(chunk, room)), text, bias)
        self.chunk_t = time.perf_counter()

    def _take_spec(self, spec: Stream) -> bool:
        """Take `spec` away from the session, unless Enter has claimed it first."""
        with self.spec_lock:
            if self.spec is not spec:
                return False
            self.spec = None
        return True

    def _absorb(self, spec: Stream, changed: bool, bridge_after: int = BRIDGE_EVERY):
        """Keep what `spec` wrote: reasoning (before </think>) to the notes, answer to answer_ahead.

        If the message changed meanwhile, the answer is dropped. A bridge then
        follows the notes, telling the model to reread the message, so it goes
        on from its earlier thoughts instead of starting over: once it had
        finished thinking, or after `bridge_after` tokens of new thoughts (not
        after every chunk written while the user types on).
        """
        spec.wait()
        self.chunk_t = time.perf_counter()
        tokens = spec.tokens
        if self.fmt == "trained":
            # The trained gatherer writes the lines for the words it was given, then ends the step.
            finished = self.end_id in tokens or self.think_end in tokens or not spec.hit_limit
            for stop in (self.end_id, self.think_end):
                if stop in tokens:
                    tokens = tokens[:tokens.index(stop)]
            self.notes = self.notes + tokens
            self.notes_fresh += len(tokens)
            self.awaiting = self.awaiting and not finished
            self.notes_text = spec.text
            return
        if self.think and not self.answer_ahead:
            cut = tokens.index(self.think_end) if self.think_end in tokens else len(tokens)
            thought, answer = tokens[:cut], tokens[cut:]
        else:
            thought, answer = [], tokens
        if thought:
            self.notes = self.notes + thought
            self.notes_fresh += len(thought)
        if changed:
            had_answer = bool(answer or self.answer_ahead)
            self.answer_ahead = []
            self._bridge(1 if had_answer else bridge_after)
        else:
            self.answer_ahead = self.answer_ahead + answer
        self.notes_text = spec.text

    def _new_words(self) -> tuple[str, int]:
        """The whole words typed since the last bridge, and where they end in the text.

        The last word is left out while it may still be typed: a bridge never
        names half a word, and the next one starts where this one stopped.
        """
        start = len(os.path.commonprefix([self.bridge_text, self.text]))
        new = self.text[start:]
        if new[-1:].isalnum():
            new = new[:max(new.rfind(" "), 0)]
        return new.strip(), start + len(new)

    def _bridge(self, after: int = 1):
        """A bridge after the notes: what the user typed since the last one, for the model to work on."""
        if self.notes_fresh < after and self.fmt != "trained":
            return
        new, end = self._new_words()
        bridge = GATHER_HEADER if self.fmt == "trained" else BRIDGE
        if self.fmt == "trained" and len(self.notes) <= len(self.seed_ids):
            bridge = bridge.lstrip("\n")           # the first step follows the frame directly, as in training
        if self.fmt == "trained" and not new:
            return                                 # the trained gatherer works on new words only
        if "{new}" in bridge and new and len(new.split()) < BRIDGE_MIN_WORDS and after > 1:
            return                                 # a word or two: wait for more before a bridge
        with self.tok_lock:
            if "{new}" in bridge and new:
                self.last_bridge = self._encode(self.tok, bridge.format(new=new[-BRIDGE_NEW_CHARS:]))
            else:
                self.last_bridge = self._encode(self.tok, BRIDGE_EDIT)
        self.notes = self.notes + self.last_bridge
        self.notes_fresh = 0
        self.bridge_text = self.text[:end]
        self.awaiting = True

    def _paused(self) -> bool:
        return time.perf_counter() - self.text_t >= self.debounce

    def _answer_complete(self) -> bool:
        return bool(self.answer_ahead) and self.answer_ahead[-1] == self.end_id

    def _ahead_room(self, extra: int = 0) -> bool:
        """Still room to write ahead: reasoning, then answer up to --ahead (or its end).

        --think-ahead caps the reasoning only while the user types. At a pause
        the text stays as it is, so the model reasons to the end and then writes
        the answer: after a long enough pause, Enter shows it at once.
        """
        if self._answer_complete() or (self.role == "gatherer" and self.answer_ahead):
            return False                           # the gatherer's job ends with the reasoning
        if self.fmt == "trained":
            return not self.notes or self.awaiting  # a step at a time: new words, then its lines
        if self.think and not self.answer_ahead:
            return self._paused() or len(self.notes) + extra < self.think_cap
        return len(self.answer_ahead) + extra < self.ahead

    def _notes_stale(self) -> bool:
        """The message was emptied or mostly rewritten: notes about the old one would mislead."""
        if not self.notes:
            return False
        kept = len(os.path.commonprefix([self.text, self.notes_text]))
        return not self.text.strip() or kept < len(self.notes_text) // 2

    def _prefill_loop(self):
        """Reads the draft in the background: a burst of new tokens at the end, or a pause.

        At a pause it reads the whole prompt, held-back tokens and end of the
        template included, as if the user had pressed Enter: if they do, a
        single token is left to compute; if they type on, the common-prefix
        check drops those few tokens on the next read.

        With --ahead (mafdet strike), once the whole prompt is read and the
        pause goes on, it also starts writing the answer. An answer only holds
        for the exact prompt it was written for, so any change cancels it.

        With --think the model reasons while the user types, not only at pauses,
        in short chunks: a chunk runs on the text as it was when it started, then
        the cache reads the new text and the next chunk goes on from there. The
        reasoning is kept across edits: it becomes notes that follow the new
        text, a prefill away (parallel and fast) rather than a decode away.
        """
        while self.state != "closed":
            time.sleep(0.02)
            if self.gatherer is not None and not self.generating and self.gatherer.notes is not self._g_notes:
                # New material from the gatherer: read at the next pause, like the rest of the prompt.
                self._g_notes = self.gatherer.notes
                self._read_material()
                self._retarget()
            target, final, final_text, done = self.target, self.final_ids, self.final_text, self.eng.ids
            spec = self.spec
            if spec is not None:
                changed, failed = spec.prompt != final, bool(spec.info.get("error"))
                if changed and self.think and not failed and not spec.done and not self._notes_stale():
                    continue                       # reasoning on a slightly older text is still worth finishing
                if (changed or failed) and self._take_spec(spec):
                    spec.cancel()                  # the text changed: that answer no longer holds
                    if failed:
                        self._failed = spec.prompt     # do not retry the same prompt in a loop
                    elif self.think:
                        self._absorb(spec, changed=True)
                    self._retarget()
                    continue
                if spec.done and spec.hit_limit and self._ahead_room(len(spec.tokens)) and self._take_spec(spec):
                    self._absorb(spec, changed=False)  # the pause goes on: carry on from there at the next turn
                    self._retarget()
                    continue
            if self.generating or self.spec is not None:
                continue
            if self.fmt == "trained" and self.notes and not self.awaiting:
                new, _ = self._new_words()
                if len(new.split()) >= GATHER_STEP_WORDS:
                    self._bridge()                 # the next step: the words typed since the last one
                    self._retarget()
                    continue
            if self.notes_text and self.text.rstrip() != self.notes_text.rstrip():
                if self.answer_ahead:              # written for a text that has changed since
                    self.answer_ahead = []
                    self._bridge()
                    self._retarget()
                    continue
                if self._notes_stale():
                    self.notes, self.notes_text, self.notes_fresh = [], "", 0
                    self._retarget()
                    continue
            paused = time.perf_counter() - self.text_t >= self.debounce
            has_text = bool(self.text.strip())
            # Reasoning starts on whole words: a pause, or a space or punctuation just typed.
            min_words = GATHER_STEP_WORDS if self.fmt == "trained" else THINK_MIN_WORDS
            can_start = self._ahead_room() and (not self.think or (
                len(self.text.split()) >= min_words and (paused or not self.text[-1:].isalnum())))
            # Reasoning goes on while the user types; an answer (no --think) waits for a pause.
            go = paused or (self.think and self.ahead and can_start)
            if (self.ahead and go and has_text and can_start and final and done == final
                    and final != self._failed):
                with self.spec_lock:
                    if not self.generating and self.final_ids is final:
                        if self.think and not self.notes:
                            self.notes = list(self.seed_ids)     # the frame first, then the thoughts
                            self.bridge_text = ""
                            if self.fmt == "trained":
                                self._bridge()                   # its first step names the words so far
                            else:
                                self.bridge_text = self.text[:self._new_words()[1]]
                            self._retarget()
                        else:
                            self._start_chunk(final, final_text)
                continue
            goal = final if go and has_text else target
            if not goal or goal == done:
                continue
            n = self._prefix_len(done, target)
            # Edits in the middle wait for a pause: recomputing the tail on every key would be a waste.
            # The tail read at the last pause (end of the template, held-back tokens) does not count.
            burst = n >= len(done) - self.tail_slack and len(target) - n >= self.min_new
            if burst or go:
                if burst and not go:
                    goal = target
                with self.slot_lock:
                    if self.generating or target is not self.target:
                        continue
                    self.busy = True
                    try:
                        self.eng.sync(goal)
                    except Exception as e:     # llama-server gone: say so in the window
                        self.state, self.error = "error", f"llama-server stopped answering ({e})."
                        return
                    finally:
                        self.busy = False
                # Reasoning starts right away on the text just read, even if the user has typed on
                # meanwhile: on a slow model that read takes longer than a keystroke, and waiting for
                # the cache to catch up with the text would never let a chunk start (measured on the
                # 14B: 0.8 s to reread the frame after every word, no reasoning ahead at all).
                if (goal is final and go and self.think and self.ahead and self.notes and not self.answer_ahead
                        and self._ahead_room() and final != self._failed):
                    with self.spec_lock:
                        if not self.generating and self.spec is None:
                            self._start_chunk(final, final_text)

    def status(self) -> dict:
        s = {"state": self.state, "error": self.error, "model": self.label, "ctx": self.args.ctx,
             "generating": self.generating}
        if self.state != "ready":
            return s
        final, target, cache = self.final_ids, self.target, self.eng.cached_tokens
        history, notes = len(self.history), len(self.notes)
        message = len(final) - history - notes
        reused = min(self._prefix_len(cache, final), len(final) - 1)
        spec = self.spec
        thought, answer = self._split_ahead(list(spec.tokens) if spec else [])
        holds = self.text.rstrip() == self.notes_text.rstrip()
        answer_ahead = len(self.answer_ahead) if holds else 0
        message -= answer_ahead
        # Between two chunks the slot is free for a few milliseconds: that is not a stop.
        writing = bool(spec and not spec.done) or (
            spec is None and time.perf_counter() - self.chunk_t < 0.5 and self._ahead_room())
        finished = bool(spec and spec.done and not spec.cancelled and not spec.hit_limit
                        and not spec.info.get("error") and answer)
        s.update(has_text=bool(self.text.strip()), busy=self.busy, context=len(final),
                 message_tokens=message, read=min(message, max(0, reused - history)),
                 left=max(0, len(final) - reused), pending=max(0, len(target) - self._prefix_len(cache, target)),
                 strike=bool(self.ahead), think=self.think, ahead=answer_ahead + answer,
                 ahead_writing=writing, thinking_ahead=notes + thought,
                 answer_ready=finished or (holds and self._answer_complete()))
        g = self.gatherer
        if g is not None and g.state == "ready":
            gspec = g.spec
            s.update(gatherer=g.label, ahead_writing=bool(gspec and not gspec.done) or (
                gspec is None and time.perf_counter() - g.chunk_t < 0.5 and g._ahead_room()))
        return s

    def _split_ahead(self, tokens: list[int]) -> tuple[int, int]:
        """How many of `tokens`, written after the current prompt, are reasoning and how many answer."""
        if not self.think or self.answer_ahead:
            return 0, len(tokens)
        cut = tokens.index(self.think_end) if self.think_end in tokens else len(tokens)
        return cut, len(tokens) - cut

    # ---- send ----------------------------------------------------------------------
    def send(self, text: str):
        """Answer `text`, as a stream of events for the page."""
        if self.state != "ready":
            yield {"type": "error", "message": "The model is not ready yet."}
            return
        t_enter = time.perf_counter()
        self.stop.clear()
        with self.spec_lock:                       # from here on no answer ahead can start
            self.generating = True
            spec, self.spec = self.spec, None
        try:
            yield from self._answer(text, t_enter, spec)
        finally:
            self.notes, self.answer_ahead, self.notes_fresh, self.notes_text = [], [], 0, ""
            if self.gatherer is not None:
                self.gatherer.next_turn()
                self._g_notes = self.gatherer.notes
            self.generating = False
            self._retarget()

    def _answer(self, text: str, t_enter: float, spec: Stream | None):
        history = len(self.history)
        final = self._prompt(text)
        reused = min(self._prefix_len(self.eng.cached_tokens, final), len(final) - 1)
        if self.gatherer is not None:
            # The gatherer stops; its material, already read at the pauses, gets the sentence saying the
            # message is complete, and this model reasons on from there.
            self.gatherer.take_notes(text)
            self._read_material(sent=True)
            final = self._prompt(text)
        elif self.think and self.notes:
            # The notes were written as for a message still being typed: keep them, say it is complete now.
            if spec is not None:
                spec.cancel()
                if not spec.info.get("error"):
                    self._absorb(spec, changed=spec.text.rstrip() != text.rstrip(), bridge_after=10 ** 9)
                spec = None
            if text.rstrip() != self.notes_text.rstrip():
                self.answer_ahead = []
            if len(self.notes) <= len(self.seed_ids):
                self.notes = []                    # nothing but the frame: no reasoning ahead after all
            elif not self.answer_ahead:
                if self.last_bridge and self.notes[-len(self.last_bridge):] == self.last_bridge:
                    self.notes = self.notes[:-len(self.last_bridge)]
                self.notes = self.notes + self.sent_ids
            final = self._prompt(text)
        if spec is not None and (spec.prompt != final or spec.cancelled or spec.info.get("error")):
            spec.cancel()                          # written for another text: it does not hold
            if self.think and not spec.info.get("error"):
                self._absorb(spec, changed=True, bridge_after=1)   # but its reasoning does
                final = self._prompt(text)
            spec = None
        room = self.args.ctx - len(final) - 8
        if room < 32:
            yield {"type": "error", "message": f"The conversation no longer fits in the context "
                                               f"({self.args.ctx} tokens). Start a new chat."}
            return
        notes = list(self.notes)
        answer_pre = self.answer_ahead if text.rstrip() == self.notes_text.rstrip() else []
        limit = min(self.args.max_tokens or room, room) - len(answer_pre)
        thought, answer = self._split_ahead(list(spec.tokens) if spec else [])
        thinking = self.think and not answer_pre
        ttft = answer_ms = None
        state = {"thinking": thinking, "started": False}

        def pieces(piece: str):
            """Reasoning and answer events for a piece of text, leading blank lines of the answer dropped."""
            if state["thinking"]:
                head, end, piece = piece.partition("</think>")
                head = head.replace("<think>", "")
                if head:
                    yield {"type": "think", "text": head if state["started"] else head.lstrip()}
                    state["started"] = True
                if not end:
                    return
                state["thinking"], state["started"] = False, False
            if not state["thinking"] and "</think>" in piece:
                piece = piece.split("</think>", 1)[1]
            if not state["started"]:
                piece = piece.lstrip()
            if piece:
                state["started"] = True
                yield {"type": "token", "text": piece}

        # What was written ahead first, all at once: the notes, then the answer if it holds.
        detok = Detokenizer(self.tok, self.tok_lock, self.end_id)
        pre = detok.push(notes + answer_pre)
        if pre:
            state["thinking"] = self.think
        for ev in pieces(pre):
            if ev["type"] == "token" and answer_ms is None:
                answer_ms = (time.perf_counter() - t_enter) * 1000
            yield ev
        # The answer written ahead, if it holds; otherwise a new one (the stream waits for a prefill still running).
        # An answer ahead that reached its end is complete: nothing is left to generate.
        complete = bool(answer_pre) and answer_pre[-1] == self.end_id
        stream = None if complete else spec or Stream(self, final, limit, text)
        first, generated, stopped = stream, [], False
        while stream is not None:
            for tokens in stream.follow(self.stop):
                if ttft is None:
                    ttft = (time.perf_counter() - t_enter) * 1000
                for ev in pieces(detok.push(tokens)):
                    if ev["type"] == "token" and answer_ms is None:
                        answer_ms = (time.perf_counter() - t_enter) * 1000
                    yield ev
            if self.stop.is_set():
                stream.cancel()
                stopped = True
            stream.wait()
            generated += stream.tokens
            if stream.info.get("error"):
                yield {"type": "error", "message": f"llama-server stopped answering ({stream.info['error']})."}
                break
            if stopped or not stream.hit_limit or len(generated) >= limit:
                break
            # The answer ahead stopped at --ahead tokens: carry on from there, with the slot's cache already holding it.
            stream = Stream(self, final + generated, limit - len(generated), text)
        written = answer_pre + generated
        sep = self.between_ids[1:] if written and written[-1] == self.end_id else self.between_ids
        self.history = final + generated + sep
        if first is not None and "cache_n" in (first.info.get("timings") or {}):
            reused = first.info["timings"]["cache_n"]   # what the server actually reused
        reasoning = len(notes) + (self._split_ahead(generated)[0] if not answer_pre else 0)
        t = (stream.info.get("timings") or {}) if stream is not None else {}
        message = len(final) - history - len(notes) - len(answer_pre)
        yield {"type": "done", "ttft_ms": round(ttft if ttft is not None else answer_ms or 0),
               "answer_ms": round(answer_ms or 0), "tokens": len(answer_pre) + len(generated),
               "tps": round(t.get("predicted_per_second", 0), 1),
               "message_tokens": message, "read": min(message, max(0, reused - history)),
               "left": max(0, len(final) - reused), "ahead": max(0, len(answer_pre) - self.think) + answer,
               "think_tokens": reasoning if self.think else 0,
               "think_ahead": len(notes) + thought if self.think else 0,
               "cut": stream is not None and stream.hit_limit and len(generated) >= limit, "stopped": stopped}

    def new_chat(self):
        with self.spec_lock:
            self.generating = True                 # no answer ahead may start meanwhile
            spec, self.spec = self.spec, None
        if spec is not None:
            spec.cancel()
        try:
            with self.slot_lock:
                if self.state == "ready":
                    self.history = list(self.start_ids)
                    self.notes, self.answer_ahead, self.notes_fresh, self.notes_text = [], [], 0, ""
                    if self.gatherer is not None:
                        self.gatherer.take_notes("")
                        self.gatherer.next_turn()
                        self._g_notes = self.gatherer.notes
                    self._retarget()
        finally:
            self.generating = False


def run(args) -> int:
    """mafdet watch / strike: the window (mafdet gui), loading the model given by this command's options."""
    if args.backend != "llama":
        raise SystemExit("mafdet watch runs on llama.cpp: drop --backend hf.")
    from .gui import run as run_window
    return run_window(args, overrides=args)
