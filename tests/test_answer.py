"""The answer after Enter, on a scripted llama-server: what was written ahead comes first and is marked as such,
the reasoning is closed when the turn ends inside it or when the thinking budget is spent, and the done event
says what happened."""
import threading
import time
from types import SimpleNamespace

import pytest

from mafdet import watch
from mafdet.watch import Session

END, NL, THINK, LF = 1, 2, 3, 4                   # <|im_end|>, "\n\n", </think>, "\n"
VOCAB = {END: "<|im_end|>", NL: "\n\n", THINK: "</think>", LF: "\n", 5: "<think>\n", 6: "\n\nThe user has now sent ",
         10: "one ", 11: "two ", 12: "three ", 13: "four ", 20: "Answer ", 21: "here. ", 50: "hello",
         30: "�"}                            # a byte of a character the model never completed
ENCODE = {"\n": [LF], "\n\n": [NL], "hello": [50]}
SEED, SENT = [5, 10, 11], [6]                     # the frame; the sentence Enter adds after the notes
NOTES = SEED + [12]                               # the frame and one token of reasoning ahead
HISTORY = [100, 101]                              # the turns before (the answer moves the session's on)


class FakeTokenizer:
    def decode(self, ids):
        return "".join(VOCAB[i] for i in ids)


class FakeStream:
    """What llama-server would write for each prompt, in turn: `scripts` is a list of token lists, one per
    Stream started; the stream ends as the model would (its turn, or the token limit)."""
    scripts: list = []
    started: list = []

    def __init__(self, session, prompt, limit, text="", bias=None, slot=None, sampling=None):
        self.prompt, self.limit, self.text = list(prompt), limit, text
        self.tokens, self.info = [], {}
        self.done = self.cancelled = False
        self.script = list(FakeStream.scripts.pop(0))
        FakeStream.started.append(self)

    @property
    def hit_limit(self):
        return self.info.get("stop_type") == "limit"

    def follow(self, stop):
        while self.script and not self.cancelled and not stop.is_set():
            chunk, self.script = self.script[:4], self.script[4:]
            self.tokens += chunk
            yield chunk
        self.finish()

    def finish(self):
        if not self.done and not self.cancelled:
            stop = "limit" if len(self.tokens) >= self.limit else "eos" if self.tokens[-1:] == [END] else "word"
            self.info = {"stop_type": stop, "timings": {"predicted_per_second": 10.0},
                         "tokens_cached": len(self.prompt) + len(self.tokens)}
        self.done = True

    def cancel(self):
        self.cancelled = True
        self.done = True

    def wait(self):
        self.finish()


def session(notes, budget=0, think=True):
    s = Session.__new__(Session)
    s.think, s.keep_reasoning, s.think_budget = think, False, budget
    s.end_id, s.think_end, s.between_ids, s.suffix = END, THINK, [END, 9, 8], ""
    s.tok, s.tok_lock, s.spec_lock = FakeTokenizer(), threading.Lock(), threading.Lock()
    s._encode = lambda tok, text: ENCODE[text]
    s._prefix_len = lambda a, b: 0
    s.eng = SimpleNamespace(cached_tokens=[])
    s.history, s.notes, s.seed_ids, s.sent_ids, s.last_bridge = list(HISTORY), list(notes), SEED, SENT, []
    s.answer_ahead, s.notes_text = [], ""
    s.n_branches, s.gatherer, s.spec, s.generating = 0, None, None, False
    s.args = SimpleNamespace(ctx=4096, max_tokens=0)
    s.stop, s.typing_t0 = threading.Event(), 0.0
    s.branch_log = {}
    return s


def s_prompt(s):
    """The prompt Enter builds: the history, the message, the notes with the sentence of Enter."""
    return HISTORY + ENCODE["hello"] + NOTES + SENT


@pytest.fixture(autouse=True)
def scripted(monkeypatch):
    monkeypatch.setattr(watch, "Stream", FakeStream)
    FakeStream.scripts, FakeStream.started = [], []


def run(s, scripts):
    FakeStream.scripts, FakeStream.started = [list(x) for x in scripts], []
    events = list(s._answer("hello", time.perf_counter(), None))
    done = next(e for e in events if e["type"] == "done")
    think = [e for e in events if e["type"] == "think"]
    answer = "".join(e["text"] for e in events if e["type"] == "token")
    return events, done, think, answer


def test_reasoning_ahead_is_marked_and_the_sentence_of_enter_is_not():
    """The notes (frame + reasoning ahead, then SENT) come first; the part before SENT carries `ahead`."""
    s = session(NOTES)
    events, done, think, answer = run(s, [[13, THINK, NL, 20, 21, END]])
    assert s.notes == NOTES + SENT                                          # Enter adds the sentence to the notes
    assert [e.get("ahead", False) for e in think] == [True, False, False]
    assert think[0]["text"] == "one two three "                             # the frame's <think> dropped
    assert think[1]["text"] == "\n\nThe user has now sent " and think[2]["text"] == "four "
    assert answer == "Answer here. " and not done["closed"] and not done["budget"]
    assert done["think_ahead"] == 5 and done["think_tokens"] == 6          # 5 notes tokens + "four"
    assert len(FakeStream.started) == 1


def test_turn_ended_inside_the_reasoning_is_closed_and_answered():
    s = session(NOTES)
    events, done, think, answer = run(s, [[13, END], [20, 21, END]])
    assert done["closed"] and not done["budget"]
    second = FakeStream.started[1]
    assert second.prompt == s_prompt(s) + [13, LF, THINK, NL]              # the reasoning closed as the template does,
    assert END not in second.prompt                                        # the end of turn inside it dropped
    assert answer == "Answer here. " and done["think_tokens"] == 5 + 2     # notes + "four" + the newline of the close


def test_thinking_budget_closes_the_reasoning_and_the_model_answers():
    """Past the budget the stream is stopped, the reasoning closed, and a new stream writes the answer."""
    s = session(NOTES, budget=8)
    scripts = [[10, 11, 12, 13] * 10 + [THINK, NL, 20, END], [20, 21, END]]
    events, done, think, answer = run(s, scripts)
    assert done["budget"] and not done["closed"] and not done["cut"] and not done["stopped"]
    first, second = FakeStream.started
    assert first.cancelled and 8 <= len(first.tokens) <= 12               # a chunk past the budget at most
    assert second.prompt == s_prompt(s) + first.tokens + [LF, THINK, NL]
    assert answer == "Answer here. "
    assert done["think_tokens"] == len(NOTES + SENT) + len(first.tokens) + 1


def test_no_budget_lets_the_model_think_to_the_end():
    s = session(NOTES)
    scripts = [[10, 11, 12, 13] * 10 + [THINK, NL, 20, END]]
    events, done, think, answer = run(s, scripts)
    assert not done["budget"] and len(FakeStream.started) == 1 and answer == "Answer "


def test_budget_does_not_touch_an_answer_already_started():
    """Once the model is answering, the budget is nothing: only reasoning counts against it."""
    s = session(NOTES, budget=4)
    scripts = [[10, 11, THINK, NL] + [20, 21] * 6 + [END]]
    events, done, think, answer = run(s, scripts)
    assert not done["budget"] and len(FakeStream.started) == 1 and answer == "Answer here. " * 6


def test_a_character_left_broken_at_the_end_does_not_lose_the_answer():
    """Seen on Qwen3-14B: an emoji with a byte missing right before the end of the turn. While more may come
    the text is held back (the other bytes may follow); at the end it comes out, as a replacement character."""
    s = session(NOTES)
    events, done, think, answer = run(s, [[THINK, NL, 20, 21, 30, END]])
    assert answer == "Answer here. �" and done["tokens"] == 6
    # The same answer written ahead in full before Enter: nothing streams, the flush still delivers it.
    s = session(NOTES)
    s.answer_ahead, s.notes_text = [THINK, NL, 20, 21, 30, END], "hello"
    events, done, think, answer = run(s, [])
    assert answer == "Answer here. �" and not FakeStream.started and done["tokens"] == 6


def test_typing_time_is_reported():
    s = session(NOTES)
    s.typing_t0 = time.perf_counter() - 12.3
    events, done, think, answer = run(s, [[THINK, NL, 20, END]])
    assert 12 <= done["typing_s"] <= 13.5
