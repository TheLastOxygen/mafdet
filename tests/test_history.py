"""The conversation kept between turns: messages and answers, the past turns' reasoning dropped as Qwen3's template does."""
from types import SimpleNamespace

from mafdet.watch import Session

END, NL, THINK, SEP = 1, 2, 3, [1, 9, 8]          # <|im_end|>, "\n\n", </think>, <|im_end|>\n<|im_start|>user\n


def session(think=True, keep=False):
    s = Session.__new__(Session)
    s.think, s.keep_reasoning, s.end_id, s.think_end, s.between_ids = think, keep, END, THINK, SEP
    s.tok = SimpleNamespace(decode=lambda ids: "\n\n" if ids == [NL] else "x")
    return s


HISTORY, MSG = [100, 101], [50, 51]               # turns before; the message with the start of the answer


def test_reasoning_dropped_answer_kept():
    notes = [70, 71, 72]                           # <think> and the reasoning ahead
    generated = [73, 74, THINK, NL, 80, 81, END]   # reasoning after Enter, then the answer
    final = HISTORY + MSG + notes
    assert session()._next_history(final, notes, [], generated) == HISTORY + MSG + [80, 81] + SEP


def test_answer_written_ahead():
    notes, answer_pre, generated = [70], [THINK, NL, 80], [81, END]
    final = HISTORY + MSG + notes + answer_pre
    assert session()._next_history(final, notes, answer_pre, generated) == HISTORY + MSG + [80, 81] + SEP


def test_stopped_inside_the_reasoning_leaves_an_empty_answer():
    notes = [70]
    assert session()._next_history(HISTORY + MSG + notes, notes, [], [71, 72]) == HISTORY + MSG + SEP


def test_keep_reasoning_and_no_think_keep_everything():
    notes, generated = [70], [THINK, NL, 80, END]
    final = HISTORY + MSG + notes
    for s in (session(keep=True), session(think=False)):
        assert s._next_history(final, notes, [], generated) == final + generated + SEP[1:]
