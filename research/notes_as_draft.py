"""Could the reasoning written ahead serve as the draft for speculative decoding after Enter?

After Enter the model reads its notes and often works the same steps out again. If it does, the tokens
it writes repeat stretches of the prompt, and prompt-lookup decoding (an n-gram of the last tokens
looked up in the context, the tokens that followed it proposed as a draft) would accept several of them
per step. This runs the word problems as research/thinking_ahead.py does, keeps each final prompt and
what the model wrote after Enter, and simulates the lookup offline. A draft token counts only if it is
the very token the model sampled: the same rule as llama.cpp's speculative decoding.

    python research/notes_as_draft.py out.json --problems 0:12 --config "reason ahead=" --config "no ahead=--ahead 0"
    python research/notes_as_draft.py out.json --analyse
"""
import argparse
import json
import random
import statistics
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mafdet import watch
from mafdet.cli import build_parser
from thinking_ahead import PROBLEMS, correct, type_message, wait


def lookup(context: list[int], generated: list[int], n_min: int = 2, n_max: int = 4, k: int = 8,
           prompt_only: bool = False) -> tuple[int, int]:
    """Prompt-lookup decoding replayed on `generated`: (decoding steps, tokens written).

    At each step the last n tokens (n_max down to n_min) are looked up in the text so far, the
    latest earlier occurrence wins, and up to k tokens that followed it are proposed; the step writes
    the accepted ones plus the one the model computes itself.
    """
    steps, i = 0, 0
    while i < len(generated):
        text = context + generated[:i]
        source = context if prompt_only else text
        draft = []
        for n in range(n_max, n_min - 1, -1):
            if len(text) < n:
                continue
            tail = text[-n:]
            for j in range(len(source) - n - 1, -1, -1):
                if source[j:j + n] == tail and j + n < len(source):
                    draft = source[j + n:j + n + k]
                    break
            if draft:
                break
        accepted = 0
        while accepted < len(draft) and i + accepted < len(generated) and draft[accepted] == generated[i + accepted]:
            accepted += 1
        i += accepted + 1
        steps += 1
    return steps, len(generated)


def analyse(rows: list[dict]):
    for name in dict.fromkeys(r["config"] for r in rows):
        rs = [r for r in rows if r["config"] == name]
        for label, prompt_only in (("prompt and own output", False), ("prompt only", True)):
            ratios, gained = [], 0
            for r in rs:
                steps, n = lookup(r["prompt"], r["generated"], prompt_only=prompt_only)
                ratios.append(n / steps)
                gained += n - steps
            total = sum(len(r["generated"]) for r in rs)
            print(f"{name:14s} draft from {label:22s}: {statistics.mean(ratios):.2f} tokens per step "
                  f"(median {statistics.median(ratios):.2f}, {gained / total:.0%} of the tokens written for free)")
        notes = statistics.mean(r["notes"] for r in rs)
        print(f"{'':14s} {len(rs)} problems, {statistics.mean(len(r['generated']) for r in rs):.0f} tokens after Enter, "
              f"{notes:.0f} tokens of notes, {sum(r['ok'] for r in rs)}/{len(rs)} right")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--problems", default=":")
    ap.add_argument("--gguf", default="")
    ap.add_argument("--max-tokens", default="3000")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--analyse", action="store_true", help="only the analysis of the rows already in `out`")
    a = ap.parse_args()
    rows = json.load(open(a.out, encoding="utf-8")) if (a.append or a.analyse) and Path(a.out).exists() else []
    if not a.analyse:
        if ":" in a.problems:
            lo, hi = (int(x) if x else None for x in a.problems.split(":"))
            picked = list(range(len(PROBLEMS)))[lo:hi]
        else:
            picked = [int(x) for x in a.problems.split(",")]
        # The prompt Enter builds is the last one the sending thread asks for.
        built = {}
        original = watch.Session._prompt

        def recording(self, text):
            ids = original(self, text)
            if threading.current_thread() is threading.main_thread() and self.generating:
                built["prompt"], built["notes"] = list(ids), len(self.notes)
            return ids
        watch.Session._prompt = recording
        for c in a.config or ["reason ahead="]:
            name, extra = c.split("=", 1)
            model = ["--gguf", a.gguf] if a.gguf else []
            s = watch.Session(build_parser().parse_args(["strike", "--max-tokens", a.max_tokens] + model + extra.split()))
            wait(lambda: s.state != "loading")
            assert s.state == "ready", s.error
            for k in picked:
                msg, expect = PROBLEMS[k]
                s.new_chat()
                type_message(s, msg, random.Random(k))
                answer = ""
                for ev in s.send(msg):
                    if ev["type"] == "token":
                        answer += ev["text"]
                prompt = built["prompt"]
                generated = s.history[len(prompt):]
                rows.append({"config": name, "k": k, "ok": correct(answer, expect), "prompt": prompt,
                             "generated": generated, "notes": built["notes"]})
                print(f"{name:14s} #{k:2d} {len(generated):5d} tokens after Enter, {built['notes']:4d} of notes", flush=True)
                json.dump(rows, open(a.out, "w", encoding="utf-8"))
            s.close()
    analyse(rows)
