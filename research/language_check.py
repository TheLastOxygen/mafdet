"""English word problems typed in real time with reasoning ahead: is the answer in English?

The frame of the notes (SEED) has an Italian example line, and one English question got its answer in Italian
(Qwen3-4B, M2 Pro, in the app). This counts the answers that are not in English (see research/LOG.md):

    python research/language_check.py 4 research/data/language_4b_m2pro_before.json    # 4 rounds of 3 problems
"""
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thinking_ahead import correct, type_message, wait
from mafdet.cli import build_parser
from mafdet.watch import Session

PROBLEMS = [
    ("A train leaves at 9:15 and arrives at 11:40. On the way it stops 3 times, 5 minutes each time. "
     "For how many minutes is the train actually moving?", ["130"]),
    ("A shop sells pens at $1.20 each or $3 for a pack of 3. What is the least I can spend on 10 pens?", ["10.20", "10.2"]),
    ("Anna is three times as old as her son, and in 10 years she will be twice his age. How old is Anna today?", ["30"]),
]
IT = re.compile(r"\b(il|di|che|della|sono|minuti|anni|risposta|quindi|per|una|delle|totale)\b", re.I)
EN = re.compile(r"\b(the|of|is|minutes|years|answer|so|for|a|total|and)\b", re.I)


def language(text):
    it, en = len(IT.findall(text)), len(EN.findall(text))
    return "it" if it > en else "en"


reps, out = int(sys.argv[1]), sys.argv[2]
s = Session(build_parser().parse_args(["strike", "--max-tokens", "3000"]))
wait(lambda: s.state != "loading")
rows = []
for r in range(reps):
    for k, (msg, expect) in enumerate(PROBLEMS):
        s.new_chat()
        type_message(s, msg, random.Random(100 * r + k))
        answer, d = "", None
        for ev in s.send(msg):
            if ev["type"] == "token": answer += ev["text"]
            elif ev["type"] == "done": d = ev
        row = {"rep": r, "k": k, "lang": language(answer), "ok": correct(answer, expect), "answer_ms": d["answer_ms"],
               "closed": d.get("closed", False), "head": answer[:120]}
        rows.append(row)
        print(json.dumps({x: row[x] for x in ("rep", "k", "lang", "ok", "answer_ms", "closed")}), flush=True)
json.dump(rows, open(out, "w"), ensure_ascii=False, indent=1)
print("italian answers:", sum(r["lang"] == "it" for r in rows), "of", len(rows), "| correct:", sum(r["ok"] for r in rows))
