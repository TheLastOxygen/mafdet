"""Does reasoning ahead on a partial message hurt the answers, and how long is the wait after Enter?
24 word problems with a known number, typed in real time (thinking mode), with and without reasoning ahead.

The README's "Reasoning ahead, Qwen3-4B" table comes from this script (data/thinking_ahead_4b.json), the
--branches one too (data/branches_4b.json), the Mac one data/thinking_ahead_4b_m2pro.json ("no ahead") and
data/thinking_ahead_4b_m2pro_fixed.json; the 14B runs use the six problems below and a warm-up answer:

    python research/thinking_ahead.py research/data/thinking_ahead_4b.json
    python research/thinking_ahead.py out.json --config "branches 1=--branches 1" --problems 0:12
    python research/thinking_ahead.py out.json --gguf Qwen/Qwen3-14B-GGUF:Qwen3-14B-Q4_K_M.gguf         --max-tokens 1800 --warm-up --problems 0,4,6,13,19,23 --config "reason ahead="

Runs longer than about ten minutes are better split with --problems and --append.
"""
import argparse
import json
import random
import re
import shlex
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mafdet.cli import build_parser
from mafdet.watch import Session

CPS = 8.0
PROBLEMS = [
    ("Luca ha 25 euro, compra 3 quaderni da 4 euro e una penna da 2 euro. Quanti euro gli restano?", ["11"]),
    ("Un rettangolo ha il perimetro di 30 cm e la base lunga il doppio dell'altezza. Qual è la sua area in cm²?", ["50"]),
    ("In una classe ci sono 32 studenti; i tre quarti hanno un cane e di questi la metà ha anche un gatto. Quanti studenti hanno sia un cane sia un gatto?", ["12"]),
    ("Una bici costa 240 euro; durante i saldi viene scontata del 15% e poi di altri 10 euro. Quanto costa alla fine?", ["194"]),
    ("Marta legge 18 pagine al giorno di un libro di 400 pagine. Dopo 12 giorni, quante pagine le mancano?", ["184"]),
    ("Un'auto consuma 6 litri ogni 100 km e il serbatoio contiene 45 litri. Quanti km può percorrere con un pieno?", ["750"]),
    ("Ho comprato 4 pacchi da 6 bottiglie d'acqua, ne ho bevute 9 e ne ho regalate 5. Quante bottiglie mi restano?", ["10"]),
    ("Un rubinetto riempie una vasca da 120 litri in 40 minuti. Se ne apro un secondo identico, in quanti minuti si riempie la vasca?", ["20"]),
    ("Anna ha il triplo dell'età di suo figlio e tra 10 anni avrà il doppio della sua età. Quanti anni ha Anna oggi?", ["30"]),
    ("Un negozio vende penne a 1,20 euro l'una oppure 3 euro per una confezione da 3. Quanto spendo al minimo per 10 penne?", ["10,20", "10.20", "10,2", "10.2"]),
    ("Un treno percorre 360 km in 3 ore e un'auto fa lo stesso tragitto a 80 km/h. Quanti minuti in più impiega l'auto?", ["90"]),
    ("In un sacchetto ci sono 5 palline rosse, 3 blu e 2 verdi. Ne aggiungo 4 blu. Quante palline non sono rosse?", ["9"]),
    ("Un giardino quadrato ha il lato di 12 metri e voglio recintarlo lasciando un cancello largo 2 metri. Quanti metri di rete servono?", ["46"]),
    ("Paolo risparmia 15 euro a settimana e vuole comprare una console da 390 euro, avendo già 120 euro. Quante settimane gli servono?", ["18"]),
    ("Un libro costa 18 euro e un fumetto costa un terzo del libro. Compro 2 libri e 3 fumetti: quanto spendo?", ["54"]),
    ("Una scala ha 15 gradini e ogni gradino è alto 18 cm. Salgo la scala 4 volte: di quanti metri sono salito in totale?", ["10,8", "10.8"]),
    ("Di giorno una lumaca sale di 3 metri e di notte scivola giù di 2 metri. In quanti giorni esce da un pozzo profondo 10 metri?", ["8"]),
    ("Un orologio va avanti di 3 minuti ogni ora e lo metto all'ora giusta a mezzogiorno. Che ora segna alle 8 di sera dello stesso giorno?", ["20:24", "8:24"]),
    ("Ho 50 euro, ne spendo il 40% al ristorante e poi un quarto di quello che resta al cinema. Quanti euro mi rimangono?", ["22,5", "22.5"]),
    ("In un parcheggio ci sono auto e moto: 30 veicoli in tutto e 100 ruote. Quante sono le moto?", ["10"]),
    ("Un pasticcere usa 250 grammi di farina per 12 biscotti. Quanti grammi di farina servono per 30 biscotti?", ["625"]),
    ("Una piscina è lunga 25 metri e Giulia nuota 36 vasche. Quanti chilometri ha nuotato?", ["0,9", "0.9"]),
    ("Io e i miei due fratelli: Marco ha 4 anni più di me, Luca 2 anni meno di me, e la somma delle nostre età è 41. Quanti anni ho?", ["13"]),
    ("Un cinema ha 18 file da 24 posti e stasera sono occupati i due terzi dei posti. Quanti posti sono liberi?", ["144"]),
]
CONFIGS = {"no ahead": ["--ahead", "0"], "reason ahead": []}


def correct(answer: str, expect: list[str]) -> bool:
    tail = answer[-250:].replace("{,}", ",")       # LaTeX decimal comma: 10{,}8
    return any(re.search(rf"(?<![\d,.]){re.escape(e)}(?![\d])", tail) for e in expect)


def wait(c, t=1800):
    t0 = time.perf_counter()
    while not c():
        assert time.perf_counter() - t0 < t
        time.sleep(0.02)


def type_message(s, msg, rng):
    for i in range(1, len(msg) + 1):
        s.draft(msg[:i])
        time.sleep(1 / CPS)
        if msg[i - 1] in ",.?;:" and i < len(msg):
            time.sleep(rng.uniform(0.6, 1.2))
        elif msg[i - 1] == " " and rng.random() < 0.1:
            time.sleep(rng.uniform(0.4, 1.0))
    time.sleep(0.1)


def summary(rows):
    for name in dict.fromkeys(x["config"] for x in rows):
        r = [x for x in rows if x["config"] == name]
        line = (f"{name:12s} correct {sum(x['ok'] for x in r)}/{len(r)} | mean wait for the answer "
                f"{sum(x['answer_ms'] for x in r) / len(r) / 1000:.2f} s, median "
                f"{statistics.median(x['answer_ms'] for x in r) / 1000:.2f} s | reasoning "
                f"{sum(x['think_tokens'] for x in r) / len(r):.0f} tok, {sum(x['think_ahead'] for x in r) / len(r):.0f} ahead")
        adopted = [x["branch"]["adopted"] for x in r if x.get("branch")]
        if adopted:
            line += f" | branch adopted: {adopted.count('exact')} exact, {adopted.count('near')} near, of {len(adopted)}"
        print(line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", help="JSON file for the rows (with --append, the rows already there are kept)")
    ap.add_argument("--config", action="append", default=[],
                    help="name=options of mafdet strike, e.g. 'branches 1=--branches 1' (default: the two of the README)")
    ap.add_argument("--problems", default=":", help="a slice of the problems, e.g. 0:12, or a list, e.g. 0,4,6 "
                                                     "(default: all)")
    ap.add_argument("--gguf", default="", help="the model, as for mafdet strike (default: its default, Qwen3-4B)")
    ap.add_argument("--max-tokens", default="3000")
    ap.add_argument("--warm-up", action="store_true", help="answer a greeting first (CUDA graphs, CPU caches)")
    ap.add_argument("--append", action="store_true")
    a = ap.parse_args()
    configs = dict(c.split("=", 1) for c in a.config) if a.config else {k: " ".join(v) for k, v in CONFIGS.items()}
    if ":" in a.problems:
        lo, hi = (int(x) if x else None for x in a.problems.split(":"))
        picked = list(range(len(PROBLEMS)))[lo:hi]
    else:
        picked = [int(x) for x in a.problems.split(",")]
    rows = json.load(open(a.out, encoding="utf-8")) if a.append and Path(a.out).exists() else []
    for name, extra in configs.items():
        model = ["--gguf", a.gguf] if a.gguf else []
        s = Session(build_parser().parse_args(["strike", "--max-tokens", a.max_tokens] + model + shlex.split(extra)))
        wait(lambda: s.state != "loading")
        assert s.state == "ready", s.error
        if a.warm_up:
            s.draft("Ciao")
            list(s.send("Ciao"))
        for k in picked:
            msg, expect = PROBLEMS[k]
            s.new_chat()
            type_message(s, msg, random.Random(k))
            answer, d = "", None
            for ev in s.send(msg):
                if ev["type"] == "token":
                    answer += ev["text"]
                elif ev["type"] == "done":
                    d = ev
            ok = correct(answer, expect)
            rows.append({"config": name, "k": k, "ok": ok, "answer_ms": d["answer_ms"], "think_tokens": d["think_tokens"],
                         "think_ahead": d["think_ahead"], "cut": d["cut"], "closed": d.get("closed", False), "tps": d["tps"],
                         "tail": answer[-160:],
                         **({"branch": d["branch"]} if "branch" in d else {})})
            b = d.get("branch")
            print(f"{name:12s} #{k:2d} {'OK ' if ok else 'BAD'} answer after {d['answer_ms']:5d} ms | reasoning "
                  f"{d['think_tokens']:4d} ({d['think_ahead']:4d} ahead){' CUT' if d['cut'] else ''}"
                  f" | {d['tps']} tok/s"
                  + (f" | branch {b['adopted'] or '-'} {b['agreement']:.2f} of {b['guesses']} guesses" if b else ""),
                  flush=True)
            json.dump(rows, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        s.close()
    summary(rows)
