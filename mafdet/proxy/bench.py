"""Measure what the proxy is for: the wait between Enter and the first token.

Two runs per prompt, same engine, same sampling, same model:
  OFF  -- straight to the engine, as the app does today;
  ON   -- the message is posted to /v1/drafts as the user types it, then committed
          through /v1/chat/completions.

Before each run the engine is left holding the system prompt and an unrelated message
(the state after a previous turn), so OFF and ON start from the same cache.

Read-ahead never changes the prompt, so the final answers must agree; the wording can
differ when llama.cpp computes a warm prefix in other batches (see engine_stability).
A different final number makes the run fail loudly.

Example:
    python -m mafdet.proxy.bench --url http://127.0.0.1:8081 --model Qwen3-4B-Q4_K_M \\
        --prompts 24 --typing-cps 12 --append-log research/LOG.md
"""
from __future__ import annotations

import argparse
import http.client
import json
import re
import statistics
import sys
import threading
import time

from .adapters import make
from .probe import probe
from .profile import EngineProfile
from .server import ProxyState, make_server

SYSTEM = "You answer arithmetic word problems with the final number on its own line."

# Twenty-four problems of the kind the app's README uses: one or two steps, a clear
# answer that can be checked.  Fixed list so two machines measure the same thing.
PROBLEMS = [
    "Luca has 25 euros, buys 3 notebooks at 4 euros each and a pen for 2 euros. How much is left?",
    "A train leaves at 9:40 and takes 1 hour 35 minutes. What time does it arrive?",
    "There are 7 boxes with 12 pencils each, and 15 pencils are removed. How many remain?",
    "A rectangle is 14 cm long and 9 cm wide. What is its perimeter?",
    "Marta reads 18 pages a day for 6 days, then 12 more pages. How many pages in total?",
    "A shop sells 45 apples in the morning and 38 in the afternoon, 17 are returned. How many sold?",
    "Three friends split a 72 euro bill equally and leave a 9 euro tip. What does each pay?",
    "A tank holds 120 litres and loses 15 litres an hour for 4 hours. How much is left?",
    "A cyclist rides 23 km, then 19 km, then rests. How far has she gone?",
    "A box has 8 rows of 9 tiles. 14 tiles are broken. How many are usable?",
    "A worker earns 14 euros an hour for 7 hours and spends 30 euros. How much is left?",
    "A garden is 6 m by 11 m. Fencing costs 5 euros per metre. What is the total cost?",
    "There are 94 chairs in a hall and 27 are taken away, then 12 brought back. How many now?",
    "Two pipes fill a pool: one adds 8 litres a minute, the other 5, for 9 minutes. How many litres?",
    "A book has 260 pages and 74 have been read. How many are left?",
    "A bus carries 52 people, 19 get off and 24 get on. How many are on board?",
    "A farmer collects 17 eggs a day for 5 days and drops 6. How many eggs remain?",
    "A jacket costs 68 euros and is discounted by 25 percent. What is the new price?",
    "A runner does 4 laps of a 450 m track. How many metres in total?",
    "A jar holds 64 sweets split among 8 children, and 3 sweets are left over. How many each?",
    "A printer prints 22 pages a minute for 6 minutes, then jams. How many pages printed?",
    "A hotel has 9 floors with 14 rooms each, and 21 rooms are empty. How many are occupied?",
    "A phone costs 240 euros paid over 8 months with no interest. What is the monthly payment?",
    "A baker makes 36 loaves, sells 27 and bakes 15 more. How many loaves now?",
]

LONG = "The following notes describe every sandbar and channel of the delta in detail. "

# Sent before each run so that OFF and ON start from the same cache (see reset() in main).
RESET = "Reply with OK."


# --------------------------------------------------------------------------------------
# HTTP helpers (http.client, like the rest of the project)
# --------------------------------------------------------------------------------------
def _chat_ttft(host: str, port: int, body: dict, *, max_tokens: int, timeout: float = 600.0,
               cache_prompt: bool = True):
    """Send a streaming chat request, return (seconds to first token, full text, finish_reason)."""
    payload = dict(body, stream=True, max_tokens=max_tokens, temperature=0.0, top_p=1.0,
                   cache_prompt=cache_prompt)
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    # The clock starts at Enter, before the request goes out: a proxy that holds the
    # request back (a warm still running) sends its headers late, and that wait counts.
    t0 = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", body=json.dumps(payload).encode(),
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    if resp.status >= 400:
        raw = resp.read().decode(errors="replace")
        conn.close()
        raise RuntimeError(f"engine answered {resp.status}: {raw[:400]}")
    first = None
    finish = ""
    text: list[str] = []
    try:
        while True:
            line = resp.readline()
            if not line:
                break
            if not line.startswith(b"data:"):
                continue
            data = line[len(b"data:"):].strip()
            if data in (b"[DONE]", b""):
                continue
            try:
                chunk = json.loads(data.decode())
            except ValueError:
                continue
            delta = ((chunk.get("choices") or [{}])[0].get("delta") or {})
            piece = delta.get("content") or ""
            think = delta.get("reasoning_content") or ""
            # A thinking model streams its reasoning before the answer: the first token
            # of either kind is when the wait after Enter ends.
            if (piece or think) and first is None:
                first = time.perf_counter() - t0
            text.append(think)
            text.append(piece)
            finish = ((chunk.get("choices") or [{}])[0].get("finish_reason") or "") or finish
            if finish in ("stop", "length"):
                break
    finally:
        conn.close()
    return (first if first is not None else float("nan")), "".join(text), finish


def _answer(text: str) -> str:
    """The number a word problem answers with: the last number in the reply."""
    found = re.findall(r"-?\d+(?:[.,]\d+)?", text or "")
    return found[-1] if found else (text or "").strip()


def _post_json(host: str, port: int, path: str, body: dict, timeout: float = 60.0):
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    conn.request("POST", path, body=json.dumps(body).encode(),
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    raw = resp.read()
    conn.close()
    try:
        return resp.status, json.loads(raw.decode())
    except ValueError:
        return resp.status, {}


def _type(proxy_host: str, proxy_port: int, conversation_id: str, model: str, text: str,
          cps: float, chunk: int = 4) -> float:
    """Post the message as it would be typed at `cps` characters a second.  Returns seconds."""
    t0 = time.perf_counter()
    step = max(chunk, 1)
    delay = step / max(cps, 0.1)
    for end in range(step, len(text) + step, step):
        piece = text[:end]
        _post_json(proxy_host, proxy_port, "/v1/drafts",
                   {"conversation_id": conversation_id, "model": model,
                    "messages": [{"role": "system", "content": SYSTEM},
                                 {"role": "user", "content": piece}]})
        if end >= len(text):
            break
        time.sleep(delay)
    return time.perf_counter() - t0


# --------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mafdet.proxy.bench", description=__doc__)
    ap.add_argument("--url", required=True, help="llama-server base URL")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", type=int, default=24, help="typed prompts (max 24)")
    ap.add_argument("--typing-cps", type=float, default=12.0, help="characters per second")
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--proxy-port", type=int, default=8077)
    ap.add_argument("--long", type=int, default=0, help="also bench N pasted long prompts")
    ap.add_argument("--settle", type=float, default=0.35, help="pause before committing (a human's last key)")
    ap.add_argument("--append-log", default="", help="append a markdown block to this file")
    ap.add_argument("--json-out", default="", help="write the full JSON report here")
    ap.add_argument("--thinking", action="store_true",
                    help="leave the model's thinking on (default: ask the template to skip it, "
                         "so the first token is the first word of the answer)")
    args = ap.parse_args(argv)

    host, _, port_s = args.url.replace("http://", "").partition(":")
    engine_host, engine_port = host, int(port_s or 8080)
    engine = make("llama", args.url)
    profile = EngineProfile.load(args.url, args.model) or probe(engine, url=args.url, model=args.model)
    profile.save()
    state = ProxyState(engine, profile, args.model)
    server = make_server(state, "127.0.0.1", args.proxy_port)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"proxy up on 127.0.0.1:{args.proxy_port}, tier {profile.tier} "
          f"({profile.spelling}, channel {profile.channel})", file=sys.stderr)

    typed = PROBLEMS[: max(1, min(args.prompts, len(PROBLEMS)))]
    results = {"typed": [], "long": [], "exactness": []}

    def reset() -> None:
        """Leave the engine's one slot holding the system prompt and an unrelated message."""
        body = {"model": args.model,
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": RESET}]}
        if not args.thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        _chat_ttft(engine_host, engine_port, body, max_tokens=1)

    def run_one(prompt: str, conversation_id: str, typed_mode: bool) -> dict:
        body = {"model": args.model,
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": prompt}]}
        if not args.thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        # Both runs start from the same cache: the system prompt held, the message not, as
        # after the previous turn of a chat.  Without this the ON run would find the whole
        # prompt cached by the OFF run just before it, drafts or no drafts.
        reset()
        # OFF: straight at the engine.
        off_ttft, off_text, off_finish = _chat_ttft(engine_host, engine_port, body, max_tokens=args.max_tokens)
        reset()
        # ON: drafts first (typed, or one paste), then commit through the proxy.
        if typed_mode:
            _type("127.0.0.1", args.proxy_port, conversation_id, args.model, prompt, args.typing_cps)
        else:
            _post_json("127.0.0.1", args.proxy_port, "/v1/drafts",
                       {"conversation_id": conversation_id, "model": args.model, "messages": body["messages"]})
        time.sleep(args.settle)
        on_ttft, on_text, on_finish = _chat_ttft("127.0.0.1", args.proxy_port,
                                                 dict(body, conversation_id=conversation_id),
                                                 max_tokens=args.max_tokens)
        return {"off_ttft": off_ttft, "on_ttft": on_ttft,
                "identical": off_text == on_text,
                "same_answer": _answer(off_text) == _answer(on_text),
                "off_finish": off_finish, "on_finish": on_finish,
                "off_text": off_text, "on_text": on_text}

    for i, prompt in enumerate(typed):
        row = run_one(prompt, f"typed-{i}-{time.time():.0f}", typed_mode=True)
        results["typed"].append(row)
        print(f"[typed {i+1}/{len(typed)}] OFF {row['off_ttft']*1000:.0f} ms  "
              f"ON {row['on_ttft']*1000:.0f} ms  identical={row['identical']}", file=sys.stderr)

    for i in range(args.long):
        # ~2k tokens (100 x an 80-char sentence) and ~4k tokens (200 x), plus the question:
        # both stay inside the 8192-token context the server is started with.
        filler = LONG * (100 if i % 2 == 0 else 200)
        prompt = filler + "\nQuestion: " + PROBLEMS[i % len(PROBLEMS)]
        row = run_one(prompt, f"long-{i}-{time.time():.0f}", typed_mode=False)
        results["long"].append(row)
        print(f"[long {i+1}/{args.long}] OFF {row['off_ttft']*1000:.0f} ms  "
              f"ON {row['on_ttft']*1000:.0f} ms  identical={row['identical']}", file=sys.stderr)

    def summarise(rows: list[dict]) -> dict:
        if not rows:
            return {}
        off = [r["off_ttft"] * 1000 for r in rows]
        on = [r["on_ttft"] * 1000 for r in rows]
        return {"n": len(rows),
                "off_mean_ms": round(statistics.mean(off), 0),
                "on_mean_ms": round(statistics.mean(on), 0),
                "off_median_ms": round(statistics.median(off), 0),
                "on_median_ms": round(statistics.median(on), 0),
                "identical": sum(1 for r in rows if r["identical"]),
                "same_answer": sum(1 for r in rows if r["same_answer"]),
                "on_slower": sum(1 for r in rows if r["on_ttft"] > r["off_ttft"])}

    # Control: is the ENGINE itself bit-stable between a cold and a warm prefill?  If it
    # is not, a text difference with the proxy is the engine's chunking, not the proxy.
    def stability(prompt: str) -> dict:
        body = {"model": args.model,
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": prompt}]}
        if not args.thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        _, cold1, _f1 = _chat_ttft(engine_host, engine_port, body, max_tokens=args.max_tokens,
                                   cache_prompt=False)
        _, cold2, _f2 = _chat_ttft(engine_host, engine_port, body, max_tokens=args.max_tokens,
                                   cache_prompt=False)
        _, warm, _f3 = _chat_ttft(engine_host, engine_port, body, max_tokens=args.max_tokens,
                                  cache_prompt=True)
        return {"cold_vs_cold_identical": cold1 == cold2,
                "cold_vs_warm_identical": cold1 == warm,
                "cold_vs_warm_same_answer": _answer(cold1) == _answer(warm)}

    controls = [stability(p) for p in PROBLEMS[:2]]
    audit = {"warms": 0, "mismatches": 0, "estimates": 0, "unknown": 0}
    for sess in state.stats()["sessions"]:
        for key in audit:
            audit[key] += sess.get(key, 0) or 0
    summary = {"typed": summarise(results["typed"]), "long": summarise(results["long"]),
               "engine_stability": controls, "audit": audit,
               "profile": profile.to_dict(), "typing_cps": args.typing_cps,
               "thinking": args.thinking,
               "max_tokens": args.max_tokens, "model": args.model, "url": args.url,
               "when": time.strftime("%Y-%m-%d %H:%M:%S")}

    rows = results["typed"] + results["long"]
    # A reply cut at max_tokens has no final answer to compare; count it as inconclusive
    # rather than as a failure of the proxy.
    comparable = [r for r in rows if r.get("off_finish") != "length" and r.get("on_finish") != "length"]
    ok = all(r["same_answer"] for r in comparable)
    identical = sum(1 for r in rows if r["identical"])
    print(json.dumps(summary, indent=1))
    if len(comparable) < len(rows):
        print(f"NOTE: {len(rows) - len(comparable)}/{len(rows)} rows hit max_tokens in at least "
              f"one run (inconclusive on the final answer).", file=sys.stderr)
    if identical < len(rows):
        print(f"NOTE: {len(rows) - identical}/{len(rows)} answers differ in wording but "
              f"{sum(1 for r in comparable if r['same_answer'])}/{len(comparable)} comparable rows "
              f"agree on the answer; see engine_stability for whether the engine is bit-stable "
              f"across cache states.", file=sys.stderr)
    if not ok:
        print("EXACTNESS FAILED: the proxy changed an answer. Read-ahead must be exact -- "
              "this is a bug, not noise.", file=sys.stderr)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "rows": results}, f, indent=1)

    if args.append_log:
        t, l = summary.get("typed", {}), summary.get("long", {})
        block = (f"\n### proxy read-ahead -- {summary['when']}\n\n"
                 f"- engine: llama.cpp via `mafdet.proxy` (tier {profile.tier}, {profile.spelling}, "
                 f"channel {profile.channel})\n"
                 f"- model: `{args.model}`, `{args.url}`, greedy (`temperature 0`), max_tokens {args.max_tokens}\n"
                 f"- typing: {args.typing_cps} chars/s, one draft every 4 characters, "
                 f"{args.settle*1000:.0f} ms settle before Enter\n"
                 f"- typed prompts: n={t.get('n', 0)}, Enter->first token: **{t.get('off_mean_ms', 'n/a')} ms -> "
                 f"{t.get('on_mean_ms', 'n/a')} ms** (median {t.get('off_median_ms', 'n/a')} -> "
                 f"{t.get('on_median_ms', 'n/a')} ms), identical answers {t.get('identical', 'n/a')}/{t.get('n', 0)}\n")
        if l:
            block += (f"- pasted long prompts: n={l.get('n', 0)}, Enter->first token: "
                      f"**{l.get('off_mean_ms', 'n/a')} ms -> {l.get('on_mean_ms', 'n/a')} ms**, "
                      f"identical answers {l.get('identical', 'n/a')}/{l.get('n', 0)}\n")
        block += ("- run with `python -m mafdet.proxy.bench --url <engine> --model <id> "
                  f"--prompts {t.get('n', 0)} --typing-cps {args.typing_cps}"
                  + (f" --long {l.get('n', 0)}" if l else "") + "`\n")
        with open(args.append_log, "a", encoding="utf-8") as f:
            f.write(block)
        print(f"appended to {args.append_log}", file=sys.stderr)

    server.shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
