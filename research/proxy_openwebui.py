"""The wait after Enter in Open WebUI, with and without MAFDET's proxy (research/LOG.md, 2026-11-06).

Open WebUI is driven the way its browser frontend does it (POST /api/chat/completions plus a
socket.io session, as Chat.svelte), so its own requests reach the engine exactly as for a person:
its 30 built-in tools, the history it rebuilds, the title, tags and follow-ups it asks for.
Two connections to the same llama-server (the one `mafdet proxy` starts):
  mafdet.<model>  through the proxy, with the drafts the user script would post while typing;
  direct.<model>  straight to llama-server, no drafts, the same typing time and pause.
Run the arms separately (--arm), each with a warm-up conversation (--warmup 1): on one
llama-server they disturb each other, since without the proxy llama-server puts a request on
whichever slot it likes, the proxy's conversation slot included.  Per turn: the time from the POST to the first
token on the socket, and llama-server's own timings for that request (tokens read on Enter,
tokens reused, prompt ms), which Open WebUI relays in its last event.

Setup (see mafdet/proxy/clients/README.md):
  mafdet proxy --model qwen3-4b                       # llama-server on 8078, proxy on 8077
  Open WebUI with OPENAI_API_BASE_URLS='http://127.0.0.1:8077/v1;http://127.0.0.1:8078/v1'
             OPENAI_API_CONFIGS='{"0":{"prefix_id":"mafdet"},"1":{"prefix_id":"direct"}}'
Run with Open WebUI's Python (it needs aiohttp and python-socketio):
  E:/openwebui/venv/Scripts/python.exe research/proxy_openwebui.py --start 0 --count 3 --out research/data/x.json
Runs over ~10 minutes get killed here: split them with --start/--count and --append.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
import uuid

import aiohttp
import socketio

BASE = "http://127.0.0.1:8080"

DOC = ("Il magazzino riceve ogni lunedì 40 casse di mele, 25 di pere e 18 di arance. Ogni cassa di mele "
       "pesa 12 kg, ogni cassa di pere 10 kg, ogni cassa di arance 15 kg. Il martedì si vendono metà delle "
       "mele e un terzo delle pere; il mercoledì arrivano altre 10 casse di arance. ") * 12

# Conversations of three turns: (text, pasted) per turn; a pasted turn is posted as one draft
# (a paste), then the question after it is typed.
CONVERSATIONS = [
    [("Rispondi in breve: Luca ha 25 euro, compra 3 quaderni da 4 euro e una penna da 2. Quanto gli resta?", False),
     ("E se i quaderni costassero 5 euro l'uno?", False),
     ("Con quello che gli resta, quante penne da 2 euro può comprare?", False)],
    [("Rispondi in breve: un treno parte alle 9:40 e viaggia per 1 ora e 35 minuti. A che ora arriva?", False),
     ("E se partisse con 25 minuti di ritardo?", False),
     ("Quanti minuti dura il viaggio in tutto?", False)],
    [(DOC, True), ("\n\nRispondi in breve: quanti kg di frutta arrivano il lunedì?", False),
     ("E quanti kg di mele restano dopo il martedì?", False)],
    [("Rispondi in breve: un rettangolo è lungo 14 cm e largo 9 cm. Qual è il perimetro?", False),
     ("E l'area?", False),
     ("Se raddoppio la lunghezza, di quanto cresce l'area?", False)],
    [("Rispondi in breve: Marta legge 18 pagine al giorno per 6 giorni, poi altre 12. Quante pagine in tutto?", False),
     ("E se il libro ne ha 200, quante ne mancano?", False),
     ("A 18 pagine al giorno, in quanti giorni lo finisce?", False)],
    [(DOC, True), ("\n\nRispondi in breve: quante casse arrivano in tutto il lunedì?", False),
     ("E quante casse di arance ci sono dopo il mercoledì?", False)],
    [("Rispondi in breve: una cisterna contiene 120 litri e ne perde 15 all'ora per 4 ore. Quanto resta?", False),
     ("E dopo altre 3 ore?", False),
     ("Dopo quante ore in tutto è vuota?", False)],
    [("Rispondi in breve: un ciclista percorre 23 km, poi 19 km, poi si ferma. Quanta strada ha fatto?", False),
     ("Se ha impiegato 2 ore, qual è la velocità media?", False),
     ("A quella velocità, quanto ci mette a fare 63 km?", False)],
]


class OpenWebUI:
    def __init__(self, proxy: str):
        self.proxy = proxy
        self.sio = socketio.AsyncClient()
        self.waiting: dict[str, dict] = {}         # response id -> what its events told us

    async def start(self):
        self.http = aiohttp.ClientSession()
        d = await (await self.http.post(f"{BASE}/api/v1/auths/signin",
                                        json={"email": "admin@localhost", "password": "admin"})).json()
        self.token, self.user = d["token"], {"id": d["id"], "name": d["name"]}
        self.h = {"Authorization": f"Bearer {self.token}"}
        self.models = {m["id"]: m for m in (await (await self.http.get(f"{BASE}/api/models", headers=self.h)).json())["data"]}

        @self.sio.on("events")
        async def on_events(data):
            rec = self.waiting.get(data.get("message_id"))
            if rec is None:
                return
            kind, payload = (data.get("data") or {}).get("type"), (data.get("data") or {}).get("data")
            if kind == "response:completion" and isinstance(payload, dict) and "delta" in str(payload.get("type", "")) \
                    and payload.get("delta") and rec["first"] is None:
                rec["first"] = time.perf_counter()
            if kind == "chat:completion" and isinstance(payload, dict):
                if payload.get("usage"):
                    rec["usage"] = payload["usage"]
                if payload.get("done"):
                    rec["done"].set()

        await self.sio.connect(BASE, socketio_path="/ws/socket.io", auth={"token": self.token}, transports=["websocket"])
        await self.sio.emit("user-join", {"auth": {"token": self.token}})
        await asyncio.sleep(0.5)

    async def draft(self, conversation_id: str, text: str):
        try:
            async with self.http.post(f"{self.proxy}/v1/drafts", json={"conversation_id": conversation_id, "text": text}) as r:
                await r.read()
        except aiohttp.ClientError:
            pass

    async def send(self, model_id: str, content: str, chat_id, parent_id, params: dict, first_turn: bool):
        model = self.models[model_id]
        uid, rid = str(uuid.uuid4()), str(uuid.uuid4())
        body = {
            "stream": True, "model": model_id, "params": params, "tool_ids": [], "tool_servers": [],
            "features": {"voice": False, "image_generation": False, "code_interpreter": False, "web_search": False},
            "variables": {}, "model_item": model, "session_id": self.sio.get_sid("/"), "id": rid,
            "message_ids": [{"model_id": model_id, "message_id": rid, "modelIdx": 0}], "parent_id": parent_id,
            "user_message": {"id": uid, "parentId": parent_id, "childrenIds": [rid], "role": "user", "user": self.user,
                             "user_id": self.user["id"], "content": content, "timestamp": int(time.time()),
                             "models": [model_id]},
            "background_tasks": {"title_generation": first_turn, "tags_generation": first_turn,
                                 "follow_up_generation": True},
            **({"chat_id": chat_id} if chat_id else {"chat_variables": {}}),
        }
        rec = {"first": None, "usage": None, "done": asyncio.Event()}
        self.waiting[rid] = rec
        t0 = time.perf_counter()
        res = await (await self.http.post(f"{BASE}/api/chat/completions", json=body, headers=self.h)).json()
        try:
            await asyncio.wait_for(rec["done"].wait(), 300)
        except asyncio.TimeoutError:
            pass
        u = rec["usage"] or {}
        return (chat_id or res.get("chat_id"), rid,
                {"ttft_ms": round((rec["first"] - t0) * 1000, 1) if rec["first"] else None,
                 "read_on_enter": u.get("prompt_n"), "reused": u.get("cache_n"),
                 "prompt_ms": round(u["prompt_ms"], 1) if u.get("prompt_ms") is not None else None,
                 "answer_tokens": u.get("predicted_n")})

    async def close(self):
        await self.sio.disconnect()
        await self.http.close()


async def type_turn(c: OpenWebUI, arm: str, conversation_id: str, typed_before: str, text: str, pasted: bool,
                    cps: float, every: float):
    """The user writes `text` after `typed_before` (what is already in the box): drafts as the user
    script posts them (at most `every` s apart while typing, at once on a paste), or the same time
    spent without drafts."""
    if pasted:
        if arm == "mafdet":
            await c.draft(conversation_id, typed_before + text)
        return
    step = max(1, int(cps * every))
    for end in range(step, len(text) + step, step):
        await asyncio.sleep(len(text[end - step:end]) / cps)      # the time to type this piece
        if arm == "mafdet":
            await c.draft(conversation_id, typed_before + text[:end])


async def run(args):
    c = OpenWebUI(args.proxy)
    await c.start()
    model = {"mafdet": f"mafdet.{args.model}", "direct": f"direct.{args.model}"}
    for arm in (["mafdet", "direct"] if args.arm == "both" else [args.arm]):
        if model[arm] not in c.models:
            await c.close()
            raise SystemExit(f"Open WebUI has no model {model[arm]}: has {sorted(c.models)}")
    params = {"max_tokens": args.max_tokens}
    if not args.think:
        params["custom_params"] = {"chat_template_kwargs": '{"enable_thinking": false}'}
    rows = []
    try:
        # Warm-up conversations first (not counted): the engine's slots then hold what this arm's own
        # traffic leaves there, as in use, and not what the other arm or an earlier run left.
        order = [(i, True) for i in range(args.start - args.warmup, args.start)] +                 [(i, False) for i in range(args.start, args.start + args.count)]
        for i, warmup in order:
            conv = CONVERSATIONS[i % len(CONVERSATIONS)]
            if args.arm != "both":
                arms = [args.arm]
            else:
                arms = ["mafdet", "direct"] if i % 2 == 0 else ["direct", "mafdet"]
            for arm in arms:
                chat_id = parent = None
                box = ""                                      # what is in the message box
                for turn, (text, pasted) in enumerate(conv):
                    cid = chat_id or "new"
                    await type_turn(c, arm, cid, box, text, pasted, args.cps, 0.6)
                    box += text
                    if pasted:                                 # the question is typed after the paste
                        continue
                    await asyncio.sleep(args.settle)
                    chat_id, parent, m = await c.send(model[arm], box, chat_id, parent, params, first_turn=parent is None)
                    row = {"conversation": i % len(CONVERSATIONS), "arm": arm, "warmup": warmup,
                           "turn": sum(1 for t, p in conv[:turn + 1] if not p), "chars": len(box), **m}
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                    box = ""
                    await asyncio.sleep(args.read)             # reading the answer before typing again
    finally:
        await c.close()
    return rows


def summary(rows):
    out = {}
    for arm in ("direct", "mafdet"):
        for turn in sorted({r["turn"] for r in rows}):
            sel = [r for r in rows if r["arm"] == arm and r["turn"] == turn and r["ttft_ms"] is not None
                   and not r.get("warmup")]
            if not sel:
                continue
            t = [r["ttft_ms"] for r in sel]
            out[f"{arm} turn {turn}"] = {
                "n": len(sel), "ttft_mean_ms": round(statistics.mean(t)), "ttft_median_ms": round(statistics.median(t)),
                "read_on_enter_mean": round(statistics.mean(r["read_on_enter"] or 0 for r in sel)),
                "prompt_ms_mean": round(statistics.mean(r["prompt_ms"] or 0 for r in sel))}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen3-4b:Q4_K_M")
    ap.add_argument("--proxy", default="http://127.0.0.1:8077")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=len(CONVERSATIONS))
    ap.add_argument("--arm", default="both", choices=["both", "mafdet", "direct"],
                    help="one arm per run is cleaner: both arms on one llama-server disturb each other "
                         "(without the proxy, llama-server puts requests on either slot)")
    ap.add_argument("--warmup", type=int, default=0, help="conversations run first and not counted")
    ap.add_argument("--cps", type=float, default=12.0, help="typing speed, characters a second")
    ap.add_argument("--settle", type=float, default=1.0, help="pause between the last key and Enter")
    ap.add_argument("--read", type=float, default=3.0, help="pause after an answer before typing again")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--think", action="store_true", help="leave Qwen3's thinking on (default off: shorter runs)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()
    rows = asyncio.run(run(args))
    if args.append and os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            rows = json.load(f)["rows"] + rows
    data = {"summary": summary(rows), "rows": rows,
            "setup": {k: getattr(args, k) for k in ("model", "cps", "settle", "read", "max_tokens", "think")}}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
    print(json.dumps(data["summary"], indent=1))


if __name__ == "__main__":
    main()
