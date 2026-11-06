"""Fake engines behind real HTTP, for the OpenAI-compatible adapter: what vLLM and Ollama
answer on the routes the adapter uses, as their documentation describes them, with a word
tokenizer and a prefix cache in place of a model.

  * FakeVLLM: /version, /tokenize (a prompt, or the messages), /v1/completions with token ids
    (max_tokens 0 refused, as vLLM does), usage.prompt_tokens_details.cached_tokens in blocks of
    16 tokens, a cache shared by every request, /metrics with the hits counter.
  * FakeOllama: /api/version, no tokenizer, /api/chat with prompt_eval_count (the whole prompt)
    and prompt_eval_duration (1 ms per token computed), one cache per slot (`parallel` of them,
    the prompt going to the slot it extends, else to the least recently used one).

Each test can set the knobs that make one behaviour (`report_cached`, `metrics`, `parallel`,
`counts_computed`) and reads back what the adapter sent (`requests`, `headers`).
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from mafdet.text import common_prefix_len

BLOCK = 16


class _Vocab:
    def __init__(self):
        self.ids: dict[str, int] = {}

    def tokenize(self, text: str) -> list[int]:
        return [self.ids.setdefault(w, len(self.ids) + 1) for w in text.split()]

    def render(self, messages, template_kwargs=None, tools=None) -> list[int]:
        text = " ".join(f"{m.get('role')}:{m.get('content')}" for m in messages or [])
        if tools:
            text = "tools:" + ",".join(t.get("function", {}).get("name", "") for t in tools) + " " + text
        if (template_kwargs or {}).get("enable_thinking") is False:
            text += " <think></think>"
        return self.tokenize(text + " assistant:")


class FakeVLLM:
    name = "vllm"

    def __init__(self, *, report_cached: bool = True, metrics: bool = False, model: str = "Qwen/Qwen3-4B"):
        self.vocab = _Vocab()
        self.report_cached = report_cached
        self.metrics = metrics
        self.model = model
        self.cached: list[list[int]] = []          # every prompt seen: the cache is shared
        self.hits = 0                               # tokens hit, for /metrics
        self.requests: list[tuple[str, dict]] = []  # (path, body) of every POST
        self.headers: list[dict] = []

    def _hit(self, ids: list[int]) -> int:
        best = max((common_prefix_len(c, ids) for c in self.cached), default=0)
        best = (best // BLOCK) * BLOCK
        return min(best, max(0, len(ids) - 1))

    def _usage(self, ids: list[int], made: int) -> dict:
        hit = self._hit(ids)
        self.hits += hit
        self.cached.append(list(ids))
        usage = {"prompt_tokens": len(ids), "completion_tokens": made, "total_tokens": len(ids) + made}
        if self.report_cached:
            usage["prompt_tokens_details"] = {"cached_tokens": hit}
        return usage

    def handle(self, method: str, path: str, body: dict, headers: dict):
        if method == "GET" and path == "/version":
            return 200, {"version": "0.11.0"}
        if method == "GET" and path == "/v1/models":
            return 200, {"object": "list", "data": [{"id": self.model, "object": "model"}]}
        if method == "GET" and path == "/metrics":
            if not self.metrics:
                return 404, {"error": "no metrics"}
            text = (f'vllm:prefix_cache_queries_total{{model_name="{self.model}"}} 0.0\n'
                    f'vllm:prefix_cache_hits_total{{model_name="{self.model}"}} {float(self.hits)}\n')
            return 200, text
        if method == "POST" and path == "/tokenize":
            if "messages" in body:
                ids = self.vocab.render(body["messages"], body.get("chat_template_kwargs"), body.get("tools"))
            elif isinstance(body.get("prompt"), str):
                ids = self.vocab.tokenize(body["prompt"])
            else:
                return 400, {"error": "prompt or messages"}
            return 200, {"count": len(ids), "max_model_len": 32768, "tokens": ids}
        if method == "POST" and path == "/v1/completions":
            prompt = body.get("prompt")
            ids = list(prompt) if isinstance(prompt, list) else self.vocab.tokenize(str(prompt or ""))
            if int(body.get("max_tokens") or 0) < 1:
                return 400, {"error": {"message": "max_tokens must be at least 1, got 0."}}
            made = min(int(body["max_tokens"]), 4)
            usage = self._usage(ids, made)
            return 200, {"id": "cmpl-1", "object": "text_completion", "model": self.model,
                         "choices": [{"index": 0, "text": " ok", "finish_reason": "length"}], "usage": usage}
        if method == "POST" and path == "/v1/chat/completions":
            ids = self.vocab.render(body.get("messages") or [], body.get("chat_template_kwargs"), body.get("tools"))
            text = ((body.get("messages") or [{}])[-1].get("content") or "")
            made = 1 if body.get("max_tokens") == 1 else 2
            usage = self._usage(ids, made)
            if body.get("stream"):
                events = [{"choices": [{"delta": {"role": "assistant", "content": "echo: "}, "finish_reason": None}]},
                          {"choices": [{"delta": {"content": text[:20]}, "finish_reason": "stop"}]}]
                if (body.get("stream_options") or {}).get("include_usage"):
                    events.append({"choices": [], "usage": usage})
                return 200, "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
            return 200, {"id": "chatcmpl-1", "object": "chat.completion", "model": self.model,
                         "choices": [{"index": 0, "message": {"role": "assistant", "content": f"echo: {text}"},
                                      "finish_reason": "stop"}], "usage": usage}
        return 404, {"error": f"no route {method} {path}"}


class FakeOllama:
    name = "ollama"

    def __init__(self, *, parallel: int = 1, counts_computed: bool = False, model: str = "qwen3:4b"):
        self.vocab = _Vocab()
        self.parallel = parallel
        self.counts_computed = counts_computed       # prompt_eval_count = only the tokens computed
        self.model = model
        self.slots: list[list[int]] = [[] for _ in range(parallel)]
        self.used = [0] * parallel                   # when each slot was last used
        self.tick = 0
        self.requests: list[tuple[str, dict]] = []
        self.headers: list[dict] = []

    def _process(self, ids: list[int]) -> int:
        """Ollama's rule (runner/common/cache.go): the prompt goes to the slot it extends whole; else
        to the least recently used slot, with the longest common prefix copied there.  Returns
        the tokens computed."""
        best = max(range(self.parallel), key=lambda i: common_prefix_len(self.slots[i], ids))
        common = common_prefix_len(self.slots[best], ids)
        if common != len(self.slots[best]) or not self.slots[best]:
            best = self.used.index(min(self.used))
        common = min(common, max(0, len(ids) - 1))
        self.slots[best] = list(ids)
        self.tick += 1
        self.used[best] = self.tick
        return len(ids) - common

    def handle(self, method: str, path: str, body: dict, headers: dict):
        if method == "GET" and path == "/api/version":
            return 200, {"version": "0.12.6"}
        if method == "GET" and path == "/v1/models":
            return 200, {"object": "list", "data": [{"id": self.model, "object": "model"}]}
        if method == "POST" and path == "/api/chat":
            ids = self.vocab.render(body.get("messages") or [], None, body.get("tools"))
            computed = self._process(ids)
            n = computed if self.counts_computed else len(ids)
            return 200, {"model": self.model, "message": {"role": "assistant", "content": "x"}, "done": True,
                         "prompt_eval_count": n, "prompt_eval_duration": (5 + computed) * 1_000_000,
                         "eval_count": 1, "eval_duration": 20_000_000}
        if method == "POST" and path == "/v1/chat/completions":
            ids = self.vocab.render(body.get("messages") or [], None, body.get("tools"))
            computed = self._process(ids)
            text = ((body.get("messages") or [{}])[-1].get("content") or "")
            usage = {"prompt_tokens": computed if self.counts_computed else len(ids), "completion_tokens": 2,
                     "total_tokens": len(ids) + 2}
            if body.get("stream"):
                events = [{"choices": [{"delta": {"role": "assistant", "content": "echo: "}, "finish_reason": None}]},
                          {"choices": [{"delta": {"content": text[:20]}, "finish_reason": "stop"}]}]
                return 200, "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
            return 200, {"id": "chatcmpl-1", "object": "chat.completion", "model": self.model,
                         "choices": [{"index": 0, "message": {"role": "assistant", "content": f"echo: {text}"},
                                      "finish_reason": "stop"}], "usage": usage}
        return 404, {"error": f"no route {method} {path}"}


class FakeLoading:
    """A server still loading its model: 503 on every route, as llama-server answers until then."""
    name = "loading"
    model = ""

    def __init__(self):
        self.requests: list = []
        self.headers: list = []

    def handle(self, method, path, body, headers):
        return 503, {"error": {"message": "Loading model", "code": 503}}


class FakeLocked:
    """An OpenAI-compatible server that wants a key: 401 without one, a model list with it."""
    name = "locked"
    model = "m"

    def __init__(self):
        self.requests: list = []
        self.headers: list = []

    def handle(self, method, path, body, headers):
        if headers.get("authorization") != "Bearer secret":
            return 401, {"error": {"message": "Unauthorized"}}
        if method == "GET" and path == "/v1/models":
            return 200, {"object": "list", "data": [{"id": self.model}]}
        return 404, {"error": f"no route {method} {path}"}


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    fake = None

    def log_message(self, format, *args):
        pass

    def _serve(self, method: str):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw.decode() or "{}")
        except ValueError:
            body = {}
        headers = {k.lower(): v for k, v in self.headers.items()}
        if method == "POST":
            self.fake.requests.append((self.path, body))
            self.fake.headers.append(headers)
        status, payload = self.fake.handle(method, self.path, body, headers)
        if isinstance(payload, str):
            data = payload.encode()
            ctype = "text/event-stream" if payload.startswith("data:") else "text/plain; version=0.0.4"
        else:
            data = json.dumps(payload).encode()
            ctype = "application/json"
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):   # noqa: N802
        self._serve("GET")

    def do_POST(self):  # noqa: N802
        self._serve("POST")


def start(fake):
    """Serve `fake` on a free port: (server, url)."""
    handler = type("FakeHandler", (_Handler,), {"fake": fake})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"
