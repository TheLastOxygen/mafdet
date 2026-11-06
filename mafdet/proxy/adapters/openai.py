"""The OpenAI-compatible adapter: vLLM, SGLang, Ollama, LM Studio, and any server with
`/v1/chat/completions`.

Two ways to read ahead, and the probe picks one (`probe.py`):
  * by token ids, when the engine renders the chat prompt for the proxy (vLLM's `/tokenize` takes
    the messages): `POST /v1/completions` with the ids and `max_tokens` 0 or 1, the reuse read from
    `usage.prompt_tokens_details.cached_tokens` (vLLM with `--enable-prompt-tokens-details`, SGLang
    with `--enable-cache-report`) or from the prefix-cache counters of `/metrics`;
  * through the chat endpoint (Ollama, LM Studio, SGLang without a template endpoint): the draft as a
    chat request with `max_tokens: 1`, and the engine applies its own template.  The proxy then
    cannot count what it warmed; the evidence of reuse is the engine's own count of the prompt tokens
    it computed, or its prefill time (Ollama's `prompt_eval_duration`, LM Studio's
    `time_to_first_token`).

Written against the engines' documented APIs and checked on fakes (tests/proxy_fake_servers.py);
which engine, at which version, was really run through it is the table in PROXY.md.
"""
from __future__ import annotations

import http.client
import json
import re
import time
from typing import Iterable
from urllib.parse import urlsplit

from ..engine import WarmResult

FLAVORS = ("openai", "vllm", "sglang", "ollama", "lmstudio")
NAMES = {"openai": "OpenAI-compatible", "vllm": "vLLM", "sglang": "SGLang", "ollama": "Ollama",
         "lmstudio": "LM Studio"}

# Prometheus counters of prefix-cache hits, in tokens (vLLM: vllm:prefix_cache_hits_total, older
# vllm:gpu_prefix_cache_hits_total), summed over their labels.
_HITS = re.compile(r"^[A-Za-z_]*:?(?:gpu_)?prefix_cache_hits(?:_total)?(?:\{[^}]*\})?\s+([0-9.eE+-]+)", re.M)


def _split(url: str) -> tuple[str, str, int, str]:
    """(scheme, host, port, base path) of an engine URL; a trailing /v1 is the routes', not the base's."""
    if "://" not in url:
        url = "http://" + url
    parts = urlsplit(url)
    scheme = "https" if parts.scheme == "https" else "http"
    host = parts.hostname or "127.0.0.1"
    port = parts.port or (443 if scheme == "https" else 80)
    path = parts.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return scheme, host, port, path


def _ids(data) -> list[int] | None:
    """The token ids of a /tokenize answer ({"tokens": [1, 2, ...]}, or pieces with an "id")."""
    tokens = data.get("tokens") if isinstance(data, dict) else None
    if not isinstance(tokens, list):
        return None
    out = []
    for t in tokens:
        if isinstance(t, bool):
            return None
        if isinstance(t, int):
            out.append(t)
        elif isinstance(t, dict) and isinstance(t.get("id"), int):
            out.append(t["id"])
        else:
            return None
    return out


class OpenAIEngine:
    dialect = "openai"
    ORDER = ("max_tokens0", "echo0", "max_tokens1")      # for token ids; the probe tries them in this order

    def __init__(self, url: str, *, model: str = "", api_key: str = "", flavor: str = "openai",
                 version: str = "", timeout: float = 600.0):
        self.scheme, self.host, self.port, self.base = _split(url)
        self.model = model
        self.api_key = api_key
        self.flavor = flavor if flavor in FLAVORS else "openai"
        self.name = NAMES[self.flavor]
        self.version = version
        self.timeout = timeout
        self._tok_style: str | None = None    # "prompt" (vLLM, SGLang) or "content" (llama-server), once one answered
        self._render: bool | None = None      # False once the engine said it has no endpoint for it (404)
        self._metrics: bool | None = None     # whether /metrics answers; None: not tried yet
        self._hits_seen: float | None = None  # the hits counter after the last warm, for the delta

    def __repr__(self) -> str:
        return f"OpenAIEngine({self.name}, {self.scheme}://{self.host}:{self.port}{self.base}, model {self.model!r})"

    # ---- plumbing ---------------------------------------------------------------
    def _conn(self, timeout: float | None = None):
        cls = http.client.HTTPSConnection if self.scheme == "https" else http.client.HTTPConnection
        return cls(self.host, self.port, timeout=self.timeout if timeout is None else timeout)

    def _headers(self, body: bool = True) -> dict[str, str]:
        headers = {"Content-Type": "application/json"} if body else {}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        return headers

    def _call(self, path: str, body: dict | None = None, method: str = "POST",
              timeout: float | None = None) -> tuple[int, dict, bytes]:
        """(status, parsed JSON object or {}, raw bytes); status 0 when nothing answered."""
        conn = self._conn(timeout)
        try:
            data = json.dumps(body).encode() if body is not None else None
            conn.request(method, self.base + path, body=data, headers=self._headers(data is not None))
            resp = conn.getresponse()
            raw = resp.read()
        except (OSError, http.client.HTTPException):
            return 0, {}, b""
        finally:
            conn.close()
        try:
            parsed = json.loads(raw.decode())
        except (ValueError, UnicodeDecodeError):
            parsed = {}
        return resp.status, parsed if isinstance(parsed, dict) else {}, raw

    # ---- tokens -----------------------------------------------------------------
    def _styles(self) -> tuple[str, ...]:
        return ("content", "prompt") if self._tok_style == "content" else ("prompt", "content")

    def tokenize(self, text: str) -> list[int] | None:
        """vLLM's and SGLang's /tokenize ({"prompt"}), or llama-server's ({"content"})."""
        for style in self._styles():
            if style == "prompt":
                body = {"prompt": text, "add_special_tokens": False}
                if self.model:
                    body["model"] = self.model
            else:
                body = {"content": text, "add_special": False}
            status, data, _ = self._call("/tokenize", body)
            ids = _ids(data) if status == 200 else None
            if ids is not None:
                self._tok_style = style
                return ids
        return None

    def render(self, messages: list[dict], template_kwargs: dict | None = None,
               tools: list | None = None) -> list[int] | None:
        """The ids of the prompt the engine builds for these messages: vLLM's /tokenize takes the
        messages (with the template options and the tools); llama-server behind its OpenAI routes has
        /apply-template.  None when the engine has neither: the drafts then go through the chat
        endpoint, and the engine applies its template itself."""
        if self._render is False:
            return None
        body = {"messages": messages, "add_generation_prompt": True, "add_special_tokens": False}
        if self.model:
            body["model"] = self.model
        if template_kwargs:
            body["chat_template_kwargs"] = template_kwargs
        if tools:
            body["tools"] = tools
        status, data, _ = self._call("/tokenize", body)
        ids = _ids(data) if status == 200 else None
        if ids is not None:
            self._render = True
            return ids
        first = status
        body = {"messages": messages, "add_generation_prompt": True}
        if template_kwargs:
            body["chat_template_kwargs"] = template_kwargs
        if tools:
            body["tools"] = tools
        status, data, _ = self._call("/apply-template", body)
        if status == 200 and data.get("prompt"):
            ids = self.tokenize(data["prompt"])
            if ids is not None:
                self._render = True
                return ids
        if first == 404 and status == 404:
            self._render = False
        return None

    # ---- warms ------------------------------------------------------------------
    def _hits(self) -> float | None:
        """The prefix-cache hits counter of /metrics (tokens), or None without one."""
        if self._metrics is False:
            return None
        text = self.metrics_text()
        if text is None:
            self._metrics = False
            return None
        self._metrics = True
        found = _HITS.findall(text)
        if not found:
            return None
        try:
            return sum(float(v) for v in found)
        except ValueError:
            return None

    def _result(self, usage: dict, ms: float, spelling: str, *, prefill_ms: float | None = None) -> WarmResult:
        """A warm's result from the OpenAI usage block: the exact reuse when the engine says it
        (prompt_tokens_details.cached_tokens), the /metrics delta when it has counters, else only
        the prompt tokens as it counted them (the probe tells whether that is what it computed)."""
        details = usage.get("prompt_tokens_details")
        cached = details.get("cached_tokens") if isinstance(details, dict) else None
        prompt_tokens = usage.get("prompt_tokens")
        if not isinstance(prompt_tokens, int):
            prompt_tokens = None
        detail = {"ok": True, "spelling": spelling, "prompt_tokens": prompt_tokens}
        if prefill_ms is not None:
            detail["prefill_ms"] = prefill_ms
        if isinstance(cached, int) and not isinstance(cached, bool):
            computed = prompt_tokens - cached if prompt_tokens is not None else None
            return WarmResult(cached_tokens=cached, computed_tokens=computed, ms=ms, channel="usage", detail=detail)
        hits = self._hits()
        if hits is not None:
            before, self._hits_seen = self._hits_seen, hits
            if before is not None:
                return WarmResult(cached_tokens=max(0, int(hits - before)), computed_tokens=prompt_tokens, ms=ms,
                                  channel="metrics", estimate=True, detail=detail)
        return WarmResult(cached_tokens=None, computed_tokens=prompt_tokens, ms=ms, channel="none", detail=detail)

    @staticmethod
    def _failed(ms: float, status: int, raw: bytes, why: str = "") -> WarmResult:
        return WarmResult(ms=ms, channel="none",
                          detail={"ok": False, "status": status, "why": why or raw[:200].decode(errors="replace")})

    def warm(self, ids: list[int], *, spelling: str = "auto") -> WarmResult:
        """Prefill token ids through /v1/completions: max_tokens 0 where the engine takes it (SGLang),
        echo with max_tokens 0 (the old OpenAI way), else one token thrown away (vLLM refuses 0).
        A spelling that generates more than one token is not prefill-only and is reported as failed."""
        if not ids:
            return WarmResult(channel="none", detail={"ok": False, "why": "empty prompt"})
        tries = self.ORDER if spelling == "auto" else (spelling,)
        last: WarmResult | None = None
        for sp in tries:
            if sp not in self.ORDER:
                return WarmResult(channel="none", detail={"ok": False, "why": f"no spelling {sp} for token ids"})
            body: dict = {"prompt": ids, "temperature": 0, "stream": False}
            if self.model:
                body["model"] = self.model
            if sp == "max_tokens0":
                body["max_tokens"] = 0
            elif sp == "echo0":
                body.update(max_tokens=0, echo=True)
            else:
                body["max_tokens"] = 1
            t0 = time.perf_counter()
            status, data, raw = self._call("/v1/completions", body)
            ms = (time.perf_counter() - t0) * 1000
            if status == 200:
                usage = data.get("usage") or {}
                made = usage.get("completion_tokens")
                if sp != "max_tokens1" and isinstance(made, int) and made > 1:
                    last = self._failed(ms, status, raw, f"{sp} generated {made} tokens")
                    continue
                return self._result(usage, ms, sp)
            last = self._failed(ms, status, raw)
        return last or WarmResult()

    def warm_chat(self, messages: list[dict], template_kwargs: dict | None = None,
                  tools: list | None = None, *, spelling: str = "chat1") -> WarmResult:
        """The draft as a chat request with one token of answer: the engine applies its own template,
        exactly as it will for the request on Enter.  Ollama's and LM Studio's native routes give the
        prefill time too, which is the evidence of reuse when nothing else is reported."""
        if spelling != "chat1":
            return WarmResult(channel="none", detail={"ok": False, "why": f"no spelling {spelling} for a chat"})
        if self.flavor == "ollama":
            res = self._warm_ollama(messages, template_kwargs, tools)
            if res.ok:
                return res
        body: dict = {"messages": messages, "max_tokens": 1, "temperature": 0, "stream": False}
        if self.model:
            body["model"] = self.model
        if template_kwargs:
            body["chat_template_kwargs"] = template_kwargs
        if tools:
            body["tools"] = tools
        paths = ("/api/v0/chat/completions", "/v1/chat/completions") if self.flavor == "lmstudio" \
            else ("/v1/chat/completions",)
        last = WarmResult()
        for path in paths:
            t0 = time.perf_counter()
            status, data, raw = self._call(path, body)
            ms = (time.perf_counter() - t0) * 1000
            if status == 200:
                stats = data.get("stats") or {}
                ttft = stats.get("time_to_first_token") if isinstance(stats, dict) else None
                prefill_ms = ttft * 1000 if isinstance(ttft, (int, float)) and not isinstance(ttft, bool) else None
                return self._result(data.get("usage") or {}, ms, "chat1", prefill_ms=prefill_ms)
            last = self._failed(ms, status, raw)
        return last

    def _warm_ollama(self, messages: list[dict], template_kwargs: dict | None, tools: list | None) -> WarmResult:
        """Ollama's /api/chat: one token, and prompt_eval_count / prompt_eval_duration (ns) back."""
        body: dict = {"model": self.model, "messages": messages, "stream": False,
                      "options": {"num_predict": 1, "temperature": 0}}
        if tools:
            body["tools"] = tools
        if template_kwargs and "enable_thinking" in template_kwargs:
            body["think"] = bool(template_kwargs["enable_thinking"])
        t0 = time.perf_counter()
        status, data, raw = self._call("/api/chat", body)
        ms = (time.perf_counter() - t0) * 1000
        if status != 200:
            return self._failed(ms, status, raw)
        n, dur = data.get("prompt_eval_count"), data.get("prompt_eval_duration")
        usage = {"prompt_tokens": n if isinstance(n, int) else None, "completion_tokens": data.get("eval_count")}
        prefill_ms = dur / 1e6 if isinstance(dur, (int, float)) and not isinstance(dur, bool) else None
        return self._result(usage, ms, "chat1", prefill_ms=prefill_ms)

    # ---- the request on Enter -----------------------------------------------------
    def chat(self, body: dict, *, stream: bool = False,
             timeout: float = 600.0) -> tuple[int, dict[str, str], Iterable[bytes]]:
        payload = dict(body)
        payload["stream"] = bool(stream)
        data = json.dumps(payload).encode()
        conn = self._conn(timeout)
        conn.request("POST", self.base + "/v1/chat/completions", body=data, headers=self._headers())
        resp = conn.getresponse()
        headers = {k.lower(): v for k, v in resp.getheaders()}
        if not stream:
            raw = resp.read()
            conn.close()
            return resp.status, headers, [raw]

        def chunks() -> Iterable[bytes]:
            try:
                while True:
                    line = resp.readline()      # a line at a time: a block read would wait for the buffer to fill
                    if not line:
                        break
                    yield line
            finally:
                conn.close()

        return resp.status, headers, chunks()

    # ---- the rest of the contract ----------------------------------------------------
    def ready(self) -> bool:
        """The engine answers /v1/models (not while it starts or loads the model)."""
        status, _, _ = self._call("/v1/models", method="GET", timeout=5)
        return status == 200

    def models(self) -> tuple[int, dict]:
        status, data, _ = self._call("/v1/models", method="GET")
        return status, data

    def slots(self) -> int | None:
        return None                               # an OpenAI-style server does not say

    def metrics_text(self) -> str | None:
        status, _, raw = self._call("/metrics", method="GET")
        return raw.decode(errors="replace") if status == 200 else None


def detect(url: str, timeout: float = 3.0, api_key: str = "") -> tuple[str, str] | None:
    """Which engine answers at `url`, and its version: ("ollama", "0.12.3"), ("llama", "b7000"),
    ("lmstudio", ""), ("sglang", ...), ("vllm", ...), ("openai", "") for anything else that has
    /v1/models (or wants a key for it), None when nothing answers yet: a server that is not up,
    or one still loading its model (llama-server answers 503 until then)."""
    e = OpenAIEngine(url, api_key=api_key, timeout=timeout)
    status, data, _ = e._call("/api/version", method="GET")
    if status == 200 and "version" in data:
        return "ollama", str(data["version"])
    status, data, _ = e._call("/props", method="GET")
    if status == 200 and ("total_slots" in data or "default_generation_settings" in data):
        return "llama", str(data.get("build_info") or "")
    status, data, _ = e._call("/api/v0/models", method="GET")
    if status == 200 and isinstance(data.get("data"), list):
        return "lmstudio", ""
    status, data, _ = e._call("/get_server_info", method="GET")
    if status == 200 and data:
        return "sglang", str(data.get("version") or "")
    status, data, _ = e._call("/version", method="GET")
    if status == 200 and "version" in data:
        return "vllm", str(data["version"])
    status, _, _ = e._call("/v1/models", method="GET")
    if status == 200 or status in (401, 403):
        return "openai", ""
    return None
