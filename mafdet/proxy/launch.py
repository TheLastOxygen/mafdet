"""`mafdet proxy`: llama-server started the right way, and the proxy in front of it; or the proxy
alone in front of an engine already running (`--url`: llama-server, Ollama, vLLM, SGLang, LM Studio or
any OpenAI-compatible server, `--engine auto` asks which).

Three llama-server options decide whether reading ahead is worth anything with a front end like
Open WebUI, and nobody would guess them (research/LOG.md, 2026-11-06):
  * two slots and one shared KV cache (`-np 2 -kvu`): the conversation in one, the front end's
    titles, tags and follow-ups in the other, or they evict the conversation every turn;
  * no idle slot saved to RAM and cleared when the other one gets a task (`--cache-ram 0`, which
    also turns off `--cache-idle-slots`): otherwise the title empties the conversation anyway;
  * a context big enough for Open WebUI's 30 built-in tools (~5.8k tokens) plus the conversation.
So `mafdet proxy` starts llama-server itself, through the app's `LlamaServer` (the same build,
GPU layers and KV-cache rules as the app), unless `--url` points at one already running.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from .adapters import make
from .adapters.openai import detect
from .engine import EngineCaps
from .probe import probe
from .profile import EngineProfile
from .server import ProxyState, make_server

ENGINE_PORT = 8078          # fixed, so the saved profile and last request are found again next time
PROXY_PORT = 8077
OPEN_WEBUI = "http://localhost:8080,http://127.0.0.1:8080"
USER_SCRIPT = Path(__file__).with_name("clients") / "open-webui.user.js"


def resolve_model(spec: str) -> tuple[str, str, tuple[str, ...]]:
    """(path of the GGUF, the name the engine serves it as, extra llama-server options) for a catalog id
    (qwen3-4b, qwen3-4b:Q8_0), a local file, or repository:file; empty means the model chosen in the app."""
    from ..core import catalog, config
    spec = spec or config.load()["model"]
    if not spec:
        raise SystemExit("No model chosen: pass --model (an id from `mafdet models`, e.g. qwen3-4b, a GGUF "
                         "file, or repository:file), or choose one in the app.")
    v = catalog.find(spec)
    if v is not None:
        path = catalog.local_path(v)
        if path is None:
            raise SystemExit(f"{v.id} is not downloaded yet: `mafdet models --download {v.id}`.")
        return path, v.id, tuple(v.model.get("llama_args", ()))
    from ..llama_backend import resolve_gguf
    path = resolve_gguf(spec)
    return path, Path(path).stem, ()


def start_engine(args):
    """llama-server for the proxy, through the app's LlamaServer.  Returns it (it stops with us)."""
    from ..llama_backend import LlamaServer, pick_device
    path, alias, model_args = resolve_model(args.model)
    device = pick_device(args.device)
    extra = model_args + ("--alias", alias) + tuple(args.llama_args.split())
    print(f"Starting llama-server: {alias} on the {'GPU' if device == 'cuda' else 'CPU'}, 2 slots sharing "
          f"{args.ctx} tokens, port {args.engine_port}", file=sys.stderr, flush=True)
    # LlamaServer always passes --cache-ram 0 (no idle slot cleared) and, with a pool, -kvu.
    server = LlamaServer(path, device, ctx=args.ctx, n_slots=2, port=args.engine_port, extra_args=extra,
                         log_name="proxy-server", pool=args.ctx)
    return server, alias


# Where a front end's side requests go when the engine has one cache and no second slot.
ONE_CACHE_HINT = {
    "llama": "start llama-server with -np 2 -kvu --no-cache-idle-slots",
    "ollama": "start Ollama with OLLAMA_NUM_PARALLEL=2",
    "lmstudio": "give the model 2 parallel requests in LM Studio's server settings, if the version has it",
}


def _wait_until(what: str, check, url: str, wait: float) -> None:
    """Wait for an engine started next to the proxy (Docker) that is still loading, or
    downloading its model: probing it now would save a profile without reading ahead."""
    if check():
        return
    print(f"waiting for the engine at {url}...", file=sys.stderr, flush=True)
    end = time.monotonic() + wait
    while not check():
        if time.monotonic() > end:
            raise SystemExit(f"The engine at {url} did not {what} in {wait:.0f} s.")
        time.sleep(2)


def _first_model(engine, url: str) -> str:
    """The first model the engine lists, for an OpenAI-style engine started without --model."""
    status, data = engine.models()
    ids = [m.get("id") for m in (data.get("data") or []) if isinstance(m, dict) and m.get("id")]
    if status != 200 or not ids:
        raise SystemExit(f"{engine.name} at {url} lists no model (GET /v1/models): pass --model.")
    print(f"model: {ids[0]} (the first {engine.name} lists; --model picks another)", file=sys.stderr, flush=True)
    return str(ids[0])


def serve(url: str, model: str, *, host: str = "127.0.0.1", port: int = PROXY_PORT, cors: str = "",
          exact: bool = False, reprobe: bool = False, no_probe: bool = False, debounce_ms: int = 300,
          min_new: int = 8, engine_name: str = "llama", banner: str = "", think_ahead: bool = False,
          wait_engine: float = 1800.0, api_key: str = "") -> int:
    """Probe (or reload the profile of) the engine at `url` and serve the proxy until Ctrl+C."""
    if engine_name == "auto":
        _wait_until("answer", lambda: detect(url, api_key=api_key) is not None, url, wait_engine)
    engine = make(engine_name, url, model=model, api_key=api_key)
    if callable(getattr(engine, "ready", None)):
        _wait_until("answer its health check", engine.ready, url, wait_engine)
    if engine.dialect == "llama":
        model = model or "default"
    elif not model or model == "default":
        model = engine.model = _first_model(engine, url)
    profile = None if reprobe else EngineProfile.load(url, model)
    if profile is None:
        if no_probe:
            profile = EngineProfile(name=engine.name, url=url, model=model, caps=EngineCaps())
        else:
            t0 = time.perf_counter()
            profile = probe(engine, url=url, model=model)
            saved = profile.save()
            print(f"probed in {time.perf_counter() - t0:.1f}s -> tier {profile.tier}, spelling {profile.spelling}, "
                  f"channel {profile.channel} (saved to {saved})", file=sys.stderr)
    origins = [o.strip() for o in cors.split(",") if o.strip()]
    state = ProxyState(engine, profile, model, debounce_ms=debounce_ms, min_new=min_new, exact=exact, cors=origins,
                       think_ahead=think_ahead)
    server = make_server(state, host, port)
    shown = "127.0.0.1" if host in ("0.0.0.0", "") else host
    lines = [banner] if banner else []
    caps = profile.caps
    how = ("no reading ahead" if profile.tier == "T-1" else
           "drafts through its chat endpoint" if caps.by_chat else "drafts as token ids")
    label = f"{engine.name} {profile.version}".strip()
    lines += [
        f"MAFDET proxy on http://{shown}:{port} (tier {'T-1, exact' if exact else ('T2, thinking ahead' if state.frame else profile.tier)})",
        f"  engine       {label} at {url}: {how}, reuse seen in {profile.channel}, "
        f"cache {'shared' if caps.shared_cache else 'one per slot'}",
        f"  front end    OpenAI API base URL http://{shown}:{port}/v1"
        + ("   (Open WebUI in Docker: http://host.docker.internal:%d/v1)" % port if host == "0.0.0.0" else ""),
        f"  drafts       {', '.join(origins) if origins else 'no web page allowed (--cors)'}",
        f"  user script  {USER_SCRIPT}",
        f"  status       http://{shown}:{port}/proxy",
    ]
    if profile.tier == "T-1" and not exact:
        why = profile.latencies.get("why") or ("no reuse seen" if caps.spelling != "none" else "no request to read ahead with")
        lines.append(f"  warning      reading ahead is off on this engine ({why}): the requests pass through")
    if think_ahead and state.frame is None and not exact:
        lines.append(f"  warning      --think-ahead: {state.no_think_reason}: reading ahead only")
    if state.side_slot is None and not exact and not caps.shared_cache and profile.tier != "T-1":
        hint = ONE_CACHE_HINT.get(getattr(engine, "flavor", ""), "give the engine a second slot, if it has the option")
        lines.append(f"  warning      one cache: a front end's titles and follow-ups will evict the conversation ({hint})")
    print("\n".join(lines), file=sys.stderr, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def run(args) -> int:
    """`mafdet proxy`."""
    if args.url:
        return serve(args.url, args.model, host=args.host, port=args.port, cors=args.cors, exact=args.exact,
                     reprobe=args.reprobe, think_ahead=args.think_ahead, engine_name=getattr(args, "engine", "auto"),
                     api_key=getattr(args, "api_key", ""))
    server, alias = start_engine(args)
    try:
        return serve(server.url, alias, host=args.host, port=args.port, cors=args.cors, exact=args.exact,
                     reprobe=args.reprobe, banner=f"llama-server log: {server.log_path}",
                     think_ahead=args.think_ahead)
    finally:
        server.close()
