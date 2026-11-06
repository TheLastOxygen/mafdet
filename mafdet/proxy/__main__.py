"""`python -m mafdet.proxy --url http://127.0.0.1:8081 --model Qwen3-4B`

Starts the proxy in front of an engine that is already running.  The first start probes
the engine (a few hundred tokens, ~10 s) and saves an `EngineProfile`; later starts read
it back.  `--reprobe` forces a fresh probe.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from .adapters import make
from .probe import probe
from .profile import EngineProfile
from .server import ProxyState, make_server


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mafdet.proxy", description=__doc__)
    ap.add_argument("--engine", default="llama", help="adapter: llama (PR #1 ships only this)")
    ap.add_argument("--url", required=True, help="engine base URL, e.g. http://127.0.0.1:8081")
    ap.add_argument("--model", required=True, help="model id the engine serves")
    ap.add_argument("--port", type=int, default=8077, help="port the proxy listens on")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--debounce-ms", type=int, default=300)
    ap.add_argument("--min-new", type=int, default=8, help="new tokens that trigger a warm")
    ap.add_argument("--reprobe", action="store_true", help="probe again instead of reusing the saved profile")
    ap.add_argument("--no-probe", action="store_true", help="start without probing (read-ahead off: T-1)")
    args = ap.parse_args(argv)

    engine = make(args.engine, args.url)

    profile = None if args.reprobe else EngineProfile.load(args.url, args.model)
    if profile is None:
        if args.no_probe:
            from .engine import EngineCaps
            profile = EngineProfile(name=engine.name, url=args.url, model=args.model, caps=EngineCaps())
        else:
            t0 = time.perf_counter()
            profile = probe(engine, url=args.url, model=args.model)
            path = profile.save()
            took = time.perf_counter() - t0
            print(f"probed in {took:.1f}s -> tier {profile.tier}, spelling {profile.spelling}, "
                  f"channel {profile.channel} (saved to {path})", file=sys.stderr)

    state = ProxyState(engine, profile, args.model, debounce_ms=args.debounce_ms, min_new=args.min_new)
    server = make_server(state, args.host, args.port)
    print(f"{profile.name} {args.url} -> proxy on http://{args.host}:{args.port} "
          f"(tier {profile.tier}); drafts to POST /v1/drafts, stats at GET /proxy/stats", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
