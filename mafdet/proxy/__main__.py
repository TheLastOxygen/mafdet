"""`python -m mafdet.proxy --url http://127.0.0.1:8081 --model Qwen3-4B`

Starts the proxy in front of an engine that is already running (`mafdet proxy` also starts
llama-server itself, with the options the proxy needs).  The first start probes
the engine (a few hundred tokens, ~10 s) and saves an `EngineProfile`; later starts read
it back.  `--reprobe` forces a fresh probe.
"""
from __future__ import annotations

import argparse

from .launch import serve


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
    ap.add_argument("--exact", action="store_true",
                    help="no read-ahead, every prompt recomputed (cache_prompt: false): answers "
                         "bit-identical to a cold run, none of the gain")
    ap.add_argument("--cors", default="",
                    help="origins allowed to call the proxy from a web page, comma-separated "
                         "(e.g. http://localhost:3000), or * for any; off by default")
    ap.add_argument("--think-ahead", action="store_true",
                    help="reason about the draft while the user types (Qwen3, thinking on): not exact")
    args = ap.parse_args(argv)

    return serve(args.url, args.model, host=args.host, port=args.port, cors=args.cors, exact=args.exact,
                 reprobe=args.reprobe, no_probe=args.no_probe, debounce_ms=args.debounce_ms,
                 min_new=args.min_new, engine_name=args.engine, think_ahead=args.think_ahead)

if __name__ == "__main__":
    raise SystemExit(main())
