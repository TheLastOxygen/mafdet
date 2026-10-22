"""
mafdet gui: the window.

A desktop window (pywebview) shows the app; behind it a small HTTP server on
127.0.0.1 connects the page to the ChatService. Only this machine can reach it,
by its own address (no DNS rebinding), and the API answers only requests that
carry the token written into the page.

    mafdet gui
    mafdet gui --browser            # in the default browser instead of a window
"""
from __future__ import annotations

import http.server
import json
import os
import secrets
import sys
import tempfile
import threading
import time
import webbrowser
from importlib.resources import files

from . import __version__
from .core.chat import ChatService
from .identity import INK, SHAPES, WORDMARK, WORDMARK_SIZE

UI = files("mafdet").joinpath("ui")

# Files the page may load: nothing else under ui/ (or outside it) is served.
STATIC = {
    "app.css": "text/css; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
    "chat.js": "text/javascript; charset=utf-8",
    "views.js": "text/javascript; charset=utf-8",
    "fonts/archivo.woff2": "font/woff2",
    "fonts/jetbrains-mono.woff2": "font/woff2",
    **{f"i18n/{lang}.json": "application/json; charset=utf-8" for lang in ("en", "it", "fr", "de")},
}


def identity_js() -> bytes:
    data = {"shapes": {k: [[d, a, o] for d, a, o in v] for k, v in SHAPES.items()},
            "wordmark": {"size": WORDMARK_SIZE, "letters": WORDMARK}, "version": __version__}
    return f"window.MAFDET_SHAPES = {json.dumps(data)};\n".encode("utf-8")


def make_handler(app: ChatService, token: str, port_box: list):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _reply(self, code: int, body: bytes = b"", ctype: str = "application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data):
            self._reply(200, json.dumps(data).encode("utf-8"))

        def _allowed(self, api: bool) -> bool:
            if self.headers.get("Host") != f"127.0.0.1:{port_box[0]}":
                return False
            return not api or secrets.compare_digest(self.headers.get("X-Mafdet-Token", ""), token)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if not self._allowed(api=path.startswith("/api/")):
                return self._reply(403)
            if path == "/":
                page = UI.joinpath("index.html").read_text(encoding="utf-8").replace("{{TOKEN}}", token)
                return self._reply(200, page.encode("utf-8"), "text/html; charset=utf-8")
            if path == "/identity.js":
                return self._reply(200, identity_js(), "text/javascript; charset=utf-8")
            if path == "/favicon.ico":
                return self._reply(200, UI.joinpath("mafdet.ico").read_bytes(), "image/x-icon")
            if path.startswith("/static/") and path[8:] in STATIC:
                return self._reply(200, UI.joinpath(*path[8:].split("/")).read_bytes(), STATIC[path[8:]])
            routes = {
                "/api/status": app.status,
                "/api/hardware": app.hardware,
                "/api/catalog": app.catalog,
                "/api/runtime": app.runtime,
                "/api/settings": lambda: app.settings,
                "/api/tasks": app.jobs,
            }
            if path in routes:
                return self._json(routes[path]())
            self._reply(404)

        def do_POST(self):
            if not self._allowed(api=True):
                return self._reply(403)
            length = int(self.headers.get("Content-Length") or 0)
            data = json.loads(self.rfile.read(length) or b"{}")
            text = str(data.get("text", ""))
            model = str(data.get("model", ""))
            routes = {
                "/api/draft": lambda: app.draft(text),
                "/api/stop": lambda: app.stop() or {},
                "/api/new": lambda: app.new_chat() or {},
                "/api/model/load": lambda: app.load(model),
                "/api/model/unload": lambda: app.unload() or {},
                "/api/download": lambda: app.download(model),
                "/api/bench": lambda: app.benchmark(model),
                "/api/runtime/install": app.install_runtime,
                "/api/tasks/cancel": lambda: app.cancel(str(data.get("id", ""))) or {},
                "/api/settings": lambda: app.update_settings(data if isinstance(data, dict) else {}),
                "/api/hardware/refresh": lambda: app.hardware(refresh=True),
            }
            if self.path in routes:
                return self._json(routes[self.path]())
            if self.path == "/api/open":
                url = str(data.get("url", ""))
                if url.startswith(("http://", "https://")):
                    webbrowser.open(url)           # links open in the browser, not in the app window
                return self._json({})
            if self.path == "/api/send":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    for event in app.send(text):
                        self.wfile.write(f"data: {json.dumps(event)}\n\n".encode("utf-8"))
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    app.stop()                     # the window went away
                return
            self._reply(404)

    return Handler


def open_window(url: str, browser: bool):
    if not browser:
        try:
            import webview
        except ImportError:
            print("pywebview is not installed (pip install pywebview): opening the browser instead.")
            browser = True
    if browser:
        webbrowser.open(url)
        print(f"MAFDET is running at {url}\nPress Ctrl+C to stop.", flush=True)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            return
    webview.create_window("MAFDET", url, width=1280, height=860, min_size=(900, 600),
                          background_color=INK, text_select=True)
    webview.start(icon=str(UI.joinpath("mafdet.ico")))


def serve(app: ChatService) -> tuple[http.server.ThreadingHTTPServer, str]:
    """Start the local server for `app`; returns it and the page's address."""
    token, port_box = secrets.token_urlsafe(24), [0]
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app, token, port_box))
    server.daemon_threads = True
    port_box[0] = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{port_box[0]}/"


def run(args, overrides=None) -> int:
    """The window. `overrides`: the options of mafdet watch/strike, which load their model directly."""
    if sys.stdout is None or sys.stderr is None:
        # Started without a console (pythonw): keep the output somewhere for when something goes wrong.
        log = open(os.path.join(tempfile.gettempdir(), "mafdet-gui.log"), "w", encoding="utf-8")
        sys.stdout = sys.stderr = log
    app = ChatService(overrides)
    app.autoload()
    server, url = serve(app)
    try:
        open_window(url, getattr(args, "browser", False))
    finally:
        server.shutdown()
        app.close()
    return 0
