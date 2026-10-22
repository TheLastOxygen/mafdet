"""The terminal interfaces: the app (mafdet tui) and the launcher that asks which one to open."""


def run(args=None) -> int:
    from .app import run as run_app
    return run_app(args)
