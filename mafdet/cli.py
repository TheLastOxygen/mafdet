"""
The `mafdet` command: one entry point with themed subcommands.

All argument parsing lives here and imports nothing heavy, so `mafdet`,
`mafdet --help` and the planned commands answer right away. PyTorch and
transformers are only loaded when a real command runs.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import textwrap
from typing import NamedTuple

from . import DEFAULT_GGUF, DEFAULT_MODEL, __version__

TAGLINE = "You type. It already knows."


class Command(NamedTuple):
    name: str
    theme: str                  # the name's meaning, as in the README table
    summary: str                # one line, for the command table
    details: str                # a paragraph, for --help
    examples: tuple[str, ...]
    group: str                  # "app", "use" or "measure"
    available: bool = True
    common: bool = True         # takes the model and backend options (the app keeps its own settings)


COMMANDS = [
    Command("gui", "the window", "the app in its own window",
            "Opens MAFDET in its own window: chat, model catalog with the speed of each model on this machine, "
            "hardware and settings, in English, Italian, French or German. The first start guides you through "
            "installing llama.cpp and choosing a model.",
            ("mafdet gui", "mafdet gui --browser"), "app", common=False),
    Command("tui", "the terminal", "the same app in the terminal",
            "The whole app in this terminal (Textual): chat, models, hardware and settings, keyboard first "
            "(F1 to F4 move between views).", ("mafdet tui",), "app", common=False),
    Command("models", "the catalog", "which models run well on this machine",
            "Lists the models MAFDET can download, with where each one runs here (all on the GPU, partly on the "
            "CPU), how fast it writes and how long the answer takes after Enter, with and without MAFDET. "
            "Estimated from the hardware, or measured once you benchmark a model in the app.",
            ("mafdet models", "mafdet models --family gemma", "mafdet models --download qwen3-4b"), "app", common=False),
    Command("doctor", "the check-up", "check the machine, llama.cpp and the models",
            "Checks the GPU, memory, llama.cpp and the downloaded models, and says what is missing. With "
            "--install it downloads the llama.cpp build for this machine by itself.",
            ("mafdet doctor", "mafdet doctor --install", "mafdet doctor --install --build vulkan"), "app", common=False),
    Command("awaken", "the awakening", "check PyTorch and the GPU (for the measurements)",
            "Checks that PyTorch sees the GPU (CUDA build, free VRAM, bf16 support) and says what to fix "
            "if it does not. With --load it also downloads the model (about 3.5 GB the first time), "
            "loads it and generates a few tokens. Run it first, and whenever something does not work.",
            ("mafdet awaken", "mafdet awaken --load", "mafdet awaken --load --backend llama"), "measure"),
    Command("watch", "the guard", "chat with your local model: it reads while you type",
            "Opens the MAFDET chat window. The model reads your message while you type, so the answer "
            "starts as soon as you press Enter, and the whole conversation stays in its cache from one "
            "turn to the next. Runs on llama.cpp (llama-server in ./llama.cpp). The step-by-step view "
            "for debugging is demo_live.py.",
            ("mafdet watch", "mafdet watch --device cpu", "mafdet watch --browser"), "use"),
    Command("strike", "the strike", "think ahead: the model reasons while you type",
            "Like watch, but the model (Qwen3 in thinking mode) starts reasoning about your message while "
            "you are still typing it. When the text changes, the reasoning is kept as notes after the new "
            "message (a fast prefill, not a new decode) and the model is told to reread it; on Enter it "
            "finishes reasoning from where it got to. If you pause and send the text unchanged, the answer "
            "written at the pause appears at once. With --no-think there is no reasoning and only that exact "
            "answer is written ahead.",
            ("mafdet strike", "mafdet strike --no-think", "mafdet strike --gguf Qwen/Qwen3-8B-GGUF:Qwen3-8B-Q4_K_M.gguf"),
            "use"),
    Command("hunt", "the hunt", "simulate people typing and measure the wait after sending",
            "Simulates a person typing a message (clean: start to finish; messy: typos, backspaces and an "
            "edit in the middle; paste: a long document followed by a question) and measures the wait from "
            "Enter to the first token, with and without incremental prefill. Every prefill really runs on "
            "the model. Results are appended to results/typing.csv.",
            ("mafdet hunt", "mafdet hunt --scenario paste --device cpu",
             "mafdet hunt --scenario messy --holdback 0 1 3 6 --debounce 100 250 500"), "measure"),
    Command("race", "the race", "benchmark prefill and decode on your machine",
            "Measures how long reading the prompt (prefill) and generating one token (decode) take on your "
            "hardware, for several prompt lengths, and how much adding k tokens to a cache that is already "
            "full costs. Results go to results/prefill_<device>.csv and results/incremental_cost_<device>.csv.",
            ("mafdet race", "mafdet race --device cpu --lengths 50 250 1000 2000"), "measure"),
    Command("judge", "the judgement", "verify results match a from-scratch computation",
            "Applies random edits to a text (appends, backspaces, edits in the middle, pastes, words glued "
            "together) and checks that the incrementally built cache gives the same next token and the same "
            "continuation as a computation from scratch. It must end with 0 errors; otherwise the exit code "
            "is 1.",
            ("mafdet judge", "mafdet judge --trials 50 --holdback 0"), "measure"),
]

GROUPS = [("app", "Open the app"), ("use", "Chat with explicit options"), ("measure", "Measure and verify")]

GETTING_STARTED = [
    ("mafdet", "choose the window or the terminal, then the setup guides you"),
    ("mafdet models", "which models run well on this machine"),
    ("mafdet doctor", "check the GPU, llama.cpp and the models"),
    ("mafdet judge --backend llama", "must end with 0 errors"),
    ("mafdet hunt --backend llama", "wait after Enter, with and without incremental prefill"),
]

_LETTERS = {  # figlet "standard" font
    "M": [r" __  __ ", r"|  \/  |", r"| |\/| |", r"| |  | |", r"|_|  |_|"],
    "A": [r"    _    ", r"   / \   ", r"  / _ \  ", r" / ___ \ ", "/_/   \\_\\"],
    "F": [r" _____ ", r"|  ___|", r"| |_   ", r"|  _|  ", r"|_|    "],
    "D": [r" ____  ", r"|  _ \ ", r"| | | |", r"| |_| |", r"|____/ "],
    "E": [r" _____ ", r"| ____|", r"|  _|  ", r"| |___ ", r"|_____|"],
    "T": [r" _____ ", r"|_   _|", r"  | |  ", r"  | |  ", r"  |_|  "],
}


# The wordmark in block characters, as in the visual identity's terminal mockup.
_BLOCKS = ["█▀▄▀█ ▄▀█ █▀▀ █▀▄ █▀▀ ▀█▀",
           "█ ▀ █ █▀█ █▀  █▄▀ ██▄  █"]

AMBER = "38;2;212;137;42"          # #D4892A, the identity's only accent colour
BOLD, DIM = "1", "2"


def _color_enabled() -> bool:
    """ANSI colours only on a real terminal, and never with NO_COLOR set (no-color.org)."""
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        return False
    if os.name != "nt":
        return True
    # The Windows console understands ANSI sequences only once asked to.
    import ctypes
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.GetStdHandle(-11)                 # standard output
    mode = ctypes.c_uint32()
    if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
        return False
    return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))   # ENABLE_VIRTUAL_TERMINAL_PROCESSING


_COLOR: bool | None = None


def paint(text: str, style: str) -> str:
    global _COLOR
    if _COLOR is None:
        _COLOR = _color_enabled()
    return f"\x1b[{style}m{text}\x1b[0m" if _COLOR and text else text


def _can_print(text: str) -> bool:
    try:
        text.encode(sys.stdout.encoding or "ascii")
        return True
    except (UnicodeEncodeError, LookupError):
        return False


def banner() -> str:
    if not _can_print("".join(_BLOCKS) + "·"):
        # e.g. output redirected to a cp1252 file: the figlet version is plain ASCII.
        rows = ["  " + " ".join(_LETTERS[c][i] for c in "MAFDET").rstrip() for i in range(5)]
        return "\n".join(rows) + f"\n\n  {TAGLINE}\n"
    rows = [f"  {paint(row, BOLD)}" for row in _BLOCKS]
    rows.append("  " + paint("reads and thinks while you type · local LLMs", AMBER))
    rows.append("  " + paint(f"v{__version__} · {TAGLINE}", DIM))
    return "\n".join(rows) + "\n"


def command_table() -> str:
    """The commands at a glance: what `mafdet` prints with no arguments."""
    name_w = max(len(c.name) for c in COMMANDS)
    lines = []
    for group, title in GROUPS:
        lines.append(paint(title, BOLD))
        cmds = [c for c in COMMANDS if c.group == group]
        summary_w = max(len(c.summary) for c in cmds)
        for c in cmds:
            status = "" if c.available else "coming soon"
            row = f"  {paint(f'{c.name:<{name_w}}', BOLD)}   {c.summary:<{summary_w}}   {paint(status, DIM)}"
            lines.append(row.rstrip())
        lines.append("")
    lines.append("First time? Just run " + paint("`mafdet`", AMBER) + " in a terminal.")
    lines.append("Run `mafdet --help` for the full guide to every command,")
    lines.append("or `mafdet <command> --help` for the options of one.")
    return "\n".join(lines)


def _wrap(text: str, indent: int) -> str:
    pad = " " * indent
    return textwrap.fill(text, width=78, initial_indent=pad, subsequent_indent=pad)


def command_guide() -> str:
    """Every command in detail, with examples: the epilog of `mafdet --help`."""
    lines = []
    for group, title in GROUPS:
        lines.append(f"{title}:")
        for c in COMMANDS:
            if c.group != group:
                continue
            lines.append(f"  {c.name} ({c.theme})" + ("" if c.available else "  [coming soon]"))
            lines.append(_wrap(c.details, 4))
            lines += [f"      {ex}" for ex in c.examples]
            lines.append("")
    lines.append("Getting started:")
    width = max(len(cmd) for cmd, _ in GETTING_STARTED)
    lines += [f"  {cmd:<{width}}   {what}" for cmd, what in GETTING_STARTED]
    lines.append("")
    lines.append("Common options (every command accepts them, see `mafdet <command> --help`):")
    lines.append("  --model, --device, --dtype, --threads, --compile,")
    lines.append("  --backend, --gguf, --ctx, --draft, --llama-args")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------

def add_common_args(p):
    """Options shared by every command (and by demo_live.py)."""
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="Hugging Face id or local folder (default: %(default)s). With --backend llama: the "
                        "tokenizer, needed only when it cannot be guessed from --gguf")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                   help="auto = the GPU if PyTorch sees one, otherwise the CPU (default: %(default)s)")
    p.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"],
                   help="--backend hf only: weight precision; auto = fp16 on GPU, fp32 on CPU "
                        "(default: %(default)s)")
    p.add_argument("--threads", type=int, default=0, help="CPU threads (0 = PyTorch default)")
    p.add_argument("--compile", action="store_true",
                   help="--backend hf only: torch.compile the model (GPU only, needs triton; "
                        "2-9 minutes of compilation at start-up)")
    p.add_argument("--backend", default="hf", choices=["hf", "llama"],
                   help="hf = PyTorch/transformers; llama = llama-server (llama.cpp) with a GGUF model "
                        "(default: %(default)s)")
    p.add_argument("--gguf", default=DEFAULT_GGUF,
                   help="--backend llama only: 'repo:file' on Hugging Face, or a local path "
                        "(default: %(default)s)")
    p.add_argument("--ctx", type=int, default=8192,
                   help="--backend llama only: context per slot, in tokens (default: %(default)s)")
    p.add_argument("--draft", default="",
                   help="--backend llama only: a small model of the same family that drafts tokens for "
                        "speculative decoding, 'repo:file' or a path (e.g. Qwen/Qwen3-0.6B-GGUF:Qwen3-0.6B-Q8_0.gguf)")
    p.add_argument("--llama-args", default="",
                   help='--backend llama only: extra llama-server options, with an equals sign: --llama-args="-ub 256"')


def _gui_args(p):
    p.add_argument("--browser", action="store_true",
                   help="open the app in the default browser instead of its own window")


def _tui_args(p):
    pass


def _models_args(p):
    p.add_argument("--all", action="store_true", help="every quantization, not only the recommended one")
    p.add_argument("--family", default="", help="only one family: qwen3, llama, gemma, mistral or phi")
    p.add_argument("--download", metavar="ID", default="",
                   help="download a model: an id from the list (qwen3-4b) or with its quantization (qwen3-4b:Q8_0)")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                   help="estimate for the GPU (auto, cuda) or as if there were none (cpu)")


def _doctor_args(p):
    p.add_argument("--install", action="store_true", help="download and install the llama.cpp build for this machine")
    p.add_argument("--build", "--cuda", dest="cuda", choices=["12", "13", "vulkan", "cpu"], default="",
                   help="with --install: this build instead of the automatic choice (CUDA 13 from driver 580 on, "
                        "CUDA 12 for older drivers and GTX 10 or older cards, Vulkan for AMD Radeon)")


def _awaken_args(p):
    p.add_argument("--load", action="store_true", help="also load the model and run a test generation")


def _watch_args(p):
    p.set_defaults(backend="llama")
    p.add_argument("--holdback", type=int, default=3,
                   help="tokens at the end kept out of the cache while typing (default: %(default)s)")
    p.add_argument("--debounce", type=float, default=250.0,
                   help="pause in ms after which the draft is read (default: 250)")
    p.add_argument("--min-new", type=int, default=16,
                   help="new tokens at the end that trigger a read without waiting for a pause "
                        "(default: %(default)s)")
    p.add_argument("--max-tokens", type=int, default=0,
                   help="longest answer, in tokens (0 = as long as the context allows)")
    p.add_argument("--browser", action="store_true",
                   help="open the chat in the default browser instead of its own window")


def _strike_args(p):
    _watch_args(p)
    p.add_argument("--ahead", type=int, default=512,
                   help="answer tokens written at a pause, before Enter (default: %(default)s)")
    p.add_argument("--no-think", dest="think", action="store_false",
                   help="no reasoning: answer right away (Qwen3 non-thinking mode), and write only the "
                        "answer ahead")
    p.add_argument("--think-ahead", type=int, default=1024,
                   help="longest reasoning kept across edits of the message, in tokens (default: %(default)s)")
    p.add_argument("--gatherer", default="",
                   help="a small model that gathers the material for the answer while you type, so the main "
                        "model only reads it: 'repo:file' or a path (e.g. Qwen/Qwen3-0.6B-GGUF:Qwen3-0.6B-Q8_0.gguf)")
    p.add_argument("--gatherer-device", default="cuda", choices=["cuda", "cpu"],
                   help="where the gatherer runs: cuda takes about 1 GB of VRAM from the main model, cpu none "
                        "(default: %(default)s)")
    p.add_argument("--branches", type=int, default=0,
                   help="experimental: the model also guesses how you will finish the message and reasons on up "
                        "to N guesses beside the main reasoning; if one guessed the exact message, its reasoning "
                        "and answer are ready on Enter. Measured on Qwen3-4B: 1 shortens the wait by 16%% on average, 2 is "
                        "slower than none; with a model larger than the VRAM the guesses come too slowly to help "
                        "(default: %(default)s, off)")
    p.add_argument("--recycle", action="store_true",
                   help="reuse the answers written for earlier versions of the message as drafts for "
                        "speculative decoding (n-gram); measured: +4%% decode speed with natural typing, "
                        "+20%% on code, slightly later first token")


def _race_args(p):
    p.add_argument("--lengths", type=int, nargs="+", default=None,
                   help="prompt lengths to measure, in tokens (default: 50 to 8000 on GPU, 50 to 2000 on CPU)")
    p.add_argument("--repeats", type=int, default=None,
                   help="runs per length, the median is reported (default: 3 on GPU, 2 on CPU)")
    p.add_argument("--base-context", type=int, default=1000,
                   help="context already in cache for the incremental-cost table (default: %(default)s)")
    p.add_argument("--out", default="results", help="output folder for the CSV files (default: %(default)s)")


def _hunt_args(p):
    p.add_argument("--scenario", nargs="+", default=["clean", "messy", "paste"],
                   choices=["clean", "messy", "paste"], help="typing scenarios to run (default: all three)")
    p.add_argument("--holdback", type=int, nargs="+", default=[3],
                   help="tokens at the end kept out of the cache, because the word being typed can still "
                        "change; several values = one row each (default: 3)")
    p.add_argument("--debounce", type=float, nargs="+", default=[250.0],
                   help="pause in ms after which a prefill starts; several values = one row each (default: 250)")
    p.add_argument("--min-new", type=int, default=16,
                   help="new stable tokens at the end that trigger a prefill "
                        "(edits in the middle of the text wait for the debounce) (default: %(default)s)")
    p.add_argument("--cps", type=float, nargs="+", default=[5.0],
                   help="typing speed in characters per second; several values = one run each (default: 5)")
    p.add_argument("--text", default=None, help="text file to type instead of the samples")
    p.add_argument("--doc-repeat", type=int, default=2,
                   help="paste scenario: how many copies of the document to paste (default: %(default)s)")
    p.add_argument("--seed", type=int, default=0,
                   help="seed of the simulated typing: same seed, same keystrokes (default: %(default)s)")
    p.add_argument("--out", default="results", help="output folder for the CSV files (default: %(default)s)")


def _judge_args(p):
    p.add_argument("--trials", type=int, default=20, help="random texts to check (default: %(default)s)")
    p.add_argument("--edits", type=int, default=40, help="edits per trial (default: %(default)s)")
    p.add_argument("--holdback", type=int, default=3,
                   help="tokens at the end kept out of the cache while typing (default: %(default)s)")
    p.add_argument("--greedy", type=int, default=12,
                   help="continuation tokens to compare (default: %(default)s)")
    p.add_argument("--seed", type=int, default=0, help="seed of the random edits (default: %(default)s)")
    p.add_argument("--tol", type=float, default=None,
                   help="threshold: relative logit difference (hf, default 0.1) "
                        "or shifted probability (llama, default 0.25)")


_ARGS = {"gui": _gui_args, "tui": _tui_args, "models": _models_args, "doctor": _doctor_args, "awaken": _awaken_args, "watch": _watch_args, "strike": _strike_args, "race": _race_args,
         "hunt": _hunt_args, "judge": _judge_args}


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------

def _runner(name: str):
    def run(args) -> int:
        return importlib.import_module(f"mafdet.{name}").run(args)
    return run


def _coming_soon(name: str):
    def run(args) -> int:
        print(f"mafdet {name}: coming soon. It is on the roadmap: see the Roadmap section of README.md.",
              file=sys.stderr)
        return 1
    return run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mafdet", description=f"MAFDET: {TAGLINE}", epilog=command_guide(),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"mafdet {__version__}")
    # The commands are described in the epilog: add_parser without help= keeps
    # argparse from listing them a second time.
    sub = parser.add_subparsers(title="commands", metavar="<command>", help="one of the commands below")
    for c in COMMANDS:
        status = "" if c.available else " (coming soon)"
        examples = "examples:\n" + "\n".join(f"  {ex}" for ex in c.examples) if c.examples else None
        sp = sub.add_parser(c.name, formatter_class=argparse.RawDescriptionHelpFormatter,
                            description=_wrap(f"mafdet {c.name} ({c.theme}): {c.summary}{status}", 0)
                                        + "\n\n" + _wrap(c.details, 0),
                            epilog=examples)
        # Each command gets its own copy of the common options (not a shared parent
        # parser), so a command can change their defaults without touching the others.
        if c.common:
            add_common_args(sp.add_argument_group("common options"))
        if c.available:
            _ARGS[c.name](sp)
            sp.set_defaults(func=_runner(c.name))
        else:
            sp.set_defaults(func=_coming_soon(c.name))
    return parser


def launch(ask: bool = False) -> int:
    """`mafdet` alone: the launcher asks which interface to open (or opens the one remembered).

    Without a terminal to ask in (output redirected, a script) it prints the commands instead.
    """
    if not (sys.stdin and sys.stdin.isatty() and sys.stdout and sys.stdout.isatty()):
        print(banner())
        print(command_table())
        return 0
    from .core import config
    choice = "" if ask else config.load()["launcher"]
    if choice not in ("gui", "tui"):
        from .tui.launcher import choose
        picked = choose()
        if picked is None:
            return 0
        choice, remember = picked
        if remember:
            config.save({"launcher": choice})
    else:
        print(paint(f"Opening the {'window' if choice == 'gui' else 'terminal app'} "
                    "(remembered; `mafdet --ask` to choose again)", DIM))
    if choice == "gui":
        from .gui import run as run_gui
        return run_gui(argparse.Namespace(browser=False))
    from .tui import run as run_tui
    return run_tui()


def main(argv: list[str] | None = None) -> int:
    # Never crash on a character the console cannot show (e.g. output redirected to a cp1252 file).
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv == ["--ask"]:
        return launch(ask=bool(argv))
    args = build_parser().parse_args(argv)
    if not hasattr(args, "func"):          # e.g. only --version-like flags handled by argparse
        print(command_table())
        return 0
    return args.func(args)
