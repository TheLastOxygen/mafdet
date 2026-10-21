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
    group: str                  # "use" or "measure"
    available: bool = True


COMMANDS = [
    Command("awaken", "the awakening", "check the GPU, download and load the model",
            "Checks that PyTorch sees the GPU (CUDA build, free VRAM, bf16 support) and says what to fix "
            "if it does not. With --load it also downloads the model (about 3.5 GB the first time), "
            "loads it and generates a few tokens. Run it first, and whenever something does not work.",
            ("mafdet awaken", "mafdet awaken --load", "mafdet awaken --load --backend llama"), "use"),
    Command("watch", "the guard", "chat with your local model: it reads while you type",
            "Opens the MAFDET chat window. The model reads your message while you type, so the answer "
            "starts as soon as you press Enter, and the whole conversation stays in its cache from one "
            "turn to the next. Runs on llama.cpp (llama-server in ./llama.cpp). The step-by-step view "
            "for debugging is demo_live.py.",
            ("mafdet watch", "mafdet watch --device cpu", "mafdet watch --browser"), "use"),
    Command("strike", "the strike", "answer ahead: the reply is already written when you hit enter",
            "Like watch, but when you pause the model also starts writing the answer, before you press "
            "Enter, and keeps it aside. If you send the message as it was at the pause, the answer written "
            "so far appears at once and the rest follows live; if you change the text, that answer is "
            "thrown away and a new one starts at the next pause. The answer is the same one the model would "
            "give after Enter: it is only written earlier. --ahead sets how many tokens it writes in advance.",
            ("mafdet strike", "mafdet strike --ahead 1024", "mafdet strike --device cpu"), "use"),
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

GROUPS = [("use", "Use it"), ("measure", "Measure and verify")]

GETTING_STARTED = [
    ("mafdet awaken --load", "check the GPU and download the model"),
    ("mafdet watch", "chat with the model: it reads while you type"),
    ("mafdet strike", "the same chat, with the answer written ahead at every pause"),
    ("mafdet judge", "must end with 0 errors"),
    ("mafdet race", "cost of prefill and decode on your hardware"),
    ("mafdet hunt", "wait after Enter, with and without incremental prefill"),
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
    rows.append("  " + paint("incremental prefill · warm KV cache · near-zero TTFT", AMBER))
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
    lines.append("First time? Start with " + paint("`mafdet awaken --load`", AMBER) + ".")
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
    lines.append("  --backend, --gguf, --ctx, --llama-args")
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
    p.add_argument("--llama-args", default="",
                   help='--backend llama only: extra llama-server options, with an equals sign: --llama-args="-ub 256"')


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


_ARGS = {"awaken": _awaken_args, "watch": _watch_args, "strike": _strike_args, "race": _race_args,
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
        add_common_args(sp.add_argument_group("common options"))
        if c.available:
            _ARGS[c.name](sp)
            sp.set_defaults(func=_runner(c.name))
        else:
            sp.set_defaults(func=_coming_soon(c.name))
    return parser


def main(argv: list[str] | None = None) -> int:
    # Never crash on a character the console cannot show (e.g. output redirected to a cp1252 file).
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(banner())
        print(command_table())
        return 0
    args = build_parser().parse_args(argv)
    if not hasattr(args, "func"):          # e.g. only --version-like flags handled by argparse
        print(command_table())
        return 0
    return args.func(args)
