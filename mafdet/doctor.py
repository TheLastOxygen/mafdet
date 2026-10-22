"""
mafdet doctor: is everything in place? GPU, memory, llama.cpp, models; with --install it puts
llama.cpp in place by itself.

    mafdet doctor
    mafdet doctor --install
    mafdet doctor --install --cuda 12     # a given build instead of the automatic choice
"""
from __future__ import annotations

from rich.console import Console

from .core import catalog, config, hardware, llama_setup
from .core.tasks import TaskManager
from .progress import follow


def run(args) -> int:
    console = Console()
    ok = "[green]✓[/]"
    warn = "[#D4892A]![/]"
    hw = hardware.detect()
    console.print(f"[bold]MAFDET doctor[/]   [dim]{config.home()}[/]\n")
    console.print(f"{ok} System     {hw.os} · {hw.arch}")
    console.print(f"{ok} Processor  {hw.cpu} · {hw.cores} cores, {hw.threads} threads")
    console.print(f"{ok} Memory     {hw.ram_mb / 1024:.1f} GB, {hw.ram_free_mb / 1024:.1f} GB free · "
                  f"~{hw.ram_bandwidth_gbs:.0f} GB/s")
    if hw.gpu:
        g = hw.gpu
        console.print(f"{ok} GPU        {g.name} · {g.vram_mb / 1024:.1f} GB VRAM ({g.vram_free_mb / 1024:.1f} GB free) · "
                      f"driver {g.driver} · {g.bandwidth_gbs:.0f} GB/s")
    else:
        console.print(f"{warn} GPU        no NVIDIA card found (nvidia-smi): models run on the CPU")

    if args.install:
        install(console, args.cuda or None)
    rt = llama_setup.status()
    if rt["installed"]:
        same = rt["build"] == rt["tested_build"]
        console.print(f"{ok if same else warn} llama.cpp  {rt['build'] or 'unknown build'} at {rt['path']}"
                      + ("" if same else f" [dim](MAFDET is tested with {rt['tested_build']})[/]"))
    else:
        console.print(f"{warn} llama.cpp  not installed: `mafdet doctor --install` installs {rt['recommended']}")

    listing = catalog.listing(hw)
    have = [e["id"] for e in listing["models"] if e["installed"]]
    if have:
        console.print(f"{ok} Models     {', '.join(have)}")
    else:
        console.print(f"{warn} Models     none downloaded yet")
    console.print(f"\n  Recommended for this machine: [bold #D4892A]{listing['recommended']}[/] · open the app with `mafdet`")
    return 0 if rt["installed"] else 1


def install(console: Console, cuda: str | None = None):
    follow(TaskManager().start("llama", llama_setup.LLAMA_TAG, lambda t: llama_setup.install(t, cuda=cuda)), console)
