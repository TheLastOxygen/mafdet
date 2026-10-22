"""
mafdet models: the catalog in the terminal, with what each model does on this machine.

    mafdet models                        # one row per model, at its recommended quantization
    mafdet models --family gemma         # one family
    mafdet models --all                  # every quantization
    mafdet models --download qwen3-4b    # download one (its default quantization, or qwen3-4b:Q8_0)
"""
from __future__ import annotations

import time

from rich import box
from rich.console import Console
from rich.progress import BarColumn, DownloadColumn, Progress, TimeRemainingColumn, TransferSpeedColumn
from rich.table import Table

from .core import catalog, hardware
from .core.downloads import download_model
from .core.tasks import TaskManager

AMBER = "#D4892A"
WHERE = {"gpu": ("all on GPU", "green"), "partial": ("{p}% on GPU", AMBER), "cpu": ("on CPU", "white"),
         "too_big": ("too large", "red")}


def dur(s):
    return "–" if s is None else f"{s:.1f} s" if s < 10 else f"{s:.0f} s" if s < 60 else f"{s / 60:.1f} min"


def run(args) -> int:
    console = Console()
    if args.download:
        return download(console, args.download)
    hw = hardware.detect()
    listing = catalog.listing(hw, args.device)
    best = listing["recommended"]
    fams = {f["id"]: f for f in listing["families"]}
    if args.family and args.family not in fams:
        console.print(f"[red]Unknown family {args.family}[/]: {', '.join(fams)}")
        return 1
    machine = f"{hw.gpu.name.replace('NVIDIA GeForce ', '')} {hw.gpu.vram_mb / 1024:.0f} GB" if hw.gpu else "CPU only"
    console.print(f"[bold]MAFDET models[/] [dim]· {machine} · {hw.ram_mb / 1024:.0f} GB RAM[/]")
    shown = [e for e in listing["models"] if (args.all or e["default"]) and args.family in ("", e["family"])]
    # One small table per family, the same column widths in all of them: they read as one list.
    columns = (("", 1, "left"), ("model", 26, "left"), ("runs", 11, "left"), ("writes", 11, "right"),
               ("thinks after Enter", 19, "right"), ("300-token answer", 16, "right"), ("size", 7, "right"),
               ("", 7, "left"))
    for i, fid in enumerate(dict.fromkeys(e["family"] for e in shown)):
        f = fams[fid]
        table = Table(box=box.SIMPLE_HEAD, show_header=i == 0, header_style="dim",
                      pad_edge=False, show_edge=False, title_justify="left",
                      title=f"\n[bold]{f['name']}[/] [dim]{f['vendor']} · {f['license']}"
                            f"{' · reasons while you type' if f['reason_ahead'] else ''}[/]")
        for name, width, justify in columns:
            table.add_column(name, width=width, justify=justify, no_wrap=True)
        for e in (x for x in shown if x["family"] == fid):
            label, colour = WHERE[e["placement"]]
            mark = "[bold #D4892A]★[/]" if e["id"] == best else "[#D4892A]·[/]" if e["family_best"] else ""
            think = (f"[#D4892A]≈ {dur(e['with_mafdet_s'])}[/] [dim]({dur(e['without_s'])})[/]" if e["reason_ahead"]
                     else "[dim]–[/]")
            table.add_row(mark, e["id"], f"[{colour}]{label.format(p=round(e['gpu_share'] * 100))}[/]",
                          f"{e['decode_tps']:.0f} tok/s" + (" [green]✓[/]" if e["measured"] else ""), think,
                          f"≈ {dur(e['answer_s'])}", f"[dim]{e['bytes'] / 2**30:.1f} GB[/]",
                          "[green]on disk[/]" if e["installed"] else "")
        console.print(table)
    console.print("[dim]★ best for you · [#D4892A]·[/] best of its family · ✓ measured · (…) without MAFDET · "
                  "download: mafdet models --download <id>[/]")
    return 0


def download(console: Console, variant_id: str) -> int:
    v = catalog.find(variant_id)
    if v is None:
        console.print(f"[red]Unknown model {variant_id}[/]: see `mafdet models --all`.")
        return 1
    task = TaskManager().start("download", v.id, lambda t: download_model(v.id, t))
    columns = ("{task.description}", BarColumn(), DownloadColumn(), TransferSpeedColumn(), TimeRemainingColumn())
    with Progress(*columns, console=console) as bar:
        row = bar.add_task(v.id, total=v.bytes)
        while task.state == "running":
            bar.update(row, completed=task.done_bytes)
            time.sleep(0.2)
        bar.update(row, completed=task.done_bytes if task.state != "done" else v.bytes)
    if task.state != "done":
        console.print(f"[red]{task.error or task.state}[/]")
        return 1
    console.print(f"[green]Downloaded[/] {v.id} to {task.result['path']}")
    return 0
