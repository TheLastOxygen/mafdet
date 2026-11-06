"""
mafdet models: the catalog in the terminal, with what each model does on this machine.

    mafdet models                        # one row per model, at its recommended quantization
    mafdet models --family gemma         # one family
    mafdet models --all                  # every quantization
    mafdet models --download qwen3-4b    # download one (the quantization shown for this machine, or qwen3-4b:Q8_0)
    mafdet models --download unsloth/Qwen3-4B-GGUF:Qwen3-4B-UD-Q4_K_XL.gguf   # any GGUF on Hugging Face
    mafdet models --delete qwen3-4b:Q8_0 # delete a downloaded model from the disk
"""
from __future__ import annotations

from rich import box
from rich.console import Console
from rich.table import Table

from .core import catalog, config, hardware
from .core.downloads import download_model, size_text
from .core.tasks import TaskManager
from .progress import follow

AMBER = "#D4892A"
WHERE = {"gpu": ("all on GPU", "green"), "partial": ("{p}% on GPU", AMBER), "cpu": ("on CPU", "white"),
         "too_big": ("too large", "red")}


def dur(s):
    return "–" if s is None else f"{s:.1f} s" if s < 10 else f"{s:.0f} s" if s < 60 else f"{s / 60:.1f} min"


def run(args) -> int:
    console = Console()
    if args.download:
        return download(console, args.download)
    if args.delete:
        return delete(console, args.delete)
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
                      title=f"\n[bold]{f['name']}[/] [dim]"
                            + " · ".join(x for x in (f["vendor"], f["license"],
                                                     "reasons while you type" if f["reason_ahead"] else "") if x)
                            + "[/]")
        for name, width, justify in columns:
            table.add_column(name, width=width, justify=justify, no_wrap=True)
        for e in (x for x in shown if x["family"] == fid):
            label, colour = WHERE[e["placement"]]
            mark = "[bold #D4892A]★[/]" if e["id"] == best else "[#D4892A]·[/]" if e["family_best"] else ""
            think = (f"[#D4892A]≈ {dur(e['with_mafdet_s'])}[/] [dim]({dur(e['without_s'])})[/]" if e["reason_ahead"]
                     else "[dim]–[/]")
            table.add_row(mark, f"{e['name']} {e['quant']}" if e["local"] else e["id"], f"[{colour}]{label.format(p=round(e['gpu_share'] * 100))}[/]",
                          f"{e['decode_tps']:.0f} tok/s" + (" [green]✓[/]" if e["measured"] else ""), think,
                          f"≈ {dur(e['answer_s'])}", f"[dim]{e['bytes'] / 2**30:.1f} GB[/]",
                          "[green]on disk[/]" if e["installed"] else "")
        console.print(table)
    console.print("[dim]★ best for you · [#D4892A]·[/] best of its family · ✓ measured · (…) without MAFDET · "
                  "download: mafdet models --download <id> · delete: --delete <id>[/]")
    disk, typing = listing["disk"], listing["typing"]
    on_disk = sum(e["installed"] for e in listing["models"])
    line = f"{on_disk} on the disk, {size_text(disk['models_bytes'])} in {disk['path']} · {size_text(disk['free_bytes'])} free"
    if typing["yours"]:
        line += f" · waits for your typing: {typing['seconds']} s per message, the median of {typing['samples']}"
    console.print(f"[dim]{line}[/]")
    return 0


def delete(console: Console, variant_id: str) -> int:
    """Delete a downloaded model from the disk: a variant id, or a bare model id when one quantization is on disk."""
    v = catalog.find(variant_id)
    if v is not None and ":" not in variant_id and not variant_id.startswith(catalog.LOCAL_PREFIX):
        on_disk = [x for x in catalog.variants() if x.model["id"] == v.model["id"] and catalog.local_path(x)]
        if len(on_disk) > 1:
            console.print(f"[red]{variant_id} is on the disk in {len(on_disk)} quantizations[/]: say which, "
                          + " or ".join(x.id for x in on_disk))
            return 1
        v = on_disk[0] if on_disk else v
    if v is None:
        console.print(f"[red]Unknown model {variant_id}[/]: see `mafdet models --all`.")
        return 1
    try:
        gone = catalog.delete_files(v)
    except (OSError, ValueError) as e:
        console.print(f"[red]✗ {v.id}: {e}[/]")
        return 1
    if config.load()["model"] == v.id:
        config.save({"model": ""})                # the app would load it at the next start
    console.print(f"[green]✓[/] {v.id} deleted: {size_text(gone['bytes'])} freed [dim]({gone['path']})[/]")
    return 0


def download(console: Console, variant_id: str) -> int:
    if "/" in variant_id:                          # repository:file, from Hugging Face
        from .core import hub
        repo, _, file = variant_id.partition(":")
        if not file.lower().endswith(".gguf"):
            console.print(f"[red]Give the file too[/]: {repo}:<name>.gguf (the files are on huggingface.co/{repo})")
            return 1
        task = TaskManager().start("download", variant_id, lambda t: hub.download(repo, file, t))
        return 0 if follow(task, console) else 1
    v = catalog.find(variant_id)
    if v is not None and ":" not in variant_id:    # a bare model id: the quantization the list shows here
        v = catalog.find(f"{v.model['id']}:{catalog.preferred_quant(v.model, hardware.detect())}") or v
    if v is None:
        console.print(f"[red]Unknown model {variant_id}[/]: see `mafdet models --all`.")
        return 1
    task = TaskManager().start("download", v.id, lambda t: download_model(v.id, t))
    return 0 if follow(task, console) else 1
