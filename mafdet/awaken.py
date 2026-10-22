"""
mafdet awaken: check that PyTorch sees the GPU and that the model loads and generates.

    mafdet awaken                 # environment only
    mafdet awaken --load          # also load the model and generate a few tokens
"""
import platform

try:
    import torch
    import transformers
except ImportError:
    raise SystemExit("mafdet awaken checks PyTorch, for the measurements on the PyTorch backend, and PyTorch is "
                     "not installed. To check the app (GPU, llama.cpp, models) run `mafdet doctor`.") from None

from .engine import Template, encode, logits_finite, make_engine, model_label, peak_mem_mb, setup, timed


def run(args) -> int:
    print(f"Python        {platform.python_version()}  ({platform.system()})")
    print(f"PyTorch       {torch.__version__}  (CUDA build: {torch.version.cuda})")
    print(f"transformers  {transformers.__version__}")
    print(f"CPU threads   {torch.get_num_threads()}")

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info()
        print(f"GPU           {props.name}  (compute capability {props.major}.{props.minor})")
        print(f"VRAM          {total / 2**30:.1f} GB total, {free / 2**30:.1f} GB free")
        print(f"native bf16   {'yes' if props.major >= 8 else 'no (Turing or older): using fp16'}")
    else:
        print("GPU           NOT available")
        if torch.version.cuda is None:
            print("              → this is the CPU-only build of PyTorch. Reinstall it from the CUDA index (see README).")
        else:
            print("              → CUDA build present but no GPU found: check the NVIDIA drivers.")

    if not args.load:
        return 0

    print(f"\nLoading {model_label(args)} ...")
    tok, model, device, dtype = setup(args)
    if args.backend == "llama":
        print(f"Loaded on {device} in {dtype} (llama-server, log in {model.log_path})")
    else:
        print(f"Loaded on {device} in {str(dtype).replace('torch.', '')}. "
              f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f} billion")

    tpl = Template.from_tokenizer(tok, thinking=False)
    ids = encode(tok, tpl.prefix + "Explain in one sentence what a KV cache is." + tpl.suffix)
    eng = make_engine(model, device)
    logits, ms = timed(device, eng.sync, ids, need_logits=True)
    if not logits_finite(logits):
        print("WARNING: non-finite logits (NaN/inf). In fp16 this happens when activations exceed "
              "the representable range: try --dtype fp32 or a smaller model.")
        return 1
    print(f"Prefill of {len(ids)} tokens: {ms:.1f} ms")
    gen, ms = timed(device, eng.greedy, logits, 24)
    print(f"Generated 24 tokens in {ms:.0f} ms ({24000 / ms:.1f} tokens/s)")
    print(f"Answer: {tok.decode(gen, skip_special_tokens=True)!r}")
    if device == "cuda":
        print(f"Peak GPU memory: {peak_mem_mb(device):.0f} MB")
    print("\nAll good: next, try `mafdet race`.")
    return 0
