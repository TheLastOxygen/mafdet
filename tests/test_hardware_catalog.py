"""Hardware parsing and the catalog's estimates, against speeds measured on the reference PC."""
import pytest

from mafdet.core import catalog
from mafdet.core.hardware import GPU, Hardware, gpu_bandwidth, parse_nvidia_smi


def reference_pc(gpu: bool = True) -> Hardware:
    """RTX 2060 6 GB (0.6 GB taken by the desktop), i5-10400, 16 GB DDR4: where the numbers were measured."""
    gpus = [GPU("NVIDIA GeForce RTX 2060", 6144, 639, "617.42", 336, True)] if gpu else []
    return Hardware(os="Windows 11", system="windows", arch="amd64", cpu="Intel(R) Core(TM) i5-10400",
                    cores=6, threads=12, ram_mb=16167, ram_free_mb=10000, ram_bandwidth_gbs=27.8, gpus=gpus)


def test_parse_nvidia_smi():
    gpus = parse_nvidia_smi("NVIDIA GeForce RTX 2060, 6144, 639, 617.42\n")
    assert len(gpus) == 1
    g = gpus[0]
    assert (g.name, g.vram_mb, g.vram_free_mb, g.bandwidth_gbs, g.bandwidth_known) == \
        ("NVIDIA GeForce RTX 2060", 6144, 5505, 336, True)
    assert parse_nvidia_smi("") == [] and parse_nvidia_smi("garbage") == []


@pytest.mark.parametrize("name,expected", [
    ("NVIDIA GeForce RTX 2060 SUPER", 448), ("NVIDIA GeForce RTX 2060", 336), ("NVIDIA GeForce RTX 4070 Ti SUPER", 672),
    ("NVIDIA GeForce RTX 5070 Ti", 896), ("NVIDIA GeForce RTX 4060 Laptop GPU", 272 * 0.75), ("Tesla T4", 320),
])
def test_gpu_bandwidth_table(name, expected):
    assert gpu_bandwidth(name, 8192) == (expected, True)


def test_unknown_gpu_guessed_from_vram():
    assert gpu_bandwidth("NVIDIA Some Future Card", 6144) == (250, False)


@pytest.mark.parametrize("variant,decode,prefill", [
    ("qwen3-4b:Q4_K_M", 87.1, 2141),          # all on the GPU
    ("qwen3-8b:Q4_K_M", 16.2, 821),           # partly on the CPU
    ("qwen3-14b:Q4_K_M", 4.9, 358),
])
def test_estimates_match_measurements(variant, decode, prefill):
    e = catalog.estimate(catalog.find(variant), reference_pc())
    assert e["decode_tps"] == pytest.approx(decode, rel=0.3)
    assert e["prefill_tps"] == pytest.approx(prefill, rel=0.3)


def test_placement():
    hw = reference_pc()
    assert catalog.estimate(catalog.find("qwen3-4b"), hw)["placement"] == "gpu"
    assert catalog.estimate(catalog.find("qwen3-14b"), hw)["placement"] == "partial"
    assert catalog.estimate(catalog.find("qwen3-32b"), hw)["placement"] == "too_big"
    assert catalog.estimate(catalog.find("qwen3-4b"), reference_pc(gpu=False))["placement"] == "cpu"


def test_recommendation():
    entries = [catalog.entry(v, reference_pc()) for v in catalog.variants()]
    assert catalog.recommend(entries) == "qwen3-4b:Q4_K_M"
    # Without a GPU the comfortable choice is a small model.
    entries = [catalog.entry(v, reference_pc(gpu=False)) for v in catalog.variants()]
    assert catalog.recommend(entries) == "qwen3-0.6b:Q8_0"


def test_waits_shorter_with_reasoning_ahead():
    w = catalog.waits(catalog.find("qwen3-4b").model, 87)
    assert 0 < w["with_mafdet_s"] < w["without_s"]


def test_find_default_quant():
    assert catalog.find("qwen3-0.6b").id == "qwen3-0.6b:Q8_0"
    assert catalog.find("qwen3-4b").id == "qwen3-4b:Q4_K_M"
    assert catalog.find("nope") is None


def test_every_family_has_models_and_a_best_pick():
    entries = catalog.listing(reference_pc())
    fams = {f["id"] for f in catalog.families()}
    assert fams == {"qwen3", "llama", "gemma", "mistral", "phi"}
    assert entries["family_best"] == {"qwen3": "qwen3-4b:Q4_K_M", "llama": "llama-3.2-3b:Q4_K_M",
                                      "gemma": "gemma-3-4b:Q4_K_M", "mistral": "mistral-7b:Q4_K_M",
                                      "phi": "phi-4-mini:Q4_K_M"}


def test_only_qwen3_reasons_ahead():
    for v in catalog.variants():
        assert v.model["reason_ahead"] == (v.model["family"] == "qwen3")
    e = catalog.entry(catalog.find("llama-3.2-3b"), reference_pc())
    assert e["with_mafdet_s"] is None and e["answer_s"] > 0


def test_gemma_keeps_the_full_cache():
    """Gemma 3's sliding window: without --swa-full the cache could not be cut back as the text changes."""
    for v in catalog.variants():
        assert ("--swa-full" in v.model.get("llama_args", [])) == (v.model["family"] == "gemma")
