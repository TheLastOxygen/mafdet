"""Hardware parsing and the catalog's estimates, against speeds measured on the reference PC."""
import pytest

from mafdet.core import catalog
from mafdet.core.hardware import GPU, Hardware, amd_gpu, gpu_bandwidth, parse_nvidia_smi


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
    ("AMD Radeon RX 7900 XTX", 960), ("AMD Radeon RX 7900 XT", 800), ("AMD Radeon RX 6700 XT", 384),
    ("AMD Radeon RX 6700", 320), ("AMD Radeon RX 9070 XT", 640), ("Radeon RX 580 Series", 256),
])
def test_gpu_bandwidth_table(name, expected):
    assert gpu_bandwidth(name, 8192) == (expected, True)


def test_unknown_gpu_guessed_from_vram():
    assert gpu_bandwidth("NVIDIA Some Future Card", 6144) == (250, False)


def test_amd_gpu():
    """A dedicated Radeon runs on the Vulkan build; an integrated one (a few hundred MB of its own) does not count."""
    g = amd_gpu("AMD Radeon RX 7800 XT", 16368, 0, "32.0.21025.1024")
    assert (g.backend, g.bandwidth_gbs, g.bandwidth_known, g.unified) == ("vulkan", 624, True, False)
    assert amd_gpu("AMD Radeon(TM) Graphics", 512, 0, "31.0.21921.1000") is None


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


def apple_m2_pro() -> Hardware:
    """MacBook Pro M2 Pro, 16 GB: Metal may use 12,124 MB of the unified memory (its recommended working set)."""
    return Hardware(os="macOS 26.2", system="darwin", arch="arm64", cpu="Apple M2 Pro", cores=12, threads=12,
                    ram_mb=16384, ram_free_mb=6000, ram_bandwidth_gbs=100,
                    gpus=[GPU("Apple M2 Pro", 12124, 0, "Metal · macOS 26.2", 200, True, unified=True)])


def test_placement_on_unified_memory():
    hw = apple_m2_pro()
    assert catalog.estimate(catalog.find("qwen3-8b"), hw)["placement"] == "gpu"
    # The GPU's share is in the same RAM: a model larger than the RAM does not fit, whatever the split.
    assert catalog.estimate(catalog.find("qwen3-14b"), hw)["placement"] in ("gpu", "partial")
    assert catalog.estimate(catalog.find("qwen3-32b"), hw)["placement"] == "too_big"


def test_recommendation():
    entries = [catalog.entry(v, reference_pc()) for v in catalog.variants()]
    assert catalog.recommend(entries) == "qwen3-4b:Q4_K_M"
    # Without a GPU the comfortable choice is a small model.
    entries = [catalog.entry(v, reference_pc(gpu=False)) for v in catalog.variants()]
    assert catalog.recommend(entries) == "qwen3-0.6b:Q8_0"


def rtx_4070_super(used_mb: int = 1120) -> Hardware:
    """RTX 4070 SUPER 12 GB, i5-14600KF, 32 GB DDR5: the third machine of research/LOG.md."""
    return Hardware(os="Windows 11", system="windows", arch="amd64", cpu="Intel(R) Core(TM) i5-14600KF",
                    cores=14, threads=20, ram_mb=32609, ram_free_mb=20000, ram_bandwidth_gbs=28,
                    gpus=[GPU("NVIDIA GeForce RTX 4070 SUPER", 12282, used_mb, "", 504, True)])


@pytest.mark.parametrize("variant,decode,prefill", [
    ("qwen3-4b:Q4_K_M", 129.8, 7441),
    ("qwen3-14b:Q4_K_M", 40.9, 2486),         # all on the GPU: the embeddings (417 MiB) stay in RAM
])
def test_estimates_on_ada(variant, decode, prefill):
    e = catalog.estimate(catalog.find(variant), rtx_4070_super())
    assert e["placement"] == "gpu"
    assert e["decode_tps"] == pytest.approx(decode, rel=0.2)
    assert e["prefill_tps"] == pytest.approx(prefill, rel=0.2)


def test_split_follows_llama_fit():
    """llama.cpp's --fit, measured on the 4070 SUPER with a larger margin standing in for a busier card:
    36 and 24 of the 14B's 41 layers (40 and the output) on the GPU."""
    for used, layers in ((1120 + 1476, 36), (1120 + 3976, 24)):
        share = catalog.estimate(catalog.find("qwen3-14b"), rtx_4070_super(used))["gpu_share"]
        assert abs(share * 40 - layers) <= 1.5


def test_smaller_quant_where_the_default_does_not_fit():
    m = catalog.find("qwen3-14b").model
    assert catalog.preferred_quant(m, rtx_4070_super(1120)) == "Q4_K_M"       # fits: the default
    assert catalog.preferred_quant(m, rtx_4070_super(1900)) == "IQ4_XS"       # a busier desktop
    assert catalog.preferred_quant(m, reference_pc()) == "IQ4_XS"             # split with the CPU
    assert catalog.preferred_quant(m, reference_pc(gpu=False)) == "Q4_K_M"    # CPU only
    assert catalog.preferred_quant(catalog.find("qwen3-4b").model, reference_pc()) == "Q4_K_M"
    entries = [catalog.entry(v, reference_pc()) for v in catalog.variants()]
    assert [e["quant"] for e in entries if e["model_id"] == "qwen3-14b" and e["default"]] == ["IQ4_XS"]


def test_recommendation_on_a_12_gb_card():
    entries = [catalog.entry(v, rtx_4070_super()) for v in catalog.variants()]
    assert catalog.recommend(entries) == "qwen3-14b:Q4_K_M"


def test_recommendation_with_a_busier_desktop():
    """1.4 GB taken on the 2060: a layer of the 4B on the CPU, still the 4B rather than the 1.7B."""
    hw = reference_pc()
    hw.gpus[0].vram_used_mb = 1400
    entries = [catalog.entry(v, hw) for v in catalog.variants()]
    assert catalog.recommend(entries) == "qwen3-4b:Q4_K_M"


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
