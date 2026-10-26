"""
What the machine has: GPU (NVIDIA through nvidia-smi, AMD Radeon through DXGI on Windows or the amdgpu
driver on Linux, Apple Silicon through Metal), VRAM, RAM, CPU.

No PyTorch: llama.cpp runs the models, so the chat must be able to tell
whether there is a GPU without importing a 4 GB library. The memory
bandwidths feed the speed estimates of the model catalog: generating a token
reads every weight once, so its time is about size / bandwidth.
"""
from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field

# Memory bandwidth in GB/s of common NVIDIA and AMD cards (manufacturer specifications).
# Matched against the name the driver reports, longest key first.
GPU_BANDWIDTH = {
    "GTX 1050 TI": 112, "GTX 1050": 112, "GTX 1060": 192, "GTX 1070 TI": 256, "GTX 1070": 256,
    "GTX 1080 TI": 484, "GTX 1080": 320, "GTX 1650 SUPER": 192, "GTX 1650": 128, "GTX 1660 SUPER": 336,
    "GTX 1660 TI": 288, "GTX 1660": 192,
    "RTX 2060 SUPER": 448, "RTX 2060": 336, "RTX 2070 SUPER": 448, "RTX 2070": 448, "RTX 2080 SUPER": 496,
    "RTX 2080 TI": 616, "RTX 2080": 448,
    "RTX 3050": 224, "RTX 3060 TI": 448, "RTX 3060": 360, "RTX 3070 TI": 608, "RTX 3070": 448,
    "RTX 3080 TI": 912, "RTX 3080": 760, "RTX 3090 TI": 1008, "RTX 3090": 936,
    "RTX 4060 TI": 288, "RTX 4060": 272, "RTX 4070 TI SUPER": 672, "RTX 4070 TI": 504, "RTX 4070 SUPER": 504,
    "RTX 4070": 504, "RTX 4080 SUPER": 736, "RTX 4080": 717, "RTX 4090": 1008,
    "RTX 5060 TI": 448, "RTX 5060": 448, "RTX 5070 TI": 896, "RTX 5070": 672, "RTX 5080": 960, "RTX 5090": 1792,
    "RTX A2000": 288, "RTX A4000": 448, "RTX A5000": 768, "RTX A6000": 768, "RTX 4000 ADA": 360,
    "RTX 6000 ADA": 960, "TESLA T4": 320, "L4": 300, "A10G": 600, "A100": 1555, "H100": 3350,
    # AMD Radeon (the Infinity Cache of the RX 6000 and later does not help here: the weights are far larger)
    "RX 470": 211, "RX 480": 256, "RX 570": 224, "RX 580": 256, "RX 590": 256, "VEGA 56": 410, "VEGA 64": 484,
    "RADEON VII": 1024, "RX 5500 XT": 224, "RX 5600 XT": 288, "RX 5700 XT": 448, "RX 5700": 448,
    "RX 6400": 128, "RX 6500 XT": 144, "RX 6600 XT": 256, "RX 6600": 224, "RX 6650 XT": 280, "RX 6700 XT": 384,
    "RX 6700": 320, "RX 6750 XT": 432, "RX 6800 XT": 512, "RX 6800": 512, "RX 6900 XT": 512, "RX 6950 XT": 576,
    "RX 7600 XT": 288, "RX 7600": 288, "RX 7700 XT": 432, "RX 7800 XT": 624, "RX 7900 GRE": 576,
    "RX 7900 XT": 800, "RX 7900 XTX": 960, "RX 9060 XT": 320, "RX 9070 XT": 640, "RX 9070": 640,
}
# Apple Silicon: the bandwidth of the unified memory, which the GPU and the CPU share (the top
# configuration of each chip; the binned M3 Max and M4 Max reach 300 and 410).
APPLE_BANDWIDTH = {
    "M1": 68, "M1 PRO": 200, "M1 MAX": 400, "M1 ULTRA": 800,
    "M2": 100, "M2 PRO": 200, "M2 MAX": 400, "M2 ULTRA": 800,
    "M3": 100, "M3 PRO": 150, "M3 MAX": 400, "M3 ULTRA": 819,
    "M4": 120, "M4 PRO": 273, "M4 MAX": 546,
    "M5": 153,
}
# Laptop parts share the name of a desktop card but run on a narrower, slower bus.
LAPTOP_FACTOR = 0.75


@dataclass
class GPU:
    name: str
    vram_mb: int
    vram_used_mb: int
    driver: str
    bandwidth_gbs: float
    bandwidth_known: bool             # from the table, not guessed from the amount of VRAM
    unified: bool = False             # Apple Silicon: the "VRAM" is the share of the RAM Metal may use
    backend: str = "cuda"             # the llama.cpp build that runs on it: cuda, vulkan (AMD) or metal

    @property
    def vram_free_mb(self) -> int:
        return max(0, self.vram_mb - self.vram_used_mb)


@dataclass
class Hardware:
    os: str                           # "Windows 11", "Linux 6.8", ...
    system: str                       # "windows", "linux", "darwin": what llama.cpp builds are for
    arch: str
    cpu: str
    cores: int
    threads: int
    ram_mb: int
    ram_free_mb: int
    ram_bandwidth_gbs: float
    gpus: list[GPU] = field(default_factory=list)

    @property
    def gpu(self) -> GPU | None:
        """The card llama.cpp uses first: the one with the most VRAM."""
        return max(self.gpus, key=lambda g: g.vram_mb) if self.gpus else None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["gpu"] = asdict(self.gpu) if self.gpu else None
        if self.gpu:
            d["gpu"]["vram_free_mb"] = self.gpu.vram_free_mb
        return d


def gpu_bandwidth(name: str, vram_mb: int) -> tuple[float, bool]:
    """Bandwidth of a card from its name; a guess from its VRAM when the name is unknown."""
    upper = re.sub(r"\s+", " ", name.upper().replace("GEFORCE", "").replace("NVIDIA", "")).strip()
    for key in sorted(GPU_BANDWIDTH, key=len, reverse=True):
        if re.search(rf"(^|\s){re.escape(key)}(\s|$)", upper):
            bw = GPU_BANDWIDTH[key]
            return (bw * LAPTOP_FACTOR if "LAPTOP" in upper or "MOBILE" in upper else bw), True
    gb = vram_mb / 1024
    for limit, bw in ((4.5, 128), (6.5, 250), (8.5, 300), (12.5, 400), (16.5, 500), (24.5, 900)):
        if gb <= limit:
            return bw, False
    return 1000, False


def parse_nvidia_smi(out: str) -> list[GPU]:
    """Rows of `nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader,nounits`."""
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            total, used = int(float(parts[1])), int(float(parts[2]))
        except ValueError:
            continue
        bw, known = gpu_bandwidth(parts[0], total)
        gpus.append(GPU(parts[0], total, used, parts[3], bw, known))
    return gpus


def nvidia_gpus() -> list[GPU]:
    exe = shutil.which("nvidia-smi")
    if exe is None and os.name == "nt":
        candidate = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvidia-smi.exe")
        exe = candidate if os.path.exists(candidate) else None
    if exe is None:
        return []
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        r = subprocess.run([exe, "--query-gpu=name,memory.total,memory.used,driver_version",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10,
                           creationflags=flags)
    except (OSError, subprocess.SubprocessError):
        return []
    return parse_nvidia_smi(r.stdout) if r.returncode == 0 else []


def _metal_device() -> tuple[str, int] | None:
    """Name and recommendedMaxWorkingSetSize (bytes) of the Metal GPU, through the Objective-C runtime."""
    import ctypes
    try:
        objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
        metal = ctypes.CDLL("/System/Library/Frameworks/Metal.framework/Metal")
    except OSError:
        return None
    metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
    objc.sel_registerName.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]
    dev = metal.MTLCreateSystemDefaultDevice()
    if not dev:
        return None

    def send(obj, sel: bytes, restype):
        return ctypes.CFUNCTYPE(restype, ctypes.c_void_p, ctypes.c_void_p)(("objc_msgSend", objc))(
            obj, objc.sel_registerName(sel))
    name = send(send(dev, b"name", ctypes.c_void_p), b"UTF8String", ctypes.c_char_p)
    return (name or b"").decode("utf-8", "replace"), send(dev, b"recommendedMaxWorkingSetSize", ctypes.c_uint64)


def apple_gpus() -> list[GPU]:
    """The GPU of an Apple Silicon Mac: llama.cpp's Metal backend may use up to the working set Metal
    recommends (about 3/4 of the RAM), so that is its "VRAM". Intel Macs: none (the x64 build is CPU only)."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return []
    import psutil
    ram_mb = psutil.virtual_memory().total >> 20
    found = _metal_device()
    name, vram_mb = (found[0], found[1] >> 20) if found and found[1] else (cpu_name(), int(ram_mb * 0.7))
    m = re.search(r"\b(M\d+(?: (?:PRO|MAX|ULTRA))?)\b", name.upper())
    known = bool(m and m.group(1) in APPLE_BANDWIDTH)
    bw = APPLE_BANDWIDTH[m.group(1)] if known else 100
    # What other apps hold is not counted: llama.cpp's split (-ngl auto) only looks at the working set.
    return [GPU(name, vram_mb, 0, f"Metal · macOS {platform.mac_ver()[0]}", bw, known, unified=True,
                backend="metal")]


AMD_VENDOR = 0x1002
# Integrated Radeons (Ryzen APUs) have a few hundred MB of their own and share the RAM: not counted, they
# keep the CPU build. A dedicated card has 4 GB or more.
AMD_MIN_VRAM_MB = 2048


def amd_gpu(name: str, vram_mb: int, used_mb: int, driver: str) -> GPU | None:
    """An AMD card, for llama.cpp's Vulkan build; None for an integrated one."""
    if vram_mb < AMD_MIN_VRAM_MB:
        return None
    bw, known = gpu_bandwidth(name, vram_mb)
    return GPU(name, vram_mb, used_mb, driver, bw, known, backend="vulkan")


def _dxgi_adapters() -> list[tuple[str, int, int, str]]:
    """Name, vendor id, dedicated memory (bytes) and driver version of the adapters Windows has running
    (DXGI). The registry also keeps cards taken out of the machine long ago: DXGI only lists the present ones."""
    import ctypes
    from ctypes import wintypes

    class GUID(ctypes.Structure):
        _fields_ = [("a", ctypes.c_uint32), ("b", ctypes.c_uint16), ("c", ctypes.c_uint16), ("d", ctypes.c_ubyte * 8)]

    def guid(text: str) -> GUID:
        h = text.replace("-", "")
        return GUID(int(h[:8], 16), int(h[8:12], 16), int(h[12:16], 16), (ctypes.c_ubyte * 8)(*bytes.fromhex(h[16:])))

    class DESC1(ctypes.Structure):
        _fields_ = [("Description", ctypes.c_wchar * 128), ("VendorId", ctypes.c_uint), ("DeviceId", ctypes.c_uint),
                    ("SubSysId", ctypes.c_uint), ("Revision", ctypes.c_uint), ("DedicatedVideoMemory", ctypes.c_size_t),
                    ("DedicatedSystemMemory", ctypes.c_size_t), ("SharedSystemMemory", ctypes.c_size_t),
                    ("LuidLow", wintypes.DWORD), ("LuidHigh", wintypes.LONG), ("Flags", ctypes.c_uint)]

    def method(obj, index: int, *argtypes):
        """The COM method at `index` of obj's vtable."""
        vtable = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        return ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *argtypes)(vtable[index])

    try:
        dxgi = ctypes.WinDLL("dxgi")
    except OSError:
        return []
    factory = ctypes.c_void_p()
    if dxgi.CreateDXGIFactory1(ctypes.byref(guid("770aae78-f26f-4dba-a829-253c83d1b387")), ctypes.byref(factory)) != 0:
        return []
    found = []
    i = 0
    while True:
        adapter = ctypes.c_void_p()
        # IDXGIFactory1::EnumAdapters1; DXGI_ERROR_NOT_FOUND past the last adapter
        if method(factory, 12, ctypes.c_uint, ctypes.c_void_p)(factory, i, ctypes.byref(adapter)) != 0:
            break
        i += 1
        desc = DESC1()
        ok = method(adapter, 10, ctypes.c_void_p)(adapter, ctypes.byref(desc)) == 0                 # GetDesc1
        version = ctypes.c_longlong()
        driver = ""
        if method(adapter, 9, ctypes.c_void_p, ctypes.c_void_p)(                                  # CheckInterfaceSupport
                adapter, ctypes.byref(guid("54ec77fa-1377-44e6-8c32-88fd5f44c84c")), ctypes.byref(version)) == 0:
            driver = ".".join(str(version.value >> shift & 0xFFFF) for shift in (48, 32, 16, 0))
        method(adapter, 2)(adapter)                                                                 # Release
        if ok and not desc.Flags & 2:                                                               # not the software renderer
            found.append((desc.Description, desc.VendorId, desc.DedicatedVideoMemory, driver))
    method(factory, 2)(factory)
    return found


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


def amd_gpus() -> list[GPU]:
    """AMD Radeon cards, for llama.cpp's Vulkan build. The memory in use is known on Linux (amdgpu), not on
    Windows, where it counts as free: llama.cpp's own split (-ngl auto) asks Vulkan, which knows it."""
    found = []
    if os.name == "nt":
        try:
            adapters = _dxgi_adapters()
        except (OSError, ValueError, AttributeError):
            adapters = []
        for name, vendor, vram, driver in adapters:
            g = amd_gpu(name, vram >> 20, 0, driver) if vendor == AMD_VENDOR else None
            if g:
                found.append(g)
    elif platform.system() == "Linux":
        import glob
        for dev in sorted(glob.glob("/sys/class/drm/card[0-9]*/device")):
            total, used = _read(f"{dev}/mem_info_vram_total"), _read(f"{dev}/mem_info_vram_used")
            if _read(f"{dev}/vendor") != "0x1002" or not total.isdigit():
                continue
            name = _read(f"{dev}/product_name") or f"AMD Radeon ({_read(f'{dev}/device')})"
            g = amd_gpu(name, int(total) >> 20, int(used) >> 20 if used.isdigit() else 0,
                        f"amdgpu · Linux {platform.release()}")
            if g:
                found.append(g)
    return found


def gpus() -> list[GPU]:
    """The cards llama.cpp can run models on: NVIDIA (CUDA build), AMD Radeon (Vulkan) or Apple Silicon (Metal).
    One kind only: the build installed runs one of them."""
    return nvidia_gpus() or amd_gpus() or apple_gpus()


def cpu_name() -> str:
    if os.name == "nt":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    if platform.system() == "Darwin":
        try:
            r = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5)
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def os_label() -> str:
    if platform.system() == "Windows":
        build = int(platform.version().split(".")[-1] or 0)
        return f"Windows {'11' if build >= 22000 else platform.release()}"
    if platform.system() == "Darwin":
        return f"macOS {platform.mac_ver()[0]}"
    return f"{platform.system()} {platform.release()}"


# llama.cpp reads the CPU's share of the weights with every core; a single-threaded
# copy reaches a fraction of that. Calibrated on the reference PC (DDR4, RTX 2060):
# with this factor the 14B estimate matches its measured 4.8 tokens/s.
RAM_COPY_FACTOR = 2.2


def measure_ram_bandwidth() -> float:
    """System memory bandwidth from copies of 128 MB, scaled to llama.cpp's multi-threaded reads.

    The best of several copies: a busy moment (a browser starting, a download) only makes a
    copy slower, never faster.
    """
    size = 128 << 20
    try:
        src = bytearray(size)
        best = float("inf")
        for _ in range(8):
            t0 = time.perf_counter()
            dst = bytes(src)                  # one read and one write of every byte
            best = min(best, time.perf_counter() - t0)
            del dst
        return round(2 * size / best / 1e9 * RAM_COPY_FACTOR, 1)
    except MemoryError:
        return 30.0


def ram_bandwidth_gbs(cpu: str, ram_mb: int, refresh: bool = False) -> float:
    """Measured once per machine and kept in ~/.mafdet/hardware.json: the estimates of the
    catalog depend on it and must not move from one start to the next."""
    from . import config
    key = f"{cpu} | {round(ram_mb / 1024)}GB"
    saved = config.read_json("hardware.json")
    if not refresh and isinstance(saved.get(key), (int, float)):
        return saved[key]
    bw = measure_ram_bandwidth()
    if refresh and isinstance(saved.get(key), (int, float)):
        bw = max(bw, saved[key])              # a new measure only replaces a slower one
    saved[key] = bw
    config.write_json("hardware.json", saved)
    return bw


_cache: Hardware | None = None


def detect(refresh: bool = False) -> Hardware:
    """The machine, measured once per process (refresh=True for the current VRAM and RAM use)."""
    global _cache
    if _cache is not None and not refresh:
        return _cache
    import psutil
    vm = psutil.virtual_memory()
    cpu, ram_mb = cpu_name(), vm.total >> 20
    bw = _cache.ram_bandwidth_gbs if _cache is not None else ram_bandwidth_gbs(cpu, ram_mb)
    found = gpus()
    if found and found[0].unified:
        bw = min(bw, found[0].bandwidth_gbs)  # the factor above is for DDR: never past the chip's own bandwidth
    _cache = Hardware(
        os=os_label(), system=platform.system().lower(), arch=platform.machine().lower(), cpu=cpu,
        cores=psutil.cpu_count(logical=False) or os.cpu_count() or 1, threads=os.cpu_count() or 1,
        ram_mb=ram_mb, ram_free_mb=vm.available >> 20, ram_bandwidth_gbs=bw, gpus=found)
    return _cache


def has_cuda_gpu() -> bool:
    """A GPU llama.cpp can use: an NVIDIA card with a working driver (CUDA build), an AMD Radeon (Vulkan)
    or Apple Silicon (Metal).
    The name stays: "cuda" is the device id of the GPU in the settings and the measurements."""
    return bool(gpus())
