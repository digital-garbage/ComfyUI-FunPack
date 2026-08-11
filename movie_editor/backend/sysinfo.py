"""Host facts for the About panel — the machine ComfyUI runs on, not the browser's.

That distinction is the whole point on a rental: the frontend already knows the laptop it
is displayed on, and nobody cares. What matters is the box doing the sampling — which GPU,
how much VRAM, which torch, how much disk is left for outputs.

Everything here is best-effort and individually guarded: a missing probe returns None and
the panel renders a dash. Nothing in this module may raise, because the About panel is the
one screen that should still work when the rest of the install is broken.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _bytes_gb(n):
    """Bytes -> GB as a float, using the 1024 base the OS's own About panels use."""
    try:
        return round(float(n) / (1024 ** 3), 2)
    except Exception:
        return None


def _cpu_name():
    """A human CPU model string. platform.processor() is famously empty on Linux and
    just 'arm' on Apple silicon, so ask the OS directly first."""
    try:
        if sys.platform == "darwin":
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                 capture_output=True, text=True, timeout=3)
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        elif sys.platform.startswith("linux"):
            cpuinfo = Path("/proc/cpuinfo")
            if cpuinfo.exists():
                for line in cpuinfo.read_text(errors="replace").splitlines():
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine() or None


def _cpu():
    info = {"name": _cpu_name(), "arch": platform.machine() or None,
            "cores": None, "threads": os.cpu_count()}
    try:
        import psutil  # a ComfyUI requirement, so normally present
        info["cores"] = psutil.cpu_count(logical=False)
        info["threads"] = psutil.cpu_count(logical=True) or info["threads"]
    except Exception:
        pass
    return info


def _memory():
    try:
        import psutil
        vm = psutil.virtual_memory()
        return {"total_gb": _bytes_gb(vm.total), "available_gb": _bytes_gb(vm.available)}
    except Exception:
        pass
    try:  # stdlib fallback (Linux/macOS)
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        return {"total_gb": _bytes_gb(total), "available_gb": None}
    except Exception:
        return {"total_gb": None, "available_gb": None}


def _gpus():
    """Every visible CUDA/ROCm device. Empty list on a CPU-only box (the local Mac)."""
    out = []
    try:
        import torch
        if not torch.cuda.is_available():
            return out
        for i in range(torch.cuda.device_count()):
            entry = {"name": None, "vram_gb": None, "capability": None}
            try:
                entry["name"] = torch.cuda.get_device_name(i)
            except Exception:
                pass
            try:
                props = torch.cuda.get_device_properties(i)
                entry["vram_gb"] = _bytes_gb(props.total_memory)
                entry["capability"] = f"sm_{props.major}{props.minor}"
            except Exception:
                pass
            out.append(entry)
    except Exception:
        pass
    return out


def _mps():
    """Apple-silicon GPU, which reports through MPS rather than CUDA."""
    try:
        import torch
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return True
    except Exception:
        pass
    return False


def _disk():
    """Free space where FunPack lives — the volume that fills up with renders."""
    try:
        usage = shutil.disk_usage(str(REPO_ROOT))
        return {"total_gb": _bytes_gb(usage.total), "free_gb": _bytes_gb(usage.free)}
    except Exception:
        return {"total_gb": None, "free_gb": None}


def _os_name():
    try:
        if sys.platform == "darwin":
            return f"macOS {platform.mac_ver()[0]}".strip()
        if sys.platform.startswith("win"):
            return f"Windows {platform.release()}"
        # Prefer the distro's own pretty name over the kernel version.
        osr = Path("/etc/os-release")
        if osr.exists():
            for line in osr.read_text(errors="replace").splitlines():
                if line.startswith("PRETTY_NAME="):
                    return line.split("=", 1)[1].strip().strip('"')
        return f"{platform.system()} {platform.release()}"
    except Exception:
        return None


def _torch():
    info = {"version": None, "cuda": None, "attention": None}
    try:
        import torch
        info["version"] = torch.__version__
        info["cuda"] = getattr(torch.version, "cuda", None)
    except Exception:
        pass
    # Which fast-attention backend is actually importable. This is the line that answers
    # "is sage actually installed on this rental?" without digging through the launch args.
    for mod, label in (("sageattention", "SageAttention"), ("flash_attn", "FlashAttention"),
                       ("xformers", "xformers")):
        try:
            m = __import__(mod)
            info["attention"] = f"{label} {getattr(m, '__version__', '')}".strip()
            break
        except Exception:
            continue
    return info


def _comfy_version():
    try:
        import comfyui_version
        return comfyui_version.__version__
    except Exception:
        return None


def collect():
    """The whole About payload. Static enough that the frontend caches it per load."""
    gpus = _gpus()
    return {
        "host": platform.node() or None,
        "os": _os_name(),
        "cpu": _cpu(),
        "memory": _memory(),
        "gpus": gpus,
        "mps": _mps() if not gpus else False,
        "disk": _disk(),
        "python": platform.python_version(),
        "torch": _torch(),
        "comfyui": _comfy_version(),
    }
