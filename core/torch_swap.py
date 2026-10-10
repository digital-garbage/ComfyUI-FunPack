"""Switch this ComfyUI's torch to the CUDA 13 build of the SAME versions -- or leave it as it was.

Why: on a Blackwell rental, torch 2.10+cu128 sampled int8 models ~2x slower than bf16; the
same torch from the cu130 index made int8 faster than bf16 (2026-10-10).

Order is the safety. Everything is DOWNLOADED first (a failed download changes nothing), then
installed from those files only, then a fresh interpreter must import all three and report
CUDA 13. If any of that fails the old build is put back. A torch that does not import means a
ComfyUI that does not start, and then there is no UI left to repair it from.

Pins carry the local tag (`torch==2.10.0+cu130`): a bare `torch==2.10.0` is already satisfied
by `2.10.0+cu128`, so pip does nothing -- which is exactly how the hand-typed version failed.
"""

import shutil
import subprocess
import sys
import tempfile
from importlib import metadata

PACKAGES = ("torch", "torchvision", "torchaudio")
INDEX = "https://download.pytorch.org/whl/{tag}"
PYPI = "https://pypi.org/simple"
TARGET = "cu130"
NEED_FREE = 12 * 1024 ** 3       # the cu13 runtime wheels alone are several GB
# Compiled against one torch: worth checking after the swap, not touched by it.
COMPILED = ("xformers", "sageattn3", "flash_attn", "flash-attn")

step = None          # what is happening now, for the UI's progress card


def _version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _pip(args, timeout):
    return subprocess.run([sys.executable, "-m", "pip", *args, "--disable-pip-version-check"],
                          capture_output=True, text=True, timeout=timeout)


def _tail(proc):
    return "\n".join(((proc.stderr or "") + (proc.stdout or "")).strip().splitlines()[-6:])


def _pins(have, tag):
    """`torch==2.10.0+cu130` for each installed package; no tag means PyPI's own build."""
    return [f"{p}=={v.split('+')[0]}{'+' + tag if tag else ''}" for p, v in have.items()]


def _constraints(path):
    """Everything else stays at the version it is now. torch's own family and the CUDA runtime
    wheels are left free: they are what changes."""
    proc = _pip(["freeze", "--exclude-editable"], 120)
    free = PACKAGES + ("triton",)
    lines = [l for l in proc.stdout.splitlines()
             if "==" in l and not l.split("==")[0].lower().startswith(free + ("nvidia-",))]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _imports(names):
    """CUDA version a FRESH interpreter's torch reports, or the error. This process keeps the old
    torch in memory until the restart, so asking here would always say the old answer."""
    proc = subprocess.run([sys.executable, "-c", f"import {', '.join(names)}; print(torch.version.cuda)"],
                          capture_output=True, text=True, timeout=300)
    return (proc.stdout.strip().splitlines() or [""])[-1] if proc.returncode == 0 else None, _tail(proc)


def swap(timeout=3600):
    """-> {"installed": bool (the new build is on disk), "message": str, "rebuild": [names]}."""
    global step
    have = {p: _version(p) for p in PACKAGES if _version(p)}
    if "torch" not in have:
        return {"installed": False, "message": "torch is not installed in this Python, so there is nothing to switch."}
    import torch
    if not getattr(torch.version, "cuda", None):
        return {"installed": False, "message": "This torch has no CUDA (a Mac, or a CPU build): there is no CUDA 13 build to switch to."}
    torch_v = have["torch"]
    old_tag = torch_v.split("+")[1] if "+" in torch_v else None
    if old_tag == TARGET or (old_tag or "").startswith("cu13"):
        return {"installed": False, "message": f"torch {torch_v} is already a CUDA 13 build."}
    if old_tag and not old_tag.startswith("cu"):
        return {"installed": False, "message": f"torch {torch_v} is not a CUDA build ({old_tag}); switching it to CUDA 13 is not something this can judge."}
    work = tempfile.mkdtemp(prefix="funpack_torch_")
    try:
        if shutil.disk_usage(work).free < NEED_FREE:
            return {"installed": False, "message": f"Not enough free disk for the download (needs ~{NEED_FREE // 1024 ** 3} GB). Nothing was changed."}
        constraints = f"{work}/constraints.txt"
        _constraints(constraints)
        new = _pins(have, TARGET)
        step = f"Downloading {', '.join(new)} (several GB)…"
        proc = _pip(["download", "-d", f"{work}/wheels", "--index-url", INDEX.format(tag=TARGET),
                     "--extra-index-url", PYPI, "-c", constraints, *new], timeout)
        if proc.returncode != 0:
            return {"installed": False, "message": f"The download failed, so nothing was changed.\n{_tail(proc)}"}
        step = "Installing the downloaded build…"
        proc = _pip(["install", "--no-index", "--find-links", f"{work}/wheels", "-c", constraints, *new], timeout)
        cuda, err = _imports(list(have)) if proc.returncode == 0 else (None, _tail(proc))
        if cuda and cuda.startswith("13"):
            rebuild = [n for n in COMPILED if _version(n)]
            return {"installed": True, "rebuild": rebuild,
                    "message": f"torch {', '.join(new)} installed and imports with CUDA {cuda}."
                               + (f" Check these after the restart (built for the old torch): {', '.join(rebuild)}." if rebuild else "")}
        step = "That did not work; putting the old build back…"
        back = _pins(have, old_tag) if old_tag else [f"{p}=={v}" for p, v in have.items()]
        index = ["--index-url", INDEX.format(tag=old_tag), "--extra-index-url", PYPI] if old_tag else []
        undo = _pip(["install", *index, "-c", constraints, *back], timeout)
        restored, _ = _imports(list(have))
        if undo.returncode == 0 and restored:
            return {"installed": False, "message": f"The CUDA 13 build failed to install or import, so the old one ({torch_v}) is back.\n{err}"}
        return {"installed": False, "broken": True,
                "message": (f"The CUDA 13 build failed AND putting {torch_v} back failed. ComfyUI will probably not start "
                            f"after a restart. Fix it in a terminal:\n  {sys.executable} -m pip install --force-reinstall "
                            f"{' '.join(back)} {' '.join(index)}\n{err}\n{_tail(undo)}")}
    except subprocess.TimeoutExpired:
        return {"installed": False, "message": "pip took too long and was stopped. Check the terminal: torch may be half-installed."}
    finally:
        step = None
        shutil.rmtree(work, ignore_errors=True)
