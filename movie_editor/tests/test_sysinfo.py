"""Host facts for the About panel.

The contract that matters is robustness, not accuracy: About is the one screen that should
still render on a half-broken install, so every probe must degrade to None rather than
raise. These tests therefore break the probes on purpose and assert the shape survives.
"""
import builtins

import pytest

from movie_editor.backend import sysinfo


def test_collect_returns_the_full_shape():
    info = sysinfo.collect()
    for key in ("host", "os", "cpu", "memory", "gpus", "disk", "python", "torch", "comfyui"):
        assert key in info, f"About reads {key!r} directly"
    assert isinstance(info["gpus"], list)
    assert isinstance(info["cpu"], dict)
    # python is the one fact that can always be determined.
    assert info["python"]


def test_collect_survives_every_import_failing(monkeypatch):
    """psutil and torch are both optional from this module's point of view."""
    real_import = builtins.__import__

    def _boom(name, *args, **kwargs):
        if name in {"psutil", "torch", "comfyui_version"} or name.startswith("torch."):
            raise ImportError(f"no {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _boom)
    info = sysinfo.collect()
    assert info["gpus"] == []
    assert info["mps"] is False
    assert info["torch"]["version"] is None
    assert info["comfyui"] is None
    assert info["python"], "python version must survive a torch-less install"


def test_collect_survives_a_failing_subprocess(monkeypatch):
    """_cpu_name shells out on macOS/Linux; a missing binary must not kill the panel."""
    def _boom(*args, **kwargs):
        raise OSError("no such binary")

    monkeypatch.setattr(sysinfo.subprocess, "run", _boom)
    assert "cpu" in sysinfo.collect()


def test_collect_survives_an_unreadable_disk(monkeypatch):
    monkeypatch.setattr(sysinfo.shutil, "disk_usage", lambda _p: (_ for _ in ()).throw(OSError))
    assert sysinfo.collect()["disk"] == {"total_gb": None, "free_gb": None}


@pytest.mark.parametrize("value,expected", [
    (1024 ** 3, 1.0),
    (0, 0.0),
    (None, None),
    ("nonsense", None),
])
def test_bytes_gb(value, expected):
    assert sysinfo._bytes_gb(value) == expected


def test_mps_is_not_reported_when_a_cuda_gpu_exists(monkeypatch):
    """`mps` exists to explain an EMPTY graphics row; with CUDA present it must stay off
    so the panel never claims both."""
    monkeypatch.setattr(sysinfo, "_gpus", lambda: [{"name": "Fake", "vram_gb": 8.0, "capability": "sm_120"}])
    monkeypatch.setattr(sysinfo, "_mps", lambda: True)
    assert sysinfo.collect()["mps"] is False
