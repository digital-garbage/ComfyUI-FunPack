"""Switching torch to its CUDA 13 build: download first, install from those files, prove it imports,
or put the old build back. pip is faked: a real run downloads several GB."""

import subprocess
import time
from collections import namedtuple

import pytest

from core import torch_swap

Disk = namedtuple("Disk", "total used free")


@pytest.fixture
def fake(monkeypatch):
    """pip calls recorded; each verb answers from `codes`; versions from `have`; imports from `cuda`."""
    state = {"calls": [], "codes": {}, "cuda": ["13.0"],
             "have": {"torch": "2.10.0+cu128", "torchvision": "0.25.0+cu128", "torchaudio": "2.10.0+cu128"}}

    def pip(args, timeout):
        state["calls"].append(args)
        out = "torch==2.10.0+cu128\nnumpy==2.1.0\nnvidia-cublas-cu12==12.8\ntriton==3.6.0\nfoo @ file:///x\n" if args[0] == "freeze" else ""
        return subprocess.CompletedProcess(args, state["codes"].get(args[0], 0), out, "pip said no")

    monkeypatch.setattr(torch_swap, "_pip", pip)
    monkeypatch.setattr(torch_swap, "_version", lambda n: state["have"].get(n))
    def imports(names):
        state["imported"] = names
        return (state["cuda"].pop(0) if state["cuda"] else None), "ImportError: boom"
    monkeypatch.setattr(torch_swap, "_imports", imports)
    monkeypatch.setattr(torch_swap.shutil, "disk_usage", lambda p: Disk(0, 0, 100 * 1024 ** 3))
    import torch
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    return state


def verbs(state):
    return [c[0] for c in state["calls"]]


def test_success_downloads_then_installs_the_tagged_pins_from_those_files_only(fake):
    r = torch_swap.swap()
    assert r["installed"] is True and "13.0" in r["message"]
    assert verbs(fake) == ["freeze", "download", "install"]
    download, install = fake["calls"][1], fake["calls"][2]
    assert "torch==2.10.0+cu130" in download, "a bare torch==2.10.0 is satisfied by +cu128 and pip does nothing"
    assert "torchaudio==2.10.0+cu130" in install and "--no-index" in install, "install must not reach the network"
    assert torch_swap.step is None


def test_constraints_hold_everything_but_torch_triton_and_the_cuda_runtime(fake, monkeypatch, tmp_path):
    seen = {}
    real = torch_swap._constraints

    def spy(path):
        real(path)
        seen["text"] = open(path).read()
    monkeypatch.setattr(torch_swap, "_constraints", spy)
    torch_swap.swap()
    assert seen["text"].split() == ["numpy==2.1.0"]


def test_a_failed_download_changes_nothing(fake):
    fake["codes"]["download"] = 1
    r = torch_swap.swap()
    assert r["installed"] is False and "nothing was changed" in r["message"]
    assert "install" not in verbs(fake)


def test_a_build_that_does_not_import_is_rolled_back_to_the_old_tagged_build(fake):
    fake["cuda"] = [None, "12.8"]          # new torch fails to import; the restored one works
    r = torch_swap.swap()
    assert r["installed"] is False and "old one" in r["message"] and "boom" in r["message"]
    undo = fake["calls"][-1]
    assert "torch==2.10.0+cu128" in undo and "https://download.pytorch.org/whl/cu128" in undo


def test_a_failed_rollback_says_it_plainly_with_the_command_to_fix_it(fake):
    fake["cuda"] = [None, None]
    r = torch_swap.swap()
    assert r.get("broken") is True and "--force-reinstall" in r["message"] and "torch==2.10.0+cu128" in r["message"]


@pytest.mark.parametrize("torch_v,why", [("2.10.0+cu130", "already"), ("2.10.0+cpu", "not a CUDA build")])
def test_refuses_what_it_should_not_touch(fake, torch_v, why):
    fake["have"]["torch"] = torch_v
    r = torch_swap.swap()
    assert r["installed"] is False and why in r["message"] and fake["calls"] == []


def test_refuses_without_disk_room(fake, monkeypatch):
    monkeypatch.setattr(torch_swap.shutil, "disk_usage", lambda p: Disk(0, 0, 1024 ** 3))
    r = torch_swap.swap()
    assert r["installed"] is False and "disk" in r["message"] and fake["calls"] == []


def test_untagged_pypi_torch_rolls_back_to_pypi(fake):
    fake["have"] = {"torch": "2.10.0", "torchvision": "0.25.0"}
    fake["cuda"] = [None, "12.8"]
    torch_swap.swap()
    assert fake["imported"] == ["torch", "torchvision"], "a missing torchaudio is not a failed import"
    undo = fake["calls"][-1]
    assert "torch==2.10.0" in undo and not any("download.pytorch.org" in a for a in undo)


# ---- the route: the git lock's rules, and a restart only when the new build is on disk ----

from core.tests.test_routes_update import _request, server  # noqa: E402,F401
from core import routes  # noqa: E402


def _finish(server):
    for _ in range(100):
        _, state = _request(server["port"], "GET", "/funpack/api/torch/cuda13")
        if not state["running"]:
            return state
        time.sleep(0.05)
    raise AssertionError("the switch never finished")


def test_route_starts_a_job_and_restarts_only_after_an_install(server, monkeypatch):
    monkeypatch.setattr(routes, "_generation_running", lambda: False)
    monkeypatch.setattr(torch_swap, "swap", lambda: {"installed": False, "message": "download failed"})
    status, body = _request(server["port"], "POST", "/funpack/api/torch/cuda13")
    assert status == 200 and body == {"started": True}, "answers at once: the download outlives one proxied request"
    state = _finish(server)
    assert state["result"]["restarting"] is False and state["result"]["message"] == "download failed"
    assert server["restarts"] == []
    monkeypatch.setattr(torch_swap, "swap", lambda: {"installed": True, "message": "ok"})
    _request(server["port"], "POST", "/funpack/api/torch/cuda13")
    assert _finish(server)["result"]["restarting"] is True


def test_a_second_start_while_one_runs_is_refused(server, monkeypatch):
    import threading
    gate = threading.Event()
    monkeypatch.setattr(routes, "_generation_running", lambda: False)
    monkeypatch.setattr(torch_swap, "swap", lambda: (gate.wait(5), {"installed": False, "message": "x"})[1])
    _request(server["port"], "POST", "/funpack/api/torch/cuda13")
    status, body = _request(server["port"], "POST", "/funpack/api/torch/cuda13")
    gate.set()
    assert status == 409 and "already running" in body["detail"]
    _finish(server)


def test_route_refuses_during_a_generation(server, monkeypatch):
    monkeypatch.setattr(routes, "_generation_running", lambda: True)
    monkeypatch.setattr(torch_swap, "swap", lambda: (_ for _ in ()).throw(AssertionError("must not run")))
    status, body = _request(server["port"], "POST", "/funpack/api/torch/cuda13")
    assert status == 409 and "generation" in body["detail"]


def test_a_torch_without_cuda_is_refused_before_any_download(fake, monkeypatch):
    import torch
    monkeypatch.setattr(torch.version, "cuda", None)
    r = torch_swap.swap()
    assert r["installed"] is False and "no CUDA" in r["message"] and fake["calls"] == []
