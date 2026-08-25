"""Value-function training runs where the sample already is.

These nets are tiny in FLOPs and not in parameters: a 5120-wide conditioning makes the first
layer 94% of ~4M weights. Twenty Adam steps over that measured ~1.1s on this Mac's CPU and
about 7.5s on the rental's vCPU — per rated generation, three value functions over. The
conditioning is already on the GPU when there is one.
"""
import sys

import pytest
import torch

sys.path.insert(0, ".")

from value_function import LatentValueFunction, OnlineValueFunction  # noqa: E402


def _trained(dim=32, samples=6):
    vf = OnlineValueFunction(hidden_dim=dim)
    for i in range(samples):
        vf.buffer_c.append(torch.randn(dim))
        vf.buffer_r.append(float(i % 3))
    return vf


def test_a_cpu_sample_trains_where_it_is():
    vf = _trained()
    assert vf._training_device(torch.zeros(4), torch.device("cpu")) == torch.device("cpu")


def test_a_non_tensor_does_not_pick_a_device():
    vf = _trained()
    home = torch.device("cpu")
    assert vf._training_device(None, home) is home
    assert vf._training_device("not a tensor", home) is home


def test_training_still_learns_on_cpu():
    """The device change must not alter what training does."""
    torch.manual_seed(0)
    vf = _trained()
    before = [p.clone() for p in vf.parameters()]
    vf.train_on(torch.randn(1, 8, 32), 0.9)
    assert vf.n_trained == 1
    assert any(not torch.equal(a, b) for a, b in zip(before, vf.parameters()))


def test_the_module_is_left_on_its_home_device():
    """Inference elsewhere expects a CPU module and the checkpoint must stay portable."""
    vf = _trained()
    vf.train_on(torch.randn(1, 8, 32), 0.5)
    assert all(p.device == torch.device("cpu") for p in vf.parameters())


def test_one_sample_is_not_trained_on():
    """Pairwise ranking needs two. n_trained must not count a step that never happened."""
    vf = OnlineValueFunction(hidden_dim=32)
    vf.train_on(torch.randn(1, 8, 32), 0.5)
    assert vf.n_trained == 0
    assert len(vf.buffer_c) == 1


def test_the_buffer_stays_on_the_cpu():
    """It is what gets written to the checkpoint; a CUDA buffer would not be portable."""
    vf = _trained()
    vf.train_on(torch.randn(1, 8, 32), 0.5)
    assert all(c.device == torch.device("cpu") for c in vf.buffer_c)


def test_the_latent_value_function_inherits_the_same_path():
    vf = LatentValueFunction(hidden_dim=64)
    for i in range(4):
        vf.buffer_c.append(torch.randn(64))
        vf.buffer_r.append(float(i % 2))
    vf.train_on(torch.randn(1, 1, 4096), 0.7)
    assert vf.n_trained == 1


def test_a_cpu_snapshot_still_goes_to_the_gpu(monkeypatch):
    """The x0 value function's sample is loaded with map_location="cpu". Keying the device on
    the sample left it training on the CPU — 10.1s a run against 46ms for the same net."""
    import value_function as V
    monkeypatch.setattr(V.torch.cuda, "is_available", lambda: True)
    home = torch.device("cpu")
    assert V.OnlineValueFunction._training_device(torch.zeros(4), home) == torch.device("cuda")


def test_without_a_gpu_it_stays_home(monkeypatch):
    import value_function as V
    monkeypatch.setattr(V.torch.cuda, "is_available", lambda: False)
    home = torch.device("cpu")
    assert V.OnlineValueFunction._training_device(torch.zeros(4), home) is home
