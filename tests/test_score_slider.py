"""Unit tests for the FreeSliders-style score-space taste slider on the
Scene Chain sampler (_build_score_slider_wrapper).

Covers the 3-pass combine eps_mod = eps_base + eta*(eps_+ - eps_-), the high-sigma
base-only warmup, the eta=0 no-op, and that it leaves a clean wrapper to restore.
"""
import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Minimal comfy stubs so `import samplers` works without a full ComfyUI env.
for _name in (
    "comfy", "comfy.k_diffusion", "comfy.k_diffusion.sampling",
    "comfy.model_sampling", "comfy.nested_tensor", "comfy.sample",
    "comfy.samplers", "comfy.utils",
):
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules["comfy.nested_tensor"].NestedTensor = object

import samplers  # noqa: E402

N = 10  # packed latent length used by the fake model


def _make_wrapper(eta, monkeypatch):
    s = samplers.FunPackLTXAVSceneChainSampler()
    # Single-stream: audio protect is an identity passthrough.
    monkeypatch.setattr(s, "_protect_audio", lambda steered, original: steered)
    model = types.SimpleNamespace(model_options={})
    liked = torch.ones(16)
    old = s._build_score_slider_wrapper(model, liked, eta=eta)
    return model.model_options["model_function_wrapper"], old


def _apply_fn(inp, ts, **c):
    """ε filled with the mean of the conditioning -> linear in c so we can predict
    the symmetric combine."""
    cc = c["c_crossattn"]
    return torch.full((1, 1, N), float(cc.float().mean()))


def _args(sigma, cond):
    return {"input": torch.zeros(1, 1, N), "timestep": torch.tensor([sigma]),
            "c": {"c_crossattn": cond}}


def test_score_slider_returns_no_previous_wrapper_when_none(monkeypatch):
    _, old = _make_wrapper(2.0, monkeypatch)
    assert old is None


def test_score_slider_warmup_is_base_only_at_high_sigma(monkeypatch):
    wrap, _ = _make_wrapper(2.0, monkeypatch)
    cond = torch.randn(1, 4, 16)
    out = wrap(_apply_fn, _args(0.9, cond))  # ramp = 1 - 0.9*2 < 0 -> base only
    assert torch.allclose(out, torch.full((1, 1, N), float(cond.float().mean())))


def test_score_slider_applies_in_quality_phase(monkeypatch):
    wrap, _ = _make_wrapper(2.0, monkeypatch)
    cond = torch.randn(1, 4, 16)
    base = torch.full((1, 1, N), float(cond.float().mean()))
    out = wrap(_apply_fn, _args(0.1, cond))  # ramp = 0.8 -> 3-pass combine
    assert not torch.allclose(out, base)
    # liked dir is all-positive and step is symmetric, so eps_+ > eps_- -> push is positive
    assert float(out.mean()) > float(base.mean())


def test_score_slider_eta_zero_is_noop(monkeypatch):
    wrap, _ = _make_wrapper(0.0, monkeypatch)
    cond = torch.randn(1, 4, 16)
    base = torch.full((1, 1, N), float(cond.float().mean()))
    out = wrap(_apply_fn, _args(0.1, cond))
    assert torch.allclose(out, base)


def test_score_slider_strength_scales_push(monkeypatch):
    cond = torch.randn(1, 4, 16)
    base = float(torch.full((1, 1, N), float(cond.float().mean())).mean())
    wrap1, _ = _make_wrapper(1.0, monkeypatch)
    wrap3, _ = _make_wrapper(3.0, monkeypatch)
    push1 = float(wrap1(_apply_fn, _args(0.1, cond)).mean()) - base
    push3 = float(wrap3(_apply_fn, _args(0.1, cond)).mean()) - base
    assert push3 > push1 > 0
    assert abs(push3 - 3.0 * push1) < 1e-4  # linear in eta
