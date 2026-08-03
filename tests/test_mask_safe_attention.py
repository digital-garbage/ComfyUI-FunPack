"""Unit tests for the Blackwell (sm_120) xformers masked-attention fallback in samplers.py.

On GPUs newer than capability (9, 0) xformers has no kernel that accepts a tensor attn_bias,
so i2v ANCHOR scenes (unmasked attention) generate fine but GUIDE scenes (LTX guide path
passes a tensor mask) crash. _funpack_install_mask_safe_attention installs a per-call override
(ComfyUI wrap_attn) that routes only MASKED attention to SDPA when that exact combo is present.

Covers: override routing (masked→pytorch, unmasked→original, mask as kwarg or positional),
and the install gating (only xformers + too-new GPU; respects/ignores other backends, existing
overrides, missing CUDA; idempotent).
"""
import sys
import types
from pathlib import Path

import _comfy_stubs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import samplers  # noqa: E402


def _install_attn_module(*, backend="xformers", with_pytorch_sentinel=True):
    """Point comfy.ldm.modules.attention at a fake with a selectable active backend.

    Installed through _comfy_stubs.install_module so the PARENT attribute moves too. Registering
    only the sys.modules entry is not enough: `import comfy.ldm.modules.attention as _am`
    (what samplers.py does) resolves through `comfy.ldm.modules.attention` the attribute, so
    a stub left behind by another test module kept winning here and the install under test
    bailed out at its first guard.
    """
    def attention_xformers(*a, **kw):
        return "xformers"

    def attention_pytorch(*a, **kw):
        return "pytorch"

    return _comfy_stubs.install_module(
        "comfy.ldm.modules.attention",
        attention_xformers=attention_xformers,
        attention_pytorch=attention_pytorch,
        optimized_attention=attention_xformers if backend == "xformers" else attention_pytorch,
    )


def _fake_model():
    return types.SimpleNamespace(model_options={})


def _force_capability(monkeypatch, cap, available=True):
    cuda = types.SimpleNamespace(
        is_available=lambda: available,
        get_device_capability=lambda *a, **k: cap,
    )
    monkeypatch.setattr(samplers.torch, "cuda", cuda, raising=False)


# ── override routing ──────────────────────────────────────────────────────────
def test_override_routes_masked_to_pytorch_kwarg():
    _install_attn_module()
    sentinel = object()
    out = samplers._funpack_mask_safe_attention_override(
        lambda *a, **k: "xformers", "q", "k", "v", 8, mask=sentinel,
    )
    assert out == "pytorch"


def test_override_routes_masked_to_pytorch_positional():
    _install_attn_module()
    # (q, k, v, heads, mask) — mask as the 5th positional arg.
    out = samplers._funpack_mask_safe_attention_override(
        lambda *a, **k: "xformers", "q", "k", "v", 8, object(),
    )
    assert out == "pytorch"


def test_override_passes_unmasked_to_original():
    _install_attn_module()
    out = samplers._funpack_mask_safe_attention_override(
        lambda *a, **k: "xformers", "q", "k", "v", 8, mask=None,
    )
    assert out == "xformers"


# ── install gating ────────────────────────────────────────────────────────────
def test_installs_on_xformers_and_blackwell(monkeypatch):
    _install_attn_module(backend="xformers")
    _force_capability(monkeypatch, (12, 0))
    m = _fake_model()
    samplers._funpack_install_mask_safe_attention(m)
    assert (m.model_options["transformer_options"]["optimized_attention_override"]
            is samplers._funpack_mask_safe_attention_override)


def test_no_install_on_supported_capability(monkeypatch):
    _install_attn_module(backend="xformers")
    _force_capability(monkeypatch, (9, 0))  # Hopper — xformers tensor-bias kernels OK
    m = _fake_model()
    samplers._funpack_install_mask_safe_attention(m)
    assert "optimized_attention_override" not in m.model_options.get("transformer_options", {})


def test_no_install_when_backend_not_xformers(monkeypatch):
    _install_attn_module(backend="pytorch")  # user already on SDPA
    _force_capability(monkeypatch, (12, 0))
    m = _fake_model()
    samplers._funpack_install_mask_safe_attention(m)
    assert "optimized_attention_override" not in m.model_options.get("transformer_options", {})


def test_no_install_without_cuda(monkeypatch):
    _install_attn_module(backend="xformers")
    _force_capability(monkeypatch, (12, 0), available=False)
    m = _fake_model()
    samplers._funpack_install_mask_safe_attention(m)
    assert "optimized_attention_override" not in m.model_options.get("transformer_options", {})


def test_does_not_stomp_existing_override(monkeypatch):
    _install_attn_module(backend="xformers")
    _force_capability(monkeypatch, (12, 0))
    other = lambda func, *a, **k: func(*a, **k)
    m = types.SimpleNamespace(model_options={"transformer_options": {"optimized_attention_override": other}})
    samplers._funpack_install_mask_safe_attention(m)
    assert m.model_options["transformer_options"]["optimized_attention_override"] is other


def test_idempotent(monkeypatch):
    _install_attn_module(backend="xformers")
    _force_capability(monkeypatch, (12, 0))
    m = _fake_model()
    samplers._funpack_install_mask_safe_attention(m)
    samplers._funpack_install_mask_safe_attention(m)
    assert (m.model_options["transformer_options"]["optimized_attention_override"]
            is samplers._funpack_mask_safe_attention_override)
