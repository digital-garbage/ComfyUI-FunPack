"""Unit tests for BachVid-style K/V identity capture + injection.

These drive _build_block_replacement directly, faking comfy's attention module
so they run without a full ComfyUI environment. They cover:
  - K/V capture populates the bank inside the sigma window (and not outside it)
  - K/V injection lerps the running gen's K/V toward the blessed identity
  - the kvlock_scale multiplier (Phase 2 hook) scales injection strength
  - bless_kv EMA-merge / load / clear roundtrip
"""
import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _install_fake_attn(record):
    """Install a fake comfy.ldm.modules.attention whose optimized_attention
    records the (k, v) it receives and returns v unchanged."""
    mod = types.ModuleType("comfy.ldm.modules.attention")

    def optimized_attention(q, k, v, heads, *a, **kw):
        record["k"] = k
        record["v"] = v
        return v

    def optimized_attention_masked(q, k, v, heads, mask, *a, **kw):
        record["k"] = k
        record["v"] = v
        return v

    mod.optimized_attention = optimized_attention
    mod.optimized_attention_masked = optimized_attention_masked
    # Build the parent package chain so `import comfy.ldm.modules.attention` works.
    for name in ("comfy", "comfy.ldm", "comfy.ldm.modules"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["comfy.ldm.modules.attention"] = mod
    sys.modules["comfy.ldm.modules"].attention = mod
    return mod


def _drive(replacement, q, k, v, heads=4):
    """Invoke a block replacement, simulating a block that calls optimized_attention."""
    import comfy.ldm.modules.attention as attn_mod

    def original_block(args):
        # During replacement, attn_mod.optimized_attention is monkeypatched.
        out = attn_mod.optimized_attention(q, k, v, heads)
        return {"img": out}

    return replacement({}, {"original_block": original_block})


def test_kv_capture_populates_bank_inside_window():
    _install_fake_attn({})
    import ltx_enhancements as L

    buf = {}
    sigma_state = [0.5]  # inside default kv_sigma_gate (0.35, 0.85)
    rep = L._build_block_replacement(
        20, None, None, None, 0.0,
        sigma_state=sigma_state,
        kv_capture_buf=buf, kv_sigma_gate=(0.35, 0.85),
    )
    assert rep is not None
    q = torch.randn(2, 8, 16)
    k = torch.randn(2, 8, 16)
    v = torch.randn(2, 8, 16)
    _drive(rep, q, k, v)
    assert 20 in buf
    assert buf[20]["k"].shape == (8, 16)
    assert buf[20]["v"].shape == (8, 16)
    # captured = batch mean of the true k/v
    assert torch.allclose(buf[20]["k"].float(), k.float().mean(dim=0), atol=2e-3)


def test_kv_capture_skips_outside_window():
    _install_fake_attn({})
    import ltx_enhancements as L

    buf = {}
    sigma_state = [0.99]  # above the gate -> no capture
    rep = L._build_block_replacement(
        20, None, None, None, 0.0,
        sigma_state=sigma_state,
        kv_capture_buf=buf, kv_sigma_gate=(0.35, 0.85),
    )
    q = torch.randn(2, 8, 16); k = torch.randn(2, 8, 16); v = torch.randn(2, 8, 16)
    _drive(rep, q, k, v)
    assert buf == {}


def test_kv_inject_lerps_toward_identity():
    rec = {}
    _install_fake_attn(rec)
    import ltx_enhancements as L

    k_id = torch.zeros(8, 16)
    v_id = torch.zeros(8, 16)
    sigma_state = [0.5]
    rep = L._build_block_replacement(
        20, None, None, None, 0.0,
        sigma_state=sigma_state,
        kv_inject={"k": k_id, "v": v_id}, kv_inject_strength=0.5,
        kv_sigma_gate=(0.35, 0.85), kvlock_scale=[1.0],
    )
    q = torch.randn(2, 8, 16)
    k = torch.ones(2, 8, 16)
    v = torch.ones(2, 8, 16)
    _drive(rep, q, k, v)
    # alpha = 0.5 -> k_use = lerp(1, 0, 0.5) = 0.5
    assert torch.allclose(rec["k"].float(), torch.full_like(k, 0.5), atol=1e-3)
    assert torch.allclose(rec["v"].float(), torch.full_like(v, 0.5), atol=1e-3)


def test_kvlock_scale_modulates_injection():
    rec = {}
    _install_fake_attn(rec)
    import ltx_enhancements as L

    lock = [0.0]  # KV-Lock schedule says "don't inject this step"
    rep = L._build_block_replacement(
        20, None, None, None, 0.0,
        sigma_state=[0.5],
        kv_inject={"k": torch.zeros(8, 16), "v": torch.zeros(8, 16)},
        kv_inject_strength=0.5, kv_sigma_gate=(0.35, 0.85), kvlock_scale=lock,
    )
    k = torch.ones(2, 8, 16); v = torch.ones(2, 8, 16)
    _drive(rep, torch.randn(2, 8, 16), k, v)
    # scale 0 -> alpha 0 -> unchanged
    assert torch.allclose(rec["k"].float(), k, atol=1e-4)

    lock[0] = 1.0  # full schedule
    _drive(rep, torch.randn(2, 8, 16), k, v)
    assert torch.allclose(rec["k"].float(), torch.full_like(k, 0.5), atol=1e-3)


def test_kv_inject_skips_on_shape_mismatch():
    rec = {}
    _install_fake_attn(rec)
    import ltx_enhancements as L

    rep = L._build_block_replacement(
        20, None, None, None, 0.0,
        sigma_state=[0.5],
        kv_inject={"k": torch.zeros(99, 16), "v": torch.zeros(99, 16)},  # wrong seq
        kv_inject_strength=0.5, kv_sigma_gate=(0.35, 0.85), kvlock_scale=[1.0],
    )
    k = torch.ones(2, 8, 16); v = torch.ones(2, 8, 16)
    _drive(rep, torch.randn(2, 8, 16), k, v)
    # mismatch -> left unchanged, no crash
    assert torch.allclose(rec["k"].float(), k, atol=1e-4)


def test_bless_kv_ema_load_clear_roundtrip():
    import ltx_enhancements as L

    key = "__pytest_kv_roundtrip__"
    tmp, bl = L._kv_temp_path(key), L._kv_blessed_path(key)
    for p in (tmp, bl):
        if os.path.exists(p):
            os.remove(p)
    torch.save({20: {"k": torch.randn(8, 16).half(), "v": torch.randn(8, 16).half()}}, tmp)
    assert L.bless_kv(key) is True
    loaded = L._load_blessed_kv(key)
    assert 20 in loaded and loaded[20]["k"].shape == (8, 16)
    # second bless triggers EMA branch
    torch.save({20: {"k": torch.ones(8, 16).half(), "v": torch.ones(8, 16).half()}}, tmp)
    assert L.bless_kv(key) is True
    L.clear_refinement_data(key)
    assert not os.path.exists(bl) and not os.path.exists(tmp)
