"""Unit tests for the native Best-FaceID identity_transfer port.

Covers the three pieces added to FunPackLTXAVSceneChainSampler:
  - _tag_last_guide_entry: best-effort marks the most-recently-appended
    guide_attention_entry, never touching earlier entries.
  - _apply_configured_guides: routes the phase tag ONLY to the guide entry the
    editor marked identity_pin (never prior-scene/mid-scene/template guides that
    happen to share the same call).
  - _install_identity_phase / _strip_identity_phase / _rotate_identity_phase: the
    tag+strip RoPE patch that reproduces the LoRA's trained-time source_phase
    tag, scoped to the exact token range of the tagged guide (not the whole
    guide-token tail, which may include co-active unrelated guides).
"""
import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

for _name in (
    "comfy", "comfy.k_diffusion", "comfy.k_diffusion.sampling",
    "comfy.model_sampling", "comfy.nested_tensor", "comfy.sample",
    "comfy.samplers", "comfy.utils",
):
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules["comfy.nested_tensor"].NestedTensor = object

import samplers  # noqa: E402


def _sampler():
    return samplers.FunPackLTXAVSceneChainSampler()


def _install_node_helpers():
    """node_helpers.conditioning_set_values, the same real implementation shape
    _append_guide_attention_entry itself relies on (copy metadata dict, overwrite
    the given keys) — reimplemented here so the test runs without a full ComfyUI env."""
    mod = types.ModuleType("node_helpers")

    def conditioning_set_values(conditioning, values=None, append=False):
        values = values or {}
        out = []
        for t in conditioning:
            n = [t[0], t[1].copy()]
            for k, v in values.items():
                n[1][k] = v
            out.append(n)
        return out

    mod.conditioning_set_values = conditioning_set_values
    sys.modules["node_helpers"] = mod
    return mod


def _cond_with_entries(entries):
    return [[torch.zeros(1, 1, 4), {"guide_attention_entries": list(entries)}]]


# ── _tag_last_guide_entry ──────────────────────────────────────────────────────

def test_tag_last_guide_entry_marks_only_newest_entry():
    _install_node_helpers()
    s = _sampler()
    cond = _cond_with_entries([{"pre_filter_count": 4}, {"pre_filter_count": 8}])
    tagged = s._tag_last_guide_entry(cond, "funpack_identity_phase", 2.0)
    entries = tagged[0][1]["guide_attention_entries"]
    assert entries[0].get("funpack_identity_phase") is None
    assert entries[1]["funpack_identity_phase"] == 2.0
    # Original untouched (conditioning_set_values copies, doesn't mutate in place).
    assert "funpack_identity_phase" not in cond[0][1]["guide_attention_entries"][1]


def test_tag_last_guide_entry_noop_without_entries():
    _install_node_helpers()
    s = _sampler()
    cond = [[torch.zeros(1, 1, 4), {}]]
    out = s._tag_last_guide_entry(cond, "funpack_identity_phase", 2.0)
    assert out is cond  # unchanged object, nothing to tag


def test_tag_last_guide_entry_noop_without_node_helpers():
    sys.modules.pop("node_helpers", None)
    s = _sampler()
    cond = _cond_with_entries([{"pre_filter_count": 4}])
    out = s._tag_last_guide_entry(cond, "funpack_identity_phase", 2.0)
    assert out is cond


# ── _apply_configured_guides phase-tag routing ─────────────────────────────────

def test_phase_tag_applies_only_to_identity_pin_entry(monkeypatch):
    s = _sampler()
    calls = []

    def fake_append_media(chunk, filename, frame_idx, apply_at, strength, positive, negative, vae,
                          phase_tag=0.0):
        calls.append((filename, phase_tag))
        return chunk, positive, negative, 0, 0

    monkeypatch.setattr(s, "_append_media_guide_at", fake_append_media)
    guide_list = [
        {"enabled": True, "source": "image", "media_ref": "pin", "identity_pin": True, "strength": 0.35},
        {"enabled": True, "source": "image", "media_ref": "prior", "strength": 0.35},  # NOT the pin
    ]
    scene_media_by_ref = {"pin": "pin.png", "prior": "prior.png"}
    chunk, pos, neg, head, tail, phase_applied = s._apply_configured_guides(
        {}, 1, guide_list, {}, [], scene_media_by_ref, [], [], object(),
        identity_transfer_enabled=True, identity_transfer_phase=3.0,
    )
    assert calls == [("pin.png", 3.0), ("prior.png", 0.0)]
    assert phase_applied is True


def test_phase_tag_off_when_identity_transfer_disabled(monkeypatch):
    s = _sampler()
    calls = []
    monkeypatch.setattr(s, "_append_media_guide_at",
                         lambda chunk, filename, *a, phase_tag=0.0, **kw: (calls.append(phase_tag), chunk, [], [], 0, 0)[1:])
    guide_list = [{"enabled": True, "source": "image", "media_ref": "pin", "identity_pin": True, "strength": 0.35}]
    _, _, _, _, _, phase_applied = s._apply_configured_guides(
        {}, 1, guide_list, {}, [], {"pin": "pin.png"}, [], [], object(),
        identity_transfer_enabled=False, identity_transfer_phase=2.0,
    )
    assert calls == [0.0]
    assert phase_applied is False


# ── _install_identity_phase / _strip_identity_phase / _rotate_identity_phase ──

class _FakeLTXV:
    def _prepare_timestep(self, timestep, batch_size, hidden_dtype, **kwargs):
        return ("timestep", batch_size, hidden_dtype)

    def _prepare_positional_embeddings(self, pixel_coords, frame_rate, x_dtype):
        # cos/sin shaped [T=10, L=4]; identity token slice under test is [6:9).
        T, L = 10, 4
        cos = torch.arange(T * L, dtype=torch.float32).reshape(T, L)
        sin = torch.arange(T * L, dtype=torch.float32).reshape(T, L) * -1.0
        return (cos, sin, False)


def _fake_model(ltxv):
    inner = types.SimpleNamespace(diffusion_model=ltxv)
    return types.SimpleNamespace(model=inner)


def test_install_is_idempotent_and_strip_restores_originals():
    s = _sampler()
    ltxv = _FakeLTXV()
    orig_pt = ltxv._prepare_timestep
    orig_pe = ltxv._prepare_positional_embeddings

    handle1 = s._install_identity_phase(_fake_model(ltxv))
    assert handle1 is not None
    assert ltxv._prepare_timestep is not orig_pt
    handle2 = s._install_identity_phase(_fake_model(ltxv))
    assert handle2 is None  # already installed — idempotent, no double-wrap

    s._strip_identity_phase(handle1)
    # Bound-method objects aren't identity-stable across separate attribute accesses in
    # CPython (a fresh wrapper is created each time) — compare by equality (same __func__ +
    # __self__) instead of `is`.
    assert ltxv._prepare_timestep == orig_pt
    assert ltxv._prepare_positional_embeddings == orig_pe
    assert not hasattr(ltxv, "_funpack_identity_range")


def test_prepare_timestep_stashes_range_for_tagged_entry_only():
    s = _sampler()
    ltxv = _FakeLTXV()
    handle = s._install_identity_phase(_fake_model(ltxv))
    entries = [
        {"pre_filter_count": 16, "surviving_count": 2},                                    # unrelated guide, first
        {"pre_filter_count": 16, "surviving_count": 3, "funpack_identity_phase": 2.5},      # the tagged one
        {"pre_filter_count": 16, "surviving_count": 4},                                    # unrelated guide, after
    ]
    ltxv._prepare_timestep(torch.zeros(1), 1, torch.float32,
                           resolved_guide_entries=entries, num_guide_tokens=9)
    assert ltxv._funpack_identity_range == (2, 3, 9)  # offset past the first entry's 2 tokens
    assert ltxv._funpack_identity_phase_value == 2.5
    s._strip_identity_phase(handle)


def test_prepare_timestep_clears_range_without_tagged_entry():
    s = _sampler()
    ltxv = _FakeLTXV()
    handle = s._install_identity_phase(_fake_model(ltxv))
    ltxv._funpack_identity_range = (0, 1, 1)  # stale from a previous scene
    entries = [{"pre_filter_count": 16, "surviving_count": 5}]
    ltxv._prepare_timestep(torch.zeros(1), 1, torch.float32,
                           resolved_guide_entries=entries, num_guide_tokens=5)
    assert ltxv._funpack_identity_range is None
    s._strip_identity_phase(handle)


def test_prepare_pe_rotates_only_the_tagged_range():
    s = _sampler()
    ltxv = _FakeLTXV()
    handle = s._install_identity_phase(_fake_model(ltxv))
    baseline_cos, baseline_sin, _ = ltxv._prepare_positional_embeddings(None, 25, torch.float32)

    entries = [
        {"pre_filter_count": 16, "surviving_count": 2},
        {"pre_filter_count": 16, "surviving_count": 3, "funpack_identity_phase": 2.5},
        {"pre_filter_count": 16, "surviving_count": 4},
    ]
    ltxv._prepare_timestep(torch.zeros(1), 1, torch.float32,
                           resolved_guide_entries=entries, num_guide_tokens=9)
    cos, sin, split_mode = ltxv._prepare_positional_embeddings(None, 25, torch.float32)

    # T=10, num_guide_tokens=9 -> guide_start=1; offset=2, length=3 -> rotated slice [3:6).
    untouched = list(range(0, 3)) + list(range(6, 10))
    for i in untouched:
        assert torch.equal(cos[i], baseline_cos[i]), f"row {i} should be untouched"
        assert torch.equal(sin[i], baseline_sin[i]), f"row {i} should be untouched"
    rotated_rows = [3, 4, 5]
    changed = any(not torch.equal(cos[i], baseline_cos[i]) for i in rotated_rows)
    assert changed, "rotated slice should differ from baseline"
    assert split_mode is False  # passthrough element preserved
    s._strip_identity_phase(handle)


def test_rotate_identity_phase_is_noop_for_zero_phase_or_length():
    s = _sampler()
    pe = (torch.ones(5, 4), torch.zeros(5, 4), False)
    assert s._rotate_identity_phase(pe, 0, 0, 5, 2.0) is pe
    assert s._rotate_identity_phase(pe, 0, 2, 5, 0.0) is pe


def test_rotate_identity_phase_rejects_out_of_bounds_range():
    s = _sampler()
    cos = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    sin = -cos.clone()
    pe = (cos, sin, False)
    out = s._rotate_identity_phase(pe, offset=10, length=3, num_guide_tokens=5, phase=2.0)
    assert out is pe  # start/end fall outside T -> safe no-op
