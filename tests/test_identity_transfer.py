"""Unit tests for the native Best-FaceID identity_transfer port (overlap tokens + ArcFace
projector), a port of ComfyUI-BFSNodes' LTX Identity Transfer.

Covers:
  - identity_transfer.rotate_overlap_freqs / append_context_tokens: pure-tensor pieces,
    no torch model needed.
  - FunPackLTXAVSceneChainSampler._apply_configured_guides: routes the identity_pin entry
    to the new mechanism (returns its filename, skips the keyframe-blend append) only when
    identity_transfer_enabled; unaffected guides append normally either way.
  - _install_identity_overlap / _strip_identity_overlap: tag+strip idempotency, ref-token
    append (_process_input), clean timestep for ref tokens (_prepare_timestep), source-phase
    RoPE rotation (_prepare_positional_embeddings), and ref-token trim (_process_output).
  - _resolve_identity_overlap: lazy, memoized once per run() call.
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
    "comfy.ldm", "comfy.ldm.lightricks", "comfy.ldm.lightricks.model", "comfy.ldm.lightricks.av_model",
):
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules["comfy.nested_tensor"].NestedTensor = object
sys.modules["comfy.ldm.lightricks.model"].latent_to_pixel_coords = (
    lambda latent_coords, scale_factors, causal_fix: latent_coords)


class _CompressedTimestep:
    def __init__(self, data, num_frames, patches_per_frame=1):
        self.data = data
        self.num_frames = num_frames
        self.patches_per_frame = patches_per_frame


sys.modules["comfy.ldm.lightricks.av_model"].CompressedTimestep = _CompressedTimestep

import samplers  # noqa: E402
import identity_transfer as idt  # noqa: E402


def _sampler():
    return samplers.FunPackLTXAVSceneChainSampler()


# ── identity_transfer.rotate_overlap_freqs ─────────────────────────────────────

def test_rotate_overlap_freqs_noop_for_zero_ref_len_or_seg():
    cos = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    sin = -cos.clone()
    pe = (cos, sin, False)
    assert idt.rotate_overlap_freqs(pe, 0, 2.0) is pe
    assert idt.rotate_overlap_freqs(pe, 2, 0.0) is pe


def test_rotate_overlap_freqs_rotates_only_last_ref_len_tokens():
    T, L = 10, 4
    cos = torch.arange(T * L, dtype=torch.float32).reshape(T, L)
    sin = torch.arange(T * L, dtype=torch.float32).reshape(T, L) * -1.0
    pe = (cos, sin, False)
    ref_len = 3
    out_cos, out_sin, split = idt.rotate_overlap_freqs(pe, ref_len, 2.0)
    untouched = list(range(0, T - ref_len))
    for i in untouched:
        assert torch.equal(out_cos[i], cos[i])
        assert torch.equal(out_sin[i], sin[i])
    rotated = list(range(T - ref_len, T))
    changed = any(not torch.equal(out_cos[i], cos[i]) for i in rotated)
    assert changed
    assert split is False


# ── identity_transfer.append_context_tokens ────────────────────────────────────

def test_append_context_tokens_pads_and_extends_mask():
    ce = torch.zeros(2, 5, 8)  # batch=2, tokens=5, dim=8
    cond = [[ce, {"attention_mask": torch.ones(2, 5)}]]
    tokens = torch.ones(1, 3, 4)  # batch=1 (broadcast), 3 new tokens, smaller dim
    out = idt.append_context_tokens(cond, tokens)
    new_ce, new_extra = out[0]
    assert new_ce.shape == (2, 8, 8)  # 5+3 tokens, dim padded back to 8
    assert torch.equal(new_ce[:, 5:, :4], torch.ones(2, 3, 4))
    assert torch.equal(new_ce[:, 5:, 4:], torch.zeros(2, 3, 4))
    assert new_extra["attention_mask"].shape == (2, 8)
    assert torch.equal(new_extra["attention_mask"][:, 5:], torch.ones(2, 3))
    # Original conditioning list untouched.
    assert ce.shape == (2, 5, 8)


def test_append_context_tokens_truncates_wider_tokens():
    ce = torch.zeros(1, 2, 4)
    cond = [[ce, {}]]
    tokens = torch.ones(1, 1, 8)  # wider than context dim
    out = idt.append_context_tokens(cond, tokens)
    new_ce, _ = out[0]
    assert new_ce.shape == (1, 3, 4)


# ── _apply_configured_guides identity-pin routing ──────────────────────────────

def test_identity_pin_routes_to_overlap_when_enabled(monkeypatch):
    s = _sampler()
    calls = []
    monkeypatch.setattr(s, "_append_media_guide_at",
                         lambda chunk, filename, *a, **kw: (calls.append(filename), chunk, [], [], 0, 0)[1:])
    guide_list = [
        {"enabled": True, "source": "image", "media_ref": "pin", "identity_pin": True, "strength": 0.35},
        {"enabled": True, "source": "image", "media_ref": "prior", "strength": 0.35},  # NOT the pin
    ]
    scene_media_by_ref = {"pin": "pin.png", "prior": "prior.png"}
    chunk, pos, neg, head, tail, identity_ref_filename = s._apply_configured_guides(
        {}, 1, guide_list, {}, [], scene_media_by_ref, [], [], object(),
        identity_transfer_enabled=True,
    )
    assert calls == ["prior.png"]  # pin entry never went through the plain append
    assert identity_ref_filename == "pin.png"


def test_identity_pin_appends_normally_when_disabled(monkeypatch):
    s = _sampler()
    calls = []
    monkeypatch.setattr(s, "_append_media_guide_at",
                         lambda chunk, filename, *a, **kw: (calls.append(filename), chunk, [], [], 0, 0)[1:])
    guide_list = [{"enabled": True, "source": "image", "media_ref": "pin", "identity_pin": True, "strength": 0.35}]
    _, _, _, _, _, identity_ref_filename = s._apply_configured_guides(
        {}, 1, guide_list, {}, [], {"pin": "pin.png"}, [], [], object(),
        identity_transfer_enabled=False,
    )
    assert calls == ["pin.png"]
    assert identity_ref_filename is None


# ── _identity_pin_filename (standalone lookup for the mixed i2v anchor branch,
#    which skips _apply_configured_guides entirely) ─────────────────────────────

def test_identity_pin_filename_finds_first_pin_entry():
    s = _sampler()
    guide_list = [
        {"enabled": True, "source": "image", "media_ref": "prior", "strength": 0.35},
        {"enabled": True, "source": "image", "media_ref": "pin", "identity_pin": True, "strength": 0.35},
    ]
    scene_media_by_ref = {"pin": "pin.png", "prior": "prior.png"}
    assert s._identity_pin_filename(guide_list, scene_media_by_ref, True) == "pin.png"


def test_identity_pin_filename_none_when_disabled_or_missing():
    s = _sampler()
    guide_list = [{"enabled": True, "source": "image", "media_ref": "pin", "identity_pin": True}]
    assert s._identity_pin_filename(guide_list, {"pin": "pin.png"}, False) is None
    assert s._identity_pin_filename(guide_list, {}, True) is None
    assert s._identity_pin_filename(None, {"pin": "pin.png"}, True) is None


# ── _install_identity_overlap / _strip_identity_overlap ────────────────────────

class _FakePatchifier:
    def patchify(self, latent):
        # latent: [1, C, 1, H, W] -> pretend 2 ref tokens: rt=[B,N,C], rlc=[B,3,N] (coords).
        b = latent.shape[0]
        return torch.zeros(b, 2, 8), torch.zeros(b, 3, 2)


class _FakeLTXV:
    def __init__(self):
        self.patchifier = _FakePatchifier()
        self.vae_scale_factors = [8, 8, 8]
        self.causal_temporal_positioning = False

    def _process_input(self, x, keyframe_idxs, denoise_mask, **kw):
        # pixel_coords mirrors x's structure: [B, 3, T] (or a [video, audio] pair for AV).
        if isinstance(x, (list, tuple)):
            pix = [torch.zeros(t.shape[0], 3, t.shape[1]) for t in x]
        else:
            pix = torch.zeros(x.shape[0], 3, x.shape[1])
        return x, pix, {}

    def patchify_proj(self, rt):
        return rt  # identity, dim already matches x for the test

    def _prepare_timestep(self, timestep, batch_size, hidden_dtype, **kwargs):
        return ("timestep", timestep, batch_size, hidden_dtype)

    def _prepare_positional_embeddings(self, pixel_coords, frame_rate, x_dtype):
        T, L = 10, 4
        cos = torch.arange(T * L, dtype=torch.float32).reshape(T, L)
        sin = torch.arange(T * L, dtype=torch.float32).reshape(T, L) * -1.0
        return (cos, sin, False)

    def _process_output(self, x, embedded_timestep, keyframe_idxs, **kw):
        return ("output", x, embedded_timestep)


def _fake_model(ltxv):
    inner = types.SimpleNamespace(diffusion_model=ltxv)
    return types.SimpleNamespace(model=inner)


def _ref_latent():
    return torch.zeros(1, 8, 1, 4, 4)


def test_install_is_idempotent_and_strip_restores_originals():
    s = _sampler()
    ltxv = _FakeLTXV()
    orig_pi = ltxv._process_input
    orig_pt = ltxv._prepare_timestep
    orig_pe = ltxv._prepare_positional_embeddings
    orig_po = ltxv._process_output

    handle1 = s._install_identity_overlap(_fake_model(ltxv), _ref_latent(), 2.0)
    assert handle1 is not None
    assert ltxv._process_input is not orig_pi
    handle2 = s._install_identity_overlap(_fake_model(ltxv), _ref_latent(), 2.0)
    assert handle2 is None  # already installed — idempotent, no double-wrap

    s._strip_identity_overlap(handle1)
    assert ltxv._process_input == orig_pi
    assert ltxv._prepare_timestep == orig_pt
    assert ltxv._prepare_positional_embeddings == orig_pe
    assert ltxv._process_output == orig_po
    assert not hasattr(ltxv, "_funpack_id_ref_len")


def test_process_input_appends_ref_tokens_and_stashes_len():
    s = _sampler()
    ltxv = _FakeLTXV()
    handle = s._install_identity_overlap(_fake_model(ltxv), _ref_latent(), 2.0)
    x = torch.zeros(1, 5, 8)  # 5 target tokens
    xx, pix, add = ltxv._process_input(x, None, None)
    assert xx.shape[1] == 7  # 5 target + 2 ref tokens
    assert ltxv._funpack_id_ref_len == 2
    assert ltxv._funpack_id_target_len == 5
    s._strip_identity_overlap(handle)


def test_prepare_timestep_gives_ref_tokens_clean_timestep():
    s = _sampler()
    ltxv = _FakeLTXV()
    handle = s._install_identity_overlap(_fake_model(ltxv), _ref_latent(), 2.0)
    ltxv._process_input(torch.zeros(1, 5, 8), None, None)  # sets ref_len=2, target_len=5
    _, timestep, batch_size, _ = ltxv._prepare_timestep(torch.full((1,), 0.7), 1, torch.float32)
    assert timestep.shape == (1, 7)
    assert torch.all(timestep[:, :5] == 0.7)
    assert torch.all(timestep[:, 5:] == 0.0)  # ref tokens: clean timestep
    s._strip_identity_overlap(handle)


def test_prepare_pe_rotates_only_the_ref_token_tail():
    s = _sampler()
    ltxv = _FakeLTXV()
    handle = s._install_identity_overlap(_fake_model(ltxv), _ref_latent(), 2.0)
    baseline_cos, baseline_sin, _ = ltxv._prepare_positional_embeddings(None, 25, torch.float32)
    ltxv._process_input(torch.zeros(1, 5, 8), None, None)  # ref_len=2
    cos, sin, split_mode = ltxv._prepare_positional_embeddings(None, 25, torch.float32)
    untouched = list(range(0, 8))
    for i in untouched:
        assert torch.equal(cos[i], baseline_cos[i])
        assert torch.equal(sin[i], baseline_sin[i])
    rotated = [8, 9]
    changed = any(not torch.equal(cos[i], baseline_cos[i]) for i in rotated)
    assert changed
    assert split_mode is False
    s._strip_identity_overlap(handle)


def test_prepare_pe_noop_when_ref_len_zero():
    s = _sampler()
    ltxv = _FakeLTXV()
    handle = s._install_identity_overlap(_fake_model(ltxv), _ref_latent(), 2.0)
    baseline = ltxv._prepare_positional_embeddings(None, 25, torch.float32)
    # No _process_input call this "forward pass" -> _funpack_id_ref_len defaults to 0.
    out = ltxv._prepare_positional_embeddings(None, 25, torch.float32)
    assert torch.equal(out[0], baseline[0])
    assert torch.equal(out[1], baseline[1])
    s._strip_identity_overlap(handle)


def test_process_output_trims_ref_tokens_from_plain_tensor():
    s = _sampler()
    ltxv = _FakeLTXV()
    handle = s._install_identity_overlap(_fake_model(ltxv), _ref_latent(), 2.0)
    ltxv._process_input(torch.zeros(1, 5, 8), None, None)  # ref_len=2
    x = torch.arange(7 * 3, dtype=torch.float32).reshape(1, 7, 3)
    embedded_timestep = torch.zeros(1, 7, 1)
    tag, out_x, out_et = ltxv._process_output(x, embedded_timestep, None)
    assert out_x.shape[1] == 5
    assert torch.equal(out_x, x[:, :5])
    assert out_et.shape[1] == 5
    s._strip_identity_overlap(handle)


def test_process_output_trims_compressed_timestep_frames():
    s = _sampler()
    ltxv = _FakeLTXV()
    handle = s._install_identity_overlap(_fake_model(ltxv), _ref_latent(), 2.0)
    ltxv._process_input([torch.zeros(1, 5, 8), torch.zeros(1, 3, 8)], None, None)
    x = [torch.arange(7 * 3, dtype=torch.float32).reshape(1, 7, 3), torch.zeros(1, 3, 3)]
    ct = _CompressedTimestep(data=torch.zeros(1, 7), num_frames=7, patches_per_frame=1)
    tag, out_x, out_et = ltxv._process_output(x, [ct, "audio_et"], None)
    assert out_x[0].shape[1] == 5
    assert out_et[0].num_frames == 5
    assert out_et[0].data.shape[1] == 5
    assert out_et[1] == "audio_et"
    s._strip_identity_overlap(handle)


# ── _resolve_identity_overlap memoization ──────────────────────────────────────

def test_resolve_identity_overlap_computes_once_per_state(monkeypatch):
    s = _sampler()
    monkeypatch.setattr(s, "_load_image_tensor", lambda filename: torch.zeros(1, 32, 32, 3))
    monkeypatch.setattr(s, "_latent_tensors", lambda chunk: [torch.zeros(1, 8, 1, 4, 4)])
    fake_comfy_utils = types.ModuleType("comfy.utils")
    calls = []

    def fake_upscale(img, w, h, mode, crop):
        calls.append((w, h))
        return img

    fake_comfy_utils.common_upscale = fake_upscale
    sys.modules["comfy.utils"] = fake_comfy_utils
    sys.modules["comfy"].utils = fake_comfy_utils

    class _FakeVAE:
        downscale_index_formula = [8, 8, 8]

        def encode(self, px):
            return torch.zeros(1, 8, 1, 4, 4)

    state = {}
    r1 = s._resolve_identity_overlap(state, "pin.png", _FakeVAE(), {}, "None", 2.0, 1.0, 1.0, "auto_adjust", False)
    assert len(calls) == 1
    r2 = s._resolve_identity_overlap(state, "pin.png", _FakeVAE(), {}, "None", 2.0, 1.0, 1.0, "auto_adjust", False)
    assert len(calls) == 1  # second call served from cache, no re-encode
    assert r1 == r2
    assert r1[2] is None and r1[3] is None  # no projector selected -> no ArcFace tokens
