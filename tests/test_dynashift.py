"""DynaShift: negative latent memory (negative_memory.py) + steer-away wrapper
(_build_dynashift_wrapper).

Covers the pending->promote/discard rating pairing, the ring-buffer cap, and the
wrapper contract: matched frames get their aligned component removed, unmatched /
anti-aligned / below-threshold frames and the audio region stay byte-identical,
prompt-similarity weighting can silence an unrelated negative, and geometry
mismatches degrade to a clean passthrough.
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

import negative_memory  # noqa: E402
import samplers  # noqa: E402

# ---------------------------------------------------------------------------
# Store lifecycle
# ---------------------------------------------------------------------------


def _patch_store(monkeypatch, tmp_path):
    monkeypatch.setattr(
        negative_memory, "_state_path",
        lambda key, mode: str(tmp_path / f"{key}.{mode}.pt"))


def test_pending_promote_and_discard(monkeypatch, tmp_path):
    _patch_store(monkeypatch, tmp_path)
    lat = torch.randn(1, 4, 3, 2, 2)
    assert negative_memory.save_pending("k", lat, torch.randn(1, 5, 8))
    #

    # Non-negative rating discards the pending candidate.
    assert negative_memory.consume_pending("k", promote=False) is None
    assert negative_memory.load_negatives("k") == []
    # Nothing pending anymore: a later bad rating cannot promote a stale latent.
    assert negative_memory.consume_pending("k", promote=True) is None

    assert negative_memory.save_pending("k", lat)
    assert negative_memory.consume_pending("k", promote=True) == 1
    entries = negative_memory.load_negatives("k")
    assert len(entries) == 1
    assert entries[0]["latent"].shape == (4, 3, 2, 2)  # batch squeezed
    assert entries[0]["latent"].dtype == torch.float16


def test_negative_profile_rule():
    # Intrusion keys bank regardless of reward.
    assert negative_memory.is_negative_profile({"key": "awful", "reward": -0.9})
    assert negative_memory.is_negative_profile({"key": "wrong_appearance", "reward": 0.0})
    # Quality-missing family banks via the reward threshold.
    assert negative_memory.is_negative_profile({"key": "missing_quality", "reward": -0.25})
    assert negative_memory.is_negative_profile({"key": "missing_action_quality", "reward": -0.55})
    assert negative_memory.is_negative_profile({"key": "missing_details_action_quality", "reward": -0.65})
    # Near-misses with positive reward stay out - their latents are mostly-wanted content.
    assert not negative_memory.is_negative_profile({"key": "missing_details", "reward": 0.35})
    assert not negative_memory.is_negative_profile({"key": "missing_action", "reward": 0.05})
    assert not negative_memory.is_negative_profile({"key": "like", "reward": 1.0})
    assert not negative_memory.is_negative_profile(None)


def test_promotion_stamps_rating_provenance(monkeypatch, tmp_path):
    _patch_store(monkeypatch, tmp_path)
    negative_memory.save_pending("k", torch.randn(1, 4, 3, 2, 2))
    assert negative_memory.consume_pending("k", promote=True, rating_key="missing_quality") == 1
    assert negative_memory.load_negatives("k")[0]["rating"] == "missing_quality"


def test_ring_buffer_caps_at_max(monkeypatch, tmp_path):
    _patch_store(monkeypatch, tmp_path)
    for i in range(negative_memory.MAX_NEGATIVES + 3):
        negative_memory.save_pending("k", torch.full((1, 2, 1, 1, 1), float(i)))
        negative_memory.consume_pending("k", promote=True)
    entries = negative_memory.load_negatives("k")
    assert len(entries) == negative_memory.MAX_NEGATIVES
    # Oldest rolled off: the first surviving entry is i=3.
    assert float(entries[0]["latent"].flatten()[0]) == 3.0


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

C, T, H, W = 4, 3, 2, 2
D = C * H * W          # 16 per-frame elements
VIDEO_N = C * T * H * W  # 48
AUDIO_N = 12


def _packed_av_model():
    ls = types.SimpleNamespace(cond=[(1, C, T, H, W), (1, AUDIO_N)])
    guider = types.SimpleNamespace(conds={"positive": [{"model_conds": {"latent_shapes": ls}}]})
    return types.SimpleNamespace(model_options={}, inner_model=guider)


def _basis_frame(i):
    """Frame pattern = one-hot basis vector in the flattened (C,H,W) space."""
    f = torch.zeros(D)
    f[i] = 1.0
    return f


def _denoised_from_frames(frames):
    """Build a packed [1,1,VIDEO_N+AUDIO_N] latent whose video frames are `frames`."""
    cur = torch.stack(frames)                    # [T, D]
    video = cur.view(T, C, H, W).permute(1, 0, 2, 3).reshape(-1)
    audio = torch.arange(AUDIO_N, dtype=torch.float32) * 0.1 + 1.0
    return torch.cat([video, audio]).view(1, 1, -1)


def _negative_entry(frame_vecs, cond=None):
    lat = torch.stack(frame_vecs).view(len(frame_vecs), C, H, W).permute(1, 0, 2, 3)
    return {"latent": lat.to(torch.float16), "cond": cond}


def _wrap(model, negatives, strength=0.5, threshold=0.6, positives=None):
    s = samplers.FunPackLTXAVSceneChainSampler()
    s._build_dynashift_wrapper(model, negatives, strength, threshold, positives=positives)
    return model.model_options["model_function_wrapper"]


def _run(wrap, denoised, sigma=0.0, c=None):
    def apply_fn(inp, ts, **kw):
        return denoised
    return wrap(apply_fn, {"input": denoised, "timestep": torch.tensor([sigma]),
                           "c": c or {}})


def _frames_of(out):
    video = out[0, 0, :VIDEO_N].view(C, T, H, W)
    return video.permute(1, 0, 2, 3).reshape(T, -1)


def test_matched_frame_steered_others_and_audio_untouched():
    model = _packed_av_model()
    neg = _negative_entry([_basis_frame(0), _basis_frame(1)])
    denoised = _denoised_from_frames([
        2.0 * _basis_frame(1),   # matches negative frame 1 (cos = 1), aligned coef 2
        3.0 * _basis_frame(2),   # orthogonal to the whole bank
        1.5 * _basis_frame(3),
    ])
    out = _run(_wrap(model, [neg], strength=0.5), denoised)
    frames = _frames_of(out)
    # gate = (1-0.6)/0.4 * 0.5 = 0.5 -> removes half the aligned component: 2 -> 1.
    assert abs(float(frames[0][1]) - 1.0) < 1e-4
    # Unmatched frames byte-identical.
    assert torch.equal(frames[1], 3.0 * _basis_frame(2))
    assert torch.equal(frames[2], 1.5 * _basis_frame(3))
    # Audio region byte-identical.
    assert torch.equal(out[..., VIDEO_N:], denoised[..., VIDEO_N:])


def test_below_threshold_and_anti_aligned_are_noop():
    model = _packed_av_model()
    neg = _negative_entry([_basis_frame(0)])
    # cos(frame, neg) = 0.5 < 0.6 threshold; last frame anti-aligned (coef clamps to 0).
    mixed = 0.5 * _basis_frame(0) + (3 ** 0.5) / 2 * _basis_frame(4)
    denoised = _denoised_from_frames([mixed, -2.0 * _basis_frame(0), _basis_frame(5)])
    out = _run(_wrap(model, [neg]), denoised)
    assert torch.equal(out, denoised)


def test_high_sigma_is_passthrough():
    model = _packed_av_model()
    neg = _negative_entry([_basis_frame(0)])
    denoised = _denoised_from_frames([2.0 * _basis_frame(0)] * T)
    out = _run(_wrap(model, [neg]), denoised, sigma=0.9)  # ramp <= 0
    assert torch.equal(out, denoised)


def test_resolution_mismatch_skipped_cleanly():
    model = _packed_av_model()
    bad_geometry = {"latent": torch.randn(C, 2, H + 1, W).to(torch.float16), "cond": None}
    denoised = _denoised_from_frames([2.0 * _basis_frame(0)] * T)
    out = _run(_wrap(model, [bad_geometry]), denoised)
    assert torch.equal(out, denoised)


def test_prompt_similarity_weight_silences_unrelated_negative():
    model = _packed_av_model()
    cond_dim = 8
    e0 = torch.zeros(cond_dim); e0[0] = 1.0
    e1 = torch.zeros(cond_dim); e1[1] = 1.0
    denoised = _denoised_from_frames([2.0 * _basis_frame(1), _basis_frame(2), _basis_frame(3)])
    cur_cond = {"c_crossattn": e0.view(1, 1, cond_dim).repeat(1, 5, 1)}

    # Negative recorded under an orthogonal prompt -> weight 0 -> no steering.
    neg_far = _negative_entry([_basis_frame(0), _basis_frame(1)], cond=e1.to(torch.float16))
    out = _run(_wrap(model, [neg_far]), denoised, c=cur_cond)
    assert torch.equal(out, denoised)

    # Same negative recorded under the same prompt -> full steering.
    neg_near = _negative_entry([_basis_frame(0), _basis_frame(1)], cond=e0.to(torch.float16))
    out = _run(_wrap(_packed_av_model(), [neg_near]), denoised, c=cur_cond)
    assert not torch.equal(out, denoised)


def test_wrapper_is_tagged_for_leak_strip():
    model = _packed_av_model()
    _wrap(model, [_negative_entry([_basis_frame(0)])])
    w = model.model_options["model_function_wrapper"]
    assert getattr(w, samplers._FUNPACK_SCENE_WRAPPER_TAG, False)
    assert samplers._strip_funpack_scene_wrappers(model) == 1


# ---------------------------------------------------------------------------
# Positive pull -- mean(liked) - mean(disliked), independent of the negative match
# ---------------------------------------------------------------------------


def test_positive_profile_rule():
    assert negative_memory.is_positive_profile({"key": "liked", "reward": 1.0})
    assert negative_memory.is_positive_profile({"key": "missing_action", "reward": 0.05})
    assert not negative_memory.is_positive_profile({"key": "missing_quality", "reward": -0.25})
    assert not negative_memory.is_positive_profile({"key": "wrong_appearance", "reward": 0.0})
    assert not negative_memory.is_positive_profile(None)
    # Mutually exclusive with is_negative_profile on every reward value -- nothing banks twice.
    for reward in (-0.9, -0.25, -0.1, 0.0, 0.05, 0.35, 1.0):
        profile = {"key": "x", "reward": reward}
        assert not (negative_memory.is_negative_profile(profile)
                    and negative_memory.is_positive_profile(profile))


def test_pull_needs_min_entries_in_each_bank():
    # Only 1 positive entry -- below MIN_BANK_FOR_PULL even though the negative bank has 2.
    model = _packed_av_model()
    negs = [_negative_entry([_basis_frame(0)]), _negative_entry([_basis_frame(0)])]
    pos = [_negative_entry([_basis_frame(1)])]
    denoised = _denoised_from_frames([3.0 * _basis_frame(2)] * T)  # orthogonal to both banks
    out = _run(_wrap(model, negs, positives=pos), denoised)
    assert torch.equal(out, denoised)  # no negative match, no pull -> pure passthrough


def test_positive_pull_moves_toward_liked_minus_disliked_direction():
    model = _packed_av_model()
    negs = [_negative_entry([_basis_frame(0)]), _negative_entry([_basis_frame(0)])]
    pos = [_negative_entry([_basis_frame(1)]), _negative_entry([_basis_frame(1)])]
    # A frame orthogonal to both banks: no negative match (gate stays 0), so any movement
    # can only come from the pull term.
    denoised = _denoised_from_frames([3.0 * _basis_frame(2)] * T)
    out = _run(_wrap(model, negs, strength=0.5, threshold=0.6, positives=pos), denoised)
    frames = _frames_of(out)
    pull_dir = torch.nn.functional.normalize(_basis_frame(1) - _basis_frame(0), dim=0)
    expected = 3.0 * _basis_frame(2) + (0.5 * 1.0 * 3.0) * pull_dir  # strength * ramp(sigma=0) * |cur|
    assert torch.allclose(frames[0], expected, atol=1e-4)


def test_pull_is_inert_with_no_positive_bank():
    model = _packed_av_model()
    negs = [_negative_entry([_basis_frame(0)]), _negative_entry([_basis_frame(0)])]
    denoised = _denoised_from_frames([3.0 * _basis_frame(2)] * T)
    out = _run(_wrap(model, negs, positives=None), denoised)
    assert torch.equal(out, denoised)
