"""Chain Sampler on MiniMax H3 — the stream-axis and keyframe routing that used to be LTX-only.

These exercise the sampler's own helpers directly rather than a full sample() run, because the
thing under test is geometry: which axis of which stream a slice lands on, and where a guide
goes when the model has no guide-attention API. Both are silent failures — the LTXAV code path
returns a tensor of a plausible shape either way.
"""
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakeNestedTensor:
    def __init__(self, tensors):
        self.tensors = list(tensors)
        self.is_nested = True

    def unbind(self):
        return self.tensors

    @property
    def shape(self):
        return self.tensors[0].shape


for _name, _attrs in (
    ("comfy", {}),
    ("comfy.k_diffusion", {}),
    ("comfy.k_diffusion.sampling", {}),
    ("comfy.model_sampling", {}),
    ("comfy.nested_tensor", {"NestedTensor": FakeNestedTensor}),
    ("comfy.sample", {"prepare_noise": lambda *a, **k: None, "sample_custom": lambda *a, **k: None}),
    ("comfy.samplers", {}),
    ("comfy.utils", {"ProgressBar": lambda total: types.SimpleNamespace(
        update_absolute=lambda *a, **k: None)}),
):
    if _name not in sys.modules:
        _mod = types.ModuleType(_name)
        for _k, _v in _attrs.items():
            setattr(_mod, _k, _v)
        sys.modules[_name] = _mod
# common_upscale is only reached by the reference-block helpers; whichever comfy.utils stub
# won registration may not carry it, so fill it in with the resize semantics that matter here.
if not hasattr(sys.modules["comfy.utils"], "common_upscale"):
    sys.modules["comfy.utils"].common_upscale = lambda samples, width, height, method, crop: \
        torch.zeros(samples.shape[0], samples.shape[1], height, width)
_comfy = sys.modules["comfy"]
for _sub in ("k_diffusion", "model_sampling", "nested_tensor", "sample", "samplers", "utils"):
    setattr(_comfy, _sub, sys.modules["comfy." + _sub])
setattr(sys.modules["comfy.k_diffusion"], "sampling", sys.modules["comfy.k_diffusion.sampling"])

from samplers import FunPackLTXAVSceneChainSampler  # noqa: E402

NT = FakeNestedTensor


@pytest.fixture(autouse=True)
def _own_nested_tensor():
    """Force the sampler to rebuild nested latents with THIS file's stub.

    Several modules in this suite install their own `comfy.nested_tensor` stub into
    sys.modules, and whichever imports first wins for the whole run — so a test that
    passes alone can fail in a full run purely on import order. Pinning it per test
    keeps these assertions about the sampler rather than about collection order.
    """
    import comfy.nested_tensor as mod
    previous = mod.NestedTensor
    mod.NestedTensor = FakeNestedTensor
    try:
        yield
    finally:
        mod.NestedTensor = previous


class H3VAE:
    """What comfy/sd.py builds for the H3 video VAE."""
    downscale_ratio = (lambda a: max(1, (a - 5) // 17 * 5 + 2) if a > 1 else 1, 16, 16)
    downscale_index_formula = (4, 16, 16)


class LTXVAE:
    downscale_ratio = (lambda a: max(0, (a + 7) // 8), 32, 32)
    downscale_index_formula = (8, 32, 32)


def h3_node(frame_count=124):
    """A Chain Sampler already told it is running on H3, as sample() would tell it."""
    node = FunPackLTXAVSceneChainSampler()
    node._is_h3 = True
    node._time_dims = (2, -1, -1, -1)
    node._h3_frame_count = frame_count
    return node


def av_latent(video_t=37, audio_t=207):
    video = torch.zeros(1, 24, video_t, 48, 84)
    audio = torch.zeros(1, 32, 2, audio_t)
    return {"samples": NT([video, audio])}


# ── stream axes ──────────────────────────────────────────────────────────────

def test_frame_counts_read_the_right_axis_per_stream():
    node = h3_node()
    video, audio = av_latent()["samples"].unbind()
    assert node._tensor_frames(video, stream=0) == 37
    assert node._tensor_frames(audio, stream=1) == 207     # not 2, the stereo axis
    # an LTXAV run keeps both streams on dim 2, exactly as before
    ltx = FunPackLTXAVSceneChainSampler()
    assert ltx._tensor_frames(torch.zeros(1, 8, 40, 4), stream=1) == 40


def test_continuation_chunk_pins_the_audio_tail_not_a_speaker():
    """The overlap prefix must land on time; on the stereo axis it would blank a channel."""
    node = h3_node()
    template = av_latent()
    previous = av_latent()
    for t in previous["samples"].unbind():
        t.fill_(1.0)

    chunk = node._build_continuation_chunk(template, previous, video_overlap=2)
    video, audio = chunk["samples"].unbind()
    vmask, amask = chunk["noise_mask"].unbind()

    assert video.shape == (1, 24, 37, 48, 84)
    assert audio.shape == (1, 32, 2, 207)                  # shape preserved, not reshaped
    # video: first 2 latent frames carried and protected
    assert video[:, :, :2].eq(1.0).all() and video[:, :, 2:].eq(0.0).all()
    assert vmask[:, :, :2].eq(0).all() and vmask[:, :, 2:].eq(1).all()
    # audio: the proportional overlap lands on the LAST axis, both speakers intact
    aov = int(round(2 * 207 / 37))
    assert audio[..., :aov].eq(1.0).all() and audio[..., aov:].eq(0.0).all()
    assert amask[..., :aov].eq(0).all() and amask[..., aov:].eq(1).all()
    assert amask.shape[2] == 2                              # stereo axis untouched


def test_blending_two_scenes_keeps_stereo_and_extends_time():
    node = h3_node()
    previous, current = av_latent(video_t=10, audio_t=56), av_latent(video_t=10, audio_t=56)
    for t in current["samples"].unbind():
        t.fill_(1.0)

    blended = node._blend_latents(previous, current, video_overlap=3)
    video, audio = blended["samples"].unbind()
    assert video.shape[2] == 10 + 10 - 3
    assert audio.shape[2] == 2                              # stereo, not consumed as time
    aov = int(round(3 * 56 / 10))
    assert audio.shape[-1] == 56 + 56 - aov


def test_cropping_the_joyai_audio_tail_trims_time():
    node = h3_node()
    latent = av_latent(video_t=10, audio_t=56)
    out = node._crop_audio_tail(latent, 6)
    video, audio = out["samples"].unbind()
    assert audio.shape == (1, 32, 2, 50)
    assert video.shape[2] == 10                             # video untouched


def test_joyai_audio_memory_appends_on_the_time_axis():
    node = h3_node()
    chunk = av_latent(video_t=10, audio_t=56)
    frame = torch.ones(1, 32, 2, 1)
    out, appended = node._append_joyai_audio_memory(chunk, [frame, frame])
    assert appended == 2
    _video, audio = out["samples"].unbind()
    assert audio.shape == (1, 32, 2, 58)
    assert audio[..., -2:].eq(1.0).all()
    _vmask, amask = out["noise_mask"].unbind()
    assert amask[..., -2:].eq(0).all()                      # pinned, never denoised


def test_joyai_audio_memory_rejects_a_frame_of_the_wrong_stereo_width():
    node = h3_node()
    chunk = av_latent(video_t=10, audio_t=56)
    mono = torch.ones(1, 32, 1, 1)
    out, appended = node._append_joyai_audio_memory(chunk, [mono])
    assert appended == 0
    assert out is chunk


# ── frame grid ───────────────────────────────────────────────────────────────

def test_template_validation_uses_the_vaes_own_count_map():
    node = h3_node()
    template = av_latent(video_t=37)
    # 124 pixel frames -> 37 latent frames on H3's 17k+5 grid
    assert node._validate_template_length(template, 124, time_scale=4, vae=H3VAE()) == 37
    with pytest.raises(ValueError):
        node._validate_template_length(template, 141, time_scale=4, vae=H3VAE())


def test_ltx_template_validation_is_unchanged():
    node = FunPackLTXAVSceneChainSampler()
    template = {"samples": torch.zeros(1, 128, 16, 4, 4)}
    assert node._validate_template_length(template, 121, time_scale=8, vae=LTXVAE()) == 16
    # and identical with no VAE passed at all (the old uniform formula)
    assert node._validate_template_length(template, 121, time_scale=8) == 16


def test_decode_tile_size_is_divided_by_the_vaes_spatial_ratio():
    node = FunPackLTXAVSceneChainSampler()
    assert node._decode_tile_latent(H3VAE(), 512) == 32     # 16x downscale
    assert node._decode_tile_latent(LTXVAE(), 512) == 16    # 32x downscale
    assert node._decode_tile_latent(object(), 512) == 64    # unknown VAE -> old behaviour


# ── guides become keyframe pins ──────────────────────────────────────────────

def test_a_first_frame_guide_becomes_a_conditioning_keyframe_with_no_latent_tail():
    node = h3_node(frame_count=124)
    chunk = av_latent()
    guide = torch.ones(1, 24, 1, 48, 84)
    positive = [[torch.zeros(1, 12, 5120), {"funpack_scene_text": "shot 2"}]]

    out_chunk, pos, neg, tail = node._append_guide_latent(
        chunk, guide, apply_at=0, strength=1.0, positive=positive, negative=[], vae=H3VAE())

    assert tail == 0                                    # nothing appended -> nothing to crop
    assert out_chunk is chunk                           # the latent is not touched at all
    meta = pos[0][1]
    assert meta["minimax_frame_count"] == 124
    assert [kf["resolved_frame_index"] for kf in meta["minimax_keyframes"]] == [0]
    assert meta["minimax_keyframes"][0]["latent"] is guide
    assert meta["funpack_scene_text"] == "shot 2"       # existing metadata preserved
    assert "minimax_visual_cond_noise_aug" not in meta  # strength 1.0 = clean pin


def test_a_last_frame_guide_is_accepted():
    node = h3_node(frame_count=124)
    pos, _neg, tail = node._append_h3_keyframe(
        torch.ones(1, 24, 1, 48, 84), apply_at=-1, strength=0.8,
        positive=[[torch.zeros(1, 12, 5120), {}]], negative=[])
    assert tail == 0
    assert pos[0][1]["minimax_keyframes"][0]["resolved_frame_index"] == 123
    assert pos[0][1]["minimax_visual_cond_noise_aug"] == pytest.approx(0.8)


def test_a_mid_clip_guide_is_refused_rather_than_crashing_the_sample(capsys):
    """PackedLayout raises for anything but first/last — better to say so up front."""
    node = h3_node(frame_count=124)
    positive = [[torch.zeros(1, 12, 5120), {}]]
    pos, _neg, tail = node._append_h3_keyframe(
        torch.ones(1, 24, 1, 48, 84), apply_at=60, strength=1.0,
        positive=positive, negative=[])
    assert tail == 0
    assert pos is positive                              # conditioning untouched
    assert "minimax_keyframes" not in positive[0][1]
    out = capsys.readouterr().out
    assert "first (0) or last (123)" in out


def test_ltx_guides_still_take_the_ltx_path():
    """The H3 branch must not fire for an LTXAV run; without comfy_extras it no-ops as before."""
    node = FunPackLTXAVSceneChainSampler()
    assert node._is_h3 is False
    chunk = {"samples": torch.zeros(1, 128, 8, 4, 4)}
    out_chunk, pos, neg, tail = node._append_guide_latent(
        chunk, torch.ones(1, 128, 1, 4, 4), apply_at=0, strength=1.0,
        positive=[[torch.zeros(1, 12, 4096), {}]], negative=[], vae=LTXVAE())
    assert tail == 0
    assert "minimax_keyframes" not in pos[0][1]


# ── ref2va reference blocks ──────────────────────────────────────────────────
# Studio bakes the presentation ("<Picture 1>: <vision block>") and records WHICH
# references it presented, in order. The sampler owns the VAE and encodes exactly that
# list. If the two orders diverge, "<Picture 2>" in the prompt points at a different
# reference than the text encoder saw — a clean-looking video of the wrong subject.

class RefVAE:
    def encode(self, pixels):
        return torch.zeros(1, 24, 1, pixels.shape[1] // 16, pixels.shape[2] // 16)


class RefAudioVAE:
    audio_sample_rate = 32000

    def encode(self, waveform):
        return torch.zeros(1, 32, 2, 5)


@pytest.fixture
def fake_media(monkeypatch):
    import minimax_h3 as h3mod
    monkeypatch.setattr(h3mod, "load_input_image",
                        lambda f: None if f.startswith("missing") else torch.zeros(1, 256, 256, 3))
    monkeypatch.setattr(h3mod, "load_input_audio",
                        lambda f: {"waveform": torch.zeros(1, 2, 800), "sample_rate": 32000})


def cond_with_refs(refs):
    return [[torch.zeros(1, 12, 5120), {"funpack_h3_refs": refs, "funpack_scene_text": "shot 1"}]]


def test_references_are_encoded_in_the_order_studio_presented_them(fake_media):
    node = h3_node()
    positive = cond_with_refs([
        {"kind": "image", "filename": "face.png"},
        {"kind": "audio", "filename": "voice.wav"},
        {"kind": "image", "filename": "style.png"},
    ])
    out, count = node._apply_h3_references(
        positive, av_latent(), RefVAE(), audio_vae=RefAudioVAE())

    assert count == 3
    blocks = out[0][1]["minimax_refs"]
    assert [b["kind"] for b in blocks] == ["image", "audio", "image"]
    assert out[0][1]["funpack_scene_text"] == "shot 1"      # scene metadata preserved


def test_reference_blocks_are_sized_against_this_scenes_canvas(fake_media):
    node = h3_node()
    # av_latent's video is [1, 24, 37, 48, 84] -> a 1344x768 canvas at 16x downscale
    out, _ = node._apply_h3_references(
        cond_with_refs([{"kind": "image", "filename": "face.png"}]), av_latent(), RefVAE())
    block = out[0][1]["minimax_refs"][0]
    # the reference is 256x256, smaller than the canvas area, so it keeps its own size
    assert block["latent_h"] * 16 == 256 and block["latent_w"] * 16 == 256


def test_no_references_leaves_the_conditioning_untouched(fake_media):
    node = h3_node()
    positive = [[torch.zeros(1, 12, 5120), {}]]
    out, count = node._apply_h3_references(positive, av_latent(), RefVAE())
    assert count == 0 and out is positive


def test_an_audio_reference_without_an_audio_vae_is_reported(fake_media, capsys):
    node = h3_node()
    out, count = node._apply_h3_references(
        cond_with_refs([{"kind": "image", "filename": "a.png"},
                        {"kind": "audio", "filename": "voice.wav"}]),
        av_latent(), RefVAE(), audio_vae=None)
    assert count == 1
    assert [b["kind"] for b in out[0][1]["minimax_refs"]] == ["image"]
    printed = capsys.readouterr().out
    assert "audio_vae" in printed
    # the renumbering consequence has to be stated, not left for the user to discover
    assert "points one reference earlier" in printed


def test_an_unloadable_reference_is_dropped_and_named(fake_media, capsys):
    node = h3_node()
    out, count = node._apply_h3_references(
        cond_with_refs([{"kind": "image", "filename": "missing.png"},
                        {"kind": "image", "filename": "ok.png"}]),
        av_latent(), RefVAE())
    assert count == 1
    assert "missing.png" in capsys.readouterr().out


def test_references_are_encoded_once_per_run_not_once_per_scene(fake_media):
    """References ride through every step of every scene — re-encoding them per scene is
    pure GPU waste, and a cache that outlived the run would violate the no-persistent-state
    rule. Cached per run, cleared by sample()."""
    node = h3_node()
    node._h3_ref_cache = {}
    encodes = []

    class CountingVAE(RefVAE):
        def encode(self, pixels):
            encodes.append(tuple(pixels.shape))
            return super().encode(pixels)

    vae = CountingVAE()
    positive = cond_with_refs([{"kind": "image", "filename": "face.png"}])
    for _scene in range(4):
        out, count = node._apply_h3_references(positive, av_latent(), vae)
        assert count == 1
        assert len(out[0][1]["minimax_refs"]) == 1
    assert len(encodes) == 1

    # a different canvas is a different encode, not a stale cache hit
    node._apply_h3_references(positive, av_latent(video_t=37), vae)
    assert len(encodes) == 1
    smaller = {"samples": NT([torch.zeros(1, 24, 37, 32, 32), torch.zeros(1, 32, 2, 207)])}
    node._apply_h3_references(positive, smaller, vae)
    assert len(encodes) == 2


# ── i2v anchors become keyframe pins ─────────────────────────────────────────
# LTX anchors an image by writing it into the starting latent (LTXVImgToVideoInplace) and
# masking that frame out. H3 has no latent i2v path at all — an anchor written into the
# latent is just noise the model may overwrite — so the same intent has to become a
# frame-0 keyframe pin. Both families must keep their own path.

def test_an_anchor_image_becomes_a_frame_zero_keyframe_pin():
    node = h3_node(frame_count=124)
    positive = [[torch.zeros(1, 12, 5120), {"funpack_scene_text": "shot 1"}]]
    out, applied = node._apply_h3_anchor(
        positive, av_latent(), RefVAE(), torch.zeros(1, 256, 256, 3), strength=1.0)

    assert applied is True
    meta = out[0][1]
    assert [kf["resolved_frame_index"] for kf in meta["minimax_keyframes"]] == [0]
    assert meta["minimax_frame_count"] == 124
    assert "minimax_visual_cond_noise_aug" not in meta      # full-strength pin stays clean
    assert meta["funpack_scene_text"] == "shot 1"


def test_a_weakened_anchor_maps_onto_condition_noise_augmentation():
    node = h3_node()
    out, applied = node._apply_h3_anchor(
        [[torch.zeros(1, 12, 5120), {}]], av_latent(), RefVAE(),
        torch.zeros(1, 256, 256, 3), strength=0.6)
    assert applied is True
    assert out[0][1]["minimax_visual_cond_noise_aug"] == pytest.approx(0.6)


def test_an_anchor_and_a_last_frame_guide_coexist_instead_of_overwriting():
    """H3's payload takes a LIST of pins — the anchor must not erase the guide, or vice
    versa. This is exactly the fl2va first-AND-last-frame case."""
    node = h3_node(frame_count=124)
    positive = [[torch.zeros(1, 12, 5120), {}]]
    positive, _neg, _tail = node._append_h3_keyframe(
        torch.ones(1, 24, 1, 48, 84), apply_at=-1, strength=1.0,
        positive=positive, negative=[])
    positive, applied = node._apply_h3_anchor(
        positive, av_latent(), RefVAE(), torch.zeros(1, 256, 256, 3))

    assert applied is True
    assert [kf["resolved_frame_index"] for kf in positive[0][1]["minimax_keyframes"]] == [0, 123]


def test_a_mixed_anchor_chunk_leaves_the_h3_latent_alone():
    """On LTX the anchor is written into the chunk; on H3 the chunk must come back
    untouched (the pin does the work) — otherwise the latent carries an anchor the model
    was never trained to read, on top of the pin."""
    node = h3_node()
    template = av_latent()
    calls = []
    node._apply_img2video_to_video_latent = lambda *a, **k: calls.append(a) or template
    node._load_image_tensor = lambda f: torch.zeros(1, 256, 256, 3)

    chunk = node._build_mixed_anchor_chunk(
        RefVAE(), {"filename": "anchor.png", "strength": 1.0}, template, None, 0)
    assert calls == []                       # the LTX node is never invoked on H3
    assert chunk["samples"].unbind()[0].shape == template["samples"].unbind()[0].shape


def test_ltx_anchors_still_go_through_img2video_inplace():
    node = FunPackLTXAVSceneChainSampler()
    assert node._is_h3 is False
    template = {"samples": torch.zeros(1, 128, 16, 4, 4)}
    calls = []
    node._load_image_tensor = lambda f: torch.zeros(1, 256, 256, 3)
    node._apply_img2video_to_video_latent = lambda vae, image, base, strength: (
        calls.append(strength) or base)
    node._build_mixed_anchor_chunk(
        LTXVAE(), {"filename": "anchor.png", "strength": 0.9}, template, None, 0)
    assert calls == [0.9]


# ── the two checkpoints ──────────────────────────────────────────────────────

def test_the_run_names_the_checkpoint_its_conditioning_needs(capsys):
    node = h3_node()
    node._h3_mode_noted = False
    node._report_h3_checkpoint_mode([[torch.zeros(1, 12, 5120), {"minimax_refs": [{"kind": "image"}]}]])
    out = capsys.readouterr().out
    assert "ref2va" in out
    # once per run, not once per scene
    node._report_h3_checkpoint_mode([[torch.zeros(1, 12, 5120), {"minimax_refs": [{"kind": "image"}]}]])
    assert capsys.readouterr().out == ""


def test_mixing_pins_and_references_is_called_out(capsys):
    node = h3_node()
    node._h3_mode_noted = False
    node._report_h3_checkpoint_mode([[torch.zeros(1, 12, 5120), {
        "minimax_keyframes": [{"resolved_frame_index": 0}],
        "minimax_refs": [{"kind": "image"}],
    }]])
    out = capsys.readouterr().out
    assert "BOTH" in out and "fl2va" in out and "ref2va" in out


# ── pins rescued from a wired MiniMax H3 Image to Video node ─────────────────
# That node hands its first_frame/last_frame pins out on its CONDITIONING output, which this
# pipeline drops (Studio owns the sampler's positive). Wiring it into h3_keyframes is the
# difference between the image conditioning the generation and being silently discarded.

def _wired_conditioning(indices, frame_count=124, aug=None):
    pins = [{"resolved_frame_index": i, "latent": torch.full((1, 24, 1, 48, 84), float(i))}
            for i in indices]
    meta = {"minimax_keyframes": pins, "minimax_frame_count": frame_count}
    if aug is not None:
        meta["minimax_visual_cond_noise_aug"] = aug
    return [[torch.zeros(1, 12, 5120), meta]]


def test_wired_pins_are_split_into_first_and_last():
    node = h3_node()
    pins = node._h3_external_pins(_wired_conditioning([0, 123], aug=0.8))
    assert [p["resolved_frame_index"] for p in pins["first"]] == [0]
    assert [p["resolved_frame_index"] for p in pins["last"]] == [123]
    assert pins["aug"] == pytest.approx(0.8)
    assert pins["source_count"] == 124


def test_conditioning_without_pins_is_reported_as_nothing_to_rescue():
    node = h3_node()
    assert node._h3_external_pins([[torch.zeros(1, 12, 5120), {}]]) is None
    assert node._h3_external_pins(None) is None


def test_a_first_frame_pin_lands_on_the_opening_scene_only():
    node = h3_node(frame_count=124)
    pins = node._h3_external_pins(_wired_conditioning([0]))
    positive = [[torch.zeros(1, 12, 5120), {}]]

    out, labels = node._apply_h3_external_pins(positive, pins, scene_index=0, scene_count=3)
    assert labels == ["first"]
    assert [kf["resolved_frame_index"] for kf in out[0][1]["minimax_keyframes"]] == [0]

    out, labels = node._apply_h3_external_pins(positive, pins, scene_index=1, scene_count=3)
    assert labels == [] and out is positive        # untouched mid-chain


def test_a_last_frame_pin_is_reindexed_onto_the_closing_scenes_own_length():
    """The source node's frame_count is its own — a stale index would land mid-clip, where
    PackedLayout refuses it."""
    node = h3_node(frame_count=209)
    pins = node._h3_external_pins(_wired_conditioning([0, 123], frame_count=124))

    out, labels = node._apply_h3_external_pins(
        [[torch.zeros(1, 12, 5120), {}]], pins, scene_index=2, scene_count=3)
    assert labels == ["last"]
    assert [kf["resolved_frame_index"] for kf in out[0][1]["minimax_keyframes"]] == [208]


def test_a_single_scene_run_gets_both_pins():
    node = h3_node(frame_count=124)
    pins = node._h3_external_pins(_wired_conditioning([0, 123]))
    out, labels = node._apply_h3_external_pins(
        [[torch.zeros(1, 12, 5120), {}]], pins, scene_index=0, scene_count=1)
    assert labels == ["first", "last"]
    assert [kf["resolved_frame_index"] for kf in out[0][1]["minimax_keyframes"]] == [0, 123]


def test_a_wired_first_frame_overrides_the_timeline_anchor_on_frame_zero():
    """Both target frame 0; the explicit wire is the one the user drew."""
    node = h3_node(frame_count=124)
    anchor = torch.zeros(1, 24, 1, 48, 84)
    positive = node._h3_add_keyframes(
        [[torch.zeros(1, 12, 5120), {}]],
        [{"resolved_frame_index": 0, "latent": anchor}], 124)

    pins = node._h3_external_pins(_wired_conditioning([0]))
    out, _labels = node._apply_h3_external_pins(positive, pins, scene_index=0, scene_count=1)
    kfs = out[0][1]["minimax_keyframes"]
    assert len(kfs) == 1 and kfs[0]["latent"] is not anchor


# ── continuing a scene from the one before it ────────────────────────────────

def _pins(cond):
    return (cond[0][1] or {}).get("minimax_keyframes") or []


def test_a_continuation_scene_is_pinned_to_the_previous_scenes_last_frame():
    """H3 has no latent conditioning: a carried tail is, in the model's terms, noise it may
    overwrite. Without a pin the seam matched and the rest of the shot knew nothing about the
    scene before it — the chain produced unrelated clips."""
    node = h3_node()
    previous = av_latent(video_t=37)
    previous["samples"].unbind()[0][:, :, -1] = 7.0        # a mark only the last frame has
    positive = [[torch.zeros(1, 8, 16), {}]]

    out, applied = node._h3_continuation_pin(positive, previous)
    assert applied is True
    pins = _pins(out)
    assert [p["resolved_frame_index"] for p in pins] == [0]
    latent = pins[0]["latent"]
    assert latent.shape == (1, 24, 1, 48, 84)              # exactly one latent frame
    assert float(latent.max()) == 7.0                      # and it is the LAST one


def test_an_explicit_anchor_outranks_the_carried_tail():
    """An anchor image (or a wired first_frame) is the user saying where this scene starts;
    the tail is inferred. Two pins cannot share frame 0."""
    node = h3_node()
    anchored = [[torch.zeros(1, 8, 16), {"minimax_keyframes": [
        {"resolved_frame_index": 0, "latent": torch.full((1, 24, 1, 48, 84), 3.0)}]}]]
    out, applied = node._h3_continuation_pin(anchored, av_latent())
    assert applied is False
    assert float(_pins(out)[0]["latent"].max()) == 3.0


def test_the_opening_scene_has_nothing_to_continue_from():
    node = h3_node()
    positive = [[torch.zeros(1, 8, 16), {}]]
    out, applied = node._h3_continuation_pin(positive, None)
    assert applied is False and _pins(out) == []


def test_an_unreadable_carry_source_skips_rather_than_failing_the_render():
    node = h3_node()
    positive = [[torch.zeros(1, 8, 16), {}]]
    out, applied = node._h3_continuation_pin(positive, {"samples": torch.zeros(1, 24, 4)})
    assert applied is False and _pins(out) == []


# ── the two carry toggles that had no H3 path at all ─────────────────────────

def _template(video_t=37, protected=1):
    latent = av_latent(video_t=video_t)
    video = latent["samples"].unbind()[0]
    mask = torch.ones_like(video)
    mask[:, :, :protected] = 0.0                       # protected = pinned prefix
    latent["noise_mask"] = NT([mask, torch.ones_like(latent["samples"].unbind()[1])])
    video[:, :, 0] = 5.0                               # the reference frame
    return latent


def test_carry_i2v_guides_pins_the_reference_instead_of_growing_the_latent():
    """The LTX path prepends the template's protected frames to the chunk. On H3 that adds
    video frames off its 5k+2 grid and attaches conditioning the model does not read."""
    node = h3_node()
    chunk = av_latent(video_t=37)
    before = chunk["samples"].unbind()[0].shape
    out_chunk, positive, _neg, carried = node._append_i2v_guides(
        chunk, _template(), [[torch.zeros(1, 8, 16), {}]], None)
    assert out_chunk["samples"].unbind()[0].shape == before   # latent untouched
    assert carried == 0                                       # nothing to crop later
    pins = _pins(positive)
    assert [p["resolved_frame_index"] for p in pins] == [0]
    assert float(pins[0]["latent"].max()) == 5.0              # the ORIGINAL reference


def test_holding_the_reference_outranks_continuing_from_the_last_shot():
    """Continuing is the default; carry_i2v_guides is the user asking for the opposite."""
    node = h3_node()
    _c, positive, _n, _t = node._append_i2v_guides(
        av_latent(), _template(), [[torch.zeros(1, 8, 16), {}]], None)
    out, applied = node._h3_continuation_pin(positive, av_latent())
    assert applied is False
    assert float(_pins(out)[0]["latent"].max()) == 5.0


def test_mid_scene_guide_declines_on_h3_instead_of_damaging_the_latent(capsys):
    """H3's packed layout places a pin at the first or last frame, nowhere between."""
    node = h3_node(frame_count=124)
    chunk = av_latent(video_t=37)
    before = chunk["samples"].unbind()[0].shape
    out_chunk, positive, _neg, tail = node._append_mid_scene_guide(
        chunk, av_latent(video_t=37), [[torch.zeros(1, 8, 16), {}]], None, LTXVAE(), 1.0)
    assert out_chunk["samples"].unbind()[0].shape == before
    assert tail == 0 and _pins(positive) == []
    assert "pins only the first" in capsys.readouterr().out


# ── pins vs a resolution-changing second pass ────────────────────────────────

def test_keyframe_pins_are_resampled_onto_the_pass_two_grid():
    """A pin is packed as condition ROWS, so its token count belongs to the grid it was
    encoded on. Handing pass 2 a pass-1 pin fails inside the model as
    "value tensor of shape [168, 96] cannot be broadcast to [672, 96]" — 2x spatial is 4x
    the tokens, and 96 is 24 channels x the 2x2 patch."""
    node = h3_node()
    positive = [[torch.zeros(1, 8, 16), {"minimax_keyframes": [
        {"resolved_frame_index": 0, "latent": torch.randn(1, 24, 1, 24, 28)}]}]]
    out, changed = node._h3_rescale_pins(positive, 48, 56)
    assert changed == 1
    assert _pins(out)[0]["latent"].shape == (1, 24, 1, 48, 56)
    assert _pins(out)[0]["resolved_frame_index"] == 0     # position is untouched
    # the original conditioning is not mutated in place
    assert _pins(positive)[0]["latent"].shape == (1, 24, 1, 24, 28)


def test_a_pin_already_on_the_right_grid_is_left_exactly_alone():
    node = h3_node()
    latent = torch.randn(1, 24, 1, 48, 56)
    positive = [[torch.zeros(1, 8, 16), {"minimax_keyframes": [
        {"resolved_frame_index": 0, "latent": latent}]}]]
    out, changed = node._h3_rescale_pins(positive, 48, 56)
    assert changed == 0
    assert _pins(out)[0]["latent"] is latent


def test_conditioning_without_pins_passes_through_untouched():
    node = h3_node()
    positive = [[torch.zeros(1, 8, 16), {"minimax_refs": [{"kind": "image"}]}]]
    out, changed = node._h3_rescale_pins(positive, 48, 56)
    assert changed == 0 and out == positive


def test_every_pin_in_a_multi_pin_conditioning_is_brought_along():
    """first_frame and last_frame can both be set; one surviving on the old grid still
    fails the whole render."""
    node = h3_node()
    positive = [[torch.zeros(1, 8, 16), {"minimax_keyframes": [
        {"resolved_frame_index": 0, "latent": torch.randn(1, 24, 1, 24, 28)},
        {"resolved_frame_index": 123, "latent": torch.randn(1, 24, 1, 24, 28)}]}]]
    out, changed = node._h3_rescale_pins(positive, 48, 56)
    assert changed == 2
    assert all(p["latent"].shape == (1, 24, 1, 48, 56) for p in _pins(out))
    assert [p["resolved_frame_index"] for p in _pins(out)] == [0, 123]


# --- taste directions across a text preprocessor ---------------------------------
# H3 refines the conditioning inside extra_conds, so what the DiT consumes is NOT what the
# taste store captured. Adding one to the other crashed embed_guidance outright and was
# silently swallowed by score_slider's except clause.

TEXT_DIM, HIDDEN = 5120, 5376


class _RefinerModel:
    """Stands in for MiniMaxH3Model: condition_proj + a nonlinear token refiner."""

    def __init__(self, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.w = torch.randn(TEXT_DIM, HIDDEN, generator=g) / (TEXT_DIM ** 0.5)

    def preprocess_text_embeds(self, text_states):
        if text_states.shape[-1] == HIDDEN:
            return text_states
        return torch.tanh(text_states @ self.w)


def _h3ish_model(with_preprocessor=True):
    inner = types.SimpleNamespace()
    if with_preprocessor:
        inner.diffusion_model = _RefinerModel()
    else:
        inner.diffusion_model = types.SimpleNamespace()
    return types.SimpleNamespace(model=inner, model_options={})


def test_a_direction_already_in_the_models_space_is_left_alone():
    node = FunPackLTXAVSceneChainSampler()
    direction = torch.randn(HIDDEN)
    cond = torch.randn(1, 8, HIDDEN)
    out = node._direction_in_cond_space(_h3ish_model(), None, direction, cond)
    assert out is direction  # LTX: capture space == consumed space


def test_a_taste_direction_is_lifted_through_the_models_text_preprocessor():
    node = FunPackLTXAVSceneChainSampler()
    raw = torch.randn(1, 12, TEXT_DIM)
    direction = torch.randn(TEXT_DIM)
    cond = torch.randn(1, 12, HIDDEN)
    out = node._direction_in_cond_space(_h3ish_model(), raw, direction, cond)
    assert out is not None
    assert tuple(out.shape) == (HIDDEN,)
    assert torch.isfinite(out).all()
    assert float(out.norm()) == pytest.approx(1.0, abs=1e-4)


def test_the_lift_follows_the_direction_rather_than_returning_a_constant():
    node = FunPackLTXAVSceneChainSampler()
    raw = torch.randn(1, 12, TEXT_DIM)
    cond = torch.randn(1, 12, HIDDEN)
    model = _h3ish_model()
    a = node._direction_in_cond_space(model, raw, torch.randn(TEXT_DIM), cond)
    b = node._direction_in_cond_space(model, raw, torch.randn(TEXT_DIM), cond)
    assert float(torch.nn.functional.cosine_similarity(a, b, dim=0)) < 0.9


def test_a_direction_that_cannot_be_mapped_is_declined_not_forced():
    node = FunPackLTXAVSceneChainSampler()
    out = node._direction_in_cond_space(
        _h3ish_model(with_preprocessor=False), torch.randn(1, 12, TEXT_DIM),
        torch.randn(TEXT_DIM), torch.randn(1, 12, HIDDEN))
    assert out is None


def _run_embed_wrapper(model, raw_cond, direction, cond):
    node = FunPackLTXAVSceneChainSampler()
    node._build_embed_guidance_wrapper(model, direction, 0.3, raw_cond=raw_cond)
    wrapper = model.model_options["model_function_wrapper"]
    seen = {}

    def apply_fn(x, t, **c):
        seen["cond"] = c.get("c_crossattn")
        return torch.zeros(1, 4)

    args = {"input": torch.zeros(1, 4), "timestep": torch.tensor([0.2]),
            "c": {"c_crossattn": cond}}
    return wrapper(apply_fn, args), seen


def test_embed_guidance_steers_the_refined_conditioning_instead_of_crashing():
    cond = torch.randn(1, 12, HIDDEN)
    _, seen = _run_embed_wrapper(_h3ish_model(), torch.randn(1, 12, TEXT_DIM),
                                 torch.randn(TEXT_DIM), cond)
    assert seen["cond"].shape == cond.shape
    assert not torch.equal(seen["cond"], cond)  # the nudge actually landed


def test_embed_guidance_passes_through_when_the_direction_cannot_be_mapped(capsys):
    cond = torch.randn(1, 12, HIDDEN)
    out, seen = _run_embed_wrapper(_h3ish_model(with_preprocessor=False),
                                   torch.randn(1, 12, TEXT_DIM), torch.randn(TEXT_DIM), cond)
    assert torch.equal(seen["cond"], cond)   # untouched, and no exception escaped
    assert out.shape == (1, 4)
    assert "steering skipped" in capsys.readouterr().out


def test_dynashift_weights_negatives_against_the_raw_prompt_not_the_refined_one():
    node = FunPackLTXAVSceneChainSampler()
    raw = torch.randn(1, 12, TEXT_DIM)
    pooled = raw.mean(dim=0).mean(dim=0)
    negatives = [{"latent": torch.randn(3, 2, 4, 4), "cond": pooled.clone()},
                 {"latent": torch.randn(3, 2, 4, 4), "cond": -pooled.clone()}]
    model = _h3ish_model()
    node._build_dynashift_wrapper(model, negatives, 0.3, 0.6, raw_cond=raw)
    captured = {}

    def apply_fn(x, t, **c):
        captured["ran"] = True
        return torch.zeros(1, 1, 8)

    # The bank's cond is 5120-wide; the model consumes 5376. Before the fix the numel guard
    # weighted every negative 1.0 regardless of how unrelated its prompt was.
    args = {"input": torch.zeros(1, 1, 8), "timestep": torch.tensor([0.2]),
            "c": {"c_crossattn": torch.randn(1, 12, HIDDEN)}}
    model.model_options["model_function_wrapper"](apply_fn, args)
    assert captured["ran"]
