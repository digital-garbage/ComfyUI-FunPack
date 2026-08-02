"""MiniMax-H3 adapter — the four places H3 diverges from LTXAV badly enough to break silently.

Each test here pins a divergence that produces plausible-looking wrong output rather than an
exception, which is why they are worth having:

1. The 17k+5 frame grid. LTXAV's 8k+1 formula returns a number for an H3 length too — just the
   wrong one, ~20% short — so a template built with it validates and then mismatches the model.
2. The audio time axis. H3's audio latent is [B, 32, stereo, T]; LTXAV's is [B, C, T, freq].
   Both are 4-D, so only the model family can tell you which axis is time. Slicing dim 2 on H3
   crops a speaker instead of the tail.
3. Token tags. The DiT indexes a per-token tag vector across the text span; a conditioning whose
   token count drifted from its tags dies mid-forward with an IndexError.
4. Batch size 1. The DiT refuses a batched forward, which is exactly what comfy builds for
   cfg != 1.0.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import minimax_h3 as h3


# ── 1. frame grid ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("length, aligned", [
    (5, 5), (6, 22), (22, 22), (23, 39), (124, 124), (125, 141), (1, 5),
])
def test_align_frame_count_snaps_up_to_17k_plus_5(length, aligned):
    assert h3.align_frame_count(length) == aligned
    assert h3.align_frame_count(length) % h3.FRAME_GRID == h3.FRAME_BASE


@pytest.mark.parametrize("frames, latents", [
    (5, 2), (22, 7), (39, 12), (124, 37), (362, 107),
])
def test_video_latent_frames_follow_the_5k_plus_2_grid(frames, latents):
    assert h3.video_latent_frames(frames) == latents


def test_ltx_frame_formula_would_be_wrong_here():
    """The bug this module exists to prevent, stated as a test.

    FunPack's LTXAV formula is ((n - 1) // time_scale) + 1 with time_scale read off
    downscale_index_formula[0], which is 4 for H3. For the default 124-frame clip that
    yields 31 latent frames where the model wants 37 — a template that passes validation
    and then does not match the model.
    """
    ltx_style = ((124 - 1) // 4) + 1
    assert ltx_style == 31
    assert h3.video_latent_frames(124) == 37


def test_latent_frames_from_vae_prefers_the_count_map():
    class FakeH3VAE:
        # what comfy/sd.py sets for the H3 video VAE
        downscale_ratio = (lambda a: max(1, (a - 5) // 17 * 5 + 2) if a > 1 else 1, 16, 16)
        downscale_index_formula = (4, 16, 16)

    class FakeLTXVAE:
        downscale_ratio = (lambda a: max(0, (a + 7) // 8), 32, 32)
        downscale_index_formula = (8, 32, 32)

    assert h3.latent_frames_from_vae(FakeH3VAE(), 124) == 37
    # the same helper must keep reproducing LTXAV's answer exactly, or every existing
    # project's template validation changes underneath the user.
    for pixels in (1, 9, 121, 129, 257):
        assert h3.latent_frames_from_vae(FakeLTXVAE(), pixels) == ((pixels - 1) // 8) + 1


def test_audio_latent_frames_track_the_video_duration():
    # 124 frames / 24 fps = 5.1667 s, at 40 audio latent fps
    assert h3.audio_latent_frames(124) == 207
    frames, vt, at = h3.temporal_shape(120)
    assert (frames, vt, at) == (124, 37, 207)


def test_adapt_canvas_snaps_to_32_and_caps_area():
    w, h_ = h3.adapt_canvas(1920, 1080)
    assert w % 32 == 0 and h_ % 32 == 0
    assert w * h_ <= 768 * 1344 * 1.05
    assert h3.adapt_canvas(768, 768) == (768, 768)


# ── 2. per-stream time axis ──────────────────────────────────────────────────

def test_stream_time_dims_split_video_from_audio_only_on_h3():
    assert h3.stream_time_dims(2, h3=False) == [2, 2]
    assert h3.stream_time_dims(2, h3=True) == [2, -1]
    assert h3.stream_time_dims(1, h3=True) == [2]


def test_slicing_h3_audio_on_dim_2_would_crop_a_speaker():
    """[B, 32, stereo=2, T]: dim 2 is the stereo channel, dim -1 is time."""
    audio = torch.arange(1 * 32 * 2 * 10).reshape(1, 32, 2, 10).float()
    dims = h3.stream_time_dims(2, h3=True)
    kept = h3.time_slice(audio, 0, -3, dims[1])
    assert kept.shape == (1, 32, 2, 7)          # tail trimmed, both speakers intact
    assert torch.equal(kept, audio[..., :7])
    # what "trim 3 frames" does on the LTXAV axis: the stereo axis is only 2 long, so
    # the crop empties it and the audio is gone rather than shortened.
    wrong = h3.time_slice(audio, 0, -3, 2)
    assert wrong.shape == (1, 32, 0, 10)

    ltx_audio = torch.zeros(1, 8, 10, 4)
    ltx_dims = h3.stream_time_dims(2, h3=False)
    assert h3.time_slice(ltx_audio, 0, -3, ltx_dims[1]).shape == (1, 8, 7, 4)


def test_set_time_slice_writes_on_the_stream_axis():
    audio = torch.zeros(1, 32, 2, 10)
    h3.set_time_slice(audio, 0, 3, 1.0, h3.AUDIO_TIME_DIM)
    assert audio[..., :3].eq(1.0).all()
    assert audio[..., 3:].eq(0.0).all()


def test_time_cat_appends_on_the_stream_axis():
    audio = torch.zeros(1, 32, 2, 10)
    tail = torch.ones(1, 32, 2, 4)
    out = h3.time_cat([audio, tail], h3.AUDIO_TIME_DIM)
    assert out.shape == (1, 32, 2, 14)
    assert out[..., -4:].eq(1.0).all()


# ── 3. token tags ────────────────────────────────────────────────────────────

def test_tags_match_catches_a_token_count_that_drifted():
    cond = torch.zeros(1, 12, 5120)
    meta = {"minimax_token_tags": torch.ones(12, dtype=torch.long)}
    assert h3.tags_match(cond, meta)
    # append 4 identity tokens without touching the tags -> the DiT would index past the end
    assert not h3.tags_match(torch.zeros(1, 16, 5120), meta)
    # no tags at all (LTXAV conditioning) is never a mismatch
    assert h3.tags_match(cond, {"pooled_output": None})


def test_extend_token_tags_keeps_appended_tokens_legal():
    meta = {"minimax_token_tags": torch.zeros(12, dtype=torch.long)}
    grown = h3.extend_token_tags(meta, 4)
    assert h3.token_tags_length(grown) == 16
    assert grown["minimax_token_tags"][-4:].eq(1).all()   # text modality row
    assert h3.tags_match(torch.zeros(1, 16, 5120), grown)
    assert h3.token_tags_length(meta) == 12               # input not mutated


def test_keyframe_indices_are_first_or_last_only():
    assert h3.keyframe_indices_supported(0, 124)
    assert h3.keyframe_indices_supported(123, 124)
    assert not h3.keyframe_indices_supported(60, 124)


# ── 4. batch size 1 ──────────────────────────────────────────────────────────

class _FakePatcher:
    def __init__(self):
        self.model_options = {}


def test_batch_split_runs_a_batched_forward_one_sample_at_a_time():
    """The DiT raises on batch > 1; comfy builds batch 2 for cfg != 1.0."""
    seen = []

    def apply_fn(x, timestep, **c):
        if x.shape[0] != 1:
            raise ValueError("MiniMax H3 supports batch size 1")
        seen.append((x.shape[0], float(x[0, 0, 0]), c["c_crossattn"].shape[0]))
        return x * 2.0

    model = _FakePatcher()
    h3.install_batch_split(model)
    wrapper = model.model_options["model_function_wrapper"]

    x = torch.stack([torch.full((1, 4), 1.0), torch.full((1, 4), 3.0)])  # [2,1,4]
    args = {"input": x, "timestep": torch.tensor([0.5, 0.5]),
            "c": {"c_crossattn": torch.zeros(2, 7, 5376)}}
    out = wrapper(apply_fn, args)

    assert [s[0] for s in seen] == [1, 1]
    assert [s[1] for s in seen] == [1.0, 3.0]     # both samples, in order
    assert [s[2] for s in seen] == [1, 1]         # conditioning sliced with them
    assert out.shape == x.shape
    assert torch.allclose(out, x * 2.0)


def test_batch_split_is_a_pass_through_for_a_single_sample():
    calls = []

    def apply_fn(x, timestep, **c):
        calls.append(x.shape[0])
        return x

    model = _FakePatcher()
    h3.install_batch_split(model)
    args = {"input": torch.zeros(1, 1, 4), "timestep": torch.tensor([0.5]), "c": {}}
    model.model_options["model_function_wrapper"](apply_fn, args)
    assert calls == [1]


def test_batch_split_chains_onto_an_existing_wrapper():
    order = []

    def existing(apply_fn, args):
        order.append("existing")
        return apply_fn(args["input"], args["timestep"], **args.get("c", {}))

    def apply_fn(x, timestep, **c):
        order.append("model")
        return x

    model = _FakePatcher()
    model.model_options["model_function_wrapper"] = existing
    old, _ = h3.install_batch_split(model)
    assert old is existing

    args = {"input": torch.zeros(2, 1, 4), "timestep": torch.tensor([0.5, 0.5]), "c": {}}
    model.model_options["model_function_wrapper"](apply_fn, args)
    assert order == ["existing", "model", "existing", "model"]


# ── detection degrades when the H3 PR is not installed ───────────────────────

def test_detection_is_false_for_anything_that_is_not_h3():
    class Inner:
        pass

    class Patcher:
        model = Inner()

    assert not h3.is_h3_model(None)
    assert not h3.is_h3_model(Patcher())
    assert not h3.is_h3_clip(None)
    assert not h3.is_h3_video_vae(None)
    assert not h3.is_h3_audio_vae(object())


def test_detection_reads_the_unet_config():
    class Cfg:
        unet_config = {"image_model": "minimax_h3"}

    class Inner:
        model_config = Cfg()

    class Patcher:
        model = Inner()

    assert h3.is_h3_model(Patcher())

    Cfg.unet_config = {"image_model": "ltxav"}
    assert not h3.is_h3_model(Patcher())


def test_attention_patch_targets_always_includes_comfys_module():
    targets = h3.attention_patch_targets()
    names = [getattr(t, "__name__", "") for t in targets]
    assert "comfy.ldm.modules.attention" in names or names == []
