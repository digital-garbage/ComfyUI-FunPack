"""second_pass — a general two-pass split on the Chain Sampler.

A scene can be finished differently from how it was started. The invariants that matter
here are the ones a split can silently get wrong:

1. BOTH schedules run in full, exactly as written. Pass 1 is not cut short and pass 2 is
   not derived from it — total steps are simply the two added up. (An earlier version cut
   pass 1 at pass 2's first sigma; it silently shortened the run and distorted the result.)
2. A hand-typed schedule is the one malformed-able input, and both ways it can be wrong
   are silent in the OUTPUT rather than loud at runtime, so they are refused up front.
3. Nothing is ever a silent no-op: every refusal comes back with a reason naming the fix.
4. The i2v anchor stays PINNED for pass 2 — without this pass 2 re-denoises the reference
   frame and the scene drifts off the image it was supposed to start from.
"""
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

for _name in (
    "comfy", "comfy.k_diffusion", "comfy.k_diffusion.sampling",
    "comfy.model_sampling", "comfy.nested_tensor", "comfy.sample",
    "comfy.samplers", "comfy.utils", "comfy.model_management",
):
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules["comfy.nested_tensor"].NestedTensor = object
# _run_upsampler (detailing.py) moves tensors to the torch device and reads the VAE's
# per-channel statistics; on CPU with vae=None both are no-ops, which is what we want here.
sys.modules["comfy.model_management"].get_torch_device = lambda: torch.device("cpu")
sys.modules["comfy"].model_management = sys.modules["comfy.model_management"]

import samplers  # noqa: E402


# The user's own 9-step schedule.
SCHEDULE = torch.tensor([1.0, 0.955, 0.893, 0.812, 0.715, 0.603, 0.482, 0.241, 0.121, 0.0])


def _node():
    return samplers.FunPackLTXAVSceneChainSampler()


# ── the pass-2 schedule ─────────────────────────────────────────────────────

def _sched(alt):
    return _node()._second_pass_schedule(alt)


def test_a_well_formed_schedule_is_taken_exactly_as_written():
    """Nothing is derived, cut or continued. Pass 1 runs the main schedule in full and pass 2
    runs this one in full, so on the user's 9-step main a 4-step second pass is 13 steps —
    not a re-slicing of the main schedule."""
    alt = torch.tensor([0.812, 0.6, 0.35, 0.15, 0.0])
    out, reason = _sched(alt)
    assert reason is None and out is alt
    assert (SCHEDULE.numel() - 1) + (out.numel() - 1) == 13


def test_an_ascending_schedule_is_refused():
    """Both malformed cases are silent in the OUTPUT rather than loud at runtime, so they
    have to be caught before sampling. An ascending pair walks the trajectory backwards."""
    out, reason = _sched(torch.tensor([0.812, 0.4, 0.55, 0.0]))
    assert out is None and "must descend" in reason


def test_a_schedule_that_stops_short_of_zero_is_refused():
    """Stopping above 0 leaves the clip partially denoised — the noise-artefact symptom."""
    out, reason = _sched(torch.tensor([0.812, 0.5, 0.25]))
    assert out is None and "not 0" in reason


def test_equal_neighbouring_sigmas_are_not_treated_as_ascending():
    out, reason = _sched(torch.tensor([0.812, 0.6, 0.6, 0.0]))
    assert reason is None and out is not None


def test_a_missing_or_degenerate_schedule_is_refused_with_a_reason():
    """second_pass with nothing to run must say so, not silently sample once."""
    for alt in (None, torch.tensor([]), torch.tensor([0.5]), "0.8, 0.0"):
        out, reason = _sched(alt)
        assert out is None and reason


def test_a_schedule_starting_at_zero_is_refused():
    out, reason = _sched(torch.tensor([0.0, 0.0]))
    assert out is None and "nothing for it to denoise" in reason


# ── the pin survives the split ──────────────────────────────────────────────

def test_the_i2v_anchor_is_re_pinned_for_pass_two():
    """_sample_chunk drops noise_mask from what it returns, so without this pass 2 runs
    unpinned and re-denoises the reference frame."""
    frames = 6
    clean = torch.arange(frames, dtype=torch.float32).view(1, 1, frames, 1, 1).repeat(1, 4, 1, 2, 2)
    mask = torch.ones_like(clean)
    mask[:, :, :1] = 0.0                       # one pinned anchor frame
    chunk = {"samples": clean, "noise_mask": mask}
    # A "state" where the anchor frame has been polluted (as pass 2's re-noise would).
    state = {"samples": torch.full_like(clean, -99.0)}

    out = _node()._restore_pinned_prefix(state, chunk)
    assert out.get("noise_mask") is not None
    # The pinned frame is back to the clean anchor...
    assert torch.equal(out["samples"][:, :, :1], clean[:, :, :1])
    # ...and nothing else was touched.
    assert torch.equal(out["samples"][:, :, 1:], state["samples"][:, :, 1:])


def test_restore_is_a_no_op_on_a_t2v_scene():
    """No pinned prefix = no anchor to protect; the state must come back untouched."""
    clean = torch.zeros(1, 4, 6, 2, 2)
    state = {"samples": torch.full_like(clean, 3.0)}
    for chunk in ({"samples": clean, "noise_mask": torch.ones_like(clean)},
                  {"samples": clean}):
        assert _node()._restore_pinned_prefix(state, chunk) is state


# ── the optional between-pass operations ────────────────────────────────────

class _FakeUpsampler:
    """Stands in for Lightricks' LatentUpsampler: 2x nearest on the spatial axes."""
    def __call__(self, x):
        return x.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)

    def parameters(self):
        yield torch.zeros(1)

    def to(self, *a, **k):
        return self

    def cpu(self):
        return self


def _video(f=4, h=8, w=8, fill=2.0):
    return {"samples": torch.full((1, 4, f, h, w), fill)}


def test_no_operation_is_the_default_and_returns_the_same_object():
    """The whole feature is opt-in: 'none' must not clone, resample or touch anything."""
    lat = _video()
    for op in ("none", None, ""):
        out, note = _node()._second_pass_operate(lat, op, _FakeUpsampler(), None)
        assert out is lat and note is None


def test_sharpen_comes_back_at_the_original_size():
    """sharpen is upsampler -> downscale: the latent pass 2 receives must be the SAME shape,
    or it is silently an upscale and costs four times as much."""
    out, note = _node()._second_pass_operate(_video(h=8, w=8), "sharpen", _FakeUpsampler(), None)
    assert out["samples"].shape == (1, 4, 4, 8, 8)
    assert "sharpen" in note


def test_upscale_2x_keeps_the_larger_latent():
    out, note = _node()._second_pass_operate(_video(h=8, w=8), "upscale_2x", _FakeUpsampler(), None)
    assert out["samples"].shape == (1, 4, 4, 16, 16)
    assert "4x the pixels" in note


def test_a_missing_upsampler_degrades_instead_of_failing_the_render():
    """A whole montage must not die because one optional sharpen could not load its model."""
    lat = _video()
    out, note = _node()._second_pass_operate(lat, "sharpen", None, None)
    assert out is lat and "no latent upsampler" in note


def test_an_upsampler_that_throws_degrades_too():
    class _Boom(_FakeUpsampler):
        def __call__(self, x):
            raise RuntimeError("out of memory")
    lat = _video()
    out, note = _node()._second_pass_operate(lat, "sharpen", _Boom(), None)
    assert out is lat and "upsampler failed" in note


def test_an_unknown_operation_is_reported_not_guessed():
    lat = _video()
    out, note = _node()._second_pass_operate(lat, "enhance", _FakeUpsampler(), None)
    assert out is lat and "unknown operation" in note


def test_the_pin_survives_an_operation_that_changed_the_resolution():
    """upscale_2x leaves the chunk's mask the wrong size, which used to drop the pin and let
    pass 2 re-denoise the anchor at 4x the pixels — the run most likely to drift.

    The anchor is not re-derived from the source image: the pinned frames in the upscaled
    state ARE the anchor, carried through the same upsampler as every other frame. Only the
    mask has to be brought to the new grid.
    """
    clean = torch.zeros(1, 4, 4, 8, 8)
    mask = torch.ones_like(clean)
    mask[:, :, :1] = 0.0
    chunk = {"samples": clean, "noise_mask": mask}
    anchor = torch.full((1, 4, 1, 16, 16), 7.0)
    upscaled = {"samples": torch.zeros(1, 4, 4, 16, 16)}
    upscaled["samples"][:, :, :1] = anchor

    out = _node()._restore_pinned_prefix(upscaled, chunk)
    assert out is not upscaled
    m = out["noise_mask"]
    assert m.shape[-2:] == (16, 16)
    assert float(m[:, :, :1].max()) == 0.0      # anchor frame still pinned...
    assert float(m[:, :, 1:].min()) == 1.0      # ...and nothing else is
    # The upsampler's anchor is what stays pinned — untouched by the restore.
    assert torch.equal(out["samples"][:, :, :1], anchor)


def test_the_pin_is_dropped_when_the_mask_cannot_be_rebuilt():
    """A mask with no spatial axes to scale is unusable at the new size. Dropping the pin and
    saying so beats pinning against a grid that isn't there."""
    clean = torch.zeros(1, 4, 4, 8, 8)
    mask = torch.ones(1, 4, 4)                  # 3-D: no H/W to rescale
    mask[:, :1] = 0.0
    chunk = {"samples": clean, "noise_mask": mask}
    upscaled = {"samples": torch.zeros(1, 4, 4, 16, 16)}
    assert _node()._restore_pinned_prefix(upscaled, chunk) is upscaled


# ── a latent op is about the upsampler, not the model family ──────────────────
# The op used to be switched off outright on MiniMax H3 because Lightricks' upsampler is
# trained on LTX's 128-channel latent. That is a fact about the FILE, not about H3: with an
# upsampler trained for the model, the op runs. What has to stay true is that a mismatched
# one is refused by name rather than blowing up inside a convolution mid-render.
import detailing  # noqa: E402


class _TypedUpsampler(_FakeUpsampler):
    def __init__(self, in_channels):
        self.in_channels = in_channels


def test_an_upsamplers_width_is_read_from_the_model():
    assert detailing.upsampler_in_channels(_TypedUpsampler(24)) == 24


def test_an_upsamplers_width_falls_back_to_its_first_convolution():
    class _Weights:
        def state_dict(self):
            return {"initial_conv.weight": torch.zeros(512, 24, 3, 3)}
    assert detailing.upsampler_in_channels(_Weights()) == 24


def test_an_upsampler_wrapped_in_a_patcher_is_still_readable():
    class _Patcher:
        model = _TypedUpsampler(24)
    assert detailing.upsampler_in_channels(_Patcher()) == 24


def test_an_upsampler_for_another_model_is_refused_by_name():
    lat = _video()                                   # 4-channel stand-in latents
    out, note = _node()._second_pass_operate(lat, "upscale_2x", _TypedUpsampler(128), None)
    assert out is lat
    assert "128-channel" in note and "4-channel" in note


def test_a_matching_upsampler_runs_on_any_family():
    out, note = _node()._second_pass_operate(_video(h=8, w=8), "upscale_2x", _TypedUpsampler(4), None)
    assert out["samples"].shape == (1, 4, 4, 16, 16)
    assert "4x the pixels" in note


def _installed(monkeypatch, files):
    """Patch the SHARED folder_paths stub in place. Replacing sys.modules["folder_paths"]
    would leak this test's view of the models folder into every later test file."""
    monkeypatch.setattr(sys.modules["folder_paths"], "get_filename_list",
                        lambda _kind: list(files))


def test_auto_never_downloads_the_ltx_upsampler_for_a_model_it_cannot_fit(monkeypatch):
    """A gigabyte, then a shape error. 'auto' takes what is installed instead."""
    _installed(monkeypatch, ["h3_latent_upscaler.safetensors"])
    monkeypatch.setattr(detailing, "DEFAULT_UPSAMPLER_FILE", "ltx.safetensors", raising=False)
    assert detailing.resolve_upsampler_name("auto", 24) == "h3_latent_upscaler.safetensors"


def test_auto_says_what_to_install_when_nothing_fits(monkeypatch):
    _installed(monkeypatch, [])
    try:
        detailing.resolve_upsampler_name("auto", 24)
    except RuntimeError as exc:
        assert "24-channel" in str(exc) and "latent_upscale_models" in str(exc)
    else:
        raise AssertionError("expected a RuntimeError naming the channel width")


def test_an_explicitly_chosen_file_is_always_honoured(monkeypatch):
    _installed(monkeypatch, [])
    assert detailing.resolve_upsampler_name("my_upsampler.safetensors", 24) == "my_upsampler.safetensors"


# ── an upsampler ComfyUI cannot load at all ───────────────────────────────────

def test_an_unknown_upsampler_architecture_is_named_not_a_python_error(monkeypatch):
    """Core's loader is an if/elif with no else, so a file it does not recognise falls off
    the end as `cannot access local variable 'model'` — which names neither the file nor
    the problem, and reads like a FunPack bug."""
    called = []
    monkeypatch.setitem(
        sys.modules, "comfy_extras.nodes_hunyuan",
        types.SimpleNamespace(LatentUpscaleModelLoader=types.SimpleNamespace(
            execute=lambda name: called.append(name))))
    sd = {"encoder.layers.0.weight": object(), "decoder.out.bias": object()}
    with pytest.raises(ValueError) as exc:
        detailing._load_upsampler_via_core("h3_upsampler.safetensors", sd)
    msg = str(exc.value)
    assert "h3_upsampler.safetensors" in msg
    assert "encoder, decoder" in msg          # what the file actually looks like
    assert "Hunyuan Video 1.5 SR (720p)" in msg
    assert not called, "core's loader should not be reached with a file it cannot branch on"


def test_an_architecture_core_does_know_is_still_handed_over(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "comfy_extras.nodes_hunyuan",
        types.SimpleNamespace(LatentUpscaleModelLoader=types.SimpleNamespace(
            execute=lambda name: ("loaded",))))
    sd = {"blocks.0.block.0.conv.weight": object()}
    assert detailing._load_upsampler_via_core("hunyuan_sr.safetensors", sd) == "loaded"
