"""ALG on H3, where the anchor is a keyframe pin in the payload, not a latent frame."""

import pytest
import torch


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy."""


def _pin_payload():
    z = torch.randn(1, 24, 1, 4, 4)
    return z, {"keyframes": [{"resolved_frame_index": 0, "latent": z}],
               "cond_video_latents": [z]}


def _apply_wrapper(tiny, **values):
    from comfy.patcher_extension import WrappersMP
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers
    patched, status = FunPackLoadModifiers.execute(
        tiny.patcher, {"alg": {"enabled": True, **values}}).result
    (fn,), = patched.wrappers[WrappersMP.APPLY_MODEL].values()
    return fn, status


def _call(fn, sigma, payload):
    seen = {}

    def executor(x, t, *a, **kw):
        seen.update(kw)
        return x

    fn(executor, torch.zeros(1), torch.tensor([sigma]), minimax_payload=payload)
    return seen["minimax_payload"]


def test_the_opening_pin_is_blurred_above_the_threshold_and_sharp_below(tiny_h3):
    fn, _ = _apply_wrapper(tiny_h3, strength=4.0, until_sigma=0.6)
    z, payload = _pin_payload()
    early = _call(fn, 0.9, payload)["cond_video_latents"][0]
    late = _call(fn, 0.3, payload)["cond_video_latents"][0]
    assert late is z
    assert early.shape == z.shape and not torch.allclose(early, z)
    assert payload["cond_video_latents"][0] is z, "the caller's payload was edited in place"


def test_a_mid_clip_pin_is_left_alone(tiny_h3):
    fn, _ = _apply_wrapper(tiny_h3)
    z = torch.randn(1, 24, 1, 4, 4)
    payload = {"keyframes": [{"resolved_frame_index": 5, "latent": z}], "cond_video_latents": [z]}
    assert _call(fn, 0.9, payload)["cond_video_latents"][0] is z


def test_the_blurred_pin_changes_a_real_h3_forward(tiny_h3):
    """The swapped latent is what the model actually reads."""
    from modules.models.minimax_h3.anchor import anchor_pin
    from modules.sampling.alg.blur import blur_frames
    z, payload = _pin_payload()
    dm = tiny_h3.patcher.model.diffusion_model
    blurred = anchor_pin({"minimax_payload": payload},
                         lambda l: blur_frames(l, 4.0, frame_indices=range(l.shape[2])))

    def fwd(p):
        with torch.inference_mode():
            return dm([tiny_h3.video, tiny_h3.audio], torch.tensor([900.0]), tiny_h3.context,
                      transformer_options={}, minimax_payload=p)[0]

    assert not torch.allclose(fwd(payload), fwd(blurred["minimax_payload"]))
