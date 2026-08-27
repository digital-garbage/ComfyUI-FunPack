"""Wrapper hygiene on the Scene Chain sampler.

Two guarantees the feature stack depends on:
1. Every per-scene model_function_wrapper (embed guidance / score slider / output
   guidance) is tagged with its predecessor so a leaked chain can be unwound
   (_strip_funpack_scene_wrappers) — the in-process analogue of
   strip_funpack_block_hooks.
2. output_guidance's correction is norm-calibrated on the video span: strength is a
   fraction of the stream norm (not a raw MLP gradient, which is numerically inert),
   and the audio region of a packed AV latent is byte-identical after correction.
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


def _model():
    return types.SimpleNamespace(model_options={})


def _install_stack(model):
    """Install embed -> slider -> output, the same order sample() uses."""
    s = samplers.FunPackLTXAVSceneChainSampler()
    liked = torch.ones(16)
    s._build_embed_guidance_wrapper(model, liked, 0.02)
    s._build_score_slider_wrapper(model, liked, eta=1.0)

    class _VF:
        def gradient(self, x):
            return torch.ones_like(x)

    s._build_output_guidance_wrapper(model, _VF(), 0.02)
    return s


def test_installed_wrappers_are_tagged_with_prev_chain():
    model = _model()
    _install_stack(model)
    w = model.model_options["model_function_wrapper"]
    depth = 0
    while w is not None and getattr(w, samplers._FUNPACK_SCENE_WRAPPER_TAG, False):
        w = w._funpack_prev_wrapper
        depth += 1
    assert depth == 3
    assert w is None


def test_strip_unwinds_leaked_stack_to_base():
    model = _model()

    def base_wrapper(apply_fn, args):  # a non-FunPack wrapper that must survive
        return apply_fn(args)

    model.model_options["model_function_wrapper"] = base_wrapper
    _install_stack(model)
    stripped = samplers._strip_funpack_scene_wrappers(model)
    assert stripped == 3
    assert model.model_options["model_function_wrapper"] is base_wrapper
    # Idempotent: nothing left to strip.
    assert samplers._strip_funpack_scene_wrappers(model) == 0


def test_strip_removes_key_when_no_base_wrapper():
    model = _model()
    _install_stack(model)
    assert samplers._strip_funpack_scene_wrappers(model) == 3
    assert "model_function_wrapper" not in model.model_options


def test_strip_leaves_foreign_wrapper_alone():
    model = _model()

    def foreign(apply_fn, args):
        return apply_fn(args)

    model.model_options["model_function_wrapper"] = foreign
    assert samplers._strip_funpack_scene_wrappers(model) == 0
    assert model.model_options["model_function_wrapper"] is foreign


class _ConstGradVF:
    """Gradient of constant direction; lets the test predict the calibrated delta."""

    def gradient(self, x):
        return torch.ones_like(x)


def _packed_av_model(video_elems, audio_elems):
    """Model stub whose guider conds carry latent_shapes for a packed AV layout."""
    ls = types.SimpleNamespace(cond=[(1, 4, video_elems // 4), (1, audio_elems)])
    cond_entry = {"model_conds": {"latent_shapes": ls}}
    guider = types.SimpleNamespace(conds={"positive": [cond_entry]})
    return types.SimpleNamespace(model_options={}, inner_model=guider)


def test_output_guidance_delta_is_norm_calibrated_and_video_only():
    video_n, audio_n = 96, 24
    model = _packed_av_model(video_n, audio_n)
    s = samplers.FunPackLTXAVSceneChainSampler()
    s._build_output_guidance_wrapper(model, _ConstGradVF(), strength=0.02)
    wrap = model.model_options["model_function_wrapper"]

    denoised = torch.randn(1, 1, video_n + audio_n)

    def apply_fn(inp, ts, **c):
        return denoised

    out = wrap(apply_fn, {"input": denoised, "timestep": torch.tensor([0.0]),
                          "c": {}})
    # Audio region untouched (identical, not merely close).
    assert torch.equal(out[..., video_n:], denoised[..., video_n:])
    # Video delta norm == strength * ramp * ||video||; sigma=0 -> ramp=1.
    delta = out[..., :video_n] - denoised[..., :video_n]
    expected = 0.02 * float(denoised[..., :video_n].norm())
    assert abs(float(delta.norm()) - expected) < 1e-5


def test_output_guidance_high_sigma_is_passthrough():
    model = _packed_av_model(96, 24)
    s = samplers.FunPackLTXAVSceneChainSampler()
    s._build_output_guidance_wrapper(model, _ConstGradVF(), strength=0.02)
    wrap = model.model_options["model_function_wrapper"]
    denoised = torch.randn(1, 1, 120)

    def apply_fn(inp, ts, **c):
        return denoised

    out = wrap(apply_fn, {"input": denoised, "timestep": torch.tensor([0.9]),
                          "c": {}})
    assert torch.equal(out, denoised)


# ── the sweep has to reach every module a hook can be installed on ────────────

class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn2 = torch.nn.Identity()               # bounded attention hooks here
        self.video_to_audio_attn = torch.nn.Identity() # v2a scale hooks here

    def forward(self, x):
        return x


class _Diffusion(torch.nn.Module):
    def __init__(self, n=2):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList(_Block() for _ in range(n))

    def forward(self, x):
        return x


def _patcher():
    return types.SimpleNamespace(model=types.SimpleNamespace(diffusion_model=_Diffusion()))


def test_the_sweep_reaches_hooks_installed_below_the_blocks():
    """v2a sits on video_to_audio_attn and bounded attention on attn2 — a block-only sweep
    left both to stack up run after run."""
    import ltx_enhancements as enh
    model = _patcher()
    hook = samplers._tag_funpack_hook(lambda *a: None)
    for blk in model.model.diffusion_model.transformer_blocks:
        blk.attn2.register_forward_pre_hook(hook)
        blk.video_to_audio_attn.register_forward_hook(hook)
    assert enh.count_module_hooks(model) == (4, 4)
    assert enh.strip_funpack_block_hooks(model) == 4
    assert enh.count_module_hooks(model) == (0, 0)


def test_the_sweep_leaves_hooks_that_are_not_funpacks():
    import ltx_enhancements as enh
    model = _patcher()
    model.model.diffusion_model.transformer_blocks[0].attn2.register_forward_hook(
        lambda *a: None)
    assert enh.strip_funpack_block_hooks(model) == 0
    assert enh.count_module_hooks(model) == (1, 1)
