"""Context-window install/remove contract on the Scene Chain sampler.

The mechanism itself is ComfyUI core's (comfy.context_windows) — nothing here tests
core's windowing math. What IS ours, and what these cover:

1. The capability gate. Core's handler class predates LTXAV support; only builds that
   also have BaseModel.map_context_window_to_modalities can unpack the packed AV latent
   and re-slice guides/audio per window. On an older core we must REFUSE, not window a
   packed tensor blindly (the DynamicConditioning failure mode).
2. Real-frame -> latent-frame conversion matches core's own LTXVContextWindows node.
3. Install/remove leaves the shared model exactly as found, including on an interrupt
   mid-scene — the same guarantee _strip_funpack_scene_wrappers gives the function
   wrappers, but for patcher wrappers and the context_handler model_option.
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


class _Handler:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _install_fake_core(monkeypatch, ltxav_support=True, accept_latent_retain=True):
    """Stand in for comfy.context_windows + comfy.patcher_extension.

    Records what the sampler asks core to do, so the test asserts on the request
    rather than on core's behaviour.
    """
    calls = {"prepare": 0, "sampler_sample": 0, "removed": []}

    cw = types.ModuleType("comfy.context_windows")

    def _handler(**kwargs):
        if not accept_latent_retain and "latent_retain_index_list" in kwargs:
            raise TypeError("unexpected keyword argument 'latent_retain_index_list'")
        return _Handler(**kwargs)

    cw.IndexListContextHandler = _handler
    cw.get_matching_context_schedule = lambda s: f"sched:{s}"
    cw.get_matching_fuse_method = lambda f: f"fuse:{f}"

    def _prep(model):
        calls["prepare"] += 1

    def _samp(model):
        calls["sampler_sample"] += 1

    cw.create_prepare_sampling_wrapper = _prep
    cw.create_sampler_sample_wrapper = _samp

    pe = types.ModuleType("comfy.patcher_extension")
    pe.WrappersMP = types.SimpleNamespace(
        PREPARE_SAMPLING="prepare_sampling", SAMPLER_SAMPLE="sampler_sample")

    monkeypatch.setitem(sys.modules, "comfy.context_windows", cw)
    monkeypatch.setitem(sys.modules, "comfy.patcher_extension", pe)

    inner = types.SimpleNamespace()
    if ltxav_support:
        inner.map_context_window_to_modalities = lambda *a, **k: None

    model = types.SimpleNamespace(
        model_options={},
        model=inner,
        remove_wrappers_with_key=lambda t, k: calls["removed"].append((t, k)),
    )
    return model, calls


def _install(sampler, model, **overrides):
    kwargs = dict(length=145, overlap=40, schedule="uniform_standard",
                  fuse="pyramid", freenoise=True, retain_first=False)
    kwargs.update(overrides)
    return sampler._install_context_windows(model, **kwargs)


def test_refuses_on_core_without_ltxav_window_support(monkeypatch):
    model, calls = _install_fake_core(monkeypatch, ltxav_support=False)
    s = samplers.FunPackLTXAVSceneChainSampler()
    assert _install(s, model) is None
    # Nothing installed: a refusal must not half-configure the shared model.
    assert "context_handler" not in model.model_options
    assert calls["prepare"] == 0 and calls["sampler_sample"] == 0


def test_refuses_when_handler_signature_drifted(monkeypatch):
    model, calls = _install_fake_core(monkeypatch, accept_latent_retain=False)
    s = samplers.FunPackLTXAVSceneChainSampler()
    assert _install(s, model) is None
    assert "context_handler" not in model.model_options


def test_real_frames_convert_to_latent_frames_like_core_node(monkeypatch):
    model, _ = _install_fake_core(monkeypatch)
    s = samplers.FunPackLTXAVSceneChainSampler()
    remove, latent_len = _install(s, model, length=145, overlap=40)
    # Core's LTXVContextWindows: ((145 - 1) // 8) + 1 = 19, 40 // 8 = 5.
    assert latent_len == 19
    handler = model.model_options["context_handler"]
    assert handler.kwargs["context_length"] == 19
    assert handler.kwargs["context_overlap"] == 5
    assert handler.kwargs["dim"] == 2
    remove()


def test_retain_first_sets_both_cond_and_latent_retain(monkeypatch):
    model, _ = _install_fake_core(monkeypatch)
    s = samplers.FunPackLTXAVSceneChainSampler()
    remove, _ = _install(s, model, retain_first=True)
    handler = model.model_options["context_handler"]
    # i2v anchors live in the noise latent, not only in conditioning channels, so both
    # lists must be set — cond alone would drop the anchor from later windows.
    assert handler.kwargs["cond_retain_index_list"] == "0"
    assert handler.kwargs["latent_retain_index_list"] == "0"
    remove()


def test_freenoise_off_skips_the_sampler_sample_wrapper(monkeypatch):
    model, calls = _install_fake_core(monkeypatch)
    s = samplers.FunPackLTXAVSceneChainSampler()
    remove, _ = _install(s, model, freenoise=False)
    assert calls["prepare"] == 1
    assert calls["sampler_sample"] == 0
    remove()


def test_remove_restores_model_options_and_drops_both_wrappers(monkeypatch):
    model, calls = _install_fake_core(monkeypatch)
    s = samplers.FunPackLTXAVSceneChainSampler()
    remove, _ = _install(s, model)
    assert "context_handler" in model.model_options
    remove()
    assert "context_handler" not in model.model_options
    assert calls["removed"] == [
        ("prepare_sampling", "ContextWindows_prepare_sampling"),
        ("sampler_sample", "ContextWindows_sampler_sample"),
    ]


def test_remove_restores_a_pre_existing_foreign_handler(monkeypatch):
    """A user-placed LTXVContextWindows node upstream must survive our scene."""
    model, _ = _install_fake_core(monkeypatch)
    foreign = object()
    model.model_options["context_handler"] = foreign
    s = samplers.FunPackLTXAVSceneChainSampler()
    remove, _ = _install(s, model)
    assert model.model_options["context_handler"] is not foreign
    remove()
    assert model.model_options["context_handler"] is foreign


def test_scene_latent_frame_count_reads_the_video_stream():
    s = samplers.FunPackLTXAVSceneChainSampler()
    chunk = {"samples": torch.zeros(1, 128, 23, 8, 8)}
    assert s._context_scene_latent_frames(chunk) == 23
    # Unreadable latent reports None rather than raising: this value is only ever used
    # for the scene report, never for control flow.
    assert s._context_scene_latent_frames({"samples": None}) is None
