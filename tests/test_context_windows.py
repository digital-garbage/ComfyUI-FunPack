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
4. The NAMES are core's, not ours. The knob shipped with readable-but-wrong spellings
   ("uniform_standard"); core calls them the other way round ("standard_uniform"), so
   every schedule except "batched" raised ValueError out of the sampler and failed the
   whole render. Core's names are the choices now, the old ones are aliased, and an
   unknown one is refused with a reason naming what core does accept.
5. A keyword this core does not take is DROPPED, not fatal. latent_retain_index_list is
   absent on some builds; passing it raised TypeError, which was reported as "ComfyUI
   core too old" — so a core that fully supports windowing had the feature silently
   switched off. Install anyway and name the knob that stops working.

Install returns (remove_fn, latent_len, note) or (None, None, reason): every failure here
is a mismatch with the installed core, and reporting the WRONG one is how (5) hid.
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
    # Core's real vocabulary (comfy.context_windows ContextSchedules / ContextFuseMethods),
    # and core's real behaviour on anything else: ValueError.
    cw.CONTEXT_MAPPING = dict.fromkeys(
        ["standard_uniform", "standard_static", "looped_uniform", "batched"], object())
    cw.FUSE_MAPPING = dict.fromkeys(["pyramid", "relative", "flat", "overlap-linear"], object())

    def _sched(name):
        if name not in cw.CONTEXT_MAPPING:
            raise ValueError(f"Unknown context_schedule '{name}'.")
        return f"sched:{name}"

    def _fuse_of(name):
        if name not in cw.FUSE_MAPPING:
            raise ValueError(f"Unknown fuse_method '{name}'.")
        return f"fuse:{name}"

    cw.get_matching_context_schedule = _sched
    cw.get_matching_fuse_method = _fuse_of

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
    remove, _, reason = _install(s, model)
    assert remove is None and "LTXAV context-window support" in reason
    # Nothing installed: a refusal must not half-configure the shared model.
    assert "context_handler" not in model.model_options
    assert calls["prepare"] == 0 and calls["sampler_sample"] == 0


def test_refuses_when_the_handler_rejects_an_argument_outright(monkeypatch):
    """A TypeError we could not see coming from the signature. Still a refusal — but one
    that reports the argument mismatch instead of blaming the core version."""
    model, calls = _install_fake_core(monkeypatch, accept_latent_retain=False)
    s = samplers.FunPackLTXAVSceneChainSampler()
    remove, _, reason = _install(s, model)
    assert remove is None and "different" in reason
    assert "context_handler" not in model.model_options


def test_real_frames_convert_to_latent_frames_like_core_node(monkeypatch):
    model, _ = _install_fake_core(monkeypatch)
    s = samplers.FunPackLTXAVSceneChainSampler()
    remove, latent_len, _ = _install(s, model, length=145, overlap=40)
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
    remove, _, _ = _install(s, model, retain_first=True)
    handler = model.model_options["context_handler"]
    # i2v anchors live in the noise latent, not only in conditioning channels, so both
    # lists must be set — cond alone would drop the anchor from later windows.
    assert handler.kwargs["cond_retain_index_list"] == "0"
    assert handler.kwargs["latent_retain_index_list"] == "0"
    remove()


def test_freenoise_off_skips_the_sampler_sample_wrapper(monkeypatch):
    model, calls = _install_fake_core(monkeypatch)
    s = samplers.FunPackLTXAVSceneChainSampler()
    remove, _, _ = _install(s, model, freenoise=False)
    assert calls["prepare"] == 1
    assert calls["sampler_sample"] == 0
    remove()


def test_remove_restores_model_options_and_drops_both_wrappers(monkeypatch):
    model, calls = _install_fake_core(monkeypatch)
    s = samplers.FunPackLTXAVSceneChainSampler()
    remove, _, _ = _install(s, model)
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
    remove, _, _ = _install(s, model)
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


# ── the names are core's, not ours ──────────────────────────────────────────

def test_every_schedule_the_node_offers_actually_resolves(monkeypatch):
    """The bug in one line: the node offered spellings core has never had, so choosing any
    of them raised ValueError out of the sampler and killed the render. Every name the combo
    accepts must now reach a real core schedule — directly or through the alias map."""
    model, _ = _install_fake_core(monkeypatch)
    s = samplers.FunPackLTXAVSceneChainSampler()
    offered = s.INPUT_TYPES()["optional"]["context_window_schedule"][0]
    known = set(sys.modules["comfy.context_windows"].CONTEXT_MAPPING)
    assert known <= set(offered)  # nothing core supports is missing from the knob
    for name in offered:
        model, _ = _install_fake_core(monkeypatch)
        assert _install(s, model, schedule=name)[0] is not None, name


def test_the_legacy_names_stay_in_the_combo_list_not_only_in_the_alias_map():
    """ComfyUI validates combo values at QUEUE time, before the node runs. Dropping the old
    spellings from the list would reject a saved project outright and the alias would never
    get a chance — which is a worse failure than the one being fixed."""
    offered = samplers.FunPackLTXAVSceneChainSampler().INPUT_TYPES()[
        "optional"]["context_window_schedule"][0]
    for legacy in samplers.FunPackLTXAVSceneChainSampler._CTX_SCHEDULE_ALIASES:
        assert legacy in offered, legacy


def test_the_old_reversed_spellings_still_resolve(monkeypatch):
    """A project saved before the names were corrected must keep generating, not fail."""
    s = samplers.FunPackLTXAVSceneChainSampler()
    for old, new in (("uniform_standard", "standard_uniform"),
                     ("static_standard", "standard_static"),
                     ("uniform_looped", "looped_uniform")):
        model, _ = _install_fake_core(monkeypatch)
        remove, _, reason = _install(s, model, schedule=old)
        assert remove is not None, (old, reason)
        assert model.model_options["context_handler"].kwargs["context_schedule"] == f"sched:{new}"


def test_an_unknown_schedule_is_refused_with_what_core_accepts(monkeypatch):
    model, calls = _install_fake_core(monkeypatch)
    s = samplers.FunPackLTXAVSceneChainSampler()
    remove, _, reason = _install(s, model, schedule="nonsense")
    assert remove is None
    assert "nonsense" in reason and "standard_uniform" in reason
    # A refusal must not half-configure the shared model.
    assert "context_handler" not in model.model_options
    assert calls["prepare"] == 0


def test_an_unknown_fuse_method_is_refused_with_what_core_accepts(monkeypatch):
    model, _ = _install_fake_core(monkeypatch)
    s = samplers.FunPackLTXAVSceneChainSampler()
    remove, _, reason = _install(s, model, fuse="wobble")
    assert remove is None
    assert "wobble" in reason and "overlap-linear" in reason


# ── a keyword this core does not take ───────────────────────────────────────

class _NoLatentRetainHandler:
    """The shape of the build that reported itself as 'core too old': core's handler minus
    latent_retain_index_list, declared in the signature so it can be seen before calling."""
    def __init__(self, context_schedule, fuse_method, context_length=1, context_overlap=0,
                 context_stride=1, closed_loop=False, dim=0, freenoise=False,
                 cond_retain_index_list=(), split_conds_to_windows=False):
        self.kwargs = dict(context_schedule=context_schedule, context_length=context_length,
                           cond_retain_index_list=cond_retain_index_list)


def test_an_unsupported_keyword_is_dropped_and_windowing_still_runs(monkeypatch):
    model, calls = _install_fake_core(monkeypatch)
    sys.modules["comfy.context_windows"].IndexListContextHandler = _NoLatentRetainHandler
    s = samplers.FunPackLTXAVSceneChainSampler()
    remove, latent_len, note = _install(s, model, retain_first=True)
    assert remove is not None and latent_len == 19
    assert calls["prepare"] == 1
    # The one knob that really does stop working is named, not swallowed.
    assert "latent_retain_index_list" in note
    remove()


def test_no_note_when_retain_first_is_off(monkeypatch):
    """Nothing was lost, so nothing to report — a note here would be noise on every scene."""
    model, _ = _install_fake_core(monkeypatch)
    sys.modules["comfy.context_windows"].IndexListContextHandler = _NoLatentRetainHandler
    s = samplers.FunPackLTXAVSceneChainSampler()
    assert _install(s, model, retain_first=False)[2] is None
