"""The RAVEN LoRA is an option on FunPack's OWN diffusion loader.

FunPack owns its loading pipeline, so the causal lane must not require a third-party loader
node in the graph. `raven_lora` on FunPack Diffusion Model Loader turns an ordinary H3
checkpoint into the chunk-causal DiT with the adapter attached — the checkpoint is the same
file either way; there is no separate RAVEN model.

Why the load is DELEGATED rather than reimplemented: the adapter is an FP32 activation
residual registered on the base Linear leaves, and it has to be attached before the
ModelPatcher exists. `ModelPatcher.model_size()` memoises, so an adapter attached afterwards
is invisible to ComfyUI's memory ledger and never moved by partial CPU offload.
"""
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import loaders


def _fields():
    spec = loaders.FunPackDiffusionModelLoader.INPUT_TYPES()
    return {**spec.get("required", {}), **spec.get("optional", {})}


def test_the_option_exists_on_our_own_loader():
    assert "raven_lora" in _fields()


def test_it_defaults_to_off():
    assert _fields()["raven_lora"][1]["default"] == "None"


def test_none_is_the_first_choice():
    """A file list that opens on a LoRA would arm the causal lane by accident."""
    assert _fields()["raven_lora"][0][0] == "None"


def test_the_tooltip_says_it_is_a_lora_not_a_model():
    """There is no 'RAVEN H3 model' — saying so would send the user hunting for a file that
    does not exist."""
    tip = _fields()["raven_lora"][1]["tooltip"]
    assert "not a separate RAVEN model file" in tip or "no separate RAVEN model" in tip


def test_the_tooltip_says_strength_is_fixed():
    assert "1.0" in _fields()["raven_lora"][1]["tooltip"]


def test_the_choices_come_from_the_loras_folder(monkeypatch):
    monkeypatch.setattr(loaders.folder_paths, "get_filename_list",
                        lambda kind: ["a.safetensors"] if kind == "loras" else [])
    assert loaders.raven_lora_choices() == ["None", "a.safetensors"]


# ── the delegating load ─────────────────────────────────────────────────────

def test_a_missing_raven_package_refuses_instead_of_loading_plain(monkeypatch):
    """Returning a model WITHOUT the adapter would leave the causal lane reading an attention
    pattern nothing was trained for — a quality problem with no visible cause."""
    import raven_causal
    monkeypatch.setattr(raven_causal, "locate_raven",
                        lambda: (None, "the pack is not installed."))
    with pytest.raises(RuntimeError, match="not installed"):
        loaders._load_with_raven_lora("h3.safetensors", "raven.safetensors", "default", {})


def test_the_refusal_says_how_to_get_a_normal_load(monkeypatch):
    import raven_causal
    monkeypatch.setattr(raven_causal, "locate_raven", lambda: (None, "missing."))
    with pytest.raises(RuntimeError, match="raven_lora to None"):
        loaders._load_with_raven_lora("h3.safetensors", "raven.safetensors", "default", {})


def test_the_load_is_delegated_with_the_causal_class(monkeypatch):
    """The class is what makes the DiT chunk-causal; loading the adapter without it would
    attach weights nothing reads."""
    import raven_causal
    seen = {}

    causal_cls = type("RavenCausalMiniMaxH3Model", (), {})
    fake_loader = types.SimpleNamespace(
        load_raven_diffusion_model=lambda unet, lora, **kw: seen.update(
            {"unet": unet, "lora": lora, **kw}) or "the-model")
    parent = types.ModuleType("raven_streaming")
    parent.loader = fake_loader
    causal_mod = types.ModuleType("raven_streaming.causal_model")
    causal_mod.RavenCausalMiniMaxH3Model = causal_cls
    parent.causal_model = causal_mod
    monkeypatch.setitem(sys.modules, "raven_streaming", parent)
    monkeypatch.setitem(sys.modules, "raven_streaming.loader", fake_loader)
    monkeypatch.setitem(sys.modules, "raven_streaming.causal_model", causal_mod)
    monkeypatch.setattr(raven_causal, "locate_raven", lambda: (parent, ""))

    model, note = loaders._load_with_raven_lora(
        "h3.safetensors", "raven.safetensors", "fp8_e4m3fn", {"opt": 1})
    assert model == "the-model"
    assert seen["unet"] == "h3.safetensors" and seen["lora"] == "raven.safetensors"
    assert seen["unet_model_cls"] is causal_cls
    assert seen["weight_dtype"] == "fp8_e4m3fn"
    assert seen["model_options"] == {"opt": 1}
    assert "strength 1.0" in note


def test_the_note_does_not_call_it_a_separate_model(monkeypatch):
    import raven_causal
    causal_mod = types.ModuleType("raven_streaming.causal_model")
    causal_mod.RavenCausalMiniMaxH3Model = type("C", (), {})
    fake_loader = types.SimpleNamespace(load_raven_diffusion_model=lambda *a, **k: "m")
    parent = types.ModuleType("raven_streaming")
    parent.loader, parent.causal_model = fake_loader, causal_mod
    monkeypatch.setitem(sys.modules, "raven_streaming", parent)
    monkeypatch.setitem(sys.modules, "raven_streaming.loader", fake_loader)
    monkeypatch.setitem(sys.modules, "raven_streaming.causal_model", causal_mod)
    monkeypatch.setattr(raven_causal, "locate_raven", lambda: (parent, ""))
    _model, note = loaders._load_with_raven_lora("h3.safetensors", "r.safetensors", "default", {})
    assert "not a separate model" in note


def test_the_sampler_points_at_our_loader_not_a_third_party_node():
    """We own the loading pipeline; telling the user to add someone else's loader node would
    be wrong as well as unnecessary."""
    import inspect
    import raven_causal
    src = inspect.getsource(raven_causal.build_session)
    assert "FunPack Diffusion Model Loader" in src
    assert "RAVEN Model Loader" not in src
