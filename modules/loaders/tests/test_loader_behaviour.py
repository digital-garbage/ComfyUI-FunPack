"""What the loaders actually do with what they are given.

Real weights cannot be loaded here (this machine's model files are zero-byte
placeholders), so the file-reading boundary is stubbed and everything on OUR side
of it runs for real: slot ordering, empty slots, no-op strengths, and where the
attention override is installed.
"""

import pytest


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Every test here imports a node module, which imports comfy."""


def test_empty_encoder_slots_are_skipped_not_looked_up(monkeypatch):
    """An empty optional slot means "no file here". Passing "" through to
    get_full_path_or_raise turns a normal one-encoder setup into a hard error."""
    import comfy.sd
    import folder_paths
    from modules.loaders.clip import nodes

    asked = []
    monkeypatch.setattr(folder_paths, "get_full_path_or_raise",
                        lambda kind, name: asked.append(name) or f"/fake/{name}")
    monkeypatch.setattr(folder_paths, "get_folder_paths", lambda kind: [])
    monkeypatch.setattr(comfy.sd, "load_clip", lambda **kw: object())

    nodes.FunPackCLIPLoader.execute(
        clip_name1="encoder.safetensors", type="ltxv",
        clip_name2="", clip_name3=None, clip_name4="")

    assert asked == ["encoder.safetensors"]


def test_encoder_slots_load_in_slot_order(monkeypatch):
    """load_clip cares about order: the encoder comes before its connector.
    Reading **kwargs in arrival order would make that arbitrary."""
    import comfy.sd
    import folder_paths
    from modules.loaders.clip import nodes

    seen = {}
    monkeypatch.setattr(folder_paths, "get_full_path_or_raise", lambda kind, name: f"/fake/{name}")
    monkeypatch.setattr(folder_paths, "get_folder_paths", lambda kind: [])
    monkeypatch.setattr(comfy.sd, "load_clip",
                        lambda **kw: seen.update(kw) or object())

    # Deliberately out of order in the call, which is how kwargs can arrive.
    nodes.FunPackCLIPLoader.execute(
        clip_name3="third.safetensors", clip_name2="second.safetensors",
        clip_name1="first.safetensors", type="ltxv")

    assert seen["ckpt_paths"] == ["/fake/first.safetensors",
                                  "/fake/second.safetensors",
                                  "/fake/third.safetensors"]


def test_an_unrecognised_encoder_fails_here_not_at_encode_time(monkeypatch):
    """comfy returns None rather than raising, and a None CLIP fails much later,
    a long way from the wrong file that caused it."""
    import comfy.sd
    import folder_paths
    from modules.loaders.clip import nodes

    monkeypatch.setattr(folder_paths, "get_full_path_or_raise", lambda kind, name: f"/fake/{name}")
    monkeypatch.setattr(folder_paths, "get_folder_paths", lambda kind: [])
    monkeypatch.setattr(comfy.sd, "load_clip", lambda **kw: None)

    with pytest.raises(RuntimeError, match="Could not load a text encoder"):
        nodes.FunPackCLIPLoader.execute(clip_name1="wrong.safetensors", type="ltxv")


def test_no_encoder_at_all_is_refused(monkeypatch):
    from modules.loaders.clip import nodes
    with pytest.raises(RuntimeError, match="at least one"):
        nodes.FunPackCLIPLoader.execute(clip_name1="", type="ltxv")


def test_a_lora_at_zero_strength_passes_the_originals_through(monkeypatch):
    """Not clones. A clone here silently discards patches an earlier loader in
    the chain applied to the same objects."""
    import comfy.sd
    from modules.loaders.lora import nodes

    def _explode(*a, **k):
        raise AssertionError("a zero-strength LoRA still read the file")

    monkeypatch.setattr(comfy.sd, "load_lora_for_models", _explode)

    model, clip = object(), object()
    out = nodes.FunPackLoraLoader.execute(
        model=model, lora_name="x.safetensors", strength_model=0.0,
        clip=clip, strength_clip=0.0)

    assert out.result[0] is model and out.result[1] is clip


def test_a_model_only_lora_still_applies_at_zero_clip_strength(monkeypatch):
    """clip is unwired, so strength_clip is meaningless -- but strength_model is
    not, and skipping the whole thing would silently drop the LoRA."""
    import comfy.sd
    import comfy.utils
    import folder_paths
    from modules.loaders.lora import nodes

    called = {}
    monkeypatch.setattr(folder_paths, "get_full_path_or_raise", lambda kind, name: f"/fake/{name}")
    monkeypatch.setattr(comfy.utils, "load_torch_file", lambda p, **kw: {"fake": 1})
    monkeypatch.setattr(comfy.sd, "load_lora_for_models",
                        lambda m, c, l, sm, sc: called.update(sm=sm, sc=sc) or ("patched", None))

    out = nodes.FunPackLoraLoader.execute(
        model=object(), lora_name="x.safetensors", strength_model=0.8, clip=None)

    assert called["sm"] == 0.8
    assert out.result[0] == "patched"


def test_the_attention_override_calls_the_unwrapped_backend(monkeypatch):
    """ComfyUI wraps each backend with wrap_attn and hands the override the
    ORIGINAL function. Calling the wrapped one re-enters that machinery."""
    from comfy.ldm.modules import attention as attn
    from modules.loaders import common

    calls = []

    def wrapped(*args, **kwargs):
        calls.append("wrapped")

    def unwrapped(*args, **kwargs):
        calls.append("unwrapped")
        return "result"

    wrapped.__wrapped__ = unwrapped
    monkeypatch.setattr(attn, "get_attention_function", lambda name, default: wrapped)

    override = common.attention_override("anything")
    assert override(wrapped, "q", "k", "v") == "result"
    assert calls == ["unwrapped"], "the override re-entered wrap_attn"


def test_default_attention_installs_nothing(monkeypatch):
    """'default' must leave ComfyUI's own launch-flag choice alone."""
    from modules.loaders import common
    assert common.attention_override("default") is None
    assert common.attention_override("") is None
    assert common.attention_override(None) is None


def test_an_unavailable_backend_does_not_silently_become_a_broken_override(monkeypatch):
    from comfy.ldm.modules import attention as attn
    from modules.loaders import common
    monkeypatch.setattr(attn, "get_attention_function", lambda name, default: None)
    assert common.attention_override("sage") is None


def test_an_unknown_encoder_family_is_refused_not_silently_downgraded(monkeypatch):
    """Falling back to STABLE_DIFFUSION loads against the wrong family and
    reports success. A family mismatch here reads as an unrelated fault."""
    from modules.loaders.clip import nodes
    with pytest.raises(RuntimeError, match="no encoder family"):
        nodes.FunPackCLIPLoader.execute(clip_name1="e.safetensors", type="not_a_family")


def test_changing_the_process_wide_fp16_flag_is_never_silent(monkeypatch, caplog):
    """torch has ONE flag for the interpreter, so this cannot mean "for this
    model". A second loader setting it differently changes the first model's
    maths, and ComfyUI's loader caching means the last value outlives its run."""
    import logging
    import torch
    from modules.loaders import common

    class FakeMatmul:
        allow_fp16_accumulation = False

    fake = FakeMatmul()
    monkeypatch.setattr(torch.backends, "cuda", type("C", (), {"matmul": fake}))

    with caplog.at_level(logging.WARNING):
        assert common.set_fp16_accumulation(True) is True
    assert fake.allow_fp16_accumulation is True
    assert any("EVERY model in this process" in r.getMessage() for r in caplog.records)

    # No transition, nothing said.
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        common.set_fp16_accumulation(True)
    assert not caplog.records

    # And turning it back off is a transition too: it is another model's setting
    # being undone.
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        common.set_fp16_accumulation(False)
    assert caplog.records


def test_a_torch_build_without_the_flag_reports_that_rather_than_pretending():
    import torch
    from modules.loaders import common
    if getattr(getattr(torch.backends, "cuda", None), "matmul", None) is None:
        assert common.set_fp16_accumulation(True) is None
