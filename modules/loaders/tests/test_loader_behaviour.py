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


def test_changing_the_process_wide_fp16_flag_is_never_silent(monkeypatch):
    """torch has ONE flag for the interpreter, so this cannot mean "for this
    model". A second loader setting it differently changes the first model's
    maths, and ComfyUI's loader caching means the last value outlives its run."""
    import torch
    from core import log
    from modules.loaders import common

    class FakeMatmul:
        allow_fp16_accumulation = False

    fake = FakeMatmul()
    monkeypatch.setattr(torch.backends, "cuda", type("C", (), {"matmul": fake}))

    log._reset()
    assert common.set_fp16_accumulation(True) is True
    assert fake.allow_fp16_accumulation is True
    assert any("EVERY model in this process" in r["message"] for r in log.history())

    # No transition, nothing said.
    log._reset()
    common.set_fp16_accumulation(True)
    assert log.history() == []

    # And turning it back off is a transition too: it is another model's setting
    # being undone.
    log._reset()
    common.set_fp16_accumulation(False)
    assert log.history()


def test_a_torch_build_without_the_flag_reports_that_rather_than_pretending():
    import torch
    from modules.loaders import common
    if getattr(getattr(torch.backends, "cuda", None), "matmul", None) is None:
        assert common.set_fp16_accumulation(True) is None


# --- GGUF routing: does the WIRING actually reach gguf_support.py, not just ---
# --- whether gguf_support.py itself works in isolation -------------------------

def test_a_gguf_model_name_is_routed_through_gguf_support(monkeypatch):
    """A .gguf model file must never reach load_torch_file -- that parser
    reads a safetensors header, and a GGUF container is not one."""
    import comfy.sd
    from modules.loaders import gguf_support
    from modules.loaders.diffusion_model import nodes

    monkeypatch.setattr(gguf_support, "gguf_path", lambda folder, name: f"/fake/{name}")
    monkeypatch.setattr(gguf_support, "load_state_dict",
                        lambda path: ({"w": 1}, {"custom_operations": "ops"}, "gguf: quantized"))
    monkeypatch.setattr(comfy.sd, "load_diffusion_model_state_dict",
                        lambda sd, model_options, metadata: object() if sd == {"w": 1}
                        and model_options.get("custom_operations") == "ops" else None)

    out = nodes.FunPackDiffusionModelLoader.execute(
        model_name="model-Q4_K_M.gguf", weight_dtype="default",
        compute_dtype="default", attention="default")

    assert out.result[0] is not None
    assert "gguf: quantized" in out.result[1]


def test_a_safetensors_named_file_that_is_actually_gguf_is_still_routed(monkeypatch):
    """The CONTENT decides, not the extension -- a .gguf renamed to
    .safetensors must not reach the safetensors parser and fail with a
    decode error that names nothing useful."""
    import comfy.sd
    import folder_paths
    from modules.loaders import gguf_support
    from modules.loaders.diffusion_model import nodes

    monkeypatch.setattr(folder_paths, "get_full_path_or_raise", lambda kind, name: "/fake/path")
    monkeypatch.setattr(gguf_support, "has_gguf_magic", lambda path: True)
    monkeypatch.setattr(gguf_support, "load_state_dict",
                        lambda path: ({"w": 1}, {}, "gguf: dequantized at load"))
    monkeypatch.setattr(comfy.sd, "load_diffusion_model_state_dict",
                        lambda sd, model_options, metadata: object())

    out = nodes.FunPackDiffusionModelLoader.execute(
        model_name="model.safetensors", weight_dtype="default",
        compute_dtype="default", attention="default")

    assert "named .safetensors but is a GGUF container" in out.result[1]


def test_an_ordinary_safetensors_model_never_touches_gguf_support(monkeypatch):
    """The common case must not pay for the uncommon one: a real
    .safetensors file goes through load_torch_file, not gguf_support."""
    import comfy.sd
    import comfy.utils
    import folder_paths
    from modules.loaders import gguf_support
    from modules.loaders.diffusion_model import nodes

    monkeypatch.setattr(folder_paths, "get_full_path_or_raise", lambda kind, name: "/fake/path")
    monkeypatch.setattr(gguf_support, "has_gguf_magic", lambda path: False)
    monkeypatch.setattr(gguf_support, "load_state_dict",
                        lambda path: (_ for _ in ()).throw(AssertionError("gguf path taken")))
    monkeypatch.setattr(comfy.utils, "load_torch_file",
                        lambda path, return_metadata: ({"w": 1}, {}))
    monkeypatch.setattr(comfy.sd, "load_diffusion_model_state_dict",
                        lambda sd, model_options, metadata: object())

    nodes.FunPackDiffusionModelLoader.execute(
        model_name="model.safetensors", weight_dtype="default",
        compute_dtype="default", attention="default")


def test_a_gguf_text_encoder_is_routed_through_gguf_support(monkeypatch):
    """A .gguf encoder cannot go through load_clip -- it reads files itself
    and does not understand the container."""
    import comfy.sd
    from modules.loaders import gguf_support
    from modules.loaders.clip import nodes

    monkeypatch.setattr(gguf_support, "gguf_path", lambda folder, name: f"/fake/{name}")
    monkeypatch.setattr(gguf_support, "load_clip_state_dict",
                        lambda path: ({"w": 1}, {}, "gguf: quantized"))
    monkeypatch.setattr(comfy.sd, "load_text_encoder_state_dicts",
                        lambda state_dicts, **kw: object() if state_dicts == [{"w": 1}] else None)

    out = nodes.FunPackCLIPLoader.execute(clip_name1="encoder-Q5_K.gguf", type="ltxv")

    assert out.result[0] is not None
    assert "gguf: quantized" in out.result[1]


def test_a_gguf_encoder_alongside_a_plain_connector_loads_both(monkeypatch):
    """LTX-2.3's normal shape: a GGUF encoder plus a .safetensors connector
    in the same list. Each slot must be read the way ITS file needs, not
    however the first slot in the list happened to be."""
    import comfy.sd
    import comfy.utils
    import folder_paths
    from modules.loaders import gguf_support
    from modules.loaders.clip import nodes

    monkeypatch.setattr(gguf_support, "gguf_path", lambda folder, name: f"/fake/{name}")
    monkeypatch.setattr(gguf_support, "load_clip_state_dict",
                        lambda path: ({"gguf": 1}, {}, "gguf: quantized"))
    monkeypatch.setattr(folder_paths, "get_full_path_or_raise", lambda kind, name: f"/fake/{name}")
    monkeypatch.setattr(comfy.utils, "load_torch_file", lambda path: {"plain": 1})
    seen = {}
    monkeypatch.setattr(comfy.sd, "load_text_encoder_state_dicts",
                        lambda state_dicts, **kw: seen.update(sd=state_dicts) or object())

    nodes.FunPackCLIPLoader.execute(
        clip_name1="encoder-Q5_K.gguf", clip_name2="connector.safetensors", type="ltxv")

    assert seen["sd"] == [{"gguf": 1}, {"plain": 1}]


def test_a_gguf_encoder_that_vanished_since_being_listed_is_refused_by_name(monkeypatch):
    """gguf_path() returns None once a listed file is gone. Without a check,
    that None reaches load_clip_state_dict and fails deep inside gguf/pack
    machinery instead of naming the missing file, the way the diffusion model
    loader already refuses the identical condition."""
    from modules.loaders import gguf_support
    from modules.loaders.clip import nodes

    monkeypatch.setattr(gguf_support, "gguf_path", lambda folder, name: None)

    with pytest.raises(RuntimeError, match="encoder-Q5_K.gguf.*no longer where it was listed"):
        nodes.FunPackCLIPLoader.execute(clip_name1="encoder-Q5_K.gguf", type="ltxv")
