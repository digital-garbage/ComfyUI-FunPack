"""FunPack's own loaders: dtype/attention plumbing, list inputs, LoRA format matching."""
import json
import sys
import types

import pytest

import _comfy_stubs
from _comfy_stubs import install_module

_comfy_stubs.install_all()


class _CLIPType:
    LTXV = "LTXV"
    STABLE_DIFFUSION = "STABLE_DIFFUSION"

    def __iter__(self):
        return iter([])


install_module("comfy.sd", CLIPType=_CLIPType, VAE=object,
               load_clip=lambda **kw: kw, load_diffusion_model_state_dict=lambda *a, **k: None)
install_module("comfy.lora_convert", convert_lora=lambda sd: sd)


def _fake_load_lora(lora, to_load, log_missing=True):
    """Only the shape that matters here: a base key matches when its lora_up/down exist."""
    return {to_load[x]: ("lora", x) for x in to_load if f"{x}.lora_up.weight" in lora}


install_module("comfy.lora", load_lora=_fake_load_lora,
               model_lora_keys_unet=lambda model, key_map={}: dict(key_map),
               model_lora_keys_clip=lambda model, key_map={}: dict(key_map))

import loaders  # noqa: E402
import model_management as mm  # noqa: E402
import widgets  # noqa: E402

FIELDS = [
    widgets.field("lora", "combo", choices=["None", "a.safetensors"], default="None"),
    widgets.field("type", "combo", choices=["general", "style"], default="general"),
    widgets.field("strength", "float", default=1.0),
    widgets.field("on", "boolean", default=True),
]


# ── list inputs ───────────────────────────────────────────────────────────────

def test_parse_rows_reads_the_json_string_a_widget_holds():
    rows = widgets.parse_rows('[{"lora": "a.safetensors", "strength": "0.5"}]', FIELDS, key="lora")
    assert rows == [{"index": 0, "lora": "a.safetensors", "type": "general",
                     "strength": 0.5, "on": True}]


def test_parse_rows_drops_unset_and_switched_off_rows_but_keeps_their_index():
    rows = widgets.parse_rows(
        [{"lora": "None"}, {"lora": "a.safetensors", "on": False}, {"lora": "a.safetensors"}],
        FIELDS, key="lora")
    assert [r["index"] for r in rows] == [2]


def test_parse_rows_survives_a_value_that_is_not_a_list():
    assert widgets.parse_rows("not json", FIELDS) == []
    assert widgets.parse_rows(None, FIELDS) == []
    assert widgets.parse_rows({"lora": "a.safetensors"}, FIELDS, key="lora")[0]["lora"] == "a.safetensors"


def test_list_widget_declares_its_row_shape_for_both_frontends():
    _, opts = widgets.list_widget("LoRA", FIELDS, add_label="+ Add LoRA")
    assert json.loads(opts["default"]) == []
    assert opts["funpack_list"]["add_label"] == "+ Add LoRA"
    assert [f["name"] for f in opts["funpack_list"]["fields"]] == ["lora", "type", "strength", "on"]


# ── diffusion model loader ────────────────────────────────────────────────────

def test_weight_dtype_default_leaves_the_file_alone():
    assert loaders.weight_model_options("default") == {}


def test_fp8_fast_asks_for_fp8_matmuls_as_well_as_fp8_storage():
    opts = loaders.weight_model_options("fp8_e4m3fn_fast")
    assert opts["fp8_optimizations"] is True
    assert opts["dtype"] is loaders.dtype_of("fp8_e4m3fn")


def test_attention_choices_only_offers_backends_this_machine_registered(monkeypatch):
    attention = install_module("comfy.ldm.modules.attention")
    monkeypatch.setattr(attention, "REGISTERED_ATTENTION_FUNCTIONS", {"sage": lambda: None,
                                                                     "pytorch": lambda: None},
                        raising=False)
    assert loaders.attention_choices() == ["default", "pytorch", "sage"]


def test_attention_override_is_none_for_default_so_the_launch_flag_wins():
    assert loaders.attention_override("default") is None
    assert loaders.attention_override("") is None


def test_attention_override_calls_the_backend_unwrapped(monkeypatch):
    attention = install_module("comfy.ldm.modules.attention")
    calls = []

    def inner(*args, **kwargs):
        calls.append(args)
        return "sage-out"

    wrapped = types.SimpleNamespace(__wrapped__=inner)
    monkeypatch.setattr(attention, "get_attention_function",
                        lambda name, default=...: wrapped if name == "sage" else default,
                        raising=False)

    override = loaders.attention_override("sage")
    assert override(lambda *a, **k: "core-out", "q", "k", "v", 8) == "sage-out"
    assert calls == [("q", "k", "v", 8)]


def test_attention_override_declines_a_backend_that_is_not_installed(monkeypatch):
    attention = install_module("comfy.ldm.modules.attention")
    monkeypatch.setattr(attention, "get_attention_function", lambda name, default=...: default,
                        raising=False)
    assert loaders.attention_override("sage3") is None


# ── LoRA: any format ──────────────────────────────────────────────────────────

class _FakeModel:
    def __init__(self, keys):
        self.model = types.SimpleNamespace(keys=keys)


@pytest.fixture
def ltx_model(monkeypatch):
    """A model whose key map names the bare LTX form comfy produces."""
    key_map = {"transformer_blocks.0.attn1.to_q": "diffusion_model.transformer_blocks.0.attn1.to_q.weight"}
    monkeypatch.setattr(sys.modules["comfy.lora"], "model_lora_keys_unet",
                        lambda model, km={}: dict(key_map))
    return _FakeModel(key_map)


def _lora(*base_keys):
    out = {}
    for base in base_keys:
        out[f"{base}.lora_up.weight"] = 1
        out[f"{base}.lora_down.weight"] = 1
    return out


def test_a_lora_already_in_the_models_own_naming_loads_as_is(ltx_model):
    patches, note = mm.resolve_lora_patches(ltx_model, _lora("transformer_blocks.0.attn1.to_q"))
    assert len(patches) == 1
    assert note == "keys=1 fmt=as-is"


def test_a_diffusers_wrapped_lora_loads_once_the_wrapper_is_stripped(ltx_model):
    patches, note = mm.resolve_lora_patches(
        ltx_model, _lora("transformer.transformer_blocks.0.attn1.to_q"))
    assert len(patches) == 1
    assert note == "keys=1 fmt=stripped transformer."


def test_a_peft_wrapped_lora_loads_too(ltx_model):
    patches, _ = mm.resolve_lora_patches(
        ltx_model, _lora("base_model.model.transformer_blocks.0.attn1.to_q"))
    assert len(patches) == 1


def test_stripping_a_wrapper_keeps_the_keys_that_were_already_bare(ltx_model):
    # A half-converted file: one key bare, one still wrapped. Stripping must add the wrapped
    # one without losing the bare one — that is why the variant is SCORED, not just tried.
    ltx_model.model.keys["transformer_blocks.9.attn1.to_q"] = "diffusion_model.x.weight"
    patches, note = mm.resolve_lora_patches(ltx_model, _lora(
        "transformer_blocks.9.attn1.to_q", "transformer.transformer_blocks.0.attn1.to_q"))
    assert len(patches) == 2
    assert note == "keys=2 fmt=stripped transformer."


def test_a_lora_for_a_different_model_family_says_so_instead_of_silently_doing_nothing(ltx_model):
    patches, note = mm.resolve_lora_patches(ltx_model, _lora("double_blocks.0.img_attn.qkv"))
    assert patches == {}
    assert note == "MATCHED NOTHING"


# ── LoRA loader without a stack ───────────────────────────────────────────────

def test_the_lora_loader_no_longer_needs_a_second_node_wired_into_it():
    spec = mm.FunPackLoraLoader.INPUT_TYPES()
    assert "lora_list" in spec["required"]
    assert "lora_stack" in spec["optional"]
    assert "funpack_list" in spec["required"]["lora_list"][1]


def test_a_list_becomes_the_same_stack_shape_apply_lora_weights_emits():
    stack = mm.stack_from_lora_list(
        '[{"lora": "a.safetensors", "type": "style", "strength": 0.8}, {"lora": "None"}]')
    assert [e["name"] for e in stack["loras"]] == ["a.safetensors"]
    entry = stack["loras"][0]
    assert entry["type"] == "style"
    assert entry["model_weight"] == entry["base_model_weight"] == 0.8
    assert entry["id"] == mm.lora_state_id("a.safetensors", "style")


def test_an_empty_lora_loader_is_a_wire_not_an_error():
    """It is seeded into every new pipeline, so its resting state has to be a working one:
    the model goes straight through and the log says so."""
    model, clip, stack, status = mm.FunPackLoraLoader().load_loras(
        model="MODEL-SENTINEL", lora_list="[]", clip="CLIP-SENTINEL")
    assert model == "MODEL-SENTINEL" and clip == "CLIP-SENTINEL"
    assert stack["loras"] == []
    assert status == "FunPack LoRA Loader | No active LoRAs"


def test_the_empty_lora_list_is_declared_a_working_state():
    """The Models panel flags an empty list as something to fix — except this one."""
    spec = mm.FunPackLoraLoader.INPUT_TYPES()
    assert spec["required"]["lora_list"][1]["funpack_list"]["allow_empty"] is True


def test_the_lora_list_is_a_picker_not_a_table_of_dropdowns():
    spec = mm.FunPackLoraLoader.INPUT_TYPES()
    lst = spec["required"]["lora_list"][1]["funpack_list"]
    assert lst["picker"] == "lora"                       # rows are picked from installed files
    assert spec["optional"]["lora_stack"][1]["advanced"] is True


def test_sla_is_a_toggle_not_a_rival_backend():
    """It layers ABOVE a backend — sparse for H3's packed sequence, the chosen backend for
    everything else — so listing it as a choice would force an either/or that is not real."""
    import sla_attention
    assert sla_attention.SLA_NAME not in loaders.attention_choices()
    spec = loaders.FunPackDiffusionModelLoader.INPUT_TYPES()
    assert spec["optional"]["sla"][1]["default"] is False


# ── changing an inference setting must not re-read the checkpoint ─────────────
# Attention and SLA live on the loader node, so ComfyUI invalidates the whole node when one
# changes and the file is read again — a minute of dequantizing a GGUF to alter a sparsity
# ratio. Only the file and the dtypes decide the weights.


class _CacheModel:
    def __init__(self):
        self.model_options = {}
        self.clones = 0

    def clone(self):
        c = _CacheModel()
        c.clones = self.clones + 1
        return c


def _key_env(tmp_path, monkeypatch):
    import loaders
    f = tmp_path / "m.safetensors"
    f.write_bytes(b"x" * 16)
    monkeypatch.setattr(loaders.folder_paths, "get_full_path", lambda folder, n: str(f),
                        raising=False)
    loaders._WEIGHT_CACHE.update({"key": None, "model": None})
    return loaders, f


def test_the_key_ignores_attention_and_sla(tmp_path, monkeypatch):
    loaders, _ = _key_env(tmp_path, monkeypatch)
    a = loaders._weight_cache_key("m.safetensors", "default", "bf16")
    b = loaders._weight_cache_key("m.safetensors", "default", "bf16")
    assert a == b and a is not None


def test_a_different_dtype_is_a_different_key(tmp_path, monkeypatch):
    loaders, _ = _key_env(tmp_path, monkeypatch)
    assert (loaders._weight_cache_key("m.safetensors", "default", "bf16")
            != loaders._weight_cache_key("m.safetensors", "fp8_e4m3fn", "bf16"))


def test_a_replaced_file_is_re_read(tmp_path, monkeypatch):
    """Same name, different bytes: size and mtime are in the key so the cache misses."""
    loaders, f = _key_env(tmp_path, monkeypatch)
    before = loaders._weight_cache_key("m.safetensors", "default", "bf16")
    f.write_bytes(b"y" * 32)
    assert loaders._weight_cache_key("m.safetensors", "default", "bf16") != before


def test_an_unreadable_file_never_matches(tmp_path, monkeypatch):
    """Failing to notice a changed file is worse than an occasional extra read."""
    import loaders
    monkeypatch.setattr(loaders.folder_paths, "get_full_path", lambda folder, n: None,
                        raising=False)
    assert loaders._weight_cache_key("gone.safetensors", "default", "bf16") is None


def test_the_cached_model_is_cloned_never_handed_out(tmp_path, monkeypatch):
    """The cached object must not collect the model_options of every run that reused it."""
    loaders, _ = _key_env(tmp_path, monkeypatch)
    base = _CacheModel()
    key = loaders._weight_cache_key("m.safetensors", "default", "bf16")
    loaders._WEIGHT_CACHE.update({"key": key, "model": base})
    node = loaders.FunPackDiffusionModelLoader()
    out, _status = node._finish(loaders._WEIGHT_CACHE["model"].clone(), [], "default",
                                False, 0.9, "64", 8192, 0, True)
    out.model_options.setdefault("transformer_options", {})["marker"] = 1
    assert base.model_options == {}      # untouched
    assert out.clones == 1
