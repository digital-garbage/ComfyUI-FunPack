"""Making a model chunk-causal is a switch on OUR loader, and the LoRA is an ordinary LoRA.

FunPack owns the loading pipeline, so neither half of this needs a third-party node or
package. The diffusion loader RE-CLASSES an already-loaded H3 in place rather than building
it through a different class — no parameter is added, no key is renamed, nothing is copied,
so ComfyUI's memory ledger and offload accounting are untouched. The adapter then goes
through FunPack Apply LoRA Weights like any other LoRA.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h3_causal as hc
import loaders
import model_management as mm


def _fields():
    spec = loaders.FunPackDiffusionModelLoader.INPUT_TYPES()
    return {**spec.get("required", {}), **spec.get("optional", {})}


def test_the_switch_is_on_our_own_loader():
    assert "chunk_causal" in _fields()


def test_it_is_off_by_default():
    assert _fields()["chunk_causal"][1]["default"] is False


def test_the_tooltip_says_a_lora_is_still_needed():
    """Without one the chunked lane runs out of distribution, which reads as a quality problem
    with no visible cause."""
    tip = _fields()["chunk_causal"][1]["tooltip"]
    assert "LoRA" in tip and "out of distribution" in tip


def test_no_raven_package_is_named_anywhere_in_the_loader():
    """The whole point of owning it: no third-party install to discover."""
    import inspect
    src = inspect.getsource(loaders.FunPackDiffusionModelLoader)
    assert "raven_streaming" not in src and "RAVEN Model Loader" not in src


# ── the LoRA is an ordinary LoRA ────────────────────────────────────────────

def test_the_peft_dit_prefix_is_one_of_the_candidates():
    """RAVEN's keys are base_model.model.dit.<path>. Stripping only `base_model.model.` leaves
    `dit.` in front of every key and the whole file silently matches nothing."""
    assert "base_model.model.dit." in mm.LORA_KEY_PREFIXES


def test_the_longer_prefix_is_tried_before_the_shorter_one_can_win():
    """Both match; the loader scores every candidate and keeps the best, so the order only has
    to include the longer one — but it must be there at all."""
    assert "base_model.model." in mm.LORA_KEY_PREFIXES
    prefixes = list(mm.LORA_KEY_PREFIXES)
    assert prefixes.index("base_model.model.dit.") > prefixes.index("base_model.model.")


def test_stripping_that_prefix_yields_plain_module_paths():
    stripped = mm._strip_key_prefix(
        {"base_model.model.dit.blocks.0.attn.qkv_proj.lora_A.weight": 1,
         "base_model.model.dit.blocks.0.attn.qkv_proj.lora_B.weight": 2},
        "base_model.model.dit.")
    assert set(stripped) == {"blocks.0.attn.qkv_proj.lora_A.weight",
                             "blocks.0.attn.qkv_proj.lora_B.weight"}


# ── re-classing an already-loaded model ─────────────────────────────────────

class _FakeBlock:
    def __init__(self):
        self.attn = type("A", (), {})()


def test_a_model_without_blocks_declines_with_a_reason():
    fake = type("M", (), {"model": type("N", (), {"diffusion_model": type("D", (), {})()})()})()
    ok, note = hc.make_causal(fake)
    assert ok is False and "no DiT blocks" in note


def test_a_model_without_a_diffusion_model_declines():
    ok, note = hc.make_causal(type("M", (), {"model": None})())
    assert ok is False and "no diffusion model" in note


def test_it_is_idempotent(monkeypatch):
    """Loading the same model twice, or re-running the node, must not stack anything."""
    diffusion = type("D", (), {})()
    diffusion.blocks = [_FakeBlock()]
    diffusion._funpack_causal = True
    fake = type("M", (), {})()
    fake.model = type("N", (), {})()
    fake.model.diffusion_model = diffusion
    ok, note = hc.make_causal(fake)
    assert ok is True and "already" in note


def test_a_comfy_without_h3_declines_instead_of_raising(monkeypatch):
    """A ComfyUI too old for H3 must not turn one optional switch into a dead node."""
    monkeypatch.setattr(hc, "_causal_classes",
                        lambda: (_ for _ in ()).throw(ImportError("no minimax")))
    diffusion = type("D", (), {})()
    diffusion.blocks = [_FakeBlock()]
    fake = type("M", (), {})()
    fake.model = type("N", (), {})()
    fake.model.diffusion_model = diffusion
    ok, note = hc.make_causal(fake)
    assert ok is False and "no MiniMax H3" in note


# ── which node the user is actually sent to ─────────────────────────────────
#
# FunPack LoRA Loader loads AND applies, from its own file list. FunPack Apply LoRA Weights is
# the optional stack producer for prompt-specific trained strengths. Naming the second one as
# the way in sends the user to add a node they do not need.

def test_the_loader_message_names_the_lora_loader():
    tip = _fields()["chunk_causal"][1]["tooltip"]
    assert "FunPack LoRA Loader" in tip
    assert "Apply LoRA Weights" not in tip


def test_the_sampler_message_names_the_lora_loader():
    import inspect
    import h3_causal
    src = inspect.getsource(h3_causal.build_session)
    assert "FunPack LoRA Loader" in src
    assert "Apply LoRA Weights" not in src


def test_the_lora_loader_does_not_advertise_itself_as_stack_only():
    """Its description used to read as though a stack node were required."""
    import model_management
    text = model_management.FunPackLoraLoader.DESCRIPTION
    assert "only needed" in text or "own list" in text


def test_the_stack_input_is_optional():
    """If it were required, 'the loader applies them on its own' would be false."""
    import model_management
    spec = model_management.FunPackLoraLoader.INPUT_TYPES()
    assert "lora_stack" in spec.get("optional", {})
    assert "lora_stack" not in spec.get("required", {})


def test_the_file_list_is_the_required_way_in():
    import model_management
    spec = model_management.FunPackLoraLoader.INPUT_TYPES()
    assert "lora_list" in spec.get("required", {})
