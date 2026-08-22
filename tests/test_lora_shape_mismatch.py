"""A LoRA that matches by NAME but not by SHAPE must say so.

The case that motivated this: MiniMax H3 ships in two forms. The full one derives adaLN's
input from a `time_embedder` (time_embed_dim 2688); the pruned "curve" one replaces that
with a small precomputed basis, so every `adaln_proj.linear` is narrower on its input side.
A turbo LoRA trained on the full model matches every key of a curve-form checkpoint and
then fails to merge into 51 of them — previously visible only as generic per-key warnings
from comfy, while FunPack's own status line still read like a clean load.
"""
import sys
import types

import pytest
import torch

sys.path.insert(0, ".")


@pytest.fixture
def mm():
    import model_management
    return model_management


class _Adapter:
    """Stands in for comfy's LoRAAdapter: weights = (up[out, rank], down[rank, in], ...)."""
    def __init__(self, out_dim, in_dim, rank=8):
        self.weights = (torch.zeros(out_dim, rank), torch.zeros(rank, in_dim), 1.0, None, None)


class _Model:
    def __init__(self, shapes, use_curves=False):
        self._sd = {k: torch.zeros(*v) for k, v in shapes.items()}
        diffusion = types.SimpleNamespace(use_adaln_curves=use_curves)
        inner = types.SimpleNamespace(state_dict=lambda: self._sd, diffusion_model=diffusion)
        self.model = inner


def test_a_fitting_lora_reports_nothing(mm):
    model = _Model({"diffusion_model.blocks.0.attn.qkv.weight": (768, 768)})
    patches = {"diffusion_model.blocks.0.attn.qkv.weight": _Adapter(768, 768)}
    assert mm._mismatched_lora_keys(model, patches) == []


def test_a_narrower_input_is_caught(mm):
    """The exact curve-form shape: the weight's input side is the small basis, the LoRA's
    is the full time-embedding width."""
    model = _Model({"diffusion_model.blocks.0.adaln_proj.linear.weight": (16128, 64)})
    patches = {"diffusion_model.blocks.0.adaln_proj.linear.weight": _Adapter(16128, 2688)}
    bad = mm._mismatched_lora_keys(model, patches)
    assert len(bad) == 1
    key, got, want = bad[0]
    assert got == (16128, 2688) and want == (16128, 64)


def test_keys_the_model_does_not_have_are_not_flagged(mm):
    model = _Model({"diffusion_model.blocks.0.attn.qkv.weight": (768, 768)})
    patches = {"diffusion_model.blocks.99.nonexistent.weight": _Adapter(10, 10)}
    assert mm._mismatched_lora_keys(model, patches) == []


def test_non_lora_adapters_are_left_to_comfy(mm):
    """LoKr/LoHa/diff entries have no up/down pair to measure — guessing at them would
    produce false alarms on formats this check does not model."""
    model = _Model({"diffusion_model.blocks.0.attn.qkv.weight": (768, 768)})
    patches = {"diffusion_model.blocks.0.attn.qkv.weight": types.SimpleNamespace(weights=None)}
    assert mm._mismatched_lora_keys(model, patches) == []


def test_curve_note_names_the_cause_and_the_remedy(mm):
    model = _Model({"diffusion_model.blocks.0.adaln_proj.linear.weight": (16128, 64)},
                   use_curves=True)
    bad = [("diffusion_model.blocks.0.adaln_proj.linear.weight", (16128, 2688), (16128, 64))]
    note = mm._adaln_curve_note(model, bad)
    assert note and "curve-form" in note
    assert "cannot be projected" in note
    # The rest of the LoRA is fine, and saying so stops it reading as a failed load.
    assert "rest of the LoRA still applies" in note


def test_no_curve_note_on_a_full_width_checkpoint(mm):
    """Same mismatched key, but this checkpoint is not curve-form — so that diagnosis would
    be a wrong explanation, which is worse than a generic one."""
    model = _Model({"diffusion_model.blocks.0.adaln_proj.linear.weight": (16128, 2688)},
                   use_curves=False)
    bad = [("diffusion_model.blocks.0.adaln_proj.linear.weight", (16128, 99), (16128, 2688))]
    assert mm._adaln_curve_note(model, bad) is None


def test_no_curve_note_when_the_mismatch_is_elsewhere(mm):
    model = _Model({"diffusion_model.blocks.0.attn.qkv.weight": (768, 768)}, use_curves=True)
    bad = [("diffusion_model.blocks.0.attn.qkv.weight", (768, 512), (768, 768))]
    assert mm._adaln_curve_note(model, bad) is None


def test_a_broken_model_never_breaks_the_load(mm):
    """Diagnostics are not worth failing a generation over."""
    class Exploding:
        model = types.SimpleNamespace(
            state_dict=lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert mm._mismatched_lora_keys(Exploding(), {"k": _Adapter(1, 1)}) == []
