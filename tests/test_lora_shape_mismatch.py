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


# --- dropping, not just reporting ------------------------------------------------------
# Reporting a mismatch and then handing it to comfy anyway means comfy attempts it: it
# materialises the full lora_A @ lora_B delta and only then discovers it cannot be reshaped
# into the weight. On a curve-form H3 checkpoint that is a 96768x2688 tensor per block, 51
# times, while dynamic VRAM staging is streaming the model in.


def _patched_resolve(mm, monkeypatch, model, patches):
    monkeypatch.setattr(mm.comfy.lora, "load_lora", lambda *a, **kw: dict(patches),
                        raising=False)
    monkeypatch.setattr(mm.comfy.lora, "model_lora_keys_unet", lambda *a, **kw: {},
                        raising=False)
    monkeypatch.setattr(mm.comfy.lora, "model_lora_keys_clip", lambda *a, **kw: {},
                        raising=False)
    monkeypatch.setattr(mm.comfy.lora_convert, "convert_lora", lambda sd: sd, raising=False)
    return mm.resolve_lora_patches(model, {"x": torch.zeros(1)})


def test_mismatched_patches_are_removed_from_what_comfy_gets(mm, monkeypatch):
    model = _Model({"diffusion_model.blocks.0.adaln_proj.linear.weight": (64, 8),
                    "diffusion_model.blocks.0.attn.qkv_proj.weight": (64, 64)},
                   use_curves=True)
    patches = {
        "diffusion_model.blocks.0.adaln_proj.linear.weight": _Adapter(64, 2688),  # cannot fit
        "diffusion_model.blocks.0.attn.qkv_proj.weight": _Adapter(64, 64),        # fits
    }
    out, note = _patched_resolve(mm, monkeypatch, model, patches)
    assert "diffusion_model.blocks.0.adaln_proj.linear.weight" not in out
    assert "diffusion_model.blocks.0.attn.qkv_proj.weight" in out   # the rest still applies
    assert "DROPPED" in note


def test_a_fitting_lora_is_handed_over_whole(mm, monkeypatch):
    """Dropping is driven by the shape check — not applied to everything on the way past."""
    model = _Model({"diffusion_model.blocks.0.attn.qkv_proj.weight": (64, 64)})
    patches = {"diffusion_model.blocks.0.attn.qkv_proj.weight": _Adapter(64, 64)}
    out, note = _patched_resolve(mm, monkeypatch, model, patches)
    assert len(out) == 1 and "DROPPED" not in note


class _MidAdapter(_Adapter):
    """A locon entry: comfy rebuilds mat2 from the mid weight, so the plain pair is not what
    the merge will require."""
    def __init__(self, out_dim, in_dim, rank=8):
        super().__init__(out_dim, in_dim, rank)
        self.weights = (self.weights[0], self.weights[1], 1.0, torch.zeros(rank, rank, 1, 1),
                        None, None)


class _ReshapeAdapter(_Adapter):
    """A `reshape` entry: comfy PADS the target weight before merging, so the weight in the
    model is not the shape the delta has to fit."""
    def __init__(self, out_dim, in_dim, target, rank=8):
        super().__init__(out_dim, in_dim, rank)
        self.weights = (self.weights[0], self.weights[1], 1.0, None, None, target)


def test_a_locon_mid_entry_is_left_to_comfy(mm):
    """Dropping it here would lose an adapter that merges correctly — the check has no model
    of what mat2 becomes, so it must abstain rather than guess."""
    model = _Model({"w": (768, 768)})
    assert mm._mismatched_lora_keys(model, {"w": _MidAdapter(768, 512)}) == []


def test_a_padded_target_is_left_to_comfy(mm):
    model = _Model({"w": (768, 768)})
    assert mm._mismatched_lora_keys(model, {"w": _ReshapeAdapter(1024, 768, (1024, 768))}) == []


def test_a_delta_that_reshapes_into_the_weight_is_kept(mm):
    """comfy merges with mm(...).reshape(weight.shape), so the element count is the real
    constraint. Requiring the two dimensions to match was stricter than the merge itself."""
    model = _Model({"w": (768, 768)})           # 589824 elements
    assert mm._mismatched_lora_keys(model, {"w": _Adapter(384, 1536)}) == []


def test_a_conv_weight_is_measured_the_way_comfy_flattens_it(mm):
    model = _Model({"w": (64, 32, 3, 3)})       # comfy flattens to (64, 288)
    assert mm._mismatched_lora_keys(model, {"w": _Adapter(64, 288)}) == []
    assert mm._mismatched_lora_keys(model, {"w": _Adapter(64, 32)}) != []


def test_the_dropped_message_carries_both_shapes(mm, monkeypatch, capsys):
    """"trained against a different variant" is not actionable on its own. The numbers say
    whether every layer is off by the same factor or only some are."""
    model = _Model({"diffusion_model.blocks.0.attn.qkv_proj.weight": (16128, 5376)})
    patches = {"diffusion_model.blocks.0.attn.qkv_proj.weight": _Adapter(5376, 5376)}
    bad = mm._mismatched_lora_keys(model, patches)

    assert len(bad) == 1
    key, dims, want = bad[0]
    assert dims == (5376, 5376)
    assert want == (16128, 5376)
