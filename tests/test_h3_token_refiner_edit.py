"""Editing the token refiner's OUTPUT rather than the Qwen tensor before it.

H3 takes `context` [1, L, 5120] through condition_proj (-> 5376) and a 2-block token
refiner that ends in an RMSNorm. Everything FunPack steers today happens BEFORE that, so
the refiner processes the edit: its attention mixes it across tokens and its final norm
renormalizes the magnitude away. An edit applied AFTER lands as set, in the space the 50
blocks consume.

Two edits: `scale` (how loudly the prompt is read through attention) and `bias` (a single
[hidden] vector added to every prompt row — prompt-LENGTH-INDEPENDENT, unlike anything
per-position, because Qwen does not pad and H3's sequence length moves with every edit).
"""
import inspect
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import minimax_h3 as h3

HID = 6


class FakeRefiner(torch.nn.Module):
    def forward(self, x, *a, **k):
        return torch.ones(x.shape[-2], HID)


class FakePatcher:
    def __init__(self, proj_weight=None):
        self.objects = {"diffusion_model.token_refiner": FakeRefiner()}
        if proj_weight is not None:
            self.objects["diffusion_model.condition_proj"] = \
                torch.nn.Linear(proj_weight.shape[1], proj_weight.shape[0])
            with torch.no_grad():
                self.objects["diffusion_model.condition_proj"].weight.copy_(proj_weight)
        self.patched = {}

    def get_model_object(self, name):
        if name in self.patched:
            return self.patched[name]
        if name not in self.objects:
            raise KeyError(name)
        return self.objects[name]

    def clone(self):
        other = FakePatcher.__new__(FakePatcher)
        other.objects, other.patched = self.objects, dict(self.patched)
        return other

    def add_object_patch(self, name, obj):
        self.patched[name] = obj


def run(edit, rows=8):
    return edit(torch.zeros(1, rows, 4))


# ── the edit ────────────────────────────────────────────────────────────────

def test_scale_multiplies_the_rows():
    out = run(h3.TokenRefinerEdit(FakeRefiner(), scale=0.5))
    assert torch.allclose(out, torch.full((8, HID), 0.5))


def test_rows_before_the_start_are_untouched():
    """The <Picture N> label and vision block sit at the head of this span."""
    out = run(h3.TokenRefinerEdit(FakeRefiner(), scale=0.0, row_start=3))
    assert torch.all(out[:3] == 1.0)
    assert torch.all(out[3:] == 0.0)


def test_a_row_end_bounds_the_other_side():
    out = run(h3.TokenRefinerEdit(FakeRefiner(), scale=2.0, row_start=2, row_end=5))
    assert torch.all(out[:2] == 1.0) and torch.all(out[5:] == 1.0)
    assert torch.all(out[2:5] == 2.0)


def test_a_bias_is_added_to_every_row_in_range():
    bias = torch.arange(HID, dtype=torch.float32)
    out = run(h3.TokenRefinerEdit(FakeRefiner(), bias=bias, row_start=1))
    assert torch.allclose(out[1], 1.0 + bias)
    assert torch.allclose(out[5], 1.0 + bias)          # same vector on every row
    assert torch.all(out[0] == 1.0)


def test_the_bias_is_length_independent():
    """The property that makes it a usable learning target: one vector, any prompt length."""
    bias = torch.full((HID,), 0.25)
    short = run(h3.TokenRefinerEdit(FakeRefiner(), bias=bias), rows=4)
    long = run(h3.TokenRefinerEdit(FakeRefiner(), bias=bias), rows=40)
    assert torch.allclose(short[0], long[0])


def test_scale_and_bias_compose_scale_first():
    bias = torch.full((HID,), 1.0)
    out = run(h3.TokenRefinerEdit(FakeRefiner(), scale=2.0, bias=bias))
    assert torch.allclose(out, torch.full((8, HID), 3.0))   # 1*2 + 1


def test_an_out_of_range_start_is_a_no_op_not_a_crash():
    out = run(h3.TokenRefinerEdit(FakeRefiner(), scale=0.0, row_start=99))
    assert torch.all(out == 1.0)


def test_the_inner_output_is_not_mutated_in_place():
    inner = FakeRefiner()
    edit = h3.TokenRefinerEdit(inner, scale=0.0)
    run(edit)
    assert torch.all(inner(torch.zeros(1, 8, 4)) == 1.0)


# ── attaching it ────────────────────────────────────────────────────────────

def test_no_edit_returns_the_model_unpatched():
    model = FakePatcher()
    out, note = h3.apply_token_refiner_edit(model, scale=1.0, bias=None)
    assert out is model and note is None and model.patched == {}


def test_it_patches_the_refiner_object():
    model = FakePatcher()
    out, note = h3.apply_token_refiner_edit(model, scale=1.2, row_start=4)
    assert isinstance(out.patched["diffusion_model.token_refiner"], h3.TokenRefinerEdit)
    assert "scale=1.2" in note and "rows 4:end" in note


def test_object_patches_so_nothing_outlives_the_run():
    src = inspect.getsource(h3.apply_token_refiner_edit)
    assert "add_object_patch" in src and "model.clone()" in src


def test_a_model_without_a_refiner_declines_with_a_reason():
    class Bare:
        def get_model_object(self, name):
            raise KeyError(name)
    out, note = h3.apply_token_refiner_edit(Bare(), scale=0.5)
    assert isinstance(out, Bare) and "no token_refiner" in note


# ── projecting a conditioning-space direction ───────────────────────────────

def test_a_direction_is_projected_by_the_weight_only():
    """A direction is a difference of two points, so the bias would move the origin."""
    weight = torch.randn(HID, 4)
    model = FakePatcher(proj_weight=weight)
    direction = torch.randn(4)
    out = h3.project_into_refiner_space(model, direction)
    assert torch.allclose(out, weight @ direction, atol=1e-5)


def test_a_mismatched_width_is_refused():
    model = FakePatcher(proj_weight=torch.randn(HID, 4))
    assert h3.project_into_refiner_space(model, torch.randn(9)) is None


def test_no_projection_available_returns_none():
    assert h3.project_into_refiner_space(FakePatcher(), torch.randn(4)) is None


# ── independence from the refinement path ───────────────────────────────────

def test_the_installer_never_reaches_past_the_conditioning():
    """The refiner edit is now partly rating-learned, but the SAMPLER still only applies.

    Studio owns the refinement key, the state file and the learning; the sampler reads what
    Studio tagged onto the conditioning and nothing else. Reaching for the key here would put
    learning on both sides of the bridge, which is the boundary this project keeps.
    """
    import samplers
    cls = samplers.FunPackLTXAVSceneChainSampler
    src = "".join(inspect.getsource(fn) for fn in
                  (cls._install_h3_token_refiner, cls._h3_taste_bias_vector, cls._h3_prompt_rows))
    for forbidden in ("refinement_key", "_v2_load_state", "value_fn", "global_state",
                      "phrase_memory"):
        assert forbidden not in src


def test_the_widget_defaults_to_untouched():
    import samplers
    spec = samplers.FunPackLTXAVSceneChainSampler.INPUT_TYPES()
    fields = {**spec.get("required", {}), **spec.get("optional", {})}
    assert fields["h3_prompt_scale"][1]["default"] == 1.0
    assert "h3_gain_prompt" in fields["h3_prompt_scale"][1]["tooltip"]   # says how they differ


# ── the learned taste bias ──────────────────────────────────────────────────
#
# The direction comes from ratings (Studio's `liked_dir`), the magnitude does not. A learned
# direction has no natural scale in the refiner's space — those norms belong to the
# checkpoint — so the bias is a UNIT vector rescaled at apply time to a fraction of the
# span's own mean row norm. "10% of a typical prompt row" means the same thing on every
# prompt and every checkpoint; an absolute vector would be inert or catastrophic with no way
# to tell which in advance.

class ScaledRefiner(torch.nn.Module):
    """Rows with a known norm, so the relative rescale can be checked exactly."""

    def __init__(self, value):
        super().__init__()
        self.value = float(value)

    def forward(self, x, *a, **k):
        return torch.full((x.shape[-2], HID), self.value)


def test_a_relative_bias_scales_to_the_rows_it_lands_on():
    unit = torch.zeros(HID)
    unit[0] = 1.0
    out = h3.TokenRefinerEdit(ScaledRefiner(2.0), bias=unit * 0.5,
                              bias_relative=True)(torch.zeros(1, 4, 4))
    row_norm = (torch.full((HID,), 2.0)).norm()             # every row has this norm
    assert out[0, 0] == pytest.approx(2.0 + 0.5 * float(row_norm), rel=1e-5)
    assert out[0, 1] == pytest.approx(2.0)                  # off-direction untouched


def test_the_same_strength_means_the_same_thing_at_a_different_scale():
    """The property the relative mode exists for: transferable between checkpoints."""
    unit = torch.zeros(HID)
    unit[0] = 1.0
    small = h3.TokenRefinerEdit(ScaledRefiner(1.0), bias=unit * 0.25,
                                bias_relative=True)(torch.zeros(1, 4, 4))
    big = h3.TokenRefinerEdit(ScaledRefiner(100.0), bias=unit * 0.25,
                              bias_relative=True)(torch.zeros(1, 4, 4))
    assert (small[0, 0] / 1.0) == pytest.approx(big[0, 0] / 100.0, rel=1e-4)


def test_an_absolute_bias_still_ignores_the_row_scale():
    unit = torch.zeros(HID)
    unit[0] = 1.0
    out = h3.TokenRefinerEdit(ScaledRefiner(5.0), bias=unit * 0.5)(torch.zeros(1, 4, 4))
    assert out[0, 0] == pytest.approx(5.5)


def test_a_negative_strength_pushes_the_other_way():
    """Away from what was liked has to be reachable, or the loop can only ever agree."""
    unit = torch.zeros(HID)
    unit[0] = 1.0
    out = h3.TokenRefinerEdit(ScaledRefiner(2.0), bias=unit * -0.5,
                              bias_relative=True)(torch.zeros(1, 4, 4))
    assert out[0, 0] < 2.0


def test_the_note_says_the_bias_is_relative():
    model = FakePatcher()
    _out, note = h3.apply_token_refiner_edit(model, bias=torch.zeros(HID), bias_relative=True)
    assert "bias (relative)" in note
