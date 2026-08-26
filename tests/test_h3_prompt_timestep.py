"""Prompt timestep: the PROMPT's own place in the denoise — not the reference label's.

H3 tells every row of its packed sequence how far along the denoise it is, and it tells the
text rows whatever it tells the picture. So early in a run, while the picture is still noise,
the prompt is modulated as unreliable too. Giving the prompt rows a level of their own makes
the model lean on the prompt from the first step.

WHICH ROWS is the whole feature, and the first version got it wrong. Measured on a rental at
0.9-1.0: it reproduced the reference image almost untouched in the output, overriding a
prompt that explicitly said the reference was not where the scene begins. An r2v conditioning
is laid out `<Picture n>: ` label / vision block / prompt, and the LABEL is text (the vision
block is not). That label already reads to the model as "compose like this"; made maximally
authoritative, the model composed like it, literally.

The prompt is the LAST text run. Nothing outside the text span ever carries the text tag, so
that identifies it without help from the conditioning — and with no reference wired there is
one text run and it is the prompt, so the rule holds either way.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import minimax_h3 as h3

MOD = 3
TEXT = h3.MODALITY_TAGS["text"]


def seg(start, stop, t_row, tag):
    return (start, stop, t_row * MOD + tag)


# An r2v sequence: label(text) / vision(video) / prompt(text) / audio / video.
R2V = [
    seg(0, 4, 0, TEXT),                          # "<Picture 1>: "
    seg(4, 20, 0, h3.MODALITY_TAGS["video"]),    # the reference itself
    seg(20, 60, 0, TEXT),                        # the prompt
    seg(60, 80, 1, h3.MODALITY_TAGS["audio"]),
    seg(80, 200, 0, h3.MODALITY_TAGS["video"]),
]
# No reference: one text run, and it is the prompt.
T2V = [
    seg(0, 40, 0, TEXT),
    seg(40, 60, 1, h3.MODALITY_TAGS["audio"]),
    seg(60, 200, 0, h3.MODALITY_TAGS["video"]),
]


# ── finding the prompt ──────────────────────────────────────────────────────

def test_the_prompt_is_the_last_text_run_not_the_first():
    """The bug in one assertion: index 0 is the reference label, index 2 is the prompt."""
    assert h3.last_prompt_segment(R2V) == 2


def test_with_no_reference_the_only_text_run_is_the_prompt():
    assert h3.last_prompt_segment(T2V) == 0


def test_a_sequence_with_no_text_at_all_has_no_prompt():
    assert h3.last_prompt_segment(
        [seg(0, 10, 0, h3.MODALITY_TAGS["video"])]) is None


def test_it_reads_the_tag_off_the_row_index():
    """Rows are t_row*3+tag, so a text run at a LATER timestep must still read as text."""
    assert h3.last_prompt_segment([seg(0, 4, 2, TEXT)]) == 0


def test_no_segments_is_not_a_crash():
    assert h3.last_prompt_segment([]) is None and h3.last_prompt_segment(None) is None


# ── running a block with it ─────────────────────────────────────────────────

def constant_row(value=0.95):
    return lambda like: torch.tensor([[value]], dtype=like.dtype, device=like.device)


def run_block(segments, t_emb, row=None):
    """Returns the args the block was actually called with."""
    seen = {}

    def original(args):
        seen.update(args)
        return args["img"]
    patch = h3.PromptTimestepBlock(row or constant_row())
    out = patch({"img": "IMG", "t_emb": t_emb, "mod_segments": segments}, 
                {"original_block": original})
    assert out == "IMG"
    return seen


def test_only_the_prompt_segment_is_re_timed():
    """The regression, stated directly: the reference label keeps the row it had."""
    got = run_block(R2V, torch.tensor([[0.2], [0.7]]))
    assert got["mod_segments"][0] == R2V[0]      # the "<Picture 1>" label — untouched
    assert got["mod_segments"][1] == R2V[1]      # the reference image — untouched
    assert got["mod_segments"][2][2] == 2 * MOD + TEXT   # the prompt — the appended row


def test_the_prompt_keeps_its_row_range():
    """Only which timestep it is modulated at changes, never which rows it covers."""
    got = run_block(R2V, torch.tensor([[0.2], [0.7]]))
    assert got["mod_segments"][2][:2] == (20, 60)


def test_every_other_segment_is_left_exactly_as_it_was():
    got = run_block(R2V, torch.tensor([[0.2], [0.7]]))
    for index, entry in enumerate(R2V):
        if index != 2:
            assert got["mod_segments"][index] == entry


def test_the_appended_row_carries_the_requested_timestep():
    got = run_block(T2V, torch.tensor([[0.2]]), row=constant_row(0.93))
    assert got["t_emb"].shape[0] == 2
    assert float(got["t_emb"][1][0]) == pytest.approx(0.93)


def test_the_original_rows_are_untouched_underneath_it():
    """Every index the model computed for its own rows must still mean what it meant."""
    t_emb = torch.tensor([[0.2], [0.7]])
    got = run_block(R2V, t_emb)
    assert torch.equal(got["t_emb"][:2], t_emb)


def test_the_segment_list_is_not_mutated_in_place():
    """It is rebuilt per forward but shared by all 50 blocks within one; editing it would
    make block 2 onward re-time a row that block 1 had already moved."""
    segments = list(R2V)
    run_block(segments, torch.tensor([[0.2]]))
    assert segments == R2V


def test_a_sequence_with_no_prompt_runs_untouched():
    got = run_block([seg(0, 10, 0, h3.MODALITY_TAGS["video"])], torch.tensor([[0.2]]))
    assert got["t_emb"].shape[0] == 1


def test_a_failed_embedding_runs_the_block_as_trained():
    """Never a hard failure: this is a refinement, not a prerequisite."""
    def boom(_like):
        raise RuntimeError("no time basis")
    got = run_block(T2V, torch.tensor([[0.2]]), row=boom)
    assert got["t_emb"].shape[0] == 1
    assert got["mod_segments"] == T2V


# ── the timestep row ────────────────────────────────────────────────────────

def test_the_curve_table_is_read_the_way_the_model_reads_it():
    table = torch.tensor([[0.0, 0.0], [10.0, 20.0], [20.0, 40.0]], dtype=torch.float32)
    row = h3.PromptTimestepRow(0.5, table=table)(torch.zeros(1, 2))
    assert torch.allclose(row, torch.tensor([[10.0, 20.0]]))    # t=0.5 -> grid row 1


def test_a_fractional_position_interpolates():
    table = torch.tensor([[0.0, 0.0], [10.0, 20.0]], dtype=torch.float32)
    row = h3.PromptTimestepRow(0.25, table=table)(torch.zeros(1, 2))
    assert torch.allclose(row, torch.tensor([[2.5, 5.0]]))


def test_the_top_of_the_range_stays_on_the_last_interval():
    """t=1.0 must read the final grid row, not one past the end of the table."""
    table = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float32)
    assert torch.allclose(h3.PromptTimestepRow(1.0, table=table)(torch.zeros(1, 1)),
                          torch.tensor([[2.0]]))


def test_a_checkpoint_without_curves_uses_its_time_embedder():
    class Embedder:
        def __call__(self, value):
            return torch.stack([value, value * 2.0], dim=-1)
    row = h3.PromptTimestepRow(0.4, embedder=Embedder())(torch.zeros(1, 2))
    assert torch.allclose(row, torch.tensor([[0.4, 0.8]]))


def test_the_row_is_cast_to_whatever_the_live_embedding_is_using():
    table = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
    row = h3.PromptTimestepRow(0.0, table=table)(torch.zeros(1, 1, dtype=torch.bfloat16))
    assert row.dtype == torch.bfloat16


def test_the_row_is_computed_once_and_reused():
    """All 50 blocks ask for the same row every step."""
    calls = []

    class Counting:
        def __call__(self, value):
            calls.append(1)
            return value.unsqueeze(-1)
    provider = h3.PromptTimestepRow(0.5, embedder=Counting())
    like = torch.zeros(1, 1)
    for _ in range(5):
        provider(like)
    assert len(calls) == 1


# ── installing it ───────────────────────────────────────────────────────────

class FakePatcher:
    def __init__(self, n=3, table=torch.tensor([[0.0], [1.0]])):
        self.objects = {"diffusion_model.blocks": [object() for _ in range(n)]}
        if table is not None:
            self.objects["diffusion_model.adaln_t_table"] = table
        self.replaced = {}

    def get_model_object(self, name):
        return self.objects[name]

    def clone(self):
        other = FakePatcher.__new__(FakePatcher)
        other.objects = self.objects
        other.replaced = dict(self.replaced)
        return other

    def set_model_patch_replace(self, patch, name, block_name, number):
        self.replaced[(name, block_name, number)] = patch


def test_zero_changes_nothing_and_does_not_clone():
    model = FakePatcher()
    out, note = h3.apply_prompt_timestep(model, 0.0)
    assert out is model and note is None and model.replaced == {}


def test_it_installs_on_every_block():
    model = FakePatcher(n=3)
    out, note = h3.apply_prompt_timestep(model, 0.9)
    assert sorted(out.replaced) == [("dit", "double_block", i) for i in range(3)]
    assert "0.9" in note


def test_every_block_shares_one_patch():
    out, _ = h3.apply_prompt_timestep(FakePatcher(n=3), 0.9)
    assert len({id(v) for v in out.replaced.values()}) == 1


def test_a_checkpoint_with_no_time_basis_declines_with_a_reason():
    model = FakePatcher(table=None)
    out, note = h3.apply_prompt_timestep(model, 0.9)
    assert out is model and "no time embedding" in note


def test_a_core_without_the_block_hook_declines_rather_than_half_working():
    class NoHook:
        def get_model_object(self, name):
            return [object()]
    out, note = h3.apply_prompt_timestep(NoHook(), 0.9)
    assert isinstance(out, NoHook)
    assert "per-block replace hook" in note


def test_the_note_names_the_value_and_when_it_applies():
    _out, note = h3.apply_prompt_timestep(FakePatcher(), 0.9)
    assert "Reference weight 0.9" in note and "every step" in note


# ── what it is called, and why ──────────────────────────────────────────────

def test_the_widget_name_is_not_renamed_with_the_label():
    """`h3_prompt_time` is stored in saved projects and saved ComfyUI workflows. The label
    was wrong and got fixed; renaming the INPUT would silently drop the setting from every
    project that already carries it."""
    import samplers
    spec = samplers.FunPackLTXAVSceneChainSampler.INPUT_TYPES()
    fields = {**spec.get("required", {}), **spec.get("optional", {})}
    assert "h3_prompt_time" in fields


def test_the_tooltip_describes_what_was_measured_not_what_was_intended():
    """It was built to improve prompt adherence and does not. A tooltip left describing the
    intent would send the next reader looking for an effect that is not there."""
    import samplers
    spec = samplers.FunPackLTXAVSceneChainSampler.INPUT_TYPES()
    tip = spec["optional"]["h3_prompt_time"][1]["tooltip"]
    assert "REFERENCE" in tip
    assert "MEASURED" in tip
    assert "stick closer to the prompt" not in tip


def test_the_editor_label_matches():
    from pathlib import Path
    js = (Path(__file__).resolve().parents[1] / "movie_editor" / "frontend"
          / "engine_settings.js").read_text(encoding="utf-8")
    row = next(ln for ln in js.splitlines() if '{ name: "h3_prompt_time"' in ln)
    assert 'label: "Reference weight"' in row


# ── ramped, so the model can commit first ───────────────────────────────────
#
# The timestep is how the model knows what stage everything is at. A prompt that disagrees
# with the picture about the stage costs it the ability to settle on ONE reading — reported
# from a rental at 0.6-0.75 as several versions of the shot attempted at once in a single
# video, strongest signal winning. Structure is chosen early, so the edit is held off until
# it has been chosen.

def ramped(segments, t_emb, sigma, ramp, row=None):
    seen = {}

    def original(args):
        seen.update(args)
        return args["img"]
    patch = h3.PromptTimestepBlock(row or constant_row(0.95), ramp=ramp)
    patch({"img": "IMG", "t_emb": t_emb, "mod_segments": segments,
           "transformer_options": {"sigmas": torch.tensor([sigma])}},
          {"original_block": original})
    return seen


def half_way(sigma):
    """Stand-in for _make_steer_ramp: nothing early, full late."""
    return 0.0 if sigma > 0.5 else 1.0


def test_early_steps_run_exactly_as_if_it_were_not_installed():
    """Not 'weaker' — identical. The disagreement is the cost, so there must be none."""
    got = ramped(T2V, torch.tensor([[0.2]]), sigma=0.9, ramp=half_way)
    assert got["t_emb"].shape[0] == 1
    assert got["mod_segments"] == T2V


def test_late_steps_get_the_full_value():
    got = ramped(T2V, torch.tensor([[0.2]]), sigma=0.1, ramp=half_way,
                 row=constant_row(0.93))
    assert float(got["t_emb"][1][0]) == pytest.approx(0.93)


def test_a_partial_gate_interpolates_from_the_prompts_own_row():
    """From its OWN row, which the segment names — so a gate of 0 is the untouched model
    rather than 'whatever row 0 happens to be'."""
    segments = [seg(0, 10, 1, TEXT)]                 # the prompt sits at t_row 1
    t_emb = torch.tensor([[0.2], [0.6]])
    got = ramped(segments, t_emb, sigma=0.0, ramp=lambda _s: 0.5, row=constant_row(1.0))
    assert float(got["t_emb"][2][0]) == pytest.approx(0.8)   # halfway 0.6 -> 1.0


def test_no_ramp_means_every_step(  ):
    """Without a ramp the patch is unconditional, which is what the tests above assume."""
    got = run_block(T2V, torch.tensor([[0.2]]))
    assert got["t_emb"].shape[0] == 2


def test_a_sampler_that_records_no_sigma_still_applies_it():
    """Better fully on than silently off: an inert feature that reports itself as active is
    the failure mode funpack_log exists to stop."""
    seen = {}

    def original(args):
        seen.update(args)
        return args["img"]
    h3.PromptTimestepBlock(constant_row(), ramp=half_way)(
        {"img": "IMG", "t_emb": torch.tensor([[0.2]]), "mod_segments": T2V,
         "transformer_options": {}}, {"original_block": original})
    assert seen["t_emb"].shape[0] == 2


def test_a_ramp_that_raises_does_not_take_the_render_with_it():
    def boom(_sigma):
        raise RuntimeError("no")
    got = ramped(T2V, torch.tensor([[0.2]]), sigma=0.1, ramp=boom)
    assert got["t_emb"].shape[0] == 2


def test_the_note_says_it_is_ramped():
    _out, note = h3.apply_prompt_timestep(FakePatcher(), 0.9, ramp=half_way)
    assert "second half of the schedule" in note


def test_the_sampler_hands_it_the_ramp_the_other_wrappers_use():
    import inspect
    import samplers
    src = inspect.getsource(
        samplers.FunPackLTXAVSceneChainSampler._install_h3_prompt_timestep)
    assert 'ramp=getattr(self, "_steer_ramp", None)' in src
