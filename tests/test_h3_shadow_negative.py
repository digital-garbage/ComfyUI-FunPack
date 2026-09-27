"""H3 shadow negative: pure-math pieces that don't need a real DiT block.

MiniMax H3 runs at CFG 1.0 always, so a plain negative prompt is otherwise dead weight
(see negative_erase's own tests for that half of the story). This mechanism gives it a
job by pushing the positive attention output away from a shadow copy of the negative,
NAG-style. These tests pin the norm-cap math and the presentation-span detection that the
rest of h3_shadow_negative.py builds on, without needing a loaded H3 model.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h3_shadow_negative as sn  # noqa: E402


# --- nag_blend ---------------------------------------------------------------------------


def test_scale_one_is_a_no_op():
    pos = torch.randn(4, 8)
    neg = torch.randn(4, 8)
    out = sn.nag_blend(pos, neg, scale=1.0, tau=2.5, alpha=1.0)
    assert out is pos  # short-circuited, not just numerically equal


def test_alpha_zero_is_a_no_op():
    pos = torch.randn(4, 8)
    neg = torch.randn(4, 8)
    out = sn.nag_blend(pos, neg, scale=3.0, tau=2.5, alpha=0.0)
    assert out is pos


def test_pushes_away_from_negative():
    """scale>1 should move the result further from neg than pos itself was."""
    pos = torch.ones(1, 4)
    neg = torch.zeros(1, 4)
    out = sn.nag_blend(pos, neg, scale=2.0, tau=8.0, alpha=1.0)
    assert torch.linalg.vector_norm(out - neg) > torch.linalg.vector_norm(pos - neg)


def test_tau_caps_the_output_norm():
    """However hard the push, the guided result's norm cannot exceed tau times pos's own."""
    pos = torch.ones(1, 4)
    neg = -100.0 * torch.ones(1, 4)  # would blow the extrapolation up without a cap
    out = sn.nag_blend(pos, neg, scale=5.0, tau=1.5, alpha=1.0)
    pos_norm = torch.linalg.vector_norm(pos, dim=1)
    out_norm = torch.linalg.vector_norm(out, dim=1)
    assert torch.all(out_norm <= pos_norm * 1.5 + 1e-4)


def test_alpha_blends_toward_positive_only():
    pos = torch.ones(1, 4)
    neg = torch.zeros(1, 4)
    full = sn.nag_blend(pos, neg, scale=2.0, tau=8.0, alpha=1.0)
    half = sn.nag_blend(pos, neg, scale=2.0, tau=8.0, alpha=0.5)
    assert torch.allclose(half, pos * 0.5 + full * 0.5)


# --- sigma_progress ------------------------------------------------------------------------


def test_no_sigmas_is_unresolvable():
    assert sn.sigma_progress(0.5, None) is None


def test_single_sigma_is_unresolvable():
    assert sn.sigma_progress(0.5, torch.tensor([1.0])) is None


def test_first_step_is_zero_progress():
    sigmas = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])
    assert sn.sigma_progress(1.0, sigmas) == pytest.approx(0.0)


def test_last_real_step_is_full_progress():
    sigmas = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])
    assert sn.sigma_progress(0.25, sigmas) == pytest.approx(1.0)


def test_nearest_sigma_wins_off_grid():
    sigmas = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])
    assert sn.sigma_progress(0.7, sigmas) == pytest.approx(1.0 / 3.0)


# --- _presentation_plan --------------------------------------------------------------------


def test_no_segments_inside_the_text_span_is_a_bypass():
    """Every segment starts at/after pos_text_len — there is no text run to replace."""
    runs, user_run = sn._presentation_plan([(10, 20, 0)], pos_text_len=5)
    assert runs is None and user_run is None


def test_a_segment_straddling_the_text_boundary_is_a_bypass():
    """A run that starts inside the text span but ends past it can't be a pure label or a
    pure user-prompt run — the layout assumption doesn't hold, so refuse rather than guess."""
    runs, user_run = sn._presentation_plan([(0, 8, 1)], pos_text_len=5)
    assert runs is None and user_run is None


def test_the_last_text_tagged_run_is_the_user_prompt():
    """Row % 3 == 1 is text. An image-label run (tag 1) precedes the user's own prompt run
    (also tag 1) — the LAST one is the one the tokenizer appended the user's text as."""
    segs = [(0, 3, 1), (3, 6, 0), (6, 10, 1)]
    runs, user_run = sn._presentation_plan(segs, pos_text_len=10)
    assert runs == segs
    assert user_run == (6, 10, 1)


def test_no_text_tagged_run_at_all_is_a_bypass():
    runs, user_run = sn._presentation_plan([(0, 5, 0)], pos_text_len=5)
    assert runs is None and user_run is None


# --- ShadowState -----------------------------------------------------------------------


def test_shadow_state_starts_with_no_negative_prepared():
    state = sn.ShadowState(dm=None, negative_context=torch.randn(1, 3, 16),
                           video_scale=3.0, audio_scale=1.0, tau=2.5, alpha=0.35,
                           start_percent=0.0, end_percent=0.6)
    assert state.neg_h is None
    assert state.hard_disabled is False


# --- _prepare_negative -------------------------------------------------------------------


def test_non_tensor_negative_context_yields_nothing():
    state = sn.ShadowState(dm=None, negative_context=None, video_scale=3.0, audio_scale=1.0,
                           tau=2.5, alpha=0.35, start_percent=0.0, end_percent=0.6)
    assert sn._prepare_negative(state, torch.zeros(1, 4), {}) is None


def test_empty_negative_context_yields_nothing():
    state = sn.ShadowState(dm=None, negative_context=torch.zeros(1, 0, 4), video_scale=3.0,
                           audio_scale=1.0, tau=2.5, alpha=0.35, start_percent=0.0,
                           end_percent=0.6)
    assert sn._prepare_negative(state, torch.zeros(1, 4), {}) is None


# --- 2026-09-27: the fix for "never ran" --------------------------------------------
# The prompt length used to be read from transformer_options["c_crossattn"], which ComfyUI
# never sets, so every block ran the stock forward. Proven against a real tiny H3 forward
# outside this (stubbed) suite; these pin the pieces the fix rests on.

def test_a_per_token_mod_row_is_accepted_not_int_cast():
    rows = torch.tensor([3, 6, 9])            # all tag 0 (video), per-token strengths
    assert sn._tag_is(rows, 0) and not sn._tag_is(rows, 2)
    assert sn._tag_is(5, 2) and not sn._tag_is(5, 0)


def test_the_user_prompt_run_is_found_when_rows_are_tensors():
    segs = [(0, 3, 1), (3, 5, torch.tensor([0, 0])), (5, 9, 1), (9, 12, 2)]
    runs, user = sn._presentation_plan(segs, 9)
    assert user == (5, 9, 1) and len(runs) == 3


def test_progress_comes_from_the_schedule_core_publishes():
    to = {"sample_sigmas": torch.tensor([1.0, 0.7, 0.3, 0.0]), "sigmas": torch.tensor([0.3])}
    assert sn.step_progress(to) == 1.0
    to["sigmas"] = torch.tensor([1.0])
    assert sn.step_progress(to) == 0.0
    assert sn.step_progress({}) is None


def test_without_a_recorded_prompt_length_it_bypasses_and_says_so(capsys):
    state = sn.ShadowState(dm=None, negative_context=torch.randn(1, 3, 16), video_scale=3.0,
                           audio_scale=1.0, tau=2.5, alpha=1.0, start_percent=0.0,
                           end_percent=1.0)
    ran = []
    hook = sn.make_block_hook(object(), state, 1)
    args = {"img": torch.zeros(4, 16), "t_emb": None,
            "mod_segments": [(0, 1, 1), (1, 2, 2), (2, 4, 0)], "rope_freqs": None,
            "transformer_options": {}}
    hook(args, {"original_block": lambda a: ran.append(1) or {"img": a["img"]}})
    assert ran == [1]
    assert "prompt length was not seen" in capsys.readouterr().out
