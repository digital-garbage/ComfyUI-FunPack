"""The four H3 render gains are learned from ratings, not typed.

Four scalars is a SMALLER search than the sigma profile, which already converges on ratings
alone — so a hand-tuned value here is the user doing work the loop can do. Same update as
the sigma profile with one difference: that profile is centred on 0, so its applied value IS
its perturbation; these are centred on 1.0, so the perturbation is `applied - value`.
"""
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import conditioning as C


@pytest.fixture
def studio():
    return C.FunPackVideoRefinerV2.__new__(C.FunPackVideoRefinerV2)


def profile(reward, **kw):
    return dict({"reward": reward, "key": "like" if reward > 0 else "awful"}, **kw)


def test_a_fresh_key_starts_neutral(studio):
    state = studio._ensure_h3_gain_state({})
    assert set(state["values"]) == set(studio.H3_GAIN_KEYS)
    assert all(v == 1.0 for v in state["values"].values())


def test_a_liked_perturbation_is_moved_toward(studio):
    g = {}
    studio._ensure_h3_gain_state(g)
    g["h3_gains"]["last_applied"]["video"] = 1.10        # rendered above centre
    studio._v2_update_h3_gains(g, profile(1.0))
    assert g["h3_gains"]["values"]["video"] > 1.0


def test_a_disliked_perturbation_is_moved_away_from(studio):
    g = {}
    studio._ensure_h3_gain_state(g)
    g["h3_gains"]["last_applied"]["video"] = 1.10
    studio._v2_update_h3_gains(g, profile(-1.0))
    assert g["h3_gains"]["values"]["video"] < 1.0


def test_the_sign_of_the_perturbation_matters_not_just_the_reward(studio):
    """Rendering BELOW centre and liking it must move the value DOWN."""
    g = {}
    studio._ensure_h3_gain_state(g)
    g["h3_gains"]["last_applied"]["video"] = 0.90
    studio._v2_update_h3_gains(g, profile(1.0))
    assert g["h3_gains"]["values"]["video"] < 1.0


def test_no_perturbation_teaches_nothing(studio):
    g = {}
    studio._ensure_h3_gain_state(g)                      # last_applied == values
    studio._v2_update_h3_gains(g, profile(1.0))
    assert all(v == 1.0 for v in g["h3_gains"]["values"].values())


def test_values_stay_inside_the_band(studio):
    g = {}
    studio._ensure_h3_gain_state(g)
    for _ in range(400):
        g["h3_gains"]["last_applied"]["video"] = studio.H3_GAIN_MAX
        studio._v2_update_h3_gains(g, profile(1.0))
    assert g["h3_gains"]["values"]["video"] <= studio.H3_GAIN_MAX


def test_exploration_narrows_on_success_and_widens_on_failure(studio):
    g = {}
    studio._ensure_h3_gain_state(g)
    start = g["h3_gains"]["explore"]
    studio._v2_update_h3_gains(g, profile(1.0))
    assert g["h3_gains"]["explore"] < start
    for _ in range(5):
        studio._v2_update_h3_gains(g, profile(-1.0))
    assert g["h3_gains"]["explore"] > start * 0.99


def test_a_forget_rating_teaches_nothing(studio):
    g = {}
    studio._ensure_h3_gain_state(g)
    g["h3_gains"]["last_applied"]["video"] = 1.2
    studio._v2_update_h3_gains(g, profile(1.0, skip_learning=True))
    assert g["h3_gains"]["values"]["video"] == 1.0
    assert g["h3_gains"]["iterations"] == 0


# ── what gets rendered ─────────────────────────────────────────────────────

def test_a_run_is_perturbed_off_the_learned_centre(studio):
    """Without a perturbation the next rating has nothing to credit."""
    g = {}
    out = studio._v2_h3_gains_for_run(g, seed=7)
    assert any(v != 1.0 for v in out.values())


def test_the_perturbation_is_recorded_for_the_next_rating(studio):
    g = {}
    out = studio._v2_h3_gains_for_run(g, seed=7)
    assert g["h3_gains"]["last_applied"] == out


def test_the_same_seed_repeats_the_same_perturbation(studio):
    a = studio._v2_h3_gains_for_run({}, seed=11)
    b = studio._v2_h3_gains_for_run({}, seed=11)
    assert a == b
    assert studio._v2_h3_gains_for_run({}, seed=12) != a


def test_exploration_can_be_switched_off(studio):
    assert all(v == 1.0 for v in studio._v2_h3_gains_for_run({}, explore=False).values())


def test_rendered_values_stay_inside_the_band(studio):
    g = {}
    studio._ensure_h3_gain_state(g)
    g["h3_gains"]["explore"] = 5.0                       # absurd, to prove the clamp
    out = studio._v2_h3_gains_for_run(g, seed=3)
    assert all(studio.H3_GAIN_MIN <= v <= studio.H3_GAIN_MAX for v in out.values())


# ── the bridge to the sampler ──────────────────────────────────────────────

def test_the_gains_ride_entry_zero_of_the_conditioning(studio, monkeypatch):
    import torch
    monkeypatch.setattr(C, "_log", types.SimpleNamespace(
        failed=lambda *a, **k: None, note_on_change=lambda *a, **k: None), raising=False)
    cond = [[torch.zeros(1, 4, 8), {}], [torch.zeros(1, 4, 8), {}]]
    out = studio._v2_tag_h3_gains(cond, {}, seed=5)
    assert "funpack_h3_gains" in out[0][1]
    assert "funpack_h3_gains" not in out[1][1]


def test_neutral_gains_tag_nothing(studio, monkeypatch):
    import torch
    monkeypatch.setattr(studio, "_v2_h3_gains_for_run",
                        lambda *a, **k: {k: 1.0 for k in studio.H3_GAIN_KEYS}, raising=False)
    cond = [[torch.zeros(1, 4, 8), {}]]
    assert studio._v2_tag_h3_gains(cond, {}, seed=5) is cond


def test_learning_runs_on_every_rating_not_behind_a_toggle():
    """The toggle gates APPLICATION; the key must keep growing either way, the same rule
    value_guidance already follows."""
    import inspect
    src = inspect.getsource(C.FunPackVideoRefinerV2._v2_learn_scene_into_state)
    assert "_v2_update_h3_gains(target_global, profile)" in src


# ── axis-aware credit assignment ───────────────────────────────────────────
#
# A rating carries three per-axis signals in [-1, +1] as well as its scalar reward, and
# they disagree with it in exactly the cases that matter. "Missing concept" is reward
# +0.10 — quality was fine — but concept_signal -1.00. Read as the scalar it nudges every
# gain toward the run that missed the prompt; read per axis it pushes the prompt gains
# away from it and leaves the audio gain out of it.

def test_missing_concept_punishes_the_prompt_gains(studio):
    prof = C.RATING_PROFILES["Missing concept"]
    assert prof["reward"] > 0                       # the scalar says "slightly good"
    assert studio._h3_gain_credit("prompt", prof) == pytest.approx(-1.0)
    assert studio._h3_gain_credit("prompt_scale", prof) == pytest.approx(-1.0)


def test_missing_concept_does_not_drag_the_video_gain_down_with_it(studio):
    prof = C.RATING_PROFILES["Missing concept"]
    assert studio._h3_gain_credit("video", prof) > 0     # detail and quality were fine


def test_missing_details_punishes_the_video_gain(studio):
    prof = C.RATING_PROFILES["Missing details"]
    assert studio._h3_gain_credit("video", prof) < 0
    assert studio._h3_gain_credit("prompt", prof) > 0     # the prompt landed


def test_audio_has_no_rated_axis_so_it_uses_the_overall_reward(studio):
    for label in ("Perfect", "Awful", "Missing concept"):
        prof = C.RATING_PROFILES[label]
        assert studio._h3_gain_credit("audio", prof) == pytest.approx(prof["reward"])


def test_perfect_credits_everything_fully(studio):
    prof = C.RATING_PROFILES["Perfect"]
    for key in studio.H3_GAIN_KEYS:
        assert studio._h3_gain_credit(key, prof) == pytest.approx(1.0)


def test_awful_punishes_everything(studio):
    prof = C.RATING_PROFILES["Awful"]
    for key in studio.H3_GAIN_KEYS:
        assert studio._h3_gain_credit(key, prof) < 0


def test_a_profile_with_no_signals_falls_back_to_the_scalar(studio):
    assert studio._h3_gain_credit("prompt", {"reward": 0.4}) == pytest.approx(0.4)


def test_a_satisfied_axis_cannot_mask_a_failed_one(studio):
    """Averaging let it: detail -1.00 with quality +0.85 came out at -0.08, so a rating that
    said details were missing barely moved the gain that governs detail."""
    prof = {"reward": 0.0, "detail_signal": -1.0, "quality_signal": 1.0}
    assert studio._h3_gain_credit("video", prof) == pytest.approx(-1.0)


def test_missing_details_now_moves_the_video_gain_decisively(studio):
    assert studio._h3_gain_credit("video", C.RATING_PROFILES["Missing details"]) \
        == pytest.approx(-1.0)


def test_both_axes_satisfied_still_credits_fully(studio):
    prof = {"reward": 1.0, "detail_signal": 1.0, "quality_signal": 1.0}
    assert studio._h3_gain_credit("video", prof) == pytest.approx(1.0)


def test_axis_credit_actually_drives_the_update(studio):
    """The end-to-end consequence: one rating moves two gains in OPPOSITE directions."""
    g = {}
    studio._ensure_h3_gain_state(g)
    for key in ("prompt", "video"):
        g["h3_gains"]["last_applied"][key] = 1.10        # both rendered above centre
    studio._v2_update_h3_gains(g, dict(C.RATING_PROFILES["Missing concept"]))
    assert g["h3_gains"]["values"]["prompt"] < 1.0       # prompt missed -> back off
    assert g["h3_gains"]["values"]["video"] > 1.0        # picture was fine -> keep


def test_every_gain_key_has_an_axis_entry(studio):
    assert set(studio.H3_GAIN_AXES) == set(studio.H3_GAIN_KEYS)
