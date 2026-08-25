"""The H3 render gains are learned from ratings, not typed.

A handful of scalars is a SMALLER search than the sigma profile, which already converges on
ratings alone — so a hand-tuned value here is the user doing work the loop can do. Same
update as the sigma profile with one difference: that profile is centred on 0, so its applied
value IS its perturbation; most of these are centred on 1.0, so the perturbation is
`applied - value`.

`refiner_bias` is the exception on both counts: it is centred on 0.0 (a signed push along the
learned taste direction, where 0 means no push) and it explores a narrower band, so its
exploration width scales down to match or its perturbations would clip against the bounds and
the reward would be credited to a move that never fully happened.
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
    # Not all neutrals are 1.0: refiner_bias is a signed push, so its neutral is 0.0.
    assert state["values"] == {k: studio._h3_gain_centre(k) for k in studio.H3_GAIN_KEYS}


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
    assert g["h3_gains"]["values"] == {k: studio._h3_gain_centre(k) for k in studio.H3_GAIN_KEYS}


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
    assert (studio._v2_h3_gains_for_run({}, explore=False)
            == {k: studio._h3_gain_centre(k) for k in studio.H3_GAIN_KEYS})


def test_rendered_values_stay_inside_the_band(studio):
    g = {}
    studio._ensure_h3_gain_state(g)
    g["h3_gains"]["explore"] = 5.0                       # absurd, to prove the clamp
    out = studio._v2_h3_gains_for_run(g, seed=3)
    for key, value in out.items():
        low, high = studio._h3_gain_bounds(key)
        assert low <= value <= high


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
                        lambda *a, **k: {key: studio._h3_gain_centre(key)
                                         for key in studio.H3_GAIN_KEYS}, raising=False)
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


# ── the learned taste direction ────────────────────────────────────────────
#
# `refiner_bias` says HOW FAR to push; this is WHICH WAY. The direction is the same
# `liked_dir` conditioning ascent already steers along — the difference is where it lands.
# Studio hands it over as a unit vector and the sampler decides the magnitude, because the
# magnitude belongs to the refiner's space and Studio cannot see it.

def _tensor(values):
    import torch
    return torch.tensor(values, dtype=torch.float32)


def _liked(count, direction=(3.0, 4.0)):
    return {"liked_dir": {"direction": C.tensor_to_serializable(_tensor(direction)),
                          "direction_count": count}}


def test_no_direction_before_enough_liked_runs(studio):
    """Two liked runs is a coincidence. The threshold is the one liked_dir's existing
    conditioning-space consumers already use — not a new number invented here."""
    assert studio.H3_TASTE_DIR_MIN_COUNT == 3
    assert studio._h3_taste_direction(_liked(2)) is None


def test_the_direction_arrives_once_the_threshold_is_met(studio):
    assert studio._h3_taste_direction(_liked(3)) is not None


def test_the_direction_is_a_unit_vector(studio):
    out = studio._h3_taste_direction(_liked(5, (3.0, 4.0)))
    assert float(out.norm()) == pytest.approx(1.0)
    assert out.tolist() == pytest.approx([0.6, 0.8])


def test_a_zero_direction_is_refused(studio):
    """Dividing by its norm would produce NaN and poison every prompt row."""
    assert studio._h3_taste_direction(_liked(9, (0.0, 0.0))) is None


def test_an_empty_state_has_no_direction(studio):
    assert studio._h3_taste_direction({}) is None
    assert studio._h3_taste_direction(None) is None


def test_the_direction_rides_entry_zero_with_the_gains(studio, monkeypatch):
    import torch
    monkeypatch.setattr(C, "_log", types.SimpleNamespace(
        failed=lambda *a, **k: None, note_on_change=lambda *a, **k: None), raising=False)
    cond = [[torch.zeros(1, 4, 8), {}], [torch.zeros(1, 4, 8), {}]]
    out = studio._v2_tag_h3_gains(cond, _liked(4), seed=5)
    assert studio.H3_TASTE_DIR_META in out[0][1]
    assert studio.H3_TASTE_DIR_META not in out[1][1]


def test_a_direction_alone_is_worth_tagging(studio, monkeypatch):
    """Neutral gains used to mean "tag nothing" — but a direction with a zero push is still
    worth carrying, because the sampler's manual mode can supply the push itself."""
    import torch
    monkeypatch.setattr(C, "_log", types.SimpleNamespace(
        failed=lambda *a, **k: None, note_on_change=lambda *a, **k: None), raising=False)
    monkeypatch.setattr(studio, "_v2_h3_gains_for_run",
                        lambda *a, **k: {key: studio._h3_gain_centre(key)
                                         for key in studio.H3_GAIN_KEYS}, raising=False)
    cond = [[torch.zeros(1, 4, 8), {}]]
    out = studio._v2_tag_h3_gains(cond, _liked(4), seed=5)
    assert out is not cond and studio.H3_TASTE_DIR_META in out[0][1]


# ── the bias explores its own band ─────────────────────────────────────────

def test_the_bias_is_centred_on_no_push(studio):
    assert studio._h3_gain_centre("refiner_bias") == 0.0
    assert studio._h3_gain_centre("video") == 1.0


def test_the_bias_can_be_learned_negative(studio):
    """Pushing AWAY from the liked direction must be reachable, or the loop can only agree."""
    low, high = studio._h3_gain_bounds("refiner_bias")
    assert low < 0.0 < high


def test_exploration_scales_to_the_band_it_explores(studio):
    """One stored `explore` drives every gain; a narrower band has to shrink it or its
    perturbations clip against the bounds and the credit goes to a move that never happened."""
    wide = studio._h3_gain_explore("video", 0.05)
    narrow = studio._h3_gain_explore("refiner_bias", 0.05)
    assert narrow < wide
    low, high = studio._h3_gain_bounds("refiner_bias")
    assert narrow == pytest.approx(0.05 * (high - low) / (studio.H3_GAIN_MAX - studio.H3_GAIN_MIN))


def test_a_liked_bias_perturbation_is_moved_toward(studio):
    g = {}
    studio._ensure_h3_gain_state(g)
    g["h3_gains"]["last_applied"]["refiner_bias"] = 0.10
    studio._v2_update_h3_gains(g, profile(1.0))
    assert g["h3_gains"]["values"]["refiner_bias"] > 0.0


def test_a_disliked_bias_perturbation_is_moved_away_from(studio):
    g = {}
    studio._ensure_h3_gain_state(g)
    g["h3_gains"]["last_applied"]["refiner_bias"] = 0.10
    studio._v2_update_h3_gains(g, profile(-1.0))
    assert g["h3_gains"]["values"]["refiner_bias"] < 0.0


def test_the_bias_stays_inside_its_own_band_not_the_shared_one(studio):
    g = {}
    studio._ensure_h3_gain_state(g)
    low, high = studio._h3_gain_bounds("refiner_bias")
    for _ in range(400):
        g["h3_gains"]["last_applied"]["refiner_bias"] = high
        studio._v2_update_h3_gains(g, profile(1.0))
    assert low <= g["h3_gains"]["values"]["refiner_bias"] <= high
