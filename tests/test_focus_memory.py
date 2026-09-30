"""Cross-prompt word frequency as a second score for where the camera aims."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import focus_memory as fm  # noqa: E402
import shot_camera as sc  # noqa: E402


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(fm, "_path", lambda key: str(tmp_path / f"{key}.focus.json"))
    return tmp_path


def test_silent_until_enough_prompts_then_more_seen_means_more_bonus(store):
    assert fm.prior("k") == {}
    fm.observe("k", "a", ["chin", "lamp"])
    fm.observe("k", "b", ["chin"])
    assert fm.prior("k") == {}                                   # two prompts: not a habit yet
    fm.observe("k", "c", ["chin", "door"])
    p = fm.prior("k")
    assert p["chin"] == pytest.approx(fm.FREQ_W) and 0 < p["lamp"] < p["chin"]
    assert p["lamp"] == p["door"]


def test_the_same_prompt_again_counts_nothing(store):
    for _ in range(5):
        fm.observe("k", "same", ["chin"])
    assert fm.observe("k", "same", ["chin"]) == 1 and fm.prior("k") == {}


def test_no_key_keeps_nothing(store):
    assert fm.observe("", "a", ["chin"]) == 0 and fm.prior("") == {}


def test_a_recurring_word_wins_between_two_otherwise_equal_targets():
    pytest.importorskip("spacy")
    P = "Intro. [Shot 1] <Subject 1> raises the lamp and lowers the mirror."
    plain = sc.add_camera_moves(P)[1][0]["target"]
    lamp = sc.add_camera_moves(P, prior={"lamp": 0.6})[1][0]["target"]
    mirror = sc.add_camera_moves(P, prior={"mirror": 0.6})[1][0]["target"]
    assert "lamp" in lamp or "->" in lamp
    assert "mirror" in mirror or "->" in mirror
    assert plain


def test_the_prior_never_outranks_an_owned_part():
    pytest.importorskip("spacy")
    P = "Intro. [Shot 1] <Subject 1> touches <Subject 1>'s chin and glances at the lamp."
    _o, rep = sc.add_camera_moves(P, prior={"lamp": sc_max()})
    assert "chin" in rep[0]["target"]


def sc_max():
    return fm.FREQ_W


def test_regenerating_with_other_cuts_and_views_is_still_one_prompt():
    base = "Intro. [Shot 1] <Subject 1> waves. [Shot 2] <Subject 2> nods."
    varied = ("Intro. [Shot 1] <Subject 1> waves. [Shot 2] At 00:04.000, the shot changes to a new view. "
              "Side view. <Subject 2> nods.")
    assert sc.content_fingerprint(base) == sc.content_fingerprint(varied)
    assert sc.content_fingerprint(base) != sc.content_fingerprint(base + " More.")


def test_add_camera_moves_reports_the_words_each_shot_dwells_on():
    pytest.importorskip("spacy")
    _o, rep = sc.add_camera_moves("Intro. [Shot 1] <Subject 1> touches <Subject 1>'s chin.")
    assert rep[0]["lemmas"] == ["chin"]


def test_refiner_wrapper_counts_prompts_and_uses_the_prior(store, monkeypatch):
    pytest.importorskip("spacy")
    import conditioning
    R = conditioning.FunPackVideoRefinerV2
    monkeypatch.setattr(conditioning, "refinement_state_path", lambda k, *a, **kw: str(store / f"{k}.j"))
    monkeypatch.setattr(fm, "_path", lambda k: str(store / f"{k}.j"))
    for i in range(3):
        R._v2_camera_moves(f"Intro. [Shot 1] <Subject 1> touches <Subject 1>'s chin{i and ' again' * i}.",
                           "prompt", 1, 1.0, "k")
    assert fm.prior("k").get("chin", 0) > 0
