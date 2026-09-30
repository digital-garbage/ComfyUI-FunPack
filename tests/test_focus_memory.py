"""Cross-prompt word frequency as a second score for where the camera aims."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import focus_memory as fm  # noqa: E402
import shot_camera as sc  # noqa: E402


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(fm, "_path", lambda: str(tmp_path / "focus_memory.json"))
    return tmp_path


def test_silent_until_enough_prompts_then_more_seen_means_more_bonus(store):
    assert fm.prior() == {}
    fm.observe("a", ["chin", "lamp"])
    fm.observe("b", ["chin"])
    assert fm.prior() == {}                                   # two prompts: not a habit yet
    fm.observe("c", ["chin", "door"])
    p = fm.prior()
    assert p["chin"] == pytest.approx(fm.FREQ_W) and 0 < p["lamp"] < p["chin"]
    assert p["lamp"] == p["door"]


def test_the_same_prompt_again_counts_nothing(store):
    for _ in range(5):
        fm.observe("same", ["chin"])
    assert fm.observe("same", ["chin"]) == 1 and fm.prior() == {}


def test_nothing_to_count_keeps_nothing(store):
    assert fm.observe("a", []) == 0 and fm.prior() == {}


def test_the_memory_is_its_own_file_named_by_the_environment_or_the_user_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("SHOT_CAMERA_MEMORY", str(tmp_path / "mine.json"))
    assert fm._path() == str(tmp_path / "mine.json")
    fm.observe("a", ["chin"])
    assert (tmp_path / "mine.json").is_file()
    monkeypatch.delenv("SHOT_CAMERA_MEMORY")
    assert fm._path().endswith("focus_memory.json") and "shot_camera" in fm._path()


def test_shot_camera_and_focus_memory_need_nothing_from_the_host_pack():
    import re
    for name in ("shot_camera.py", "focus_memory.py"):
        src = (Path(__file__).resolve().parents[1] / name).read_text(encoding="utf-8")
        assert not re.search(r"^\s*(?:from|import)\s+(?:conditioning|samplers|templates|movie_editor)", src, re.M)


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
    monkeypatch.setattr(fm, "_path", lambda: str(store / "focus_memory.json"))
    for i in range(3):
        R._v2_camera_moves(f"Intro. [Shot 1] <Subject 1> touches <Subject 1>'s chin{i and ' again' * i}.",
                           "prompt", 1, 1.0)
    assert fm.prior().get("chin", 0) > 0


def test_a_habit_does_not_win_every_shot_it_merely_appears_in():
    """100 videos with bananas, then a shot with a banana AND a cherry: the cherry still gets
    the camera some of the time, and only between equals."""
    pytest.importorskip("spacy")
    P = "Intro. [Shot 1] <Subject 1> holds a banana and a cherry."
    heard = {"banana": 0, "cherry": 0}
    for seed in range(200):
        _o, rep = sc.add_camera_moves(P, seed=seed, chance=1.0, prior={"banana": fm.FREQ_W})
        t = rep[0]["target"]
        heard["banana"] += "banana" in t
        heard["cherry"] += "cherry" in t
    assert heard["banana"] > heard["cherry"] > 20, heard          # favoured, not decisive


def test_a_habit_never_beats_a_real_target():
    pytest.importorskip("spacy")
    P = "Intro. [Shot 1] <Subject 1> touches <Subject 1>'s chin and glances at a banana."
    for seed in range(60):
        _o, rep = sc.add_camera_moves(P, seed=seed, chance=1.0, prior={"banana": fm.FREQ_W})
        assert "chin" in rep[0]["target"], rep[0]["target"]


def test_a_word_the_shot_keeps_returning_to_is_what_it_is_about():
    pytest.importorskip("spacy")
    P = "Intro. [Shot 1] <Subject 1> lifts the cherry, then the lamp, then looks at the cherry again."
    hits = sum("cherry" in sc.add_camera_moves(P, seed=s, chance=1.0)[1][0]["target"] for s in range(80))
    assert hits > 50, hits


def test_choices_reinforce_what_was_picked_and_mark_down_what_was_replaced(store):
    fm.learn([{"auto": "lamp", "picked": "chin", "mode": "auto"}] * 3)
    p = fm.prior()
    assert p["chin"] > 0.5 > 0 > p["lamp"]
    fm.learn([{"auto": "lamp", "picked": "lamp", "mode": "auto"}])
    assert fm.prior()["lamp"] > p["lamp"] - 1                          # agreeing does not punish


def test_no_move_counts_against_moves_not_against_words(store):
    fm.learn([{"auto": "chin", "picked": None, "mode": "none"}] * 12)
    assert fm.prior() == {}
    assert fm.effective_chance(0.7) < 0.4
    assert fm.effective_chance(1.0) < 0.6


def test_the_chance_is_left_alone_until_enough_shots_were_reviewed(store):
    fm.learn([{"auto": "chin", "picked": None, "mode": "none"}] * 3)
    assert fm.effective_chance(0.7) == 0.7


def test_a_chosen_target_and_mode_are_obeyed():
    pytest.importorskip("spacy")
    P = "Intro. [Shot 1] <Subject 1> raises the lamp and lowers the mirror."
    opts = sc.focus_options(P)[0]
    key = opts["key"]
    assert {c["lemma"] for c in opts["candidates"]} >= {"lamp", "mirror"}
    for seed in range(10):
        _o, rep = sc.add_camera_moves(P, seed=seed, chance=1.0,
                                      choices={key: {"mode": "hold", "lemma": "mirror"}})
        assert "mirror" in rep[0]["move"] and ("holds" in rep[0]["move"] or "stays" in rep[0]["move"])
        _o, rep = sc.add_camera_moves(P, seed=seed, chance=1.0,
                                      choices={key: {"mode": "move", "lemma": "lamp"}})
        assert "lamp" in rep[0]["move"] and "mirror" not in rep[0]["move"] and " Then" not in rep[0]["move"]
        _o, rep = sc.add_camera_moves(P, seed=seed, chance=1.0, choices={key: {"mode": "none"}})
        assert rep[0]["move"] is None and "you chose" in rep[0]["why"]


def test_a_choice_survives_cut_openers_and_views_added_to_the_shot():
    pytest.importorskip("spacy")
    base = "Intro. [Shot 1] <Subject 1> waves. [Shot 2] <Subject 2> raises the lamp."
    viewed = ("Intro. [Shot 1] <Subject 1> waves. [Shot 2] At 00:04.000, the camera cuts to a new angle. "
              "Side view. <Subject 2> raises the lamp.")
    assert sc.focus_options(base)[1]["key"] == sc.focus_options(viewed)[1]["key"]


def test_focus_scene_texts_and_options_follow_the_generation_path(monkeypatch):
    """The review shows the text the camera-move step will see: shortcuts expanded, anchor and
    postfix folded in, one text per scene."""
    pytest.importorskip("spacy")
    import conditioning
    import templates
    db = {"shortcuts": {"k": {"name": "k", "enabled": True, "triggers": ["tip"],
                              "replacements": ["[Shot 1] <Subject 1> touches <Subject 1>'s chin and raises the lamp."],
                              "refinement_key": "", "category": "", "sub_category": ""}}}
    monkeypatch.setattr(templates, "load_shortcut_db", lambda: db)
    monkeypatch.setattr(templates, "load_custom_transition_triggers", lambda: {})
    texts = conditioning.focus_scene_texts(
        {"anchor": "Intro <Subject 1>.", "scenes": ["tip", "[Shot 1] <Subject 1> waves."], "postfix": ""},
        [], "start")
    assert len(texts) == 2 and texts[0].startswith("Intro <Subject 1>.")
    assert "touches <Subject 1>'s chin" in texts[0]
    opts = sc.focus_options(texts[0])
    assert opts[0]["candidates"] and opts[0]["auto_lemma"]
