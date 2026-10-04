"""Cross-prompt word frequency as a second score for where the camera aims."""
from pathlib import Path

import pytest

from modules.conditioning.shot_camera import memory as fm
from modules.conditioning.shot_camera import engine as sc


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


# ── rated, trait-aware views ─────────────────────────────────────────────────────────
# ── several picks per shot, and readable previews ────────────────────────────────────
def test_several_targets_in_order_become_one_travelling_move():
    P = "[Shot 1] A woman raises the lamp. Her hand holds a cup near the window."
    opts = sc.focus_options(P)[0]["candidates"]
    assert len(opts) >= 3
    a, b, c = opts[0]["lemma"], opts[1]["lemma"], opts[2]["lemma"]
    key = sc.shot_key(P.split("]", 1)[1])
    out, rep = sc.add_camera_moves(P, seed=1, chance=0.0,
                                   choices={key: {"mode": "move", "lemmas": [b, a, c]}})
    assert rep[0]["target"].count("->") == 2 and rep[0]["lemmas"]
    names = [t.strip() for t in rep[0]["target"].split("->")]
    assert names[0].endswith(next(o["text"] for o in opts if o["lemma"] == b).split()[-1])
    assert "from" in out and out.count("Then the camera") == 1
    one, rep1 = sc.add_camera_moves(P, seed=1, chance=0.0, choices={key: {"mode": "hold", "lemmas": [a, b]}})
    assert "->" not in rep1[0]["target"]                      # Hold aims at the first pick only


def test_shot_texts_name_what_each_shot_says():
    t = sc.shot_texts("[Shot 1] A cat sits.  [Shot 2] At 00:03.000, the camera cuts to a new angle. A dog runs.")
    assert t == {1: "A cat sits.", 2: "A dog runs."}


# ── ratings teach moves, styles, targets and splits ──────────────────────────────────
def test_ratings_teach_the_camera_its_own_choices(store):
    assert fm.effective_chance(0.5) == 0.5 and fm.split_chance(0.5) == 0.5 and fm.prior() == {}
    for _ in range(4):
        fm.rate_arms(["move:yes", "style:hold", "word:lamp", "split:no"], 1)
        fm.rate_arms(["move:no", "word:door", "split:yes"], -1)
    assert fm.effective_chance(0.5) > 0.5 and fm.split_chance(0.5) < 0.5
    p = fm.prior()
    assert p["lamp"] > 0 > p["door"]
    assert fm.split_chance(0.0) == 0.0                         # a chance the user turned off stays off
    assert fm.forget("word", "lamp") and "lamp" not in fm.prior()


def test_plan_reports_what_it_drew_and_a_liked_style_is_drawn_more():
    P = "[Shot 1] A woman raises the lamp. Her hand holds a cup near the window."
    rep = sc.add_camera_moves(P, seed=3, chance=1.0)[1][0]
    assert "move:yes" in rep["arms"] and any(a.startswith("word:") for a in rep["arms"])
    assert sc.add_camera_moves(P, seed=3, chance=0.0)[1][0]["arms"] == ["move:no"]
    forced = {sc.shot_key(P.split("]", 1)[1]): {"mode": "move"}}
    assert sc.add_camera_moves(P, seed=3, chance=1.0, choices=forced)[1][0]["arms"] == []   # a person's call is not learned
    like = {"style:hold": (20.0, 0.0), "style:k1": (0.0, 20.0)}
    holds = sum("style:hold" in sc.add_camera_moves(P, seed=s, chance=1.0, arms=like)[1][0]["arms"] for s in range(60))
    plain = sum("style:hold" in sc.add_camera_moves(P, seed=s, chance=1.0)[1][0]["arms"] for s in range(60))
    assert holds > plain

