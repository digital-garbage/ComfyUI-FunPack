"""Camera moves for [Shot N] blocks. Fixtures are neutral stand-ins with the real shape:
<Subject N> tags everywhere, one shot per block, a music sentence at the end."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import shot_camera as sc  # noqa: E402

HEAD = ("Two friends in a sunlit loft, <Subject 1> is a tall woman with red hair, "
        "<Subject 2> is a man in a grey coat. Cinematic 35mm look. ")
S1 = "[Shot 1] <Subject 1> is waving <Subject 1>'s hand, greeting <Subject 2>. "
S2 = "[Shot 2] <Subject 1> bites <Subject 1>'s lower lip while <Subject 2> smiles at <Subject 1>. "
S3 = "[Shot 3] <Subject 2> hands <Subject 1> a glass of wine across the table. <Subject 1> laughs. "
MUSIC = "Upbeat jazz music with a soft bassline plays throughout."
PROMPT = HEAD + S1 + S2 + S3 + MUSIC


def test_no_shot_markers_leaves_the_prompt_alone():
    assert sc.add_camera_moves("A woman walks. Music plays.") == ("A woman walks. Music plays.", [])


def test_move_aims_at_a_part_of_the_subject_never_the_subject():
    pytest.importorskip("spacy")
    out, rep = sc.add_camera_moves(PROMPT)
    assert rep[0]["target"] == "<Subject 1>'s hand"
    assert rep[1]["target"] == "<Subject 1>'s lower lip"
    for r in rep:
        assert r["move"] and "<Subject 1>." not in r["move"].replace("'s", "")
    assert rep[0]["move"] in out and "<Subject 1>'s hand" in rep[0]["move"]


def test_moves_sit_inside_their_shot_and_before_the_music():
    pytest.importorskip("spacy")
    out, rep = sc.add_camera_moves(PROMPT)
    assert out.startswith(HEAD + "[Shot 1]")                 # intro untouched
    assert out.endswith(" " + MUSIC)                         # music untouched, last
    assert out.index(rep[2]["move"]) < out.index(MUSIC)
    assert out.index("[Shot 2]") < out.index(rep[1]["move"]) < out.index("[Shot 3]")


def test_a_shot_with_a_move_already_is_left_alone():
    pytest.importorskip("spacy")
    p = HEAD + "[Shot 1] The camera zooms in on <Subject 1>'s hand as it waves. " + S2
    out, rep = sc.add_camera_moves(p)
    assert rep[0]["move"] is None and "already" in rep[0]["why"]
    assert "[Shot 1] The camera zooms in on <Subject 1>'s hand as it waves. [Shot 2]" in out


def test_a_thing_in_most_shots_is_not_a_target():
    pytest.importorskip("spacy")
    p = (HEAD + "[Shot 1] <Subject 1> holds a rose. [Shot 2] <Subject 1> smells the rose. "
         "[Shot 3] <Subject 2> hugs <Subject 1>'s shoulder near the rose.")
    _out, rep = sc.add_camera_moves(p)
    assert all(r["target"] != "the rose" for r in rep)
    assert rep[2]["target"] == "<Subject 1>'s shoulder"


def test_a_shot_with_only_subjects_gets_nothing():
    pytest.importorskip("spacy")
    _out, rep = sc.add_camera_moves(HEAD + "[Shot 1] <Subject 1> laughs with <Subject 2>. "
                                    "[Shot 2] <Subject 2> nods at <Subject 1>.")
    assert [r["move"] for r in rep] == [None, None]


def test_the_same_move_does_not_repeat_between_shots():
    pytest.importorskip("spacy")
    p = HEAD + "[Shot 1] <Subject 1> raises <Subject 1>'s hand. [Shot 2] <Subject 2> lifts <Subject 2>'s chin."
    _out, rep = sc.add_camera_moves(p)
    assert rep[0]["move"].split(" ")[1:3] != rep[1]["move"].split(" ")[1:3]


def test_without_spacy_the_rules_still_find_owned_parts(monkeypatch):
    monkeypatch.setattr(sc, "_nlp", None)
    monkeypatch.setattr(sc, "_tried", True)
    _out, rep = sc.add_camera_moves(PROMPT)
    assert rep[0]["target"] == "<Subject 1>'s hand"


def test_tags_survive_the_round_trip():
    pytest.importorskip("spacy")
    out, _rep = sc.add_camera_moves(PROMPT)
    assert out.count("<Subject 1>") >= PROMPT.count("<Subject 1>")
    assert not any(n in out for n in sc.NAMES)


def test_refiner_wrapper_returns_the_prompt_and_says_what_it_did(capsys):
    pytest.importorskip("spacy")
    import conditioning
    out = conditioning.FunPackVideoRefinerV2._v2_camera_moves(PROMPT, "prompt")
    assert "The camera" in out and "<Subject 1>'s hand" in out
    assert "camera moves: Active" in capsys.readouterr().out
    assert conditioning.FunPackVideoRefinerV2._v2_camera_moves("no shots here", "prompt") == "no shots here"


def test_refiner_wrapper_never_breaks_a_run(monkeypatch, capsys):
    import conditioning
    monkeypatch.setattr(sc, "add_camera_moves", lambda t: 1 / 0)
    assert conditioning.FunPackVideoRefinerV2._v2_camera_moves(PROMPT, "prompt") == PROMPT
    assert "camera moves: failed" in capsys.readouterr().out


def test_two_topics_in_a_shot_become_a_move_from_the_first_to_the_last():
    pytest.importorskip("spacy")
    p = HEAD + "[Shot 1] <Subject 1> touches <Subject 1>'s chin, then reaches for the lamp."
    _out, rep = sc.add_camera_moves(p)
    assert "<Subject 1>'s chin" in rep[0]["move"] and "the lamp" in rep[0]["move"]
    assert rep[0]["move"].index("chin") < rep[0]["move"].index("lamp")
    assert " from " in rep[0]["move"]


def test_three_topics_go_from_the_first_to_the_last():
    pytest.importorskip("spacy")
    p = (HEAD + "[Shot 1] <Subject 1> touches <Subject 1>'s chin, picks up the lamp, "
         "then opens the window.")
    _out, rep = sc.add_camera_moves(p)
    assert "chin" in rep[0]["move"] and "window" in rep[0]["move"] and "lamp" not in rep[0]["move"]


TWO = HEAD + "[Shot 1] <Subject 1> touches <Subject 1>'s chin, then reaches for the lamp."
ONE = HEAD + "[Shot 1] <Subject 1> touches <Subject 1>'s chin."


def test_varied_moves_are_reproducible_from_the_seed():
    pytest.importorskip("spacy")
    assert sc.add_camera_moves(PROMPT, seed=7) == sc.add_camera_moves(PROMPT, seed=7)
    assert len({sc.add_camera_moves(PROMPT, seed=s)[0] for s in range(20)}) > 5


def test_chance_zero_leaves_every_shot_alone_and_says_why():
    pytest.importorskip("spacy")
    out, rep = sc.add_camera_moves(PROMPT, seed=1, chance=0.0)
    assert out == PROMPT and all("chance" in r["why"] for r in rep)


def test_no_single_pattern_is_every_prompt():
    """Over many seeds a shot with two topics gets one, two and three moves, some travelling
    X -> Y and some not: 'zoom in, move to Y, zoom out' is never the rule."""
    pytest.importorskip("spacy")
    sentences, travel = set(), set()
    for seed in range(200):
        _out, rep = sc.add_camera_moves(TWO, seed=seed)
        sentences.add(rep[0]["move"].count(". ") + 1)
        travel.add(" from " in rep[0]["move"])
    assert sentences == {1, 2, 3} and travel == {True, False}


def test_a_single_topic_shot_can_still_chain_with_a_pull_back():
    pytest.importorskip("spacy")
    moves = {sc.add_camera_moves(ONE, seed=s)[1][0]["move"] for s in range(60)}
    assert any("Then the camera" in m for m in moves) and any("Then" not in m for m in moves)
    assert all(m.startswith("The camera") for m in moves)


def test_a_shot_that_opens_with_a_cut_is_not_taken_for_having_a_camera_move():
    """H3 writes cuts at a shot's opening: "At 00:03.500, the camera cuts to ..."."""
    pytest.importorskip("spacy")
    p = (HEAD + "[Shot 1] <Subject 1> raises <Subject 1>'s hand. "
         "[Shot 2] At 00:03.500, the camera cuts to <Subject 2> who lifts <Subject 2>'s chin.")
    _out, rep = sc.add_camera_moves(p)
    assert rep[1]["move"] and "already" not in rep[1]["why"]
    p2 = HEAD + "[Shot 1] <Subject 1> raises <Subject 1>'s hand as the camera pushes in slowly."
    assert "already" in sc.add_camera_moves(p2)[1][0]["why"]


@pytest.mark.parametrize("text", [
    "<Subject 1> tilts <Subject 1>'s head up and down.",
    "<Subject 1> rolls onto <Subject 1>'s side and pulls out a knife.",
    "<Subject 1> looks at the camera and smiles, pushes in the last piece.",
    "<Subject 1>'s back arcs, then <Subject 2> focuses on the screen.",
    "<Subject 1> starts to pull out slowly and pushes in again, then will push in once more.",
    "Pull out slowly. Push in deeper.",
    "<Subject 1> moves up and down, then pans <Subject 1>'s gaze across the room.",
])
def test_body_movement_and_looking_at_the_camera_are_not_camera_moves(text):
    assert not sc.CAMERA.search(sc.CUT.sub("", text))


@pytest.mark.parametrize("text", [
    "The camera pushes in slowly toward <Subject 1>.",
    "Camera pans left across the room.",
    "Slow zoom in on <Subject 1>'s face.",
    "<Subject 1> waves. Pan right to reveal the door.",
    "A tracking shot follows <Subject 1>.",
    "Rack focus to the window.",
    "The camera slowly drifts upward.",
    "The camera pulls back to a wider view.",
    "The camera pushes in toward <Subject 1>.",
])
def test_real_camera_moves_are_still_found(text):
    assert sc.CAMERA.search(sc.CUT.sub("", text))


def test_a_cut_alone_is_not_a_camera_move():
    assert not sc.CAMERA.search(sc.CUT.sub("", "At 00:03.500, the camera cuts to <Subject 2>."))


def test_a_noun_shared_by_two_of_three_shots_is_still_a_target():
    """Only a noun in (nearly) every shot is a constant. Two of three is a shared topic."""
    pytest.importorskip("spacy")
    p = (HEAD + "[Shot 1] <Subject 1> holds the rose near the door. "
         "[Shot 2] <Subject 2> smells the rose beside a window. "
         "[Shot 3] <Subject 1> closes the door behind a curtain.")
    _out, rep = sc.add_camera_moves(p)
    assert all(r["move"] for r in rep), [r["why"] for r in rep]


def test_a_noun_in_every_shot_is_still_a_constant():
    pytest.importorskip("spacy")
    p = (HEAD + "[Shot 1] <Subject 1> holds the rose. [Shot 2] <Subject 2> smells the rose. "
         "[Shot 3] <Subject 1> drops the rose.")
    assert [r["move"] for r in sc.add_camera_moves(p)[1]] == [None, None, None]


def test_the_final_prompt_is_published_for_the_composer_with_the_enhancer_off():
    import conditioning
    import run_phase
    node = conditioning.FunPackVideoRefinerV2.__new__(conditioning.FunPackVideoRefinerV2)
    node._v2_enhanced_prompts = []
    run_phase.reset_enhanced()
    node._v2_note_final("before", "after with moves", 1)
    item = node._v2_enhanced_prompts[0]
    assert item["kind"] == "final" and item["scene"] == 1
    assert (item["before"], item["after"]) == ("before", "after with moves")
    assert run_phase._state()["enhanced"]["items"][0]["after"] == "after with moves"


PA = "<Subject 1> touches <Subject 1>'s chin and smiles."
PB = "Then <Subject 2> opens the window and the curtain sways."
PC = "<Subject 1> laughs and pours a glass of wine."
PIECES = [PA, PB, PC]
CUTS = ("Two friends in a sunlit loft, <Subject 1> is a tall woman, <Subject 2> is a man. "
        f"[Shot 1] {PA} {PB} [Shot 2] {PC} Upbeat jazz music plays.")


def test_a_shot_whose_point_changes_is_cut_in_two_with_whole_second_times():
    pytest.importorskip("spacy")
    out, info = sc.add_shot_cuts(CUTS, 12, seed=1, chance=1.0, pieces=PIECES)
    assert (info["before"], info["after"]) == (2, 3)
    assert info["times"] == ["00:04.000", "00:08.000"]
    assert "[Shot 3] At 00:08.000," in out and "[Shot 2] At 00:04.000," in out
    assert out.count("[Shot ") == 3 and out.endswith("Upbeat jazz music plays.")
    assert "Then <Subject 2>" not in out


def test_cut_times_rise_and_stay_inside_the_video():
    pytest.importorskip("spacy")
    for secs in (5, 9, 14, 30):
        _out, info = sc.add_shot_cuts(CUTS, secs, seed=2, chance=1.0, pieces=PIECES)
        ts = [int(t[3:5]) + 60 * int(t[:2]) for t in info["times"]]
        assert ts == sorted(set(ts)) and all(0 < t < secs for t in ts)


def test_no_length_or_existing_times_or_no_room_leave_the_prompt_alone():
    pytest.importorskip("spacy")
    assert sc.add_shot_cuts(CUTS, None)[0] == CUTS and "length" in sc.add_shot_cuts(CUTS, None)[1]["why"]
    assert sc.add_shot_cuts(CUTS, 1)[0] == CUTS
    stamped = CUTS.replace("[Shot 2] ", "[Shot 2] At 00:05.000, the camera cuts to ")
    out, info = sc.add_shot_cuts(stamped, 12, chance=1.0, pieces=PIECES)
    assert out == stamped and "already" in info["why"]
    assert sc.add_shot_cuts("no shots", 10)[0] == "no shots"


def test_chance_zero_still_stamps_the_existing_shots_but_splits_nothing():
    pytest.importorskip("spacy")
    out, info = sc.add_shot_cuts(CUTS, 10, seed=1, chance=0.0)
    assert info["after"] == 2 and len(info["times"]) == 1 and "[Shot 2] At 00:" in out


def test_shot_cuts_are_reproducible_from_the_seed():
    pytest.importorskip("spacy")
    assert (sc.add_shot_cuts(CUTS, 12, seed=5, chance=1.0, pieces=PIECES)
            == sc.add_shot_cuts(CUTS, 12, seed=5, chance=1.0, pieces=PIECES))


def test_refiner_wrapper_reports_cuts(capsys, monkeypatch):
    pytest.importorskip("spacy")
    import conditioning
    monkeypatch.setattr(conditioning, "_shortcut_texts", lambda: PIECES)
    R = conditioning.FunPackVideoRefinerV2
    out = R._v2_shot_cuts(CUTS, "scene 1", 1, 1.0, 12)
    assert "shot cuts: Active" in capsys.readouterr().out and "00:04.000" in out
    assert R._v2_shot_cuts(CUTS, "prompt", 1, 1.0, None) == CUTS
    assert "shot cuts: Inactive" in capsys.readouterr().out


def test_views_use_the_trusted_words_skip_shot_one_and_never_repeat():
    P = HEAD + "[Shot 1] <Subject 1> waves. [Shot 2] <Subject 2> nods. [Shot 3] <Subject 1> smiles. [Shot 4] <Subject 2> sits."
    for seed in range(30):
        out, added = sc.add_shot_views(P, seed=seed, chance=1.0)
        assert [a["shot"] for a in added] == [2, 3, 4]
        assert all(a["view"] in sc.VIEWS for a in added)
        assert all(a["view"] != b["view"] for a, b in zip(added, added[1:]))
        assert out.startswith(HEAD + "[Shot 1] <Subject 1> waves. [Shot 2]")
    assert len({tuple(a["view"] for a in sc.add_shot_views(P, seed=s, chance=1.0)[1]) for s in range(40)}) > 5


def test_views_respect_chance_zero_and_a_view_already_stated():
    P = HEAD + "[Shot 1] <Subject 1> waves. [Shot 2] Front view. <Subject 2> nods. [Shot 3] <Subject 1> smiles."
    assert sc.add_shot_views(P, seed=1, chance=0.0) == (P, [])
    _out, added = sc.add_shot_views(P, seed=1, chance=1.0)
    assert [a["shot"] for a in added] == [3]
    assert sc.add_shot_views("[Shot 1] only one.", seed=1, chance=1.0)[1] == []


def test_a_view_goes_after_the_cut_opener_and_before_the_action():
    P = HEAD + "[Shot 1] <Subject 1> waves. [Shot 2] At 00:04.000, the camera cuts to a new angle. <Subject 2> nods. " + MUSIC
    out, added = sc.add_shot_views(P, seed=3, chance=1.0)
    v = added[0]["view"]
    assert f"a new angle. {v}. <Subject 2> nods." in out and out.endswith(MUSIC)


def test_refiner_wrapper_reports_views(capsys):
    import conditioning
    P = HEAD + "[Shot 1] <Subject 1> waves. [Shot 2] <Subject 2> nods."
    out = conditioning.FunPackVideoRefinerV2._v2_shot_views(P, "scene 1", 1, 1.0)
    assert "shot views: Active" in capsys.readouterr().out and out != P


def test_a_shot_is_never_cut_inside_one_shortcut():
    """The topic changes between PA and PB. If PA+PB is ONE shortcut, nothing may be split."""
    pytest.importorskip("spacy")
    one = f"{PA} {PB}"
    out, info = sc.add_shot_cuts(CUTS, 12, seed=1, chance=1.0, pieces=[one, PC])
    assert info["after"] == 2 and out.count("[Shot ") == 2
    out, info = sc.add_shot_cuts(CUTS, 12, seed=1, chance=1.0)          # no shortcut texts known
    assert info["after"] == 2


def test_shortcut_texts_come_from_the_library(monkeypatch):
    import conditioning, templates
    monkeypatch.setattr(templates, "load_shortcut_db", lambda: {"shortcuts": {
        "a": {"enabled": True, "replacements": ["A long enough replacement text."]},
        "b": {"enabled": False, "replacements": ["Disabled but long enough text."]},
        "c": {"enabled": True, "replacements": ["short"]}}})
    assert conditioning._shortcut_texts() == ["A long enough replacement text."]


@pytest.mark.parametrize("view", sc.VIEWS)
def test_a_view_is_not_a_camera_move(view):
    assert not sc.CAMERA.search(sc.CUT.sub("", view + "."))


def test_a_shot_that_was_given_a_view_still_gets_its_move():
    pytest.importorskip("spacy")
    p = HEAD + "[Shot 1] <Subject 1> waves. [Shot 2] <Subject 2> touches <Subject 2>'s chin."
    for seed in range(12):
        viewed, added = sc.add_shot_views(p, seed=seed, chance=1.0)
        assert added
        _out, rep = sc.add_camera_moves(viewed, seed=seed, chance=1.0)
        assert rep[1]["move"], (added, rep[1]["why"])


def test_cut_times_split_the_scene_evenly_with_no_tiny_shots():
    """Text-length weighting gave 00:03 / 00:04 / 00:09 over 15.08s: a one-second shot."""
    pytest.importorskip("spacy")
    P = HEAD + "[Shot 1] <Subject 1> waves. [Shot 2] <Subject 2> nods. [Shot 3] <Subject 1> smiles. [Shot 4] <Subject 2> sits."
    _out, info = sc.add_shot_cuts(P, 15.0833)
    secs = [int(t[3:5]) for t in info["times"]]
    assert secs == [4, 8, 11]
    edges = [0] + secs + [15.0833]
    assert all(b - a >= 2 for a, b in zip(edges, edges[1:]))
    assert sc.add_shot_cuts(P, 8)[1]["times"] == ["00:02.000", "00:04.000", "00:06.000"]
    assert sc.add_shot_cuts(P, 4)[1]["times"] == ["00:01.000", "00:02.000", "00:03.000"]


def test_the_phrases_we_write_never_say_the_camera_pulls_out():
    """Only what the rewriter writes; a shortcut may say it as an action."""
    for group in (sc.MOVES_DETAIL, sc.MOVES_OTHER, sc.MOVES_TRAVEL, sc.MOVES_THEN, sc.MOVES_FINISH):
        assert not any("pulls out" in m for m in group), group


def test_words_the_rewriter_wrote_never_become_targets():
    """'The camera pans from the shot transitions to the tip': the cut opener was read as
    the shot's own content."""
    pytest.importorskip("spacy")
    P = (HEAD + "[Shot 1] <Subject 1> waves. [Shot 2] At 00:04.000, the shot transitions to the next "
         "moment. Side view. <Subject 2> touches <Subject 2>'s chin, then reaches for the lamp.")
    _out, rep = sc.add_camera_moves(P, seed=1, chance=1.0)
    m = rep[1]["move"] or ""
    assert "transition" not in m and "moment" not in m and "view" not in m.lower(), m
    assert "chin" in m or "lamp" in m


def test_the_full_pipeline_output_has_no_own_words_in_a_move():
    pytest.importorskip("spacy")
    P = (HEAD + f"[Shot 1] {PA} [Shot 2] {PB} {PC} " + MUSIC)
    for seed in range(20):
        a, _ = sc.add_shot_cuts(P, 12, seed=seed, chance=1.0, pieces=PIECES)
        b, _ = sc.add_shot_views(a, seed=seed, chance=1.0)
        _out, rep = sc.add_camera_moves(b, seed=seed, chance=1.0)
        for r in rep:
            t = (r["target"] or "").lower()
            assert not any(w in t for w in ("transition", "moment", "beat", "new angle", "new view",
                                            "switch", "changes")), t


def test_a_relational_noun_is_kept_with_what_it_is_part_of_or_dropped():
    pytest.importorskip("spacy")
    c = {x[0]: x[1] for x in sc.candidates(" <Subject 1> reaches for the base of the lamp.")}
    assert c.get("base") == "base of the lamp"
    assert "base" not in {x[0] for x in sc.candidates(" <Subject 1> looks at the base, then leaves.")}


def test_an_unowned_body_part_needs_a_single_owner():
    pytest.importorskip("spacy")
    one = {x[0]: x[1] for x in sc.candidates(" <Subject 1> touches the skin.")}
    assert one.get("skin") == "<Subject 1>'s skin" or "Mariel's skin" in one.values()
    two = {x[0] for x in sc.candidates(" <Subject 1> touches the skin while <Subject 2> watches.")}
    assert "skin" not in two


def test_a_move_never_ends_on_a_dangling_word():
    pytest.importorskip("spacy")
    P = HEAD + "[Shot 1] <Subject 1> touches the base, then the skin, while <Subject 2> watches."
    _out, rep = sc.add_camera_moves(P, seed=1, chance=1.0)
    assert rep[0]["move"] is None


def test_some_shots_hold_still_on_one_object_and_pans_are_the_minority():
    """Not every shot is a move: a static focus is part of the mix, and X -> Y is rarer than it."""
    pytest.importorskip("spacy")
    kinds = {"hold": 0, "pan": 0, "other": 0}
    for seed in range(300):
        _o, rep = sc.add_camera_moves(TWO, seed=seed)
        m = rep[0]["move"]
        if "holds steady" in m or "stays still" in m or "stays fixed" in m:
            kinds["hold"] += 1
        elif " from " in m and " to " in m and " Then" not in m:
            kinds["pan"] += 1
        else:
            kinds["other"] += 1
    assert kinds["hold"] > 20 and 0 < kinds["pan"] < kinds["other"], kinds
    assert kinds["pan"] < 0.3 * 300, kinds


def test_a_body_action_from_behind_is_not_a_stated_view():
    for text in ("She is taken from behind by him.", "Water drips from above onto her.",
                 "He enters from below the stairs."):
        assert not sc.VIEW_STATED.search(text), text
    for text in ("POV view, she waves.", "Seen from behind, she walks.", "Shot from above, the room.",
                 "Side view of the table.", "The camera films from below."):
        assert sc.VIEW_STATED.search(text), text
    out, added = sc.add_shot_views("[Shot 1] A.\n[Shot 2] She is taken from behind by him.", seed=1, chance=1.0)
    assert added and added[0]["shot"] == 2
