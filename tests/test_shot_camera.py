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
