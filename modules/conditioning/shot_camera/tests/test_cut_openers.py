"""Every cut opener is a hard cut that reads as a cut, never as a camera move."""

import random
import re

from .. import SETTINGS, engine, nodes


def test_every_opener_is_a_hard_cut_and_no_opener_reads_as_a_camera_move():
    for opener in engine.CUT_OPENERS:
        assert not engine.CAMERA.search(opener), opener
        assert not any(w in opener for w in ("dissolve", "fade", "wipe")), opener
        assert engine.CUT.search(opener), opener
        assert re.search(r"\b(?:cuts|transitions|changes|switches) to\b", opener), opener



def test_the_transition_slider_at_zero_is_always_a_plain_cut_and_at_one_always_a_mixed_transition():
    for seed in range(200):
        opener, kind = engine._opener(random.Random(seed), 0.0, None)
        assert kind is None and opener in engine.CUT_OPENERS
    last, kinds = None, []
    for seed in range(300):
        opener, last = engine._opener(random.Random(seed), 1.0, last)
        assert last is not None, "1 is never a plain cut"
        kinds.append(last)
    assert len(set(kinds)) == 3, "a mixture of dissolve, fade and wipe"
    assert all(a != b for a, b in zip(kinds, kinds[1:])), "never the same kind twice running"


def test_every_phrase_the_module_writes_after_a_cut_time_is_its_own_words_never_content():
    for opener in engine.OUR_OPENERS:
        assert not engine.CAMERA.search(opener), opener
        picture = f" At 00:05.000, {opener}. She waves at the camera."
        assert engine.own_words_removed(picture).strip() == "She waves at the camera.", opener
    prints = {engine.content_fingerprint(f"[Shot 1] A woman walks in. [Shot 2] At 00:05.000, {o}. She waves at the camera.")
              for o in engine.OUR_OPENERS}
    assert len(prints) == 1, "a re-roll of the cut wording is the same prompt"


DANCE = "She sways her hips from side to side, smiling at the room."
WAVE = "She raises her hand and waves at the camera with her fingers."
KINDS = ("dissolve", "fades", "wipes")


def _cut(monkeypatch, tmp_path, **values):
    monkeypatch.setenv("SHOT_CAMERA_MEMORY", str(tmp_path / "m.json"))
    settings = {"cut_same_shot": True, "shot_cuts_chance": 1, **values}
    return nodes.rewrite(f"[Shot 1] {DANCE} {WAVE} {DANCE}", settings, seconds=12, pieces=[DANCE, WAVE])[0]


def test_the_slider_default_is_zero_and_zero_leaves_every_cut_plain(monkeypatch, tmp_path):
    assert SETTINGS["cut_difference_chance"]["default"] == 0.0
    assert engine.add_shot_cuts.__defaults__[-1] == 0.0, "no transition unless asked"
    for values in ({}, {"cut_difference_chance": 0}):
        out = _cut(monkeypatch, tmp_path, **values)
        assert out.count("At 00:") == 2 and not any(k in out for k in KINDS), out


def test_the_slider_set_in_the_node_makes_every_cut_a_transition_and_they_are_mixed(monkeypatch, tmp_path):
    out = _cut(monkeypatch, tmp_path, cut_difference_chance=1)
    openers = [o for o in engine.OUR_OPENERS for _ in range(out.count(o))]
    assert out.count("At 00:") == 2 and len(openers) == 2, out
    kinds = [dict((p, k) for k, p in engine.TRANSITIONS).get(o) for o in openers]
    assert None not in kinds, "every cut is a transition at 1"
    assert kinds[0] != kinds[1], "mixed: never the same kind twice running"


def test_a_transition_the_user_wrote_is_not_given_a_second_opener(monkeypatch, tmp_path):
    monkeypatch.setenv("SHOT_CAMERA_MEMORY", str(tmp_path / "m.json"))
    text = f"[Shot 1] {DANCE} [Shot 2] The shot cross-dissolves to her hands. {WAVE}"
    out, _ = engine.add_shot_cuts(text, 10, seed=1, chance=0, pieces=[DANCE, WAVE])
    assert out.count("cross-dissolves") == 1 and sum(out.count(o) for o in engine.OUR_OPENERS) == 0, out
    assert "At 00:" in out and "At 00:05.000, the shot cross-dissolves to her hands." in out, out


def test_a_move_is_never_aimed_at_a_transition_word(monkeypatch, tmp_path):
    for variation in range(20):
        out = _cut(monkeypatch, tmp_path, cut_difference_chance=1, camera_moves=True, camera_moves_chance=1, variation=variation)
        for sentence in out.replace("\n", " ").split(". "):
            if "camera" in sentence and "At 00:" not in sentence:
                assert not any(w in sentence for w in ("black", "white", "angle", "moment", "follows")), sentence


import pytest


@pytest.mark.parametrize("phrase", ["The shot fades to black.", "The shot fades out to black.", "The shot dissolves to her hand.",
                                    "The camera wipes across to her hand.", "The shot cross-dissolves to her hand.",
                                    "Fade to black.", "Cut to her hand.", "Dissolve to her hand.", "A cross-dissolve to her hand.",
                                    "Smash cut to her hand.", "The shot cross-fades to her hand.", "Then the camera cuts to her hand.",
                                    "The camera then cuts to her hand.", "The shot slowly fades to black.",
                                    "The shot quickly wipes across to her hand.", "The shot dissolves into her hand.",
                                    "The scene fades to black."])
def test_the_cut_phrases_the_hint_names_are_kept_without_a_second_opener(monkeypatch, tmp_path, phrase):
    monkeypatch.setenv("SHOT_CAMERA_MEMORY", str(tmp_path / "m.json"))
    out, _ = engine.add_shot_cuts(f"[Shot 1] {DANCE} [Shot 2] {phrase} {WAVE}", 10, seed=1, chance=0, pieces=[DANCE, WAVE], transition=1.0)
    assert sum(out.count(o) for o in engine.OUR_OPENERS) == 0 and phrase.lower()[:12] in out.lower(), out


def test_the_time_before_a_user_written_cut_is_not_part_of_the_prompts_identity(monkeypatch, tmp_path):
    monkeypatch.setenv("SHOT_CAMERA_MEMORY", str(tmp_path / "m.json"))
    text = f"[Shot 1] {DANCE} [Shot 2] The shot cross-dissolves to her hands. {WAVE}"
    prints = {engine.content_fingerprint(engine.add_shot_cuts(text, secs, seed=1, chance=0, pieces=[DANCE, WAVE])[0]) for secs in (12, 20, 30)}
    assert prints == {engine.content_fingerprint(text)}, "one prompt however long the video"


def test_between_zero_and_one_the_share_of_transitions_follows_the_slider():
    for difference in (0.1, 0.5, 0.9):
        hits = sum(engine._opener(random.Random(seed), difference, None)[1] is not None for seed in range(800))
        assert abs(hits / 800 - difference) < 0.08, difference


def test_each_cut_rolls_for_itself_so_a_shot_can_mix_plain_and_transition_cuts(monkeypatch, tmp_path):
    monkeypatch.setenv("SHOT_CAMERA_MEMORY", str(tmp_path / "m.json"))
    text = f"[Shot 1] {DANCE} {WAVE} {DANCE} {WAVE}"
    mixes = 0
    for seed in range(40):
        out, _ = engine.add_shot_cuts(text, 16, seed=seed, chance=1, pieces=[DANCE, WAVE], transition=0.5)
        plain = sum(out.count(o) for o in engine.CUT_OPENERS)
        mixes += plain > 0 and sum(out.count(p) for _, p in engine.TRANSITIONS) > 0
    assert mixes > 5


@pytest.mark.parametrize("stamp", ["At 00:04.000, the camera cuts to the door.", "At 00:03, the camera cuts to her hand.",
                                   "At 0:03.5, the shot cuts to her hand."])
def test_cut_times_the_user_wrote_anywhere_leave_the_prompt_alone(monkeypatch, tmp_path, stamp):
    monkeypatch.setenv("SHOT_CAMERA_MEMORY", str(tmp_path / "m.json"))
    text = f"[Shot 1] {DANCE} [Shot 2] {WAVE} " + "and she keeps going " * 6 + stamp
    out, info = engine.add_shot_cuts(text, 12, seed=1, chance=1, pieces=[DANCE, WAVE], transition=1.0)
    assert out == text and "already written" in info["why"], stamp


def test_the_status_line_says_which_transitions_were_used(monkeypatch, tmp_path):
    monkeypatch.setenv("SHOT_CAMERA_MEMORY", str(tmp_path / "m.json"))
    said = nodes.rewrite(f"[Shot 1] {DANCE} {WAVE} {DANCE}", {"cut_same_shot": True, "shot_cuts_chance": 1, "cut_difference_chance": 1},
                         seconds=12, pieces=[DANCE, WAVE])[1]
    assert "transitions: " in said and any(k in said for k in ("dissolve", "fade", "wipe")), said
    plain = nodes.rewrite(f"[Shot 1] {DANCE} {WAVE} {DANCE}", {"cut_same_shot": True, "shot_cuts_chance": 1}, seconds=12, pieces=[DANCE, WAVE])[1]
    assert "transitions" not in plain
