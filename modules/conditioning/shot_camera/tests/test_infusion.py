"""Focus infusion: a camera focus now and then aims at a word the user's prompts keep returning to."""

import pytest

from .. import engine, memory, nodes

WALK = "She walks down the street, her legs moving steadily."
CAMERA_ON = {"camera_moves": True, "camera_moves_chance": 1}


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOT_CAMERA_MEMORY", str(tmp_path / "m.json"))


def _habit(*lemmas, prompts=3):
    for k in range(prompts):
        memory.observe(f"p{k}", list(lemmas), "P")


def _move(**values):
    out, said, chose = nodes.rewrite(f"[Shot 1] {WALK}", {**CAMERA_ON, **values}, seconds=5, pieces=[], project="P")
    return out, said, chose["arms"]


def test_at_zero_nothing_changes_even_with_a_habit_on_record():
    _habit("hand", "head")
    base = _move()[0]
    assert _move(focus_infusion_chance=0)[0] == base and "hand" not in base and "head" not in base


def test_at_one_the_focus_is_a_word_from_other_prompts_not_the_shots_own():
    _habit("hand", "head")
    for variation in range(15):
        out, said, arms = _move(focus_infusion_chance=1, variation=variation)
        sentence = out.split(WALK)[1]
        assert "hand" in sentence or "head" in sentence, out
        assert "legs" not in sentence and "street" not in sentence, "instead of the shot's own focus"
        assert "infuse:yes" in arms and any(a in arms for a in ("infused:hand", "infused:head"))
        assert "infused:" in said or "infused" in said


def test_nothing_is_infused_before_there_is_a_habit_or_from_a_word_seen_in_one_prompt():
    out, _, arms = _move(focus_infusion_chance=1)
    assert "infuse:yes" not in arms
    _habit("hand", prompts=1)
    assert memory.infusion_bank("P") == []
    _habit("hand", prompts=3)
    assert dict(memory.infusion_bank("P")).keys() == {"hand"}


def test_the_modules_own_words_are_never_infused():
    _habit("camera", "angle", "moment", "view", "zoom", "fade", "black")
    for variation in range(10):
        _, _, arms = _move(focus_infusion_chance=1, variation=variation)
        assert "infuse:yes" not in arms and not any(a.startswith("infused:") for a in arms)


def test_a_reframe_can_be_infused_too():
    _habit("hand")
    a, b = "She sways her hips from side to side.", "She turns and walks out through the door."
    out, info = engine.add_reframes(f"[Shot 1] {a} {b}", seed=1, chance=1, pieces=[a, b], infuse=1.0, bank=memory.infusion_bank("P"))
    assert "hand" in out.split(a)[1] and "infuse:yes" in info["arms"] and "infused:hand" in info["arms"]
    plain, plain_info = engine.add_reframes(f"[Shot 1] {a} {b}", seed=1, chance=1, pieces=[a, b])
    assert "infuse:yes" not in plain_info["arms"] and "hand" not in plain.split(a)[1]


def test_the_chance_follows_the_slider_exactly_at_the_ends_and_is_tilted_by_ratings_between(tmp_path):
    assert memory.infuse_chance(0.0) == 0.0 and memory.infuse_chance(1.0) == 1.0 and memory.infuse_chance(0.5) == 0.5
    memory.record_run("good", {"views": [], "arms": ["infuse:yes", "infused:hand"]})
    memory.on_rating("good", "liked")
    assert memory.infuse_chance(0.5) > 0.5 and memory.infuse_chance(1.0) == 1.0
    memory.on_rating("good", "liked")                      # same rating again changes nothing
    for k in range(8):
        memory.record_run(f"bad{k}", {"views": [], "arms": ["infuse:yes", "infused:hand"]})
        memory.on_rating(f"bad{k}", "disliked")
    assert 0.0 < memory.infuse_chance(0.5) < 0.5 and memory.infuse_chance(0.0) == 0.0


def test_a_dislike_blames_the_infused_shots_first_and_a_change_of_mind_takes_it_back():
    memory.record_run("r", {"views": [], "arms": ["move:yes", "style:hold", "word:legs", "infuse:yes", "infused:hand"]})
    memory.on_rating("r", "disliked")
    stats = memory.arm_stats()
    assert stats["infuse:yes"][1] > 0 and stats["infused:hand"][1] > 0
    assert stats.get("style:hold", (0, 0))[1] == 0 and stats.get("move:yes", (0, 0))[1] == 0, "the rest of the run is not blamed"
    memory.on_rating("r", "liked")
    stats = memory.arm_stats()
    assert stats["infuse:yes"] == (0.5, 0.0) or stats["infuse:yes"][1] == 0, "the blame was taken back"
    assert stats["style:hold"][0] > 0


def test_a_run_without_infusion_shares_the_blame_as_before():
    memory.record_run("r", {"views": [], "arms": ["move:yes", "style:hold", "infuse:no"]})
    memory.on_rating("r", "disliked")
    stats = memory.arm_stats()
    assert stats["move:yes"][1] > 0 and stats["style:hold"][1] > 0 and stats["infuse:no"][1] > 0


def test_one_dislike_lowers_a_word_but_never_removes_it_and_a_like_brings_it_back():
    _habit("hand", "head")
    start = dict(memory.infusion_bank("P"))["hand"]
    memory.record_run("r", {"views": [], "arms": ["infuse:yes", "infused:hand"]})
    memory.on_rating("r", "disliked")
    after = dict(memory.infusion_bank("P"))
    assert 0 < after["hand"] < start and after["head"] == dict(memory.infusion_bank("P"))["head"] == start
    for k in range(3):
        memory.record_run(f"l{k}", {"views": [], "arms": ["infuse:yes", "infused:hand"]})
        memory.on_rating(f"l{k}", "liked")
    assert dict(memory.infusion_bank("P"))["hand"] > after["hand"]


def test_the_probe_lists_each_subject_with_what_ratings_taught_and_forgetting_clears_it():
    _habit("hand", "head")
    memory.record_run("r", {"views": [], "arms": ["infuse:yes", "infused:hand"]})
    memory.on_rating("r", "disliked")
    rows = {r["word"]: r for r in memory.summary()["infusion"]}
    assert rows["hand"]["bad"] > 0 and rows["hand"]["seen"] == 3 and rows["head"]["bad"] == 0
    assert all(not a["arm"].startswith("infused:") for a in memory.summary()["arms"]), "shown once, as a subject"
    assert memory.forget("word", "hand")
    assert "infused:hand" not in memory.arm_stats()


def test_a_project_only_aims_at_words_its_own_prompts_keep_returning_to():
    _habit("hand", "head")                                    # project P
    assert dict(memory.infusion_bank("P")).keys() == {"hand", "head"}
    assert memory.infusion_bank("other") == [] and memory.infusion_bank("") == [], "an unrelated project, or no project, has no bank"
    out, _, chose = nodes.rewrite(f"[Shot 1] {WALK}", {**CAMERA_ON, "focus_infusion_chance": 1}, seconds=5, pieces=[], project="other")
    assert "infuse:yes" not in chose["arms"] and "hand" not in out and "head" not in out


def test_the_oldest_projects_are_forgotten_past_the_cap_and_a_forgotten_word_leaves_every_project():
    for k in range(memory.MAX_PROJECTS + 3):
        memory.observe("x", ["lamp"], f"proj{k}")
    kept = memory._read()["projects"]
    assert len(kept) == memory.MAX_PROJECTS and "proj0" not in kept and f"proj{memory.MAX_PROJECTS + 2}" in kept
    assert memory.forget("word", "lamp") and all("lamp" not in r["seen"] for r in memory._read()["projects"].values())
