import pytest

from core import config, projects, shortcuts, suggestions


@pytest.fixture(autouse=True)
def _store(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "SHORTCUTS_FILE", tmp_path / "s.json")
    monkeypatch.setattr(config, "SHORTCUT_CATEGORIES_FILE", tmp_path / "c.json")
    monkeypatch.setattr(config, "PROJECTS_DIR", tmp_path / "p")


def _project(*texts):
    p = projects.Project(name="x", scenes=[projects.Scene(text=t) for t in texts])
    projects.save(p)


def test_pairs_follows_counts_come_from_typed_triggers_only():
    for n in ("rain", "neon", "fox"):
        shortcuts.save({"name": n, "triggers": [n], "replacements": ["r"]})
    shortcuts.save({"name": "off", "triggers": ["dog"], "replacements": ["r"], "enabled": False})
    _project("rain and neon here", "", "a fox, dog", "raining")        # "raining" is not "rain"
    s = suggestions.stats()
    assert s["scenes"] == 3 and s["counts"] == {"rain": 1, "neon": 1, "fox": 1}
    assert s["pairs"] == [["neon", "rain", 1]] or s["pairs"] == [["rain", "neon", 1]]
    assert sorted(map(tuple, s["follows"])) == [("neon", "fox", 1), ("rain", "fox", 1)]


def test_no_projects_or_no_shortcuts_is_an_empty_answer():
    assert suggestions.stats() == {"scenes": 0, "counts": {}, "pairs": [], "follows": [], "scores": {}, "rated_pairs": []}


def test_excluded_scenes_and_sub_clips_are_skipped_and_follows_use_timeline_order():
    for n in ("rain", "neon", "fox"):
        shortcuts.save({"name": n, "triggers": [n], "replacements": ["r"]})
    scenes = [projects.Scene(id="a", text="rain"), projects.Scene(id="b", text="neon"),
              projects.Scene(id="c", text="fox", excluded=True),
              projects.Scene(id="d", text="rain", gen_unit_id="a", cut_offset_frames=24)]
    projects.save(projects.Project(name="x", scenes=scenes, timeline_order=["b", "a", "c", "d"]))
    s = suggestions.stats()
    assert s["scenes"] == 2 and s["counts"] == {"rain": 1, "neon": 1}
    assert s["follows"] == [["neon", "rain", 1]]


def test_ratings_score_shortcuts_and_pairs_and_a_bad_image_blames_nothing():
    for n in ("rain", "neon", "fox"):
        shortcuts.save({"name": n, "triggers": [n], "replacements": ["r"]})
    projects.save(projects.Project(name="x", scenes=[
        projects.Scene(text="rain neon", rating="10"), projects.Scene(text="rain neon", rating="10|loved"),
        projects.Scene(text="neon fox", rating="Disliked: bad composition"),
        projects.Scene(text="fox", rating="Disliked: bad image"), projects.Scene(text="fox", rating="")]))
    s = suggestions.stats()
    assert s["scores"] == {"rain": 2, "neon": 1, "fox": -1}
    assert sorted(map(tuple, s["rated_pairs"])) == [("fox", "neon", -1), ("neon", "rain", 2)]
    assert [suggestions.vote(x) for x in ("1", "6", "5", "Disliked: bad image", "odd")] == [-1, 1, -1, 0, 0]


def test_a_rating_counts_for_the_text_it_was_given_to_not_a_rewrite():
    for n in ("rain", "neon"):
        shortcuts.save({"name": n, "triggers": [n], "replacements": ["r"]})
    projects.save(projects.Project(name="x", scenes=[projects.Scene(text="neon", rating="10", rated_text="rain")]))
    s = suggestions.stats()
    assert s["scores"] == {"rain": 1} and s["counts"] == {"neon": 1}
    assert projects.Scene.from_dict({"rated_text": "rain"}).rated_text == "rain"


def test_the_miner_finds_exactly_what_the_expander_replaces():
    import random
    words = ["rain", "golden", "hour", "fox", "tail", "neon-lit", "neon", "a", "the", "end", ",", "fox's", "\n", "raining", "GOLDEN"]
    items = [shortcuts.Shortcut(name=f"n{i}", triggers=[w], replacements=[f"<{i}>"])
             for i, w in enumerate(["rain", "golden hour", "golden", "fox", "fox tail", "neon-lit", "a", "the end"])]
    fired, rng = shortcuts.matcher(items), random.Random(7)
    for _ in range(500):
        text = " ".join(rng.choice(words) for _ in range(12))
        out = shortcuts.expand(text, shortcuts=items, seed=1)
        assert fired(text) == {f"n{i}" for i in range(len(items)) if f"<{i}>" in out}, text


def test_ratings_outlive_a_regenerate_a_cleared_text_and_a_left_out_clip():
    for n in ("rain", "neon", "fox"):
        shortcuts.save({"name": n, "triggers": [n], "replacements": ["r"]})
    shown = {"media": {"filename": "3.mp4"}, "promptId": "p3"}
    projects.save(projects.Project(name="x", scenes=[
        projects.Scene(id="a", text="fox", rating="", rated_text=""),
        projects.Scene(id="b", text="", rating="10", rated_text="neon"),
        projects.Scene(id="c", text="fox", rating="10", rated_text="fox", excluded=True)],
        scene_renders={"a": shown},
        scene_variants={"a": [{"media": {"filename": "1.mp4"}, "promptId": "p1", "rating": "1", "rated_text": "rain"},
                              {**shown, "rating": "10", "rated_text": "fox"}]}))   # the take on the clip: its head speaks for it
    assert suggestions.stats()["scores"] == {"rain": -1, "neon": 1, "fox": 1}


def test_a_trigger_lower_reshapes_is_still_found_as_the_expander_finds_it():
    items = [shortcuts.Shortcut(name="big", triggers=["İstanbul"], replacements=["<b>"]),
             shortcuts.Shortcut(name="small", triggers=["istanbul"], replacements=["<s>"])]
    out = shortcuts.expand("istanbul", shortcuts=items, seed=1)
    assert shortcuts.matcher(items)("istanbul") == {"big" if "<b>" in out else "small"}
