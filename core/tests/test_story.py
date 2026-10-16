import pytest

from core import config, story


@pytest.fixture(autouse=True)
def _store(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "MARKERS_FILE", tmp_path / "markers.json")


@pytest.mark.parametrize("scenes", [
    [""], ["a"], ["a", "b"], ["a", "", "b"], ["", "a"], ["a", ""], ["", ""],
    ["a cat\nsleeps", "dog, running"],
])
def test_round_trip_is_lossless(scenes):
    assert story.split(story.join(scenes)) == scenes


def test_only_whole_words_cut():
    assert story.split("a qcutting b") == ["a qcutting b"]
    assert story.split("a QCUT b") == ["a", "b"]


def test_longest_marker_wins_and_spaces_are_flexible():
    assert story.split("a scene   cut b cut c", ["cut", "scene cut"]) == ["a", "b", "c"]


def test_custom_markers_persist_and_join_uses_the_first():
    story.save_markers(["Cut", "cut", " next  shot "])
    assert story.markers() == ["Cut", "next shot"]
    assert story.join(["a", "b"]) == "a\nCut\nb"


def test_empty_marker_list_is_refused_and_keeps_the_old_one():
    story.save_markers(["zap"])
    with pytest.raises(ValueError):
        story.save_markers(["  ", 5 and ""])
    assert story.markers() == ["zap"]


def test_unreadable_store_falls_back_to_default():
    config.MARKERS_FILE.write_text("{nope", encoding="utf-8")
    assert story.markers() == story.DEFAULT_MARKERS


def test_non_string_markers_are_ignored():
    assert story.save_markers([None, 5, "cut"]) == ["cut"]
    with pytest.raises(ValueError):
        story.save_markers([None, 5])
