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
    assert suggestions.stats() == {"scenes": 0, "counts": {}, "pairs": [], "follows": []}
