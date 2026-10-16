"""Two Generates at once must not draw the same replacement."""

import threading

from core import config, shortcuts


def test_concurrent_commits_each_draw_a_different_replacement(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "REVOLVER_FILE", tmp_path / "r.json")
    sc = shortcuts.Shortcut.from_dict({"name": "t", "triggers": ["tt"], "replacements": ["a", "b", "c", "d"]})
    shortcuts.set_revolver_settings(enabled=True)
    out = []
    ts = [threading.Thread(target=lambda: out.append(shortcuts.expand("tt", [sc], commit=True))) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert sorted(out) == ["a", "b", "c", "d"]
