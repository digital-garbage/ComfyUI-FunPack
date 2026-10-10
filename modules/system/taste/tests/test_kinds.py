"""A key's kinds: listed with what each is for, and one kind deleted without touching the rest."""

import pytest


@pytest.fixture(autouse=True)
def root(tmp_path, monkeypatch):
    from modules.system.taste import store
    monkeypatch.setattr(store, "ROOT", tmp_path / "taste")


def test_each_kind_is_listed_with_its_purpose_and_the_pending_capture_counts_toward_it(tmp_path):
    from modules.system import taste
    from modules.system.taste import store
    folder = store.ROOT / "fox"
    folder.mkdir(parents=True)
    (folder / "reins.pt").write_bytes(b"x" * 10)
    (folder / "reins.pending.pt").write_bytes(b"y" * 5)
    (folder / "stas.pt").write_bytes(b"z" * 3)
    kinds = taste.key_kinds("fox")
    assert [k["kind"] for k in kinds] == ["reins", "stas"]
    assert kinds[0]["title"] == "Taste steering (REINS)" and kinds[0]["bytes"] == 15
    assert "steer" in kinds[0]["hint"].lower()


def test_clearing_one_kind_leaves_the_others_and_the_key(tmp_path):
    from modules.system import taste
    from modules.system.taste import store
    folder = store.ROOT / "fox"
    folder.mkdir(parents=True)
    (folder / "reins.pt").write_bytes(b"x")
    (folder / "stas.pt").write_bytes(b"z")
    store.clear_kind("fox", "reins")
    assert [k["kind"] for k in taste.key_kinds("fox")] == ["stas"]
    assert store.keys() == ["fox"]


def test_an_unknown_key_lists_nothing_and_a_bad_name_is_refused():
    from modules.system import taste
    assert taste.key_kinds("nosuch") == []
    with pytest.raises(ValueError):
        taste.key_kinds("../etc")
