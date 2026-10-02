"""Taste keys: capture -> rate pairing, re-rating, and the direction math."""

import pytest
import torch

from modules.system import taste
from modules.system.taste import store


@pytest.fixture(autouse=True)
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "ROOT", tmp_path / "taste")
    return tmp_path / "taste"


def _v(x):
    return torch.tensor([float(x), 0.0])


def test_latest_run_is_recorded_once_and_a_replaced_one_is_refused():
    store.capture("fox", "reins", {25: _v(1)}, prompt_id="p1")
    store.capture("fox", "reins", {25: _v(2)}, prompt_id="p2")
    out = store.rate("p1", "liked")
    assert out["recorded"] == [] and "most recent" in out["why"]
    assert store.rate("p2", "liked")["recorded"] == ["reins"]
    assert store.counts("fox", "reins") == (1, 0)
    # The pending capture is spent: rating p2 again changes it, never adds a row.
    assert store.rate("p2", "disliked")["updated"] == ["reins"]
    assert store.counts("fox", "reins") == (0, 1)
    store.rate("p2", None)
    assert store.counts("fox", "reins") == (0, 0)


def test_every_kind_of_one_run_is_recorded_together():
    store.capture("fox", "reins", {1: _v(1)}, prompt_id="p1")
    store.capture("fox", "dynashift", {1: _v(1)}, prompt_id="p1")
    assert sorted(store.rate("p1", "liked")["recorded"]) == ["dynashift", "reins"]


def test_direction_needs_two_of_each_and_points_liked_minus_disliked():
    for i, (x, r) in enumerate([(3, "liked"), (-3, "disliked"), (5, "liked")]):
        store.capture("fox", "k", {7: _v(x)}, prompt_id=f"p{i}")
        store.rate(f"p{i}", r)
    assert store.direction("fox", "k", 7)[0] is None
    store.capture("fox", "k", {7: _v(-1)}, prompt_id="p9")
    store.rate("p9", "disliked")
    d, liked, disliked = store.direction("fox", "k", 7)
    assert (liked, disliked) == (2, 2)
    assert torch.allclose(d, torch.tensor([1.0, 0.0]))


def test_recent_ratings_outvote_old_ones():
    rows = [(10, "liked"), (-10, "disliked"), (10, "liked"), (-10, "disliked")]
    rows += [(0, "liked")] * 30 + [(1, "disliked")] * 2
    for i, (x, r) in enumerate(rows):
        store.capture("fox", "k", {0: torch.tensor([float(x), float(i % 2)])}, prompt_id=f"p{i}")
        store.rate(f"p{i}", r)
    d, *_ = store.direction("fox", "k", 0)
    assert d[0] < 0                              # the recent disliked +1 dominates the old +-10


def test_bad_names_are_refused():
    for bad in ("", "../x", "a/b", ".hidden", "x."):
        assert not store.valid(bad)
    with pytest.raises(ValueError):
        store.capture("../x", "k", {0: _v(1)}, prompt_id="p")


class _Patcher:
    def __init__(self):
        self.model_options = {}


def test_modifier_names_the_key_the_run_teaches():
    p = _Patcher()
    assert taste.install(p, {"key": ""}, "k") is None
    assert taste.taste_store(p) is None
    assert "fox" in taste.install(p, {"key": "fox"}, "k")
    handle = taste.taste_store(p)
    handle.capture("reins", {3: _v(1)})
    assert (store.ROOT / "fox" / "reins.pending.pt").exists()
    with pytest.raises(ValueError):
        taste.install(p, {"key": "a/b"}, "k")


def test_a_dislike_can_name_the_axis_and_a_blind_learner_reads_it_as_neutral():
    for i, axis in enumerate([None, "image", "composition"]):
        store.capture("fox", "k", {1: _v(i)}, prompt_id=f"p{i}")
        store.rate(f"p{i}", "disliked", axis)
    rows = store.load("fox", "k")["rows"]
    assert [r.get("axis") for r in rows] == [None, "image", "composition"]
    # composition learners cannot judge a bad PICTURE; the picture learner cannot judge a bad plan
    assert [r["reward"] for r in store.blind(rows, "image")] == [-1.0, 0.0, -1.0]
    assert [r["reward"] for r in store.blind(rows, "composition")] == [-1.0, -1.0, 0.0]
    assert [r["reward"] for r in store.blind(rows, None)] == [-1.0] * 3
    assert rows[1]["reward"] == -1.0                       # the stored rows are never edited


def test_changing_the_rating_drops_or_sets_the_axis_and_bad_axes_are_refused():
    store.capture("fox", "k", {1: _v(1)}, prompt_id="p1")
    store.rate("p1", "disliked", "image")
    store.rate("p1", "liked")
    assert "axis" not in store.load("fox", "k")["rows"][0]
    store.rate("p1", "disliked", "composition")
    assert store.load("fox", "k")["rows"][0]["axis"] == "composition"
    for bad in [("liked", "image"), ("disliked", "colour"), (None, "image")]:
        with pytest.raises(ValueError):
            store.rate("p1", *bad)


def test_the_learners_read_through_the_blind_view():
    for i, axis in enumerate(["image", "composition"]):
        store.capture("fox", "k", {1: _v(i)}, prompt_id=f"p{i}")
        store.rate(f"p{i}", "disliked", axis)
    h = taste.Handle("fox")
    assert [r["reward"] for r in h.rows("k", blind_to="image")] == [0.0, -1.0]
    assert [r["reward"] for r in h.rows("k")] == [-1.0, -1.0]
