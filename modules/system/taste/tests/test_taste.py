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


def test_a_run_is_recorded_once_and_a_re_rating_changes_it_never_adds_a_row():
    store.capture("fox", "reins", {25: _v(1)}, prompt_id="p1")
    assert store.rate("p1", "liked")["recorded"] == ["reins"]
    assert store.counts("fox", "reins") == (1, 0)
    # The pending capture is spent: rating p1 again changes it, never adds a row.
    assert store.rate("p1", "disliked")["updated"] == ["reins"]
    assert store.counts("fox", "reins") == (0, 1)
    store.rate("p1", None)
    assert store.counts("fox", "reins") == (0, 0)


def test_every_run_of_one_generate_waits_for_its_own_rating():
    store.capture("fox", "reins", {25: _v(1)}, prompt_id="p1")
    store.capture("fox", "reins", {25: _v(2)}, prompt_id="p2")
    store.capture("fox", "reins", {25: _v(3)}, prompt_id="p3")
    assert store.rate("p2", "liked")["recorded"] == ["reins"]       # not the latest run, still rateable
    assert store.rate("p1", "disliked")["recorded"] == ["reins"]
    assert store.rate("p3", "liked")["recorded"] == ["reins"]
    assert store.counts("fox", "reins") == (2, 1)
    rows = store.load("fox", "reins")["rows"]
    assert [float(r["rows"]["25"][0]) for r in rows] == [2.0, 1.0, 3.0]    # each row holds ITS run's capture


def test_a_new_generate_forgets_what_nobody_rated():
    store.capture("fox", "reins", {25: _v(1)}, prompt_id="p1")
    store.capture("fox", "reins", {25: _v(2)}, prompt_id="p2")
    assert store.new_generation() == 2
    out = store.rate("p1", "liked")
    assert out["recorded"] == [] and "new Generate" in out["why"]
    assert store.rate("p2", "liked")["recorded"] == []
    assert not list((store.ROOT / "fox").glob("*.pending.pt"))
    store.capture("fox", "reins", {25: _v(3)}, prompt_id="p3")           # the next Generate's runs still work
    assert store.rate("p3", "liked")["recorded"] == ["reins"]


def test_rated_rows_survive_a_new_generate():
    store.capture("fox", "reins", {25: _v(1)}, prompt_id="p1")
    store.rate("p1", "liked")
    store.new_generation()
    assert store.counts("fox", "reins") == (1, 0)
    assert store.rate("p1", "disliked")["updated"] == ["reins"]            # changing your mind still works


def test_an_id_that_cannot_name_a_file_is_never_kept_aside_and_never_crashes():
    store.capture("fox", "reins", {25: _v(1)}, prompt_id="../evil")
    store.capture("fox", "reins", {25: _v(2)}, prompt_id="p2")
    assert not list(store.ROOT.rglob("*evil*"))
    assert store.rate("../evil", "liked")["recorded"] == []
    assert store.rate("p2", "liked")["recorded"] == ["reins"]


def test_pending_runs_are_not_part_of_an_export_and_clearing_a_kind_forgets_them(tmp_path):
    store.capture("fox", "reins", {25: _v(1)}, prompt_id="p1")
    store.capture("fox", "reins", {25: _v(2)}, prompt_id="p2")
    store.rate("p2", "liked")
    out = tmp_path / "k.zip"
    store.export_key("fox", out)
    import zipfile
    assert zipfile.ZipFile(out).namelist() == ["reins.pt"]
    store.clear_kind("fox", "reins")
    assert not list((store.ROOT / "fox").glob("reins*"))


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


def test_a_cleared_rating_can_be_given_again_while_its_generate_is_the_latest():
    store.capture("fox", "reins", {25: _v(1)}, prompt_id="p1")
    store.rate("p1", "liked")
    store.rate("p1", None)                                  # "I take it back"
    assert store.counts("fox", "reins") == (0, 0)
    out = store.rate("p1", "disliked")
    assert out["recorded"] == ["reins"] and store.counts("fox", "reins") == (0, 1)
    store.rate("p1", None)
    store.new_generation()                                  # the next Generate: nothing waits any more
    out = store.rate("p1", "liked")
    assert out["recorded"] == [] and out["why"]


def test_a_cleared_rating_beside_a_newer_run_goes_aside_not_over_it():
    store.capture("fox", "reins", {25: _v(1)}, prompt_id="p1")
    store.rate("p1", "liked")
    store.capture("fox", "reins", {25: _v(2)}, prompt_id="p2")
    store.rate("p1", None)
    assert store.rate("p2", "liked")["recorded"] == ["reins"]
    assert store.rate("p1", "disliked")["recorded"] == ["reins"]
    rows = {r["prompt_id"]: float(r["rows"]["25"][0]) for r in store.load("fox", "reins")["rows"]}
    assert rows == {"p2": 2.0, "p1": 1.0}


def test_a_rating_says_why_when_nothing_was_kept_for_the_clip():
    out = store.rate("never-captured", "liked")
    assert out["recorded"] == [] and "captured nothing" in out["why"]


def test_deleting_a_key_leaves_other_keys_runs_ratable_and_one_run_can_wait_under_two_keys():
    store.capture("fox", "reins", {25: _v(1)}, prompt_id="p1")
    store.capture("wolf", "reins", {25: _v(2)}, prompt_id="p1")        # one run, two keys
    store.capture("wolf", "reins", {25: _v(3)}, prompt_id="p2")
    store.delete("fox")
    assert sorted(store.rate("p1", "liked")["recorded"]) == ["reins"]
    assert store.counts("wolf", "reins") == (1, 0) and store.counts("fox", "reins") == (0, 0)
    assert store.rate("p2", "liked")["recorded"] == ["reins"]


def test_a_run_waiting_under_two_keys_teaches_both():
    store.capture("fox", "reins", {25: _v(1)}, prompt_id="p1")
    store.capture("wolf", "reins", {25: _v(2)}, prompt_id="p1")
    assert store.rate("p1", "liked")["recorded"] == ["reins", "reins"]
    assert store.counts("fox", "reins") == (1, 0) and store.counts("wolf", "reins") == (1, 0)
