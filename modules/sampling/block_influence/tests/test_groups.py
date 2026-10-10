"""The four-group comparison and the recording default, on synthetic rows (no model needed)."""

import torch

from modules.sampling.block_influence import measure


def _row(reward, axis, profile):
    return {"reward": reward, "axis": axis, "rows": {"ratio": torch.tensor(profile, dtype=torch.float32)}}


def test_recording_is_on_when_no_switch_file_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(measure, "SWITCH", tmp_path / "switch")
    assert measure.enabled() is True
    measure.set_enabled(False)
    assert measure.enabled() is False
    measure.set_enabled(True)
    assert measure.enabled() is True


def test_groups_sort_dislikes_by_axis():
    assert measure._group(_row(1.0, None, [0])) == "liked"
    assert measure._group(_row(-1.0, "image", [0])) == "image"
    assert measure._group(_row(-1.0, "composition", [0])) == "composition"
    assert measure._group(_row(-1.0, None, [0])) == "both"
    assert measure._group(_row(0.0, None, [0])) is None


def test_opposite_directions_show_as_opposed_cosines():
    rows = [_row(1, None, [1, 0, 0]) for _ in range(3)] + [_row(-1, "image", [0, 1, 0]) for _ in range(3)] \
         + [_row(-1, "composition", [0, 0, 1]) for _ in range(3)] + [_row(-1, None, [0, 0, 0]) for _ in range(3)]
    out = measure.groups(rows, shuffles=50)
    assert out["counts"] == {"liked": 3, "image": 3, "composition": 3, "both": 3}
    # Each dislike is "liked minus a different block": the two directions share the "away from liked"
    # part (the -1 on block 0), so they agree half way, not fully and not at all.
    assert abs(out["cos"]["composition vs image"] - 0.5) < 1e-5
    assert set(out["chance"]) == set(out["cos"])


def test_too_few_clips_says_so_instead_of_comparing():
    out = measure.groups([_row(1, None, [1, 0]), _row(-1, "image", [0, 1])], shuffles=5)
    assert out["cos"] == {} and "not enough" in out["note"]


def test_nothing_recorded_is_said():
    assert measure.groups([])["note"] == "nothing recorded yet"
