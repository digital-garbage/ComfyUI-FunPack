import sys

import pytest
import torch

sys.path.insert(0, ".")
import h3_repr_steering as rs  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_state(monkeypatch, tmp_path):
    monkeypatch.setattr(rs, "state_path", lambda key: str(tmp_path / f"{key}.pt"))


# --- video_mask_from_mod_segments -------------------------------------------

def test_scalar_rows_tag_video_by_mod_3():
    # row = t_row*3 + tag; tag 0 = video, 1 = text, 2 = audio (seg_tag in model.py)
    mod_segments = [(0, 4, 6), (4, 10, 7), (10, 14, 8)]  # video, text, audio
    mask = rs.video_mask_from_mod_segments(mod_segments, seq_len=14, device="cpu")
    assert mask.tolist() == [True] * 4 + [False] * 10


def test_tensor_rows_use_the_same_mod_3_rule():
    rows = torch.tensor([6, 7, 6, 8])  # video, text, video, audio
    mod_segments = [(0, 4, rows)]
    mask = rs.video_mask_from_mod_segments(mod_segments, seq_len=4, device="cpu")
    assert mask.tolist() == [True, False, True, False]


def test_no_video_segment_returns_none_rather_than_an_empty_mask():
    mod_segments = [(0, 4, 7), (4, 8, 8)]  # text, audio only
    assert rs.video_mask_from_mod_segments(mod_segments, seq_len=8, device="cpu") is None


def test_empty_mod_segments_returns_none():
    assert rs.video_mask_from_mod_segments([], seq_len=10, device="cpu") is None


# --- capture -----------------------------------------------------------------

def test_capture_means_over_video_rows_only():
    h = torch.tensor([[1.0, 1.0], [3.0, 3.0], [9.0, 9.0]])
    mask = torch.tensor([True, True, False])
    desc = rs.capture(h, mask)
    assert torch.allclose(desc, torch.tensor([2.0, 2.0]))


def test_capture_with_no_video_rows_is_none():
    h = torch.zeros(3, 2)
    mask = torch.zeros(3, dtype=torch.bool)
    assert rs.capture(h, mask) is None


def test_capture_with_no_mask_is_none():
    assert rs.capture(torch.zeros(3, 2), None) is None


# --- persistence + direction --------------------------------------------------

def test_direction_needs_min_per_group_of_each():
    for i in range(rs.MIN_PER_GROUP - 1):
        rs.save_pending("k", torch.tensor([1.0, 0.0]))
        rs.commit("k", 1.0)
    for i in range(rs.MIN_PER_GROUP):
        rs.save_pending("k", torch.tensor([-1.0, 0.0]))
        rs.commit("k", -0.9)
    direction, n_pos, n_neg = rs.direction("k")
    assert direction is None
    assert n_pos == rs.MIN_PER_GROUP - 1
    assert n_neg == rs.MIN_PER_GROUP


def test_direction_points_from_negative_mean_to_positive_mean_when_weights_balance():
    for _ in range(rs.MIN_PER_GROUP):
        rs.save_pending("k", torch.tensor([5.0, 0.0]))
        rs.commit("k", 1.0)
    for _ in range(rs.MIN_PER_GROUP):
        rs.save_pending("k", torch.tensor([-5.0, 0.0]))
        rs.commit("k", -1.0)
    direction, n_pos, n_neg = rs.direction("k")
    assert direction is not None
    assert n_pos == rs.MIN_PER_GROUP and n_neg == rs.MIN_PER_GROUP
    assert torch.allclose(direction, torch.tensor([1.0, 0.0]), atol=1e-5)


def test_shared_content_cancels_regardless_of_weight_imbalance():
    """Both groups carry the same huge shared offset (simulating prompt content), AND the
    weights are deliberately unbalanced (+1.0 vs -0.05) -- unlike a plain mean difference,
    centring the WEIGHTS makes the shared term cancel exactly even when sum(weights) != 0.
    (Loose tolerance: subtracting a 1000-magnitude shared offset in float32 costs real
    precision, and that cost is not what this test is checking.)"""
    shared = torch.tensor([1000.0, 1000.0])
    for _ in range(rs.MIN_PER_GROUP):
        rs.save_pending("k", shared + torch.tensor([1.0, 0.0]))
        rs.commit("k", 1.0)
    for _ in range(rs.MIN_PER_GROUP):
        rs.save_pending("k", shared + torch.tensor([-1.0, 0.0]))
        rs.commit("k", -0.05)
    direction, _, _ = rs.direction("k")
    assert torch.allclose(direction, torch.tensor([1.0, 0.0]), atol=1e-3)


def test_a_weak_weight_loses_to_a_strong_one_on_a_competing_axis():
    """Two DIFFERENT positive signals competing (not the same axis at different strengths --
    equal group counts on one axis always cancel to the plain difference regardless of
    magnitude, so that case can't show this): x is rated strongly (weight 1.0), y only
    weakly (weight 0.2). The direction must lean toward x, not split evenly between them."""
    for _ in range(rs.MIN_PER_GROUP):
        rs.save_pending("k", torch.tensor([0.0, 0.0]))
        rs.commit("k", -0.9)
    for _ in range(rs.MIN_PER_GROUP):
        rs.save_pending("k", torch.tensor([1.0, 0.0]))
        rs.commit("k", 1.0)
    for _ in range(rs.MIN_PER_GROUP):
        rs.save_pending("k", torch.tensor([0.0, 1.0]))
        rs.commit("k", 0.2)
    direction, _, _ = rs.direction("k")
    assert direction[0] > 0.9 and direction[1] < 0.2


def test_pending_is_discarded_on_a_neutral_rating():
    rs.save_pending("k", torch.tensor([1.0, 0.0]))
    rs.commit("k", None)
    direction, n_pos, n_neg = rs.direction("k")
    assert n_pos == 0 and n_neg == 0


def test_pending_overwrites_not_accumulates():
    rs.save_pending("k", torch.tensor([1.0, 0.0]))
    rs.save_pending("k", torch.tensor([2.0, 0.0]))  # a second capture before any rating
    rs.commit("k", 1.0)
    data = rs._load("k")
    assert len(data["rows"]) == 1
    assert torch.allclose(data["rows"][0]["desc"], torch.tensor([2.0, 0.0]))


def test_clear_all_removes_the_state_file():
    rs.save_pending("k", torch.tensor([1.0, 0.0]))
    rs.commit("k", 1.0)
    rs.clear_all("k")
    direction, n_pos, n_neg = rs.direction("k")
    assert direction is None and n_pos == 0 and n_neg == 0


def test_no_refinement_key_is_a_safe_no_op():
    rs.save_pending("", torch.tensor([1.0]))  # must not raise
    rs.commit("", 1.0)
    rs.clear_all("")
