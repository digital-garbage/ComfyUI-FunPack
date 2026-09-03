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
        rs.commit("k", "like")
    for i in range(rs.MIN_PER_GROUP):
        rs.save_pending("k", torch.tensor([-1.0, 0.0]))
        rs.commit("k", "awful")
    direction, n_liked, n_disliked = rs.direction("k")
    assert direction is None
    assert n_liked == rs.MIN_PER_GROUP - 1
    assert n_disliked == rs.MIN_PER_GROUP


def test_direction_points_from_disliked_mean_to_liked_mean():
    for _ in range(rs.MIN_PER_GROUP):
        rs.save_pending("k", torch.tensor([5.0, 0.0]))
        rs.commit("k", "like")
    for _ in range(rs.MIN_PER_GROUP):
        rs.save_pending("k", torch.tensor([-5.0, 0.0]))
        rs.commit("k", "awful")
    direction, n_liked, n_disliked = rs.direction("k")
    assert direction is not None
    assert n_liked == rs.MIN_PER_GROUP and n_disliked == rs.MIN_PER_GROUP
    assert torch.allclose(direction, torch.tensor([1.0, 0.0]), atol=1e-5)


def test_shared_content_cancels_without_needing_explicit_centring():
    """Both groups carry the same huge shared offset (simulating prompt content); the
    direction must still point along the true liked-vs-disliked axis, not the shared one."""
    shared = torch.tensor([1000.0, 1000.0])
    for _ in range(rs.MIN_PER_GROUP):
        rs.save_pending("k", shared + torch.tensor([1.0, 0.0]))
        rs.commit("k", "nailed_it")
    for _ in range(rs.MIN_PER_GROUP):
        rs.save_pending("k", shared + torch.tensor([-1.0, 0.0]))
        rs.commit("k", "missing_quality")
    direction, _, _ = rs.direction("k")
    assert torch.allclose(direction, torch.tensor([1.0, 0.0]), atol=1e-5)


def test_pending_is_discarded_on_a_neutral_rating():
    rs.save_pending("k", torch.tensor([1.0, 0.0]))
    rs.commit("k", None)
    direction, n_liked, n_disliked = rs.direction("k")
    assert n_liked == 0 and n_disliked == 0


def test_an_ambiguous_near_miss_is_excluded_from_both_sides():
    """Missing action: reward +0.05 in the shared table, but that scalar is a quality-
    landscape score for other mechanisms, not a liked/disliked verdict for this one -- it
    must not count as 'liked' just because its reward sign is positive."""
    rs.save_pending("k", torch.tensor([1.0, 0.0]))
    rs.commit("k", "missing_action")
    direction, n_liked, n_disliked = rs.direction("k")
    assert n_liked == 0 and n_disliked == 0


def test_pending_overwrites_not_accumulates():
    rs.save_pending("k", torch.tensor([1.0, 0.0]))
    rs.save_pending("k", torch.tensor([2.0, 0.0]))  # a second capture before any rating
    rs.commit("k", "like")
    data = rs._load("k")
    assert len(data["rows"]) == 1
    assert torch.allclose(data["rows"][0]["desc"], torch.tensor([2.0, 0.0]))


def test_clear_all_removes_the_state_file():
    rs.save_pending("k", torch.tensor([1.0, 0.0]))
    rs.commit("k", "like")
    rs.clear_all("k")
    direction, n_liked, n_disliked = rs.direction("k")
    assert direction is None and n_liked == 0 and n_disliked == 0


def test_no_refinement_key_is_a_safe_no_op():
    rs.save_pending("", torch.tensor([1.0]))  # must not raise
    rs.commit("", "like")
    rs.clear_all("")
