"""Studio's liked/bad/axis directions fade with recency, like REINS' direction()."""
import sys

import torch

sys.path.insert(0, ".")
import conditioning  # noqa: E402


def _refiner():
    return conditioning.FunPackVideoRefinerV2.__new__(conditioning.FunPackVideoRefinerV2)


def _payload(vec):
    return conditioning.tensor_to_serializable(torch.tensor(vec).view(1, 1, -1))


def test_newer_ratings_outvote_older_ones():
    r = _refiner()
    slot = {}
    zero = _payload([0.0, 0.0])
    for _ in range(20):
        r._v2_store_direction(slot, _payload([5.0, 0.0]), zero)     # old style: +x
    for _ in range(10):                                            # ~half-life is 7
        r._v2_store_direction(slot, _payload([0.0, 5.0]), zero)     # new style: +y
    d = conditioning.serializable_to_tensor(slot["direction"])
    assert slot["direction_count"] == 30                           # gates still count rows
    assert d[1] > d[0] > 0                                         # y leads, x lingers


def test_plain_average_would_have_kept_the_old_style(monkeypatch):
    monkeypatch.setattr(conditioning, "V2_PATH_OUTCOME_DECAY", 1.0)
    r = _refiner()
    slot = {}
    zero = _payload([0.0, 0.0])
    for _ in range(20):
        r._v2_store_direction(slot, _payload([5.0, 0.0]), zero)
    for _ in range(10):
        r._v2_store_direction(slot, _payload([0.0, 5.0]), zero)
    d = conditioning.serializable_to_tensor(slot["direction"])
    assert d[0] > d[1]
