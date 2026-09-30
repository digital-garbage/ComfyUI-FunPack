"""Two-axis dislikes route to the learners that own that axis; a rating counts more when the
run's measured effect was unusually strong for the key."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import conditioning  # noqa: E402
import rated_dial  # noqa: E402
import samplers  # noqa: E402


def test_axis_dislikes_are_awful_to_everyone_else_and_are_valid_choices():
    for label, axis in (("Disliked: bad image", "image"), ("Disliked: bad composition", "composition")):
        p = conditioning.normalize_refiner_v2_rating(label)
        assert p["key"] == "awful" and p["reward"] == -0.90 and p["axis"] == axis
        assert label in conditioning.V2_RATING_LABELS          # the graph builder keeps it
        assert label + "|loved" not in conditioning.V2_RATING_LABELS
    assert "axis" not in conditioning.normalize_refiner_v2_rating("Awful")


def test_unusualness_needs_history_and_is_clamped():
    assert rated_dial.unusualness(0.5, [0.1, 0.1]) == 1.0          # too little history
    assert rated_dial.unusualness(None, [0.1] * 5) == 1.0          # not measured
    assert rated_dial.unusualness(0.4, [0.1] * 5) == 2.0           # clamped high
    assert rated_dial.unusualness(0.01, [0.1] * 5) == 0.25         # clamped low
    assert abs(rated_dial.unusualness(0.15, [0.1] * 5) - 1.5) < 1e-9


def test_dial_weights_a_strong_run_more(tmp_path, monkeypatch):
    d = rated_dial.Dial("t", 0.5, 0.0, 1.0, 0.1)
    store = {}
    monkeypatch.setattr(d, "read", lambda k: dict(store.get(k, {})))
    monkeypatch.setattr(d, "_write", lambda k, data: store.__setitem__(k, data))
    for _ in range(3):
        d.save_pending("k", 0.8, 0.1)
        d.commit("k", 1.0)
    d.save_pending("k", 0.8, 0.2)
    d.commit("k", 1.0)
    assert store["k"]["history"][-1]["w"] == 2.0
    assert store["k"]["history"][0]["w"] == 1.0


def test_input_steer_records_how_hard_it_pushed():
    s = samplers._InputSteer(False, "x")
    assert s.effect() is None
    d = torch.ones(1, 2, 2, 2)
    s.keep({}, d, d * 1.5)
    assert abs(s.effect() - 0.5) < 1e-6
