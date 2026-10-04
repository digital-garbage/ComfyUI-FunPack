"""The node's rewrite, the rating hook and the module contract."""
import importlib

from .. import memory, nodes


def _mod():
    return importlib.import_module(nodes.__package__)


def test_off_by_default_leaves_text_alone(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOT_CAMERA_MEMORY", str(tmp_path / "m.json"))
    text = "[Shot 1] A woman walks.\n[Shot 2] She sits down."
    assert nodes.rewrite(text, {}, seconds=5.0, pieces=[])[0] == text


def test_cuts_add_times(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOT_CAMERA_MEMORY", str(tmp_path / "m.json"))
    text = "[Shot 1] A woman walks along the street.\n[Shot 2] She sits down at a table."
    out, said, _ = nodes.rewrite(text, {"shot_cuts": True}, seconds=6.0, pieces=[])
    assert out != text and said


def test_rating_teaches_once_and_skips_picture_only(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOT_CAMERA_MEMORY", str(tmp_path / "m.json"))
    chose = {"views": [{"view": "side view", "traits": ["none"]}], "arms": ["a1"]}
    memory.record_run("p1", chose)
    assert memory.on_rating("p1", "disliked", "image") == 0          # the picture was at fault, not the camera
    assert memory.on_rating("p1", "liked") == 2
    assert memory.on_rating("p1", "disliked") == 0                   # changing your mind does not count twice
    assert memory.view_stats()["side view"] == (1.0, 0.0)
    assert memory.on_rating("unknown", "liked") == 0


def test_contract():
    m = _mod()
    assert m.PROVIDES["on_rating"]("nope", "liked") == 0
    assert m.CATEGORY == "conditioning" and m.NODES
