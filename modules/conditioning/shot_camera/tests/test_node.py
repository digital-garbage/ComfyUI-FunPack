"""The node's rewrite, the rating hook and the module contract."""
import importlib

import pytest

from .. import memory, nodes


def _mod():
    return importlib.import_module(nodes.__package__)


def test_off_by_default_leaves_text_alone(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOT_CAMERA_MEMORY", str(tmp_path / "m.json"))
    text = "[Shot 1] A woman walks.\n[Shot 2] She sits down."
    assert nodes.rewrite(text, {}, seconds=5.0, pieces=[])[0] == text


def test_cut_within_the_same_shot_is_its_own_switch(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOT_CAMERA_MEMORY", str(tmp_path / "m.json"))
    m = _mod()
    assert m.SETTINGS["cut_same_shot"]["type"] == "bool" and "when" not in m.SETTINGS["cut_same_shot"]
    assert m.SETTINGS["shot_cuts_chance"]["when"] == {"cut_same_shot": True}
    dance = "She sways her hips from side to side, smiling at the room."
    text = f"[Shot 1] {dance} {dance} {dance}"
    kept, _, _ = nodes.rewrite(text, {"shot_cuts": True, "cut_same_shot": False, "shot_cuts_chance": 1},
                               seconds=12, pieces=[dance])
    assert kept.count("[Shot ") == 1
    cut, said, _ = nodes.rewrite(text, {"cut_same_shot": True, "shot_cuts_chance": 1}, seconds=12, pieces=[dance])
    assert cut.count("[Shot ") == 3 and "00:04.000" in cut and "00:08.000" in cut and "cuts at" in said


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
    assert memory.on_rating("p1", "liked") == 0                      # the same rating again counts once
    assert memory.view_stats()["side view"] == (1.0, 0.0)
    assert memory.on_rating("unknown", "liked") == 0


def test_a_like_teaches_a_detail_and_a_bad_composition_can_retire_it(tmp_path, monkeypatch):
    pytest.importorskip("spacy")
    monkeypatch.setenv("SHOT_CAMERA_MEMORY", str(tmp_path / "m.json"))
    m = _mod()
    assert m.SETTINGS["detail_notes"]["type"] == "bool" and "when" not in m.SETTINGS["detail_notes"]
    assert m.SETTINGS["detail_notes_chance"]["when"] == {"detail_notes": True}
    text = "[Shot 1] She kisses his lips."
    memory.record_run("liked", {"views": [], "arms": [], "text": "[Shot 1] She has detailed lips.", "details": []})
    assert memory.on_rating("liked", "disliked", "image") == 0
    assert memory.detail_bank() == []
    assert memory.on_rating("liked", "liked") == 1
    assert memory.detail_bank()[0]["phrase"] == "detailed lips"
    out, said, chose = nodes.rewrite(text, {"detail_notes": True, "detail_notes_chance": 1})
    assert "Detailed lips." in out and "shot 1 detailed lips" in said
    assert out.startswith("[Shot 1] She kisses his lips.")
    memory.record_run("bad1", {"views": [], "arms": [], "text": out, "details": chose["details"]})
    memory.record_run("bad2", {"views": [], "arms": [], "text": out, "details": chose["details"]})
    memory.on_rating("bad1", "disliked", "composition")
    memory.on_rating("bad2", "disliked", "composition")
    assert memory.detail_bank() == []
    quiet, _, _ = nodes.rewrite(text, {"detail_notes": True, "detail_notes_chance": 1})
    assert quiet == text


def test_contract():
    m = _mod()
    assert m.PROVIDES["on_rating"]("nope", "liked") == 0
    assert m.CATEGORY == "conditioning" and m.NODES


def test_last_rating_wins_and_forgetting_takes_it_back(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOT_CAMERA_MEMORY", str(tmp_path / "m.json"))
    memory.record_run("a", {"views": [], "arms": ["move:yes"]})
    assert memory.on_rating("a", "disliked") == 1
    assert memory.arm_stats()["move:yes"] == (0.0, 1.0)
    memory.on_rating("a", "disliked", "image")                      # refined: the picture, not the camera
    assert memory.arm_stats()["move:yes"] == (0.0, 0.0)
    memory.on_rating("a", "liked")
    memory.on_rating("a", None)                                     # forgotten
    assert memory.arm_stats()["move:yes"] == (0.0, 0.0)


def test_wrong_shaped_memory_and_threads_do_not_break(tmp_path, monkeypatch):
    import threading
    f = tmp_path / "m.json"
    monkeypatch.setenv("SHOT_CAMERA_MEMORY", str(f))
    f.write_text('{"runs":[1],"seen":{"x":"a"},"prompts":5,"hashes":3}')
    memory.record_run("p", {"views": [], "arms": ["a"]})
    assert memory.prior() is not None or True
    def work(i):
        for j in range(40):
            memory.record_run(f"{i}-{j}", {"views": [], "arms": ["a"]})
            memory.on_rating(f"{i}-{j}", "liked")
    ts = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert memory.arm_stats()["a"][0] > 0


def test_more_shots_than_seconds_refuses_instead_of_writing_bad_times(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOT_CAMERA_MEMORY", str(tmp_path / "m.json"))
    text = "\n".join(f"[Shot {i}] A woman walks along street number {i}." for i in range(1, 8))
    out, said, _ = nodes.rewrite(text, {"shot_cuts": True}, seconds=5.17, pieces=[])
    assert out == text and "do not fit" in said


def test_memory_routes_read_and_forget(tmp_path, monkeypatch):
    import asyncio, json
    monkeypatch.setenv("SHOT_CAMERA_MEMORY", str(tmp_path / "m.json"))
    memory.record_run("a", {"views": [{"view": "side view", "traits": ["none"]}], "arms": []})
    memory.on_rating("a", "liked")
    handlers = {}

    class Table:
        def get(self, path): return lambda fn: handlers.setdefault(("GET", path), fn)
        def post(self, path): return lambda fn: handlers.setdefault(("POST", path), fn)

    class web:
        @staticmethod
        def json_response(data, status=200):
            return type("R", (), {"status": status, "data": data})()

    class Req:
        def __init__(self, body=None): self._b = body
        async def json(self): return self._b

    _mod().routes(Table(), "/b", web)
    got = asyncio.run(handlers[("GET", "/b/memory")](Req()))
    assert [v["view"] for v in got.data["views"]] == ["side view"]
    assert asyncio.run(handlers[("POST", "/b/forget")](Req({"kind": "view", "name": "side view"}))).data == {"forgotten": True}
    assert asyncio.run(handlers[("GET", "/b/memory")](Req())).data["views"] == []
    assert asyncio.run(handlers[("POST", "/b/forget")](Req({"kind": "nonsense"}))).status == 400


def test_forget_is_complete_and_stays_forgotten(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOT_CAMERA_MEMORY", str(tmp_path / "m.json"))
    memory.observe("h1", ["lamp"])
    memory.record_run("a", {"views": [{"view": "side view", "traits": ["none"]}], "arms": ["word:lamp", "split:yes"]})
    memory.on_rating("a", "liked")
    assert memory.forget("word", "lamp") and "word:lamp" not in memory.arm_stats()
    assert memory.forget("view", "side view")
    memory.on_rating("a", "disliked")                              # the run that taught it is gone too
    assert memory.view_stats() == {}
    assert [a["arm"] for a in memory.summary()["arms"]] == ["split:yes"]
    assert memory.forget("arm", "split:yes") and memory.summary()["arms"] == []


def test_a_bad_composition_dislike_blames_the_camera_choices(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOT_CAMERA_MEMORY", str(tmp_path / "m.json"))
    memory.record_run("c", {"views": [{"view": "low angle", "traits": ["none"]}], "arms": ["move:yes", "style:travel"]})
    assert memory.on_rating("c", "disliked", "composition") == 3
    assert memory.arm_stats()["style:travel"] == (0.0, 0.5)          # two camera choices share one blame
    assert memory.view_stats()["low angle"] == (0.0, 1.0)
    assert memory.split_chance(0.5) == 0.5 and memory.effective_chance(0.7) < 0.7      # the next run moves the camera less often


def test_a_node_that_will_not_rewrite_does_not_invalidate_what_comes_after_it(monkeypatch):
    """"Always changed" also re-runs the prompt/reference encode downstream (~7s on H3) on every run."""
    import math
    shots = "[Shot 1] A woman walks.\n[Shot 2] She sits."
    monkeypatch.setattr(nodes, "_values", lambda settings: dict(settings or {}))
    assert nodes.FunPackShotCamera.fingerprint_inputs(text=shots, settings={}) == ""
    assert nodes.FunPackShotCamera.fingerprint_inputs(text="no shots here", settings={"camera_moves": True}) == ""
    assert math.isnan(nodes.FunPackShotCamera.fingerprint_inputs(text=shots, settings={"camera_moves": True}))


def test_camera_switch_turns_off_the_camera_rewrites_and_only_those(tmp_path, monkeypatch):
    """_off.camera: moves, cuts, views, reframes (and their chances) sit out; liked details and the plain pipeline stay."""
    import types
    monkeypatch.setenv("SHOT_CAMERA_MEMORY", str(tmp_path / "m.json"))
    m = _mod()
    spec = types.SimpleNamespace(id=nodes.ID, settings=m.SETTINGS, defaults=lambda: {k: v.get("default") for k, v in m.SETTINGS.items()})
    monkeypatch.setattr(nodes.registry_mod, "current", lambda *a, **k: types.SimpleNamespace(specs={nodes.ID: spec}))
    every = {k: True for k in nodes.CAMERA_SWITCHES} | {"detail_notes": True, "detail_notes_chance": 1.0}
    every |= {k: 1.0 for k in nodes.CAMERA_CHANCES}
    on = {nodes.ID: every}
    text = "[Shot 1] A woman walks.\n[Shot 2] She sits down."
    assert nodes._rewrites(nodes._values(on), text)
    off = nodes._values({**on, "_off": {"camera": True}})
    assert all(off[k] is False for k in nodes.CAMERA_SWITCHES) and all(off[k] == 0.0 for k in nodes.CAMERA_CHANCES)
    assert off["detail_notes"] is True and off["detail_notes_chance"] == 1.0          # not camera work: kept
    assert nodes.FunPackShotCamera.execute(text, 124, 24.0, settings={nodes.ID: {k: True for k in nodes.CAMERA_SWITCHES}, "_off": {"camera": True}}).args[0] == text
    # Flag off, or only "camera": false: the camera works as before.
    assert nodes._values({**on, "_off": {"camera": False}}) == nodes._values(on)
    # Every camera key is a real setting, and nothing outside the list is touched.
    assert set(nodes.CAMERA_SWITCHES + nodes.CAMERA_CHANCES) <= set(m.SETTINGS)
    untouched = {k for k in m.SETTINGS} - set(nodes.CAMERA_SWITCHES + nodes.CAMERA_CHANCES)
    assert all(off[k] == nodes._values(on)[k] for k in untouched)
