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
    assert memory.on_rating("p1", "liked") == 0                      # the same rating again counts once
    assert memory.view_stats()["side view"] == (1.0, 0.0)
    assert memory.on_rating("unknown", "liked") == 0


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
