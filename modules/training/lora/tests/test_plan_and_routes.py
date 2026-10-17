"""The graph a run is queued as, and the module's routes."""

import asyncio

import pytest


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Needs ComfyUI's node schemas."""


def pipeline():
    return [
        {"id": "model", "node": "FunPackDiffusionModelLoader", "group": "Loaders", "inputs": {"model_name": "x.safetensors"}},
        {"id": "clip", "node": "FunPackCLIPLoader", "group": "Loaders", "inputs": {}},
        {"id": "vae", "node": "FunPackVAELoader", "group": "Loaders", "inputs": {}},
        {"id": "audio_vae", "node": "FunPackVAELoader", "group": "Loaders", "inputs": {}},
        {"id": "lora", "node": "FunPackLoraLoader", "group": "Loaders", "inputs": {"model": ["model", 0]}},
    ]


def test_the_trainer_is_fed_by_the_pipelines_loaders(monkeypatch):
    from core import graph
    from modules.training.lora import plan
    monkeypatch.setattr(graph, "build", lambda slots, schemas=None, optional=None: ({s["id"]: s for s in slots}, []))

    class Schemas:
        def outputs(self, node):
            return {"FunPackDiffusionModelLoader": ["MODEL"], "FunPackCLIPLoader": ["CLIP", "STRING"], "FunPackVAELoader": ["VAE"],
                    "FunPackLoraLoader": ["MODEL", "CLIP"]}[node]

        def inputs(self, node):
            return {"FunPackLoraLoader": {"model": "MODEL"}}.get(node, {})
    built, problems = plan.assemble(pipeline(), "mine", {"steps": 10}, Schemas())
    assert problems == []
    inputs = built["train_lora"]["inputs"]
    assert "model" not in inputs and inputs["model_name"] == "x.safetensors" and inputs["weight_dtype"] == "default"    # the file, not a loader node
    assert "model" not in built                                                                                  # and no loader node runs in front of the trainer
    assert inputs["clip"] == ["clip", 0] and inputs["audio_vae"] == ["audio_vae", 0] and inputs["set_name"] == "mine" and inputs["steps"] == 10


def test_a_model_slot_that_is_not_a_plain_loader_is_still_linked(monkeypatch):
    from core import graph
    from modules.training.lora import plan
    monkeypatch.setattr(graph, "build", lambda slots, schemas=None, optional=None: ({s["id"]: s for s in slots}, []))
    slots = pipeline()
    slots[0]["inputs"] = {}                                   # no file named: the loader node itself has to run
    class Schemas:
        def outputs(self, node):
            return {"FunPackDiffusionModelLoader": ["MODEL"], "FunPackCLIPLoader": ["CLIP"], "FunPackVAELoader": ["VAE"], "FunPackLoraLoader": ["MODEL"]}[node]

        def inputs(self, node):
            return {"FunPackLoraLoader": {"model": "MODEL"}}.get(node, {})
    built, _ = plan.assemble(slots, "mine", {}, Schemas())
    assert built["train_lora"]["inputs"]["model"] == ["model", 0] and "model" in built
    assert "lora" not in built, "a patch on the model is not part of the training graph"


def test_a_pipeline_without_a_loader_is_refused_with_which(monkeypatch):
    from modules.training.lora import plan

    class Schemas:
        def outputs(self, node):
            return ["MODEL"]

        def inputs(self, node):
            return {}
    built, problems = plan.assemble(pipeline()[:1], "mine", {}, Schemas())
    assert built == {} and "clip, vae" in problems[0]


def test_routes_make_and_edit_a_set(tmp_path, monkeypatch):
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer
    from core import config
    from modules.training import lora
    from core import addons
    monkeypatch.setattr(config, "TRAINING_DIR", tmp_path)
    monkeypatch.setattr(addons, "enabled", lambda _id: False)       # the add-on is off: a plan is refused

    async def go():
        app, table = web.Application(), web.RouteTableDef()
        lora.routes(table, "/m", web)
        app.add_routes(table)
        async with TestClient(TestServer(app)) as c:
            bad = await c.post("/m/sets", json={"name": "../x"})
            made = await c.post("/m/sets", json={"name": "cats"})
            added = await (await c.post("/m/sets/cats/add", json={"entries": [
                {"src": {"kind": "output", "type": "output", "subfolder": "", "filename": "a.png"}, "caption": "cat, dog"}]})).json()
            bulk = await (await c.post("/m/sets/cats/bulk", json={"op": "remove", "tag": "dog"})).json()
            missing = await c.get("/m/sets/nope")
            plan_off = await c.post("/m/plan", json={"set": "cats", "slots": []})
            gone = await (await c.delete("/m/sets/cats")).json()
            return bad.status, made.status, added, bulk, missing.status, plan_off.status, gone
    bad, made, added, bulk, missing, plan_off, gone = asyncio.run(go())
    assert bad == 400 and made == 200 and added["added"] == 1
    assert bulk["changed"] == 1 and bulk["set"]["items"][0]["caption"] == "cat"
    assert missing == 404 and plan_off == 409 and gone == {"deleted": True}


def test_rated_results_list_only_the_videos_that_are_still_here(tmp_path, monkeypatch):
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer
    from core import config, projects
    from modules.training import lora
    from modules.training.lora import train
    (tmp_path / "here.mp4").write_bytes(b"x")
    doc = {"scenes": [{"id": "a", "rating": "10"}, {"id": "b", "rating": "10"}],
           "scene_renders": {"a": {"media": {"type": "output", "subfolder": "", "filename": "here.mp4"}},
                             "b": {"media": {"type": "output", "subfolder": "", "filename": "gone.mp4"}}}}

    class Project:
        def to_dict(self):
            return doc
    monkeypatch.setattr(config, "TRAINING_DIR", tmp_path)
    monkeypatch.setattr(projects, "listing", lambda: [{"id": "p"}])
    monkeypatch.setattr(projects, "get", lambda _id: Project())
    monkeypatch.setattr(train, "roots", lambda: {"output": str(tmp_path)})

    async def go():
        app, table = web.Application(), web.RouteTableDef()
        lora.routes(table, "/m", web)
        app.add_routes(table)
        async with TestClient(TestServer(app)) as c:
            return await (await c.get("/m/rated")).json()
    got = asyncio.run(go())
    assert [r["media"]["filename"] for r in got["results"]] == ["here.mp4"] and got["counts"]["liked"] == 1


def test_a_finished_run_can_be_downloaded_and_only_its_own_file(tmp_path, monkeypatch):
    import folder_paths
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer
    from core import config
    from modules.training import lora
    from modules.training.lora import jobs
    monkeypatch.setattr(config, "TRAINING_DIR", tmp_path)
    loras = tmp_path / "loras"
    loras.mkdir()
    (loras / "cats.safetensors").write_bytes(b"weights")
    (tmp_path / "secret.safetensors").write_bytes(b"no")
    monkeypatch.setattr(folder_paths, "get_folder_paths", lambda kind: [str(loras)])
    runs = jobs._dir()
    import json
    for run, name in (("aaaaaaaaaaaa", "cats.safetensors"), ("bbbbbbbbbbbb", "../secret.safetensors"), ("cccccccccccc", "gone.safetensors"), ("dddddddddddd", "")):
        (runs / f"{run}.json").write_text(json.dumps({"run": run, "lora": name}))

    async def go():
        app, table = web.Application(), web.RouteTableDef()
        lora.routes(table, "/m", web)
        app.add_routes(table)
        async with TestClient(TestServer(app)) as c:
            ok = await c.get("/m/runs/aaaaaaaaaaaa/lora")
            return ok.status, ok.headers.get("Content-Disposition"), await ok.read(), [(await c.get(f"/m/runs/{r}/lora")).status for r in ("bbbbbbbbbbbb", "cccccccccccc", "dddddddddddd", "eeeeeeeeeeee")]
    status, disposition, body, refused = asyncio.run(go())
    assert status == 200 and body == b"weights" and "attachment" in disposition and "cats.safetensors" in disposition
    assert refused == [404, 404, 404, 404]


def test_a_run_lists_its_saved_copies_and_preview_and_serves_each(tmp_path, monkeypatch):
    import json
    import folder_paths
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer
    from core import config
    from modules.training import lora
    from modules.training.lora import jobs
    monkeypatch.setattr(config, "TRAINING_DIR", tmp_path)
    loras = tmp_path / "loras"
    loras.mkdir()
    for f in ("cats.safetensors", "cats_step200.safetensors", "cats_step50.safetensors", "cats_step50_2.safetensors", "cats_preview.safetensors", "dogs_step50.safetensors", "cats_stepx.safetensors"):
        (loras / f).write_bytes(f.encode())
    monkeypatch.setattr(folder_paths, "get_folder_paths", lambda kind: [str(loras)])
    (jobs._dir() / "aaaaaaaaaaaa.json").write_text(json.dumps({"run": "aaaaaaaaaaaa", "name": "cats", "lora": "cats.safetensors"}))

    async def go():
        app, table = web.Application(), web.RouteTableDef()
        lora.routes(table, "/m", web)
        app.add_routes(table)
        async with TestClient(TestServer(app)) as c:
            rec = await (await c.get("/m/runs/aaaaaaaaaaaa")).json()
            one = await c.get("/m/runs/aaaaaaaaaaaa/lora?file=cats_step200.safetensors")
            other = await c.get("/m/runs/aaaaaaaaaaaa/lora?file=dogs_step50.safetensors")
            return rec["files"], await one.read(), other.status
    files, body, other = asyncio.run(go())
    assert files == ["cats.safetensors", "cats_step50.safetensors", "cats_step50_2.safetensors", "cats_step200.safetensors", "cats_preview.safetensors"]
    assert body == b"cats_step200.safetensors" and other == 404
