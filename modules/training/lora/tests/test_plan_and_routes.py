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
    assert inputs["model"] == ["model", 0] and inputs["audio_vae"] == ["audio_vae", 0] and inputs["set_name"] == "mine" and inputs["steps"] == 10
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
