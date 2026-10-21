"""Every kind of reference the Generate window can hand the H3 reference-to-video preset ends up in ComfyUI.

The chain is: what is picked in the UI -> `buildInputs` (JavaScript, run here through node) -> `graph.override` ->
`graph.build` -> ComfyUI's own `validate_prompt`. A reference that is neither wired nor reported as left out is a
silent drop, which is what this file exists to forbid: nine pictures, three videos and three sounds are the model's
ceiling, and each must reach its own socket.
"""

import asyncio
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from core import graph

pytestmark = pytest.mark.skipif(not shutil.which("node"), reason="node needed to run the UI's input builder")

BUILDER = Path(__file__).parent / "fixtures" / "build_inputs.mjs"


@pytest.fixture(scope="module")
def schemas(comfyui):
    import nodes as comfy_nodes

    async def load():
        await comfy_nodes.init_extra_nodes(init_custom_nodes=False)
        await comfy_nodes.load_custom_node(str(Path(__file__).resolve().parents[2]), module_parent="custom_nodes")

    asyncio.run(load())
    if "ImageTransformKJ" not in comfy_nodes.NODE_CLASS_MAPPINGS:       # a third-party pack the preset uses; same sockets
        class ImageTransformKJ:
            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {"image": ("IMAGE",), "target_width": ("INT", {"default": 512, "min": 0, "max": 16384}),
                                     "target_height": ("INT", {"default": 512, "min": 0, "max": 16384}),
                                     "upscale_method": (["nearest-exact", "bilinear", "area", "bicubic", "lanczos"],),
                                     "keep_proportion": (["stretch", "resize", "pad", "pad_edge", "crop", "total_pixels"],),
                                     "divisible_by": ("INT", {"default": 2, "min": 0, "max": 512}),
                                     "extra_padding": (["disabled", "enabled"],), "invert_crop": (["disabled", "enabled"],)},
                        "optional": {"bboxes": ("STRING", {"default": ""})}}
            RETURN_TYPES = ("IMAGE", "MASK", "BBOX", "INT", "INT", "INT")
            FUNCTION = "run"
            CATEGORY = "test"

            def run(self, **kw):
                return ()
        comfy_nodes.NODE_CLASS_MAPPINGS["ImageTransformKJ"] = ImageTransformKJ
    return graph.from_comfyui()


def _ids(n, prefix):
    return [f"{prefix}{i}" for i in range(n)]


def _scenario(pictures=0, videos=0, audios=0, silent_videos=0):
    pics, vids, auds = _ids(pictures, "p"), _ids(videos, "v"), _ids(audios, "a")
    kinds = {**{i: "image" for i in pics}, **{i: "video" for i in vids}, **{i: "audio" for i in auds}}
    return dict(refs=pics + vids + auds, kinds=kinds, sounds=vids[:videos - silent_videos])


def _run(slots, schemas, optional, refs, kinds, sounds):
    req = dict(slots=slots, scene=dict(text="a cat", references=refs), kinds=kinds, sounds=sounds,
               project={"video": {}, "negative": "", "fixed_references": {}}, frames=None,
               nodes={s["node"]: ({"inputs": f["inputs"], "outputs": f["outputs"]} if (f := schemas.of(s["node"])) else {"inputs": None, "outputs": None})
                      for s in slots})
    done = subprocess.run(["node", str(BUILDER)], input=json.dumps(req), capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-800:]
    built = json.loads(done.stdout)
    new, refused = graph.override(slots, built["inputs"])
    prompt, problems = graph.build(new, schemas, optional=optional)
    return built, refused, prompt, problems


def _validate(prompt):
    import execution
    ok, err, _, node_errors = asyncio.run(execution.validate_prompt("pid", prompt, None))
    return ok, err, node_errors


def _real_problems(items):
    """What is left when only the models and files this machine does not have are set aside."""
    return [p for p in items if "not one of" not in p and "nothing fills" not in p]


MATRIX = {
    "2 pictures": _scenario(pictures=2),
    "9 pictures": _scenario(pictures=9),
    "1 video with sound": _scenario(videos=1),
    "1 silent video": _scenario(videos=1, silent_videos=1),
    "3 videos": _scenario(videos=3),
    "1 audio": _scenario(audios=1),
    "3 audios": _scenario(audios=3),
    "a mix": _scenario(pictures=2, videos=1, audios=1),
    "every slot full": _scenario(pictures=9, videos=3, audios=3),
}


@pytest.mark.parametrize("name", MATRIX)
def test_h3_reference_reaches_its_socket_and_comfyui_accepts_the_graph(name, schemas):
    from modules.models.minimax_h3 import pipeline as h3
    scene = MATRIX[name]
    slots = h3.h3_reference_to_video()
    built, refused, prompt, problems = _run(slots, schemas, h3.OPTIONAL_NODES, scene["refs"], scene["kinds"], scene["sounds"])

    assert not refused
    assert not _real_problems(problems), problems
    assert built["left"] == [] and not built["unwired"], f"dropped: {built['left']} {built['unwired']}"

    revealed = [k.split(".")[0] for k in prompt["r2v"]["inputs"] if "." in k]
    for kind, group in (("image", "ref_images"), ("video", "ref_videos"), ("audio", "ref_audios")):
        assert revealed.count(group) == sum(v == kind for v in scene["kinds"].values()), (group, revealed)

    ok, err, node_errors = _validate(prompt)
    # No model is installed here, so the loaders' own inputs are unfilled; the reference sockets are what is under test.
    leftovers = {n: [(e["message"], e.get("extra_info", {}).get("input_name")) for e in v["errors"]
                     if str(e.get("extra_info", {}).get("input_name", "")).startswith("ref_")]
                 for n, v in (node_errors or {}).items()}
    assert not any(leftovers.values()), leftovers


def test_more_than_the_models_ceiling_is_reported_not_dropped_silently(schemas):
    from modules.models.minimax_h3 import pipeline as h3
    scene = _scenario(pictures=10)
    built, _, prompt, _ = _run(h3.h3_reference_to_video(), schemas, h3.OPTIONAL_NODES, scene["refs"], scene["kinds"], scene["sounds"])
    assert [item["id"] for item in built["left"]] == ["p9"] and "only 9 images" in built["left"][0]["why"]
    assert sum(k.startswith("ref_images.") for k in prompt["r2v"]["inputs"]) == 9


def _presets():
    from modules.models.ltx import pipeline as ltx
    from modules.models.minimax_h3 import pipeline as h3
    from modules.models.sd15 import text_to_image
    found = [("sd15", text_to_image, {})]
    found += [(p["id"], (lambda p=p: p["slots"]), ltx.OPTIONAL_NODES) for p in ltx.presets()]
    found += [(p["id"], (lambda p=p: p["slots"]), h3.OPTIONAL_NODES) for p in h3.presets()]
    return found


SCENES = {
    "two pictures": dict(refs=["m0", "m1"], kinds={"m0": "image", "m1": "image"}, sounds=[], source=None),
    "a video with sound": dict(refs=["v"], kinds={"v": "video"}, sounds=["v"], source=None),
    "an audio": dict(refs=["a"], kinds={"a": "audio"}, sounds=[], source=None),
    "a start picture": dict(refs=[], kinds={"s": "image"}, sounds=[], source="s"),
    "start picture and a reference": dict(refs=["p"], kinds={"s": "image", "p": "image"}, sounds=[], source="s"),
}


def _placed(inputs):
    """Every media id some slot was told to load."""
    return {v for fields in inputs.values() for k, v in fields.items() if k == "media_id" and v}


@pytest.mark.parametrize("scene", SCENES)
@pytest.mark.parametrize("preset", [p[0] for p in _presets()])
def test_in_every_preset_a_picked_file_is_used_or_named_as_left_out(preset, scene, schemas):
    """The silent drop this audit started from: a file picked in the UI that reaches no node and is not mentioned."""
    _, make, optional = next(p for p in _presets() if p[0] == preset)
    spec = SCENES[scene]
    slots = make()
    req_scene = dict(text="a cat", references=spec["refs"], **({"source_image": spec["source"]} if spec["source"] else {}))
    req = dict(slots=slots, scene=req_scene, kinds=spec["kinds"], sounds=spec["sounds"],
               project={"video": {}, "negative": "", "fixed_references": {}}, frames=None,
               nodes={s["node"]: ({"inputs": f["inputs"], "outputs": f["outputs"]} if (f := schemas.of(s["node"])) else {"inputs": None, "outputs": None})
                      for s in slots})
    done = subprocess.run(["node", str(BUILDER)], input=json.dumps(req), capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-800:]
    built = json.loads(done.stdout)
    picked = set(spec["refs"]) | ({spec["source"]} if spec["source"] else set())
    named = {item["id"] for item in built["left"]}
    assert picked <= _placed(built["inputs"]) | named, f"silently ignored: {sorted(picked - _placed(built['inputs']) - named)}"
    new, refused = graph.override(slots, built["inputs"])
    assert not refused
    assert not _real_problems(graph.build(new, schemas, optional=optional)[1])
