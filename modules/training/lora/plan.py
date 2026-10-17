"""The graph a training run is queued as: the pipeline's own loaders, and the training node fed by them."""

from __future__ import annotations

from ..._core import graph

NODE = "FunPackTrainLora"
WANTED = (("model", "MODEL"), ("clip", "CLIP"), ("vae", "VAE"), ("audio_vae", "VAE"))


def _loaders(slots, schemas):
    """{input name: slot id}: the slots that hand the trainer its model, text encoder and VAEs.

    The pipeline names them (a model's pipeline calls them model, clip, vae and audio_vae); a
    pipeline that names them otherwise falls back to the first slot that makes that kind and
    takes none of that kind in, which is a loader and not a patch on one.
    """
    live = {s["id"]: s for s in slots if not s.get("bypassed")}
    found = {}
    for name, kind in WANTED:
        if name in live and kind in schemas.outputs(live[name]["node"]):
            found[name] = name
    for name, kind in WANTED[:3]:
        if name in found:
            continue
        for slot in live.values():
            if slot["id"] in found.values() or slot["id"] == "audio_vae":
                continue
            if kind in schemas.outputs(slot["node"]) and kind not in schemas.inputs(slot["node"]).values():
                found[name] = slot["id"]
                break
    return found


def _upstream(slots, roots):
    """The given slots and every slot they read from."""
    by_id = {s["id"]: s for s in slots}
    keep, todo = set(), list(roots)
    while todo:
        sid = todo.pop()
        if sid in keep or sid not in by_id:
            continue
        keep.add(sid)
        for value in (by_id[sid].get("inputs") or {}).values():
            if graph.is_link(value):
                todo.append(value[0])
    return [s for s in slots if s["id"] in keep]


def _model_file(slot):
    """The file and dtypes a plain diffusion model loader slot names, or {} when the slot is anything else."""
    given = slot.get("inputs") or {}
    name = given.get("model_name")
    if slot.get("node") != "FunPackDiffusionModelLoader" or not isinstance(name, str) or not name or name == "None":
        return {}
    return {"model_name": name, "weight_dtype": given.get("weight_dtype", "default"), "compute_dtype": given.get("compute_dtype", "default")}


def assemble(slots, set_name: str, values: dict, schemas=None):
    """(prompt, problems). `values` are the training node's settings; the loaders come from `slots`."""
    schemas = schemas or graph.from_comfyui()
    bad = graph.shape_problems(slots)
    if bad:
        return {}, bad
    found = _loaders(slots, schemas)
    missing = [n for n in ("model", "clip", "vae") if n not in found]
    if missing:
        return {}, [f"the pipeline has no {', '.join(missing)} loader: training needs the model, the text encoder and the VAE"]
    trainer = {"id": "train_lora", "node": NODE, "inputs": {"set_name": set_name, **values}}
    by_id = {s["id"]: s for s in slots}
    path = _model_file(by_id[found["model"]])
    if path:                                              # the trainer loads this file itself, once: no loader node runs in front of it
        trainer["inputs"].update(path)
        found = {n: sid for n, sid in found.items() if n != "model"}
    kept = _upstream([s for s in slots if not s.get("bypassed")], found.values())
    for name, sid in found.items():
        trainer["inputs"][name] = [sid, 0]
    return graph.build(kept + [trainer], schemas)
