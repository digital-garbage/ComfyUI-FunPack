"""LoRA training: a training set made from your media and rated results, tagged, then learned on ComfyUI's own trainer.

An add-on (Settings > Modules): it asks for onnxruntime, which only the automatic tagger needs. The window is the
Training feature of the app; this module holds what it stands on: the saved sets, the tagger models, rated results
as training data, the graph a run is queued as, and the training node. The learning itself is ComfyUI's Train LoRA
node, fed latents and conditioning built from the set, with the loaders the pipeline already has.

H3 trains picture and sound together; the repairs its latent and its rotary op need live in the H3 module and are
used when this module asks. UNTESTED on real weights or hardware: proven only on tiny stand-in models on CPU.
"""

from . import captions, jobs, plan, rated, store, taggers, train
from .nodes import FunPackTrainLora

ID = "training_lora"
TITLE = "LoRA training"
STAGE = "post"
CATEGORY = "system"
STATUS = "experimental"

NODES = [FunPackTrainLora]


def _bad(web, why, status=400):
    return web.json_response({"why": why}, status=status)


def routes(table, base, web):
    from ..._core import addons, projects, shortcuts

    def enabled():
        return addons.enabled("lora_training")

    @table.get(base + "/state")
    async def _state(_req):
        return web.json_response({"enabled": enabled(), "uses": train.USES, "algorithms": list(train.ALGORITHMS),
                                  "labels": list(store.LABELS), "rated_labels": rated.LABEL_TITLES})

    # --- training sets
    @table.get(base + "/sets")
    async def _sets(_req):
        return web.json_response({"sets": store.summaries()})

    @table.post(base + "/sets")
    async def _create(req):
        try:
            body = await req.json()
            name = str(body.get("name") or "")
            if not store.valid_name(name):
                return _bad(web, "A set's name is letters, numbers, spaces, - and _ (up to 60).")
            return web.json_response(store.create(name))
        except (ValueError, TypeError, AttributeError) as exc:
            return _bad(web, str(exc))

    @table.get(base + "/sets/{name}")
    async def _get(req):
        doc = store.get(req.match_info["name"]) if store.valid_name(req.match_info["name"]) else None
        return web.json_response(doc) if doc else _bad(web, "There is no such set.", 404)

    @table.put(base + "/sets/{name}")
    async def _put(req):
        name = req.match_info["name"]
        if not store.valid_name(name):
            return _bad(web, "That is not a set name.")
        try:
            return web.json_response(store.save(name, await req.json()))
        except (ValueError, TypeError) as exc:
            return _bad(web, str(exc))

    @table.delete(base + "/sets/{name}")
    async def _delete(req):
        name = req.match_info["name"]
        return web.json_response({"deleted": store.valid_name(name) and store.delete(name)})

    @table.post(base + "/sets/{name}/add")
    async def _add(req):
        name = req.match_info["name"]
        if not store.valid_name(name):
            return _bad(web, "That is not a set name.")
        try:
            body = await req.json()
            doc, added = store.add(name, body.get("entries"))
        except (ValueError, TypeError, AttributeError) as exc:
            return _bad(web, str(exc))
        return web.json_response({"set": doc, "added": added})

    @table.post(base + "/sets/{name}/bulk")
    async def _bulk(req):
        name = req.match_info["name"]
        doc = store.get(name) if store.valid_name(name) else None
        if doc is None:
            return _bad(web, "There is no such set.", 404)
        try:
            body = await req.json()
            ids = body.get("ids")
            changed = captions.bulk(doc["items"], body.get("op"), body.get("tag"), body.get("new"),
                                    set(ids) if isinstance(ids, list) else None)
            return web.json_response({"set": store.save(name, doc), "changed": changed})
        except (ValueError, TypeError, AttributeError) as exc:
            return _bad(web, str(exc))

    # --- rated results as training data
    @table.get(base + "/rated")
    async def _rated(req):
        docs = []
        for entry in projects.listing():
            project = projects.get(entry["id"])
            if project is not None:
                docs.append({**project.to_dict(), "id": entry["id"]})
        found = rated.scan(docs, lambda t: shortcuts.expand(t, seed=1))
        want = req.rel_url.query.get("label")
        if want:
            found = [r for r in found if r["label"] == want]
        counts = {label: sum(1 for r in found if r["label"] == label) for label in rated.LABEL_TITLES}
        return web.json_response({"results": found, "counts": counts})

    # --- automatic tagging
    @table.get(base + "/taggers")
    async def _taggers(_req):
        return web.json_response({"taggers": taggers.listing(), "job": {k: v for k, v in taggers.job.items() if k != "results"}})

    @table.post(base + "/taggers/download")
    async def _download(req):
        body = await req.json()
        started = taggers.download(str(body.get("tagger") or ""))
        return web.json_response({"started": started})

    @table.post(base + "/taggers/remove")
    async def _remove(req):
        body = await req.json()
        return web.json_response({"removed": taggers.remove(str(body.get("tagger") or ""))})

    @table.post(base + "/autotag")
    async def _autotag(req):
        body = await req.json()
        name = str(body.get("set") or "")
        doc = store.get(name) if store.valid_name(name) else None
        if doc is None:
            return _bad(web, "There is no such set.", 404)
        tagger = str(body.get("tagger") or "")
        if not taggers.ready(tagger):
            return _bad(web, "Download that tagger first.")
        ids = body.get("ids")
        chosen = [i for i in doc["items"] if not isinstance(ids, list) or i["id"] in ids]
        targets = []
        for item in chosen:
            path = store.resolve(item["src"], train.roots(), train.bin_path)
            if path is not None:
                targets.append((item["id"], path))
        options = {k: body[k] for k in ("general", "character", "characters", "blocked", "limit") if k in body}
        from . import frames
        started = taggers.autotag(tagger, targets, options, lambda p: frames.picture(p))
        return web.json_response({"started": started, "total": len(targets)})

    @table.get(base + "/autotag")
    async def _autotag_state(_req):
        return web.json_response(dict(taggers.job))

    @table.post(base + "/autotag/cancel")
    async def _autotag_cancel(_req):
        taggers.cancel()
        return web.json_response({"ok": True})

    # --- a run
    @table.post(base + "/plan")
    async def _plan(req):
        if not enabled():
            return _bad(web, "LoRA training is switched off. Turn it on in Settings > Modules.", 409)
        body = await req.json()
        name = str(body.get("set") or "")
        values = body.get("settings") if isinstance(body.get("settings"), dict) else {}
        use = str(values.get("use") or "normal")
        if not store.valid_name(name):
            return _bad(web, "Pick a training set first.")
        _items, problems = train.plan_items(name, use) if use in train.USES else ([], [f"use is one of {sorted(train.USES)}"])
        if problems:
            return web.json_response({"prompt": {}, "problems": problems})
        prompt, problems = plan.assemble(body.get("slots"), name, values)
        return web.json_response({"prompt": prompt, "problems": problems})

    @table.get(base + "/run")
    async def _run(req):
        try:
            since = int(req.rel_url.query.get("since") or 0)
        except ValueError:
            since = 0
        return web.json_response(jobs.snapshot(since))

    @table.post(base + "/run/stop")
    async def _stop(_req):
        stopped = jobs.request_stop()
        if stopped:
            try:
                import comfy.model_management
                comfy.model_management.interrupt_current_processing()
            except Exception:                              # noqa: BLE001 -- the next step ends the run anyway
                pass
        return web.json_response({"stopped": stopped})

    @table.get(base + "/runs")
    async def _runs(_req):
        return web.json_response({"runs": jobs.history()})

    @table.get(base + "/runs/{run}")
    async def _record(req):
        rec = jobs.record(req.match_info["run"])
        return web.json_response(rec) if rec else _bad(web, "There is no such run.", 404)

    @table.delete(base + "/runs/{run}")
    async def _forget(req):
        return web.json_response({"deleted": jobs.forget(req.match_info["run"])})


PROVIDES = {"routes": routes}
