"""LoRA training: a training set made from your media and rated results, tagged, then learned on ComfyUI's own trainer.

An add-on (Settings > Modules): it asks for onnxruntime, which only the automatic tagger needs. The window is the
Training feature of the app; this module holds what it stands on: the saved sets, the tagger models, rated results
as training data, the graph a run is queued as, and the training node. The learning itself is ComfyUI's Train LoRA
node, fed latents and conditioning built from the set, with the loaders the pipeline already has.

H3 trains picture and sound together; the repairs its latent and its rotary op need live in the H3 module and are
used when this module asks. UNTESTED on real weights or hardware: proven only on tiny stand-in models on CPU.
"""

from . import captions, jobs, loop, media_io, plan, rated, store, taggers, train
from .nodes import FunPackTrainLora

ID = "training_lora"
TITLE = "LoRA training"
STAGE = "post"
CATEGORY = "system"
STATUS = "experimental"

NODES = [FunPackTrainLora]


def _bad(web, why, status=400):
    return web.json_response({"why": why}, status=status)


def _loras_folders():
    from pathlib import Path
    try:
        import folder_paths
        return [Path(f) for f in folder_paths.get_folder_paths("loras")]
    except Exception:                                      # noqa: BLE001 -- no ComfyUI folders here
        return []


def saved_files(rec) -> list[str]:
    """The files a run left in the loras folder: its finished LoRA, the copies saved along the way (name_stepN) and its preview."""
    import re
    name = (rec or {}).get("name") or ""
    stem = "".join(c for c in name if c.isalnum() or c in "-_ .").strip() or "funpack_lora"
    pattern = re.compile(rf"{re.escape(stem)}_(step(\d+)|preview)(_\d+)?\.safetensors")
    found = set()
    for folder in _loras_folders():
        try:
            found.update(p.name for p in folder.iterdir() if p.is_file() and pattern.fullmatch(p.name))
        except OSError:
            pass
    final = (rec or {}).get("lora") or ""
    order = lambda f: (0, 0, f) if f == final else ((2, 0, f) if "_preview" in f else (1, int(re.search(r"_step(\d+)", f).group(1)), f))
    return sorted(found | ({final} if final and any((d / final).is_file() for d in _loras_folders()) else set()), key=order)


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
            entries = body.get("entries")
            roots = train.roots()
            for entry in entries if isinstance(entries, list) else []:           # a file added without words is captioned with the prompt it was made from, read from the file itself
                src = store.clean_src(entry.get("src")) if isinstance(entry, dict) else None
                if src and not str(entry.get("caption") or "").strip():
                    path = store.resolve(src, roots, train.bin_path)
                    entry["caption"] = rated.embedded_prompt(path) if path else ""
            doc, added = store.add(name, entries)
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
        parts = [p for p in (req.rel_url.query.get("include") or "main").split(",") if p in rated.INCLUDE]
        found = rated.scan(docs, lambda t: shortcuts.expand(t, seed=1), shortcuts.resolve_variables, parts)
        roots = train.roots()
        here = [(r, store.resolve({"kind": "output", **r["media"]}, roots, train.bin_path)) for r in found]
        here = [(r, path) for r, path in here if path]      # a rating is not a video: only results still on this machine
        found = [r for r, path in here if rated.same_video(r["prompt_text"], path)]       # ...and the same video: a project carried to another instance meets other files under the same names
        different = len(here) - len(found)
        want = req.rel_url.query.get("label")
        if want:
            found = [r for r in found if r["label"] == want]
        counts = {label: sum(1 for r in found if r["label"] == label) for label in rated.LABEL_TITLES}
        return web.json_response({"results": found, "counts": counts, "different": different})

    @table.post(base + "/clips")
    async def _clips(req):
        """Frame count and rate of video sources, by source key: what the pickers show beside a clip."""
        body = await req.json()
        out, roots = {}, train.roots()
        for raw in (body.get("srcs") or [])[:500] if isinstance(body, dict) else []:
            src = store.clean_src(raw)
            if src is None:
                continue
            path = store.resolve(src, roots, train.bin_path)
            info = media_io.clip_info(path) if path and store.kind_of(path.name) == "video" else None
            if info:
                out[store.src_key(src)] = info
        return web.json_response({"clips": out})

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

    @table.post(base + "/run/pause")
    async def _pause(_req):
        return web.json_response({"paused": jobs.request_pause()})

    @table.post(base + "/run/stop")
    async def _stop(_req):
        now = jobs.snapshot()
        was_paused = now["state"] == "paused"
        stopped = jobs.request_stop()
        if stopped and was_paused:
            train.discard(now["run"], now["name"])
            return web.json_response({"stopped": True})
        if stopped:
            try:
                import comfy.model_management
                comfy.model_management.interrupt_current_processing()
            except Exception:                              # noqa: BLE001 -- the next step ends the run anyway
                pass
        return web.json_response({"stopped": stopped})

    @table.get(base + "/runs")
    async def _runs(_req):
        return web.json_response({"runs": [{**r, "resumable": loop.has_checkpoint(r["run"])} for r in jobs.history()]})

    @table.get(base + "/runs/{run}")
    async def _record(req):
        rec = jobs.record(req.match_info["run"])
        return web.json_response({**rec, "resumable": loop.has_checkpoint(rec.get("run")), "files": saved_files(rec)}) if rec else _bad(web, "There is no such run.", 404)

    @table.get(base + "/runs/{run}/lora")
    async def _download(req):
        from pathlib import Path
        from urllib.parse import quote
        rec = jobs.record(req.match_info["run"])
        name = req.query.get("file") or (rec or {}).get("lora") or ""
        if not rec or name not in saved_files(rec) or Path(name).name != name:
            return _bad(web, "This run has no such saved LoRA file.", 404)
        path = next((f / name for f in _loras_folders() if (f / name).is_file()), None)
        if path is None:
            return _bad(web, f"{name} is no longer in your loras folder.", 404)
        return web.FileResponse(path, headers={"Content-Type": "application/octet-stream", "Content-Disposition": f"attachment; filename*=UTF-8''{quote(name)}"})

    @table.delete(base + "/runs/{run}")
    async def _forget(req):
        run = req.match_info["run"]
        gone = jobs.forget(run)
        if gone:
            loop.forget_checkpoint(run)
        return web.json_response({"deleted": gone})


PROVIDES = {"routes": routes}
