"""Route registration on ComfyUI's aiohttp server.

`register()` takes the route table rather than reaching for PromptServer itself,
so the real routes can be mounted on a throwaway app in a test. The handlers are
thin adapters over pure functions in `serve`.
"""

import asyncio
import json

from . import (backend_log, config, graph as graph_mod, log, media, nodes_manager, probe as probe_mod,
               projects,
               prompt_build,
               shortcuts as shortcuts_mod,
               sysinfo,
               temp_files,
               update as update_mod,
               registry as registry_mod, serve as static, widgets)
from .contract import CONTRACT_VERSION
from .relations import order
from .traits import split

# Two independent imports, two independent guards: aiohttp is installable on its
# own, and folding them together nulled `web` whenever ComfyUI's `server` was
# absent -- which is every test run.
try:
    from aiohttp import web
except Exception:
    web = None

try:
    from server import PromptServer
except Exception:  # not running inside ComfyUI
    PromptServer = None


_pending_restart = False
"""Set when a git change landed but a generation blocked the restart it needs.

Deliberately module state, not per-request: the change is already on disk and
ComfyUI is already running stale code for it. The one way out is `/api/git/
restart`, gated on the same running-check -- nothing re-runs `pull`/`checkout`/
`rollback` to "retry", because by the time this fires HEAD has already moved and
running that action again would no longer mean what it did the first time
(rollback especially: run twice, it undoes itself).
"""

def _generation_running():
    """Whether ComfyUI is mid-generation right now, if this can even be asked.

    The client already disables the update/checkout/rollback buttons while a run
    is in flight, but that check goes stale the moment the dialog is left open
    across a run starting -- this is the one that actually stops the restart.
    Module-level (rather than nested in `register`) so a test can replace it
    without a real PromptServer.
    """
    instance = getattr(PromptServer, "instance", None)
    if instance is None:
        return False
    try:
        return instance.prompt_queue.get_tasks_remaining() > 0
    except Exception:
        return False


def _schedule_restart():
    """Restart ComfyUI shortly, unless a generation started in the meantime.

    The 0.7s delay (so the response reaches the client before the process
    goes) is itself a check-then-act window: `_generation_running()` was
    false when this was scheduled, but nothing stops a generation being
    queued in the gap. Checked again right here, at the moment that matters,
    instead of trusting the answer from 0.7s ago -- a restart scheduled while
    idle and fired while running is exactly the thing this whole mechanism
    exists to prevent.
    """
    global _pending_restart

    def _fire():
        global _pending_restart
        if _generation_running():
            _pending_restart = True
            return
        from . import restart as restart_mod
        restart_mod.restart()

    asyncio.get_event_loop().call_later(0.7, _fire)


def _respond(served):
    if served.status != 200:
        return web.Response(status=served.status, headers=served.headers)
    return web.Response(
        body=served.body,
        content_type=served.content_type,
        headers=served.headers,
    )


def _serve_under(req, root, allowed):
    # match_info keeps dot-segments intact for a request as it arrives on the
    # wire (aiohttp does not collapse them during matching), so a traversal
    # reaches resolve() and is refused there. Note that aiohttp's own TestClient
    # normalises the URL before sending -- a test written with it cannot see
    # this path at all, which is why test_routes.py uses a raw socket.
    return _respond(static.serve(root, req.match_info["tail"], allowed))


def modules(rescan=False):
    """The one shared scan. Held in `registry` so the nodes and the app cannot
    end up describing different sets of modules in the same session."""
    return registry_mod.current(rescan)


def manifest(traits=None):
    """What the browser is told: only modules that loaded and validated.

    A module that failed is absent from `modules` entirely -- the UI renders
    what announced itself, so absence here is what makes it absent on screen.
    `failed` is carried alongside for the modules dump, never for rendering.
    """
    reg = modules()
    specs = list(reg.specs.values())

    incompatible = []
    if traits is not None:
        specs, incompatible = split(specs, traits)

    ordered, rejected = order(specs)

    return {
        "contract": CONTRACT_VERSION,
        "modules": [spec.to_manifest() for spec in ordered],
        "failed": (
            [{"where": where, "why": why} for where, why in reg.failed]
            + [{"where": spec.source, "why": why} for spec, why in rejected]
        ),
        "incompatible": [
            {"id": spec.id, "requires": spec.requires} for spec in incompatible
        ],
    }


def register(routes, prefix=None):
    """Attach FunPack's routes to an aiohttp route table."""
    P = config.UI_PREFIX if prefix is None else prefix

    # Serializes every git-mutating route (including the remote fetch inside
    # status). Created fresh here, not at module import: an `asyncio.Lock`
    # binds to whichever event loop first acquires it, and register() runs
    # once per real server process but once per TEST too, each on its own
    # loop -- a module-level lock reused across those raised "bound to a
    # different event loop" the moment a second test tried it.
    #
    # Without it, two requests -- two tabs, or a double-click before the
    # client's own disable takes effect -- could both pass the
    # `_pending_restart` check before either had set it, and then run two real
    # `git` subprocesses against the same working tree at once: index.lock
    # contention at best, a checkout's branch switch interleaved with a
    # pull's fetch/merge at worst.
    _git_lock = asyncio.Lock()

    @routes.get(P + "/api/health")
    async def _health(_req):
        return web.json_response({"ok": True})

    @routes.get(P + "/api/modules")
    async def _modules(req):
        raw = req.query.get("traits")
        traits = [t for t in raw.split(",") if t] if raw is not None else None
        return web.json_response(manifest(traits))

    def _pipeline():
        """Whatever module offers a default pipeline, or nothing.

        Core does not hold one: a default pipeline is feature content, and the
        point of it being data is that it can be replaced by other data.
        """
        for _spec, make in modules().providers("default_pipeline"):
            return make()
        return []

    def _sinks():
        """Every place a module says the app's settings can be put."""
        found = []
        for _spec, make in modules().providers("settings_sink"):
            try:
                sink = make()
            except Exception as exc:  # noqa: BLE001
                log.failed("settings_sink", exc)
                continue
            if isinstance(sink, dict):
                found.append(sink)
        return found

    @routes.get(P + "/api/pipeline")
    async def _pipeline_get(_req):
        slots = _pipeline()
        prompt, incomplete = graph_mod.build(slots)
        return web.json_response({"slots": slots, "refused": [],
                                  "incomplete": incomplete,
                                  "queueable": not incomplete})

    @routes.post(P + "/api/pipeline")
    async def _pipeline_edit(req):
        """Replace or remove a slot, and say what that did to the graph.

        This is what makes "a built-in node can be swapped or removed" a thing
        the running server can do rather than a property of a library nothing
        calls. Refusals come back as data, so the app can show the reason.
        """
        try:
            body = await req.json()
        except Exception:  # noqa: BLE001
            return web.json_response({"problems": ["that is not JSON"]}, status=400)

        # Everything below reads named fields off the body, so the body has to
        # be a thing with names. `[]` and `"slots"` are both valid JSON and
        # neither is a request; asking them for `.get` is an AttributeError and
        # a 500 nothing can read.
        if not isinstance(body, dict):
            return web.json_response(
                {"problems": [f"a request is an object, not a {type(body).__name__}"],
                 "queueable": False}, status=400)

        # `or` would resurrect the default here: an explicitly empty pipeline is
        # falsy, and a client that has removed every slot is entitled to be told
        # it has none rather than handed the defaults back.
        slots = body.get("slots")
        if slots is None:
            slots = _pipeline()
        action, slot_id = body.get("action"), body.get("slot")

        # A malformed payload is a bad REQUEST, not an unfinished pipeline --
        # so it comes back as a refusal with a reason and a 400, rather than
        # among the "you have not picked a model yet" notes.
        malformed = graph_mod.shape_problems(slots)
        if malformed:
            return web.json_response(
                {"refused": malformed, "incomplete": [], "queueable": False,
                 "slots": [], "prompt": None},
                status=400)

        # An id and a node name are strings, and only a string can be looked up
        # in the dicts below: `{"slot": ["a"]}` reaches a `in`-on-dict as an
        # unhashable key and the refusal turns into a 500. The slots array is
        # shape-checked above; these two fields are read from the same body and
        # were not.
        if action in ("replace", "remove", "wire", "unwire") and not isinstance(slot_id, str):
            return web.json_response(
                {"problems": [f"which slot to {action} is named by a string, "
                              f"not a {type(slot_id).__name__}"],
                 "queueable": False}, status=400)
        if action == "replace" and not isinstance(body.get("node"), str):
            return web.json_response(
                {"problems": [f"a node is named by a string, not a "
                              f"{type(body.get('node')).__name__}"],
                 "queueable": False}, status=400)
        # wire/unwire both name an INPUT on `slot`; wire additionally names
        # where it is fed from. Checked as a group because a malformed field in
        # either reaches graph.wire()/unwire() as a dict key or an index and
        # fails somewhere that is not this readable.
        if action in ("wire", "unwire") and not isinstance(body.get("input"), str):
            return web.json_response(
                {"problems": [f"which input to {action} is named by a string, "
                              f"not a {type(body.get('input')).__name__}"],
                 "queueable": False}, status=400)
        if action == "wire":
            if not isinstance(body.get("from_slot"), str):
                return web.json_response(
                    {"problems": [f"which slot feeds it is named by a string, not a "
                                  f"{type(body.get('from_slot')).__name__}"],
                     "queueable": False}, status=400)
            from_output = body.get("from_output")
            if isinstance(from_output, bool) or not isinstance(from_output, int):
                return web.json_response(
                    {"problems": [f"which output is a number, not a "
                                  f"{type(from_output).__name__}"],
                     "queueable": False}, status=400)

        if action == "replace":
            slots, problems = graph_mod.replace(slots, slot_id, body.get("node"))
        elif action == "remove":
            slots, problems = graph_mod.remove(slots, slot_id)
        elif action == "wire":
            slots, problems = graph_mod.wire(
                slots, slot_id, body.get("input"), body.get("from_slot"), body.get("from_output"))
        elif action == "unwire":
            slots, problems = graph_mod.unwire(slots, slot_id, body.get("input"))
        elif action in (None, "check"):
            problems = []
        else:
            return web.json_response({"problems": [f"unknown action {action!r}"]}, status=400)

        # Two different things, kept apart. "refused" means the edit did not
        # happen. "incomplete" means it did and the pipeline still is not ready
        # -- an unset file picker on a fresh install is the normal case, not a
        # failed edit, and an app that showed them together would say the wrong
        # thing about both.
        # Values the app is showing for named inputs -- the prompt box, and
        # whatever else the pipeline says belongs on the main window. Addressed
        # by slot rather than sent as a whole pipeline: the window that owns the
        # STRUCTURE and a control that owns one VALUE then cannot disagree about
        # the slots between them.
        edits = body.get("inputs")
        if edits is not None:
            slots, refused = graph_mod.override(slots, edits)
            problems = list(problems) + refused

        # What the UI holds, on its way into the graph. Sent with the pipeline
        # rather than held on the server: two stores of "what the user picked"
        # is two answers to what a run used, and the one believed would be
        # whichever was written last.
        values, notes = body.get("values"), []
        if values is not None:
            if not isinstance(values, dict):
                return web.json_response(
                    {"problems": [f"settings are an object keyed by module id, "
                                  f"not a {type(values).__name__}"],
                     "queueable": False}, status=400)
            slots, placed = graph_mod.place(slots, json.dumps(values), _sinks())
            # Said, not swallowed. A pipeline with nothing to accept them is a
            # legitimate pipeline, so this does not stop the run -- but a panel
            # full of switches that do nothing has to say so somewhere, and
            # beside Generate is where the run is started.
            if values and not placed:
                notes.append("nothing in this pipeline accepts module settings, "
                             "so what is set in the panels will not be applied")

        prompt, incomplete = graph_mod.build(slots)
        return web.json_response({
            "slots": slots,
            "refused": problems,
            "incomplete": incomplete,
            "notes": notes,
            "queueable": not (problems or incomplete),
            "prompt": prompt if not (problems or incomplete) else None,
        })

    # --- keeping the install alive -------------------------------------------
    #
    # Updating is the most-used feature in a pack whose value is experimental
    # work shipped continuously, so it is in the app rather than in a terminal.
    #
    # Every one of these ends by relaunching ComfyUI, which means the answer has
    # to be written BEFORE the process goes. They schedule the restart and return
    # `restarting: true`; the app then polls /api/health until the server is back.
    #
    # `restart` is looked up on the module rather than captured, so a test can
    # replace it -- otherwise running the suite would relaunch the test runner.

    async def _git(action, **kwargs):
        """Run one git operation off the event loop, or say why it did not."""
        global _pending_restart
        async with _git_lock:
            if _pending_restart:
                # A change already landed and is only waiting on the restart --
                # running another git action on top of it before that restart
                # happens is not "retrying", it is a second change stacked on an
                # unapplied one. Send them to finish the first restart instead.
                return web.json_response({
                    "detail": "An earlier update is applied and waiting on a restart "
                              "(Settings ▸ Updates). Restart before doing anything else here.",
                }, status=409)
            try:
                result = await asyncio.to_thread(action, **kwargs)
            except update_mod.GitUpdateError as exc:
                # A refusal, not a crash: no git, not a checkout, a dirty tree, a
                # branch that exists on neither side. Each names what to do.
                return web.json_response({"detail": str(exc)}, status=400)
            except Exception as exc:  # noqa: BLE001
                log.broke("git", exc, doing=getattr(action, "__name__", "git"))
                return web.json_response({"detail": f"{type(exc).__name__}: {exc}"}, status=500)

            result = result or {}
            # ONLY when the checkout actually moved. Pressing Update while already
            # up to date is a normal thing to do -- there is no way to know until
            # it has been asked -- and restarting for it costs a boot and, if a
            # generation is running, the generation.
            moved = bool(result.get("updated")) or (
                result.get("before") is not None and result.get("before") != result.get("after")
            ) or (
                result.get("before_branch") is not None
                and result.get("before_branch") != result.get("branch")
            )
            if not moved:
                return web.json_response({"restarting": False, **result})

            if _generation_running():
                _pending_restart = True
                return web.json_response({
                    "restarting": False,
                    "blocked": "A generation is running. The update is applied -- "
                               "restart it from this window once the generation finishes.",
                    **result,
                })

            _pending_restart = False
            # After the response is written, not before.
            _schedule_restart()
            return web.json_response({"restarting": True, **result})

    @routes.post(P + "/api/git/restart")
    async def _git_restart(_req):
        """Restart ComfyUI: finishing one a running generation deferred, or one
        asked for directly with nothing owed. Either way there is exactly one
        thing to do -- relaunch -- so there is one route for both rather than
        a second copy of this same guard living behind its own button.

        Never re-runs a git action: if a restart WAS pending, HEAD already
        moved when the change first landed, and there is nothing left to do
        here but the restart itself.
        """
        global _pending_restart
        async with _git_lock:
            if _generation_running():
                return web.json_response({
                    "restarting": False,
                    "blocked": "A generation is running. Wait for it to finish, "
                              "or cancel it, then restart.",
                })
            _pending_restart = False
            _schedule_restart()
            return web.json_response({"restarting": True})

    @routes.get(P + "/api/git/status")
    async def _git_status(req):
        # `?remote=0` answers from the checkout alone. The fetch is the only part
        # that touches the network and so the only part that can hang; the app
        # asks without it first so it has something to draw.
        #
        # `remote=1` runs its own `git fetch` -- a real mutation of this
        # checkout's refs, same as pull/checkout/rollback -- so it shares their
        # lock. Without it, opening this window mid-update raced two `git fetch`
        # invocations against the same repo, which can interleave writes to
        # FETCH_HEAD and produce a "cannot fast-forward to multiple branches"
        # failure, or a false "you have local commits at risk" refusal, that had
        # nothing to do with either fetch's own branch.
        remote = req.query.get("remote", "1") not in ("0", "false", "no")
        async with _git_lock:
            payload = await asyncio.to_thread(update_mod.status, remote=remote)
        payload["restart_pending"] = _pending_restart
        return web.json_response(payload)

    @routes.get(P + "/api/system")
    async def _system_info(_req):
        # The machine ComfyUI runs on, not the browser -- on a rental those are
        # different machines, and the host is the one worth looking at.
        # to_thread: this shells out (sysctl) and can wait on a GPU query.
        payload = await asyncio.to_thread(sysinfo.collect)
        return web.json_response(payload)

    @routes.get(P + "/api/probe")
    async def _probe(req):
        """Which model family a checkpoint file is, before anything loads it.

        Reads a safetensors header off disk -- to_thread because that is
        real (if small) file I/O -- and asks every installed model module
        whether the tensor names are its own. Nothing here names a family:
        that answer comes back from whichever module claimed it.
        """
        filename = (req.query.get("file") or "").strip()
        if not filename:
            return web.json_response(
                {"problems": ["which file is named by ?file="]}, status=400)
        path = probe_mod.resolve_diffusion_model(filename)
        if path is None:
            return web.json_response({
                "module": None, "title": None, "detected": False,
                "reason": f"{filename}: not found in the models folders",
            })
        payload = await asyncio.to_thread(probe_mod.detect, path)
        return web.json_response(payload)

    @routes.post(P + "/api/git/update")
    async def _git_pull(req):
        body = await req.json() if req.can_read_body else {}
        branch = str((body or {}).get("branch") or "").strip() or None
        return await _git(update_mod.pull, branch=branch, install_deps=True)

    @routes.post(P + "/api/git/checkout")
    async def _git_checkout(req):
        body = await req.json() if req.can_read_body else {}
        branch = str((body or {}).get("branch") or "").strip()
        if not branch:
            return web.json_response({"detail": "Which branch? Name one."}, status=400)
        return await _git(update_mod.checkout, branch=branch,
                          pull_after=True, install_deps=True)

    @routes.post(P + "/api/git/rollback")
    async def _git_rollback(_req):
        return await _git(update_mod.rollback)

    @routes.get(P + "/api/log")
    async def _backend_log(req):
        """What ComfyUI printed. Read on every poll rather than cached: the
        reason anyone is looking is that something is happening right now."""
        import asyncio
        try:
            limit = int(req.query.get("limit", 600))
        except (TypeError, ValueError):
            limit = 600
        return web.json_response(await asyncio.to_thread(backend_log.recent, limit))

    @routes.get(P + "/api/temp")
    async def _temp(req):
        """Transient outputs, newest first. Walking a directory is filesystem
        work, so it goes off the loop like everything else here."""
        import asyncio
        try:
            limit = int(req.query.get("limit", temp_files.MAX_FILES))
        except (TypeError, ValueError):
            limit = temp_files.MAX_FILES
        return web.json_response(await asyncio.to_thread(temp_files.listing, limit))

    # --- node packs -----------------------------------------------------------
    #
    # A stand-in for ComfyUI-Manager's three operations, so a user is not sent to
    # another UI to add the one pack a workflow needs. There is no catalogue: the
    # URL is theirs, which keeps this honest about being git in a directory.
    #
    # Every one of these can take minutes (clone, pip), so they run off the loop.

    async def _pack(action, *args):
        import asyncio
        try:
            return web.json_response(await asyncio.to_thread(action, *args))
        except nodes_manager.CustomNodeError as exc:
            # Refusals name the pack and what to do: not a name, not installed,
            # not inside custom_nodes, or FunPack itself.
            return web.json_response({"detail": str(exc)}, status=400)
        except Exception as exc:  # noqa: BLE001
            log.broke("node packs", exc, doing=getattr(action, "__name__", "pack"))
            return web.json_response({"detail": f"{type(exc).__name__}: {exc}"}, status=500)

    @routes.get(P + "/api/packs")
    async def _packs(_req):
        return await _pack(nodes_manager.list_nodes)

    @routes.post(P + "/api/packs/check")
    async def _packs_check(_req):
        return await _pack(nodes_manager.check_updates)

    @routes.post(P + "/api/packs/install")
    async def _packs_install(req):
        body = await req.json() if req.can_read_body else {}
        return await _pack(nodes_manager.install, str((body or {}).get("url") or ""))

    @routes.post(P + "/api/packs/update")
    async def _packs_update(req):
        body = await req.json() if req.can_read_body else {}
        return await _pack(nodes_manager.update, str((body or {}).get("name") or ""))

    @routes.post(P + "/api/packs/remove")
    async def _packs_remove(req):
        body = await req.json() if req.can_read_body else {}
        return await _pack(nodes_manager.remove, str((body or {}).get("name") or ""))

    @routes.get(P + "/api/nodes")
    async def _nodes(req):
        """What the nodes in a pipeline look like to someone editing them.

        Asked for by name -- `?classes=A,B,C` -- rather than served whole.
        ComfyUI's own /object_info answers with every installed node, which on a
        machine with a few packs is megabytes, and a pipeline needs a dozen.

        A node that is not installed comes back as null rather than missing, so
        the app can tell "this slot points at something absent" from "I did not
        ask about that one".
        """
        raw = req.query.get("classes") or ""
        wanted = [name for name in (part.strip() for part in raw.split(",")) if name]
        if not wanted:
            return web.json_response({"nodes": {}})
        # Bounded: the query string is the caller's, and describing a thousand
        # nodes one by one on the event loop is a request that stops the server
        # answering anything else.
        if len(wanted) > 200:
            return web.json_response(
                {"problems": [f"asked about {len(wanted)} nodes at once; 200 is the limit"]},
                status=400)
        return web.json_response({"nodes": widgets.describe_all(wanted)})

    @routes.get(P + "/api/nodes/search")
    async def _nodes_search(req):
        """What could go in a slot, by name.

        Separate from the route above because it answers a different question:
        that one describes nodes the caller already named, this one finds the
        name. Bounded the same way and for the same reason -- the query string
        is the caller's, and the answer is built on the event loop.
        """
        try:
            limit = int(req.query.get("limit", 40))
        except ValueError:
            limit = 40
        return web.json_response(
            widgets.search(req.query.get("q", ""), max(1, min(limit, 200))))

    # ── projects ──────────────────────────────────────────────────────────
    # What the timeline IS. A run produces one clip; a project is the ordered
    # list of them the user is making, and the only part of the app that has to
    # survive a reload.

    async def _body(req):
        """The request's JSON object, or None. A body that is not an object is
        not a project edit -- `[]` and `"x"` both reach `.get` otherwise."""
        try:
            data = await req.json()
        except Exception:  # noqa: BLE001 - a malformed body is the caller's
            return None
        return data if isinstance(data, dict) else None

    @routes.get(P + "/api/projects")
    async def _projects_list(_req):
        return web.json_response({"projects": projects.listing()})

    @routes.post(P + "/api/projects")
    async def _projects_create(req):
        body = await _body(req) or {}
        return web.json_response(projects.create(body.get("name")).to_dict())

    @routes.get(P + "/api/projects/{pid}")
    async def _projects_get(req):
        found = projects.get(req.match_info["pid"])
        if found is None:
            return web.json_response({"problems": ["no such project"]}, status=404)
        return web.json_response(found.to_dict())

    @routes.put(P + "/api/projects/{pid}")
    async def _projects_save(req):
        pid = req.match_info["pid"]
        body = await _body(req)
        if body is None:
            return web.json_response({"problems": ["expected a project object"]}, status=400)
        if projects.get(pid) is None:
            return web.json_response({"problems": ["no such project"]}, status=404)
        # The id comes from the URL, never the body: a PUT that names its own
        # target is a PUT that can write over a different project.
        return web.json_response(
            projects.save(projects.Project.from_dict({**body, "id": pid})).to_dict())

    @routes.delete(P + "/api/projects/{pid}")
    async def _projects_delete(req):
        if not projects.delete(req.match_info["pid"]):
            return web.json_response({"problems": ["no such project"]}, status=404)
        return web.json_response({"deleted": True})

    @routes.post(P + "/api/projects/import")
    async def _projects_import(req):
        body = await _body(req)
        if body is None or "scenes" not in body:
            return web.json_response(
                {"problems": ["that does not look like a project file"]}, status=400)
        # Never the id in the file: a project imported twice, or imported on
        # the machine it came from, must land as its OWN project -- an id that
        # happens to match one already here would overwrite it on save.
        body = {**body, "id": None}
        # from_dict() itself is lenient by design -- everything it cannot make
        # sense of degrades to a safe default rather than raising, matching
        # every other project load. Only save()'s own disk I/O can actually
        # fail here, and that is not the file's fault: calling it "invalid"
        # sends someone to inspect their JSON when the real problem is on this
        # end (disk full, permissions).
        try:
            saved = projects.save(projects.Project.from_dict(body))
        except Exception as exc:  # noqa: BLE001
            log.broke("project import", exc, doing="saving the imported project")
            return web.json_response(
                {"problems": [f"could not save the imported project: {exc}"]}, status=500)
        return web.json_response(saved.to_dict())

    @routes.get(P + "/api/projects/{pid}/download")
    async def _projects_download(req):
        found = projects.get(req.match_info["pid"])
        if found is None:
            return web.json_response({"problems": ["no such project"]}, status=404)
        safe = "".join(c if c.isalnum() or c in "._- " else "_" for c in found.name).strip()[:64]
        return web.json_response(
            found.to_dict(),
            headers={"Content-Disposition":
                     f'attachment; filename="{safe or found.id}.funpack_project.json"'})

    # ── media ─────────────────────────────────────────────────────────────
    # Files the user brought IN, as opposed to something a run produced. A
    # reference image today; whatever a model module wires a reference input
    # to tomorrow. Kept apart from projects (its own store, its own ids) and
    # from ComfyUI's own /upload/image (that one lands in the output tree and
    # is meant for a single frame save, not something the user manages).

    @routes.get(P + "/api/media")
    async def _media_list(_req):
        return web.json_response({"media": media.listing()})

    @routes.post(P + "/api/media")
    async def _media_upload(req):
        if not req.content_type or not req.content_type.startswith("multipart/"):
            return web.json_response(
                {"problems": ["expected a multipart upload"]}, status=400)
        saved, problems = [], []
        reader = await req.multipart()
        while True:
            part = await reader.next()
            if part is None:
                break
            if not part.filename:
                continue  # a form field that is not a file
            data = await part.read(decode=False)
            try:
                saved.append(media.save_upload(part.filename, data))
            except ValueError as exc:
                problems.append(f"{part.filename}: {exc}")
        if not saved and problems:
            return web.json_response({"problems": problems}, status=400)
        return web.json_response({"media": saved, "problems": problems})

    @routes.get(P + "/api/media/{mid}/file")
    async def _media_file(req):
        path = media.path_for(req.match_info["mid"])
        if path is None:
            return web.json_response({"problems": ["no such media"]}, status=404)
        # FileResponse, not a hand-read body: it answers Range requests on its
        # own, which is what lets a <video> scrub an imported clip instead of
        # re-downloading it whole on every seek.
        return web.FileResponse(path, headers={
            "Content-Type": media.content_type(req.match_info["mid"])})

    @routes.delete(P + "/api/media/{mid}")
    async def _media_delete(req):
        if not media.delete(req.match_info["mid"]):
            return web.json_response({"problems": ["no such media"]}, status=404)
        return web.json_response({"deleted": True})

    # ── shortcuts ─────────────────────────────────────────────────────────
    # A trigger -> replacement text library, global across every project --
    # see core/shortcuts.py. CRUD here; the expansion itself is stateless
    # (below), not tied to saving or loading a shortcut.

    @routes.get(P + "/api/shortcuts")
    async def _shortcuts_list(_req):
        return web.json_response({"shortcuts": [s.to_dict() for s in shortcuts_mod.listing()]})

    @routes.post(P + "/api/shortcuts")
    async def _shortcuts_save(req):
        try:
            body = await req.json()
        except Exception:  # noqa: BLE001
            return web.json_response({"problems": ["that is not JSON"]}, status=400)
        if not isinstance(body, dict):
            return web.json_response(
                {"problems": [f"a shortcut is an object, not a {type(body).__name__}"]}, status=400)
        original_name = body.get("original_name")
        try:
            items = shortcuts_mod.save(body, original_name if isinstance(original_name, str) else None)
        except ValueError as exc:
            return web.json_response({"problems": [str(exc)]}, status=400)
        return web.json_response({"shortcuts": [s.to_dict() for s in items]})

    @routes.delete(P + "/api/shortcuts/{name}")
    async def _shortcuts_delete(req):
        items = shortcuts_mod.delete(req.match_info["name"])
        return web.json_response({"shortcuts": [s.to_dict() for s in items]})

    @routes.post(P + "/api/shortcuts/clear")
    async def _shortcuts_clear(_req):
        shortcuts_mod.clear()
        return web.json_response({"shortcuts": []})

    @routes.post(P + "/api/prompt/expand")
    async def _prompt_expand(req):
        """What a scene's typed text becomes at generation: anchor + shortcuts
        + $variables + postfix, applied here rather than client-side so the
        library and the algorithm have exactly one implementation. Stateless
        -- everything it needs travels in the body -- for the same reason
        /api/pipeline is: this runs for a preview as freely as for a real run,
        and neither should require a project to exist on the server's disk."""
        try:
            body = await req.json()
        except Exception:  # noqa: BLE001
            return web.json_response({"problems": ["that is not JSON"]}, status=400)
        if not isinstance(body, dict):
            return web.json_response(
                {"problems": [f"a request is an object, not a {type(body).__name__}"]}, status=400)
        variables = body.get("variables")
        seed = body.get("seed")
        expanded = await asyncio.to_thread(
            prompt_build.build,
            body.get("text") or "",
            anchor=body.get("anchor") or "",
            postfix=body.get("postfix") or "",
            postfix_enabled=body.get("postfix_enabled") is not False,
            variables=variables if isinstance(variables, list) else None,
            seed=seed if isinstance(seed, int) and not isinstance(seed, bool) else 0,
        )
        return web.json_response({"text": expanded})

    @routes.get(P + "/api/log/funpack")
    async def _log(req):
        level = req.query.get("level") or None
        try:
            limit = int(req.query.get("limit", log.HISTORY))
        except ValueError:
            limit = log.HISTORY
        return web.json_response({"levels": list(log.LEVELS),
                                  "records": log.history(level, limit)})

    @routes.get(P + "/app/{tail:.*}")
    async def _app_asset(req):
        return _serve_under(req, config.APP_DIR, config.APP_EXTS)

    @routes.get(P + "/modules/{tail:.*}")
    async def _module_asset(req):
        return _serve_under(req, config.MODULES_DIR, config.MODULE_EXTS)

    @routes.get(P + "/")
    async def _index(_req):
        return _respond(static.serve(config.APP_DIR, "index.html", config.APP_EXTS))

    @routes.get(P)
    async def _index_bare(_req):
        raise web.HTTPFound(P + "/")

    return routes


if web is not None and PromptServer is not None:
    # `PromptServer.instance` only exists once a server has been constructed.
    # Importing `server` successfully is not the same as running inside one --
    # a test run with ComfyUI on the path gets the class and no instance -- and
    # an unguarded attribute here takes the whole pack down with it.
    try:
        register(PromptServer.instance.routes)
    except Exception as exc:  # noqa: BLE001
        log.failed("route registration", exc)
    else:
        log.info("routes", f"serving the app at {config.UI_PREFIX}/")
