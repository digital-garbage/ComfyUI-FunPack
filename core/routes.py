"""Route registration on ComfyUI's aiohttp server.

`register()` takes the route table rather than reaching for PromptServer itself,
so the real routes can be mounted on a throwaway app in a test. The handlers are
thin adapters over pure functions in `serve`.
"""

import json

from . import (config, graph as graph_mod, log, nodes_manager, projects,
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
        if action in ("replace", "remove") and not isinstance(slot_id, str):
            return web.json_response(
                {"problems": [f"which slot to {action} is named by a string, "
                              f"not a {type(slot_id).__name__}"],
                 "queueable": False}, status=400)
        if action == "replace" and not isinstance(body.get("node"), str):
            return web.json_response(
                {"problems": [f"a node is named by a string, not a "
                              f"{type(body.get('node')).__name__}"],
                 "queueable": False}, status=400)

        if action == "replace":
            slots, problems = graph_mod.replace(slots, slot_id, body.get("node"))
        elif action == "remove":
            slots, problems = graph_mod.remove(slots, slot_id)
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
        import asyncio
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
        # ONLY when the checkout actually moved. Pressing Update while already up
        # to date is a normal thing to do -- there is no way to know until it has
        # been asked -- and restarting for it costs a boot and, if a generation
        # is running, the generation.
        moved = bool(result.get("updated")) or (
            result.get("before") is not None and result.get("before") != result.get("after")
        ) or (
            result.get("before_branch") is not None
            and result.get("before_branch") != result.get("branch")
        )
        if not moved:
            return web.json_response({"restarting": False, **result})

        import asyncio as _asyncio
        from . import restart as restart_mod
        # After the response is written, not before.
        _asyncio.get_event_loop().call_later(0.7, restart_mod.restart)
        return web.json_response({"restarting": True, **result})

    @routes.get(P + "/api/git/status")
    async def _git_status(req):
        # `?remote=0` answers from the checkout alone. The fetch is the only part
        # that touches the network and so the only part that can hang; the app
        # asks without it first so it has something to draw.
        import asyncio
        remote = req.query.get("remote", "1") not in ("0", "false", "no")
        return web.json_response(await asyncio.to_thread(update_mod.status, remote=remote))

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

    @routes.get(P + "/api/log")
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
