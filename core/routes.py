"""Route registration on ComfyUI's aiohttp server.

`register()` takes the route table rather than reaching for PromptServer itself,
so the real routes can be mounted on a throwaway app in a test. The handlers are
thin adapters over pure functions in `serve`.
"""

from . import (config, graph as graph_mod, log, registry as registry_mod,
               serve as static, widgets)
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
        prompt, incomplete = graph_mod.build(slots)
        return web.json_response({
            "slots": slots,
            "refused": problems,
            "incomplete": incomplete,
            "queueable": not (problems or incomplete),
            "prompt": prompt if not (problems or incomplete) else None,
        })

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
