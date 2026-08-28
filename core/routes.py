"""Route registration on ComfyUI's aiohttp server.

`register()` takes the route table rather than reaching for PromptServer itself,
so the real routes can be mounted on a throwaway app in a test. The handlers are
thin adapters over pure functions in `serve`.
"""

from . import config, log, registry as registry_mod, serve as static
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


# Scanned once at startup. A module list that changed under the UI mid-session
# would mean a panel could refer to something that is no longer there.
_registry = None


def modules(rescan=False):
    global _registry
    if _registry is None or rescan:
        _registry = registry_mod.scan()
    return _registry


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
    register(PromptServer.instance.routes)
    log.note(f"serving the app at {config.UI_PREFIX}/")
