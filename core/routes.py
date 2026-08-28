"""Route registration on ComfyUI's aiohttp server.

`register()` takes the route table rather than reaching for PromptServer itself,
so the real routes can be mounted on a throwaway app in a test. The handlers are
thin adapters over pure functions in `serve`.
"""

from . import config, log, serve as static

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


def register(routes, prefix=None):
    """Attach FunPack's routes to an aiohttp route table."""
    P = config.UI_PREFIX if prefix is None else prefix

    @routes.get(P + "/api/health")
    async def _health(_req):
        return web.json_response({"ok": True})

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
