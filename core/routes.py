"""Route registration on ComfyUI's aiohttp server.

Imported for its side effects. Everything here is a thin adapter over pure
functions elsewhere in core, so the logic stays testable without a server.
"""

from . import config, log, serve as static

try:
    from aiohttp import web
    from server import PromptServer
except Exception:  # not running inside ComfyUI (tests, tooling)
    web = None
    PromptServer = None


def _respond(served: "static.Served") -> "web.Response":
    if served.status != 200:
        return web.Response(status=served.status, headers=served.headers)
    return web.Response(
        body=served.body,
        content_type=served.content_type,
        headers=served.headers,
    )


if web is not None and PromptServer is not None:
    routes = PromptServer.instance.routes
    P = config.UI_PREFIX

    @routes.get(P + "/api/health")
    async def _health(_req):
        return web.json_response({"ok": True})

    @routes.get(P + "/app/{tail:.*}")
    async def _app_asset(req):
        return _respond(static.serve(
            config.APP_DIR, req.match_info["tail"], config.APP_EXTS))

    @routes.get(P + "/modules/{tail:.*}")
    async def _module_asset(req):
        return _respond(static.serve(
            config.MODULES_DIR, req.match_info["tail"], config.MODULE_EXTS))

    @routes.get(P + "/")
    async def _index(_req):
        return _respond(static.serve(config.APP_DIR, "index.html", config.APP_EXTS))

    @routes.get(P)
    async def _index_bare(_req):
        raise web.HTTPFound(P + "/")

    log.note(f"serving the app at {P}/")
