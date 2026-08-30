"""The routes, exercised through a real aiohttp router over a real socket.

test_serve_allowlist.py calls serve() with literal strings and cannot see what a
router does to a path before the handler runs, so routes.py had no coverage at
all. This closes that.

Requests are written onto the socket by hand rather than through aiohttp's
TestClient, because TestClient normalises the URL BEFORE sending: ask it for
`/funpack/modules/%2e%2e/app/boot.js` and the server receives a plain
`/funpack/app/boot.js`. A test written that way silently checks nothing about
traversal -- it looks like a routing bypass when it is only the client tidying up
after itself.

The loop is driven by hand rather than with pytest-asyncio: ComfyUI brings
aiohttp, and the suite should not need a plugin users would have to install.
"""

import asyncio
import json

import pytest

pytest.importorskip("aiohttp")
from aiohttp import web                    # noqa: E402
from aiohttp.test_utils import TestServer  # noqa: E402

from core import config, routes as routes_mod  # noqa: E402

P = config.UI_PREFIX


@pytest.fixture()
def app(tmp_path, monkeypatch):
    app_dir = tmp_path / "app"
    (app_dir / "composer").mkdir(parents=True)
    (app_dir / "boot.js").write_text("export const app = 1;")
    (app_dir / "composer" / "composer.css").write_text(":root{}")
    (app_dir / "index.html").write_text("<p>shell</p>")

    mod_dir = tmp_path / "modules" / "timing" / "audio_clock"
    mod_dir.mkdir(parents=True)
    (mod_dir / "ui.js").write_text("export default 1;")
    (mod_dir / "sneaky.css").write_text(".x{}")

    monkeypatch.setattr(config, "APP_DIR", app_dir)
    monkeypatch.setattr(config, "MODULES_DIR", tmp_path / "modules")

    # The registry caches its scan for the whole session, so anything that
    # triggers one WHILE MODULES_DIR points here caches this temp directory --
    # one fake module -- as the installed set, and every later test in the
    # session runs against it. That happened, and cost twenty-three failures in
    # files that have nothing to do with routes. Whoever redirects the directory
    # owns putting the cache back.
    from core import registry as registry_mod
    monkeypatch.setattr(registry_mod, "_current", registry_mod._current, raising=False)

    # A builder, not an Application: aiohttp binds an app to the loop that first
    # ran it, and each request here gets its own asyncio.run().
    def build():
        table = web.RouteTableDef()
        routes_mod.register(table)
        application = web.Application()
        application.add_routes(table)
        return application

    return build


def get(app, target):
    """(status, body) for a request line written verbatim onto the socket."""
    async def go():
        server = TestServer(app())
        await server.start_server()
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
            writer.write(
                f"GET {target} HTTP/1.1\r\nHost: test\r\nConnection: close\r\n\r\n".encode()
            )
            await writer.drain()
            raw = await reader.read()
            writer.close()
        finally:
            await server.close()
        head, _, body = raw.partition(b"\r\n\r\n")
        status = int(head.split(b"\r\n", 1)[0].split()[1])
        return status, body.decode("utf-8", "replace")
    return asyncio.run(go())


# --- traversal, as it actually arrives on the wire -------------------------

@pytest.mark.parametrize("target", [
    P + "/modules/../app/boot.js",
    P + "/modules/%2e%2e/app/boot.js",
    P + "/modules/%2e%2e/app/composer/composer.css",
    P + "/modules/%2e%2e%2f%2e%2e/app/boot.js",
    P + "/app/../../etc/passwd",
    P + "/app/%2e%2e/%2e%2e/etc/passwd",
])
def test_traversal_never_leaves_its_root(app, target):
    status, body = get(app, target)
    assert status in (403, 404), f"{target} -> {status} {body[:80]!r}"


def test_a_modules_url_cannot_reach_app_content(app):
    # The route that matched must be the root that serves: a .css is reachable
    # under /app/ and must stay unreachable under /modules/.
    assert get(app, P + "/modules/../app/composer/composer.css")[0] in (403, 404)
    assert get(app, P + "/app/composer/composer.css")[0] == 200


# --- the allowlist on ordinary URLs ----------------------------------------

def test_module_css_is_refused(app):
    assert get(app, P + "/modules/timing/audio_clock/sneaky.css")[0] == 404


def test_module_js_is_served(app):
    status, body = get(app, P + "/modules/timing/audio_clock/ui.js")
    assert status == 200 and "export default" in body


def test_app_asset_is_served(app):
    status, body = get(app, P + "/app/boot.js")
    assert status == 200 and "export const app" in body


def test_query_string_does_not_change_the_target(app):
    assert get(app, P + "/app/boot.js?v=2")[0] == 200


def test_index_is_served(app):
    status, body = get(app, P + "/")
    assert status == 200 and "shell" in body


def test_health(app):
    status, body = get(app, P + "/api/health")
    assert status == 200 and '"ok"' in body



# --- the nodes route -------------------------------------------------------

def _nodes(app, target):
    status, body = get(app, target)
    return status, json.loads(body)


def test_the_nodes_route_describes_what_a_pipeline_points_at(app, comfyui):
    """The window that edits a pipeline needs the widgets, and asks by name.

    ComfyUI's own /object_info answers with every installed node, which on a
    machine with a few packs is megabytes for a question about a dozen.
    """
    # ComfyUI's own registry only. Installing FunPack's nodes here would call
    # collect(), which scans MODULES_DIR -- and the `app` fixture has pointed
    # that at a temp directory holding one fake module. The scan is CACHED, so
    # every later test in the session then ran against an empty module registry:
    # twenty-three failures in files that never mention routes. What this test
    # needs from the registry is a node, and KSampler is one.
    import nodes  # noqa: F401

    status, body = _nodes(app, P + "/api/nodes?classes=KSampler,NoSuchNodeAnywhere")
    assert status == 200
    assert body["nodes"]["KSampler"]["widgets"], "a node came back with nothing to edit"
    assert body["nodes"]["NoSuchNodeAnywhere"] is None, (
        "an absent node was dropped from the answer rather than reported as absent")


def test_asking_about_nothing_is_not_an_error(app):
    status, body = _nodes(app, P + "/api/nodes")
    assert status == 200 and body["nodes"] == {}


def test_asking_about_a_thousand_nodes_is_refused(app):
    """The query string is the caller's, and describing a thousand nodes one by
    one on the event loop stops the server answering anything else."""
    many = ",".join(f"Node{i}" for i in range(1000))
    status, body = _nodes(app, P + f"/api/nodes?classes={many}")
    assert status == 400
    assert any("200" in problem for problem in body["problems"])
