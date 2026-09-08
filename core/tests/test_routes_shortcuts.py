"""/api/shortcuts and /api/prompt/expand, over real HTTP -- proves the routes
actually reach core/shortcuts.py and core/prompt_build.py, not just that
those modules are correct in isolation (test_shortcuts.py, test_prompt_build.py).
"""

import json
import socket
import threading

import pytest

pytest.importorskip("aiohttp")

from core import config, routes  # noqa: E402


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SHORTCUTS_FILE", tmp_path / "shortcuts.json")
    from aiohttp import web as aioweb

    app = aioweb.Application()
    table = aioweb.RouteTableDef()
    routes.register(table, prefix="/funpack")
    app.add_routes(table)

    holder = {"ready": threading.Event()}

    def run():
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        runner = aioweb.AppRunner(app)
        loop.run_until_complete(runner.setup())
        site = aioweb.TCPSite(runner, "127.0.0.1", 0)
        loop.run_until_complete(site.start())
        holder["port"] = site._server.sockets[0].getsockname()[1]
        holder["loop"] = loop
        holder["ready"].set()
        loop.run_forever()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert holder["ready"].wait(20), "the test server did not start"
    yield holder["port"]
    holder["loop"].call_soon_threadsafe(holder["loop"].stop)


def _request(port, method, path, body=None):
    payload = b"" if body is None else json.dumps(body).encode()
    head = (f"{method} {path} HTTP/1.1\r\nHost: localhost\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n").encode()
    with socket.create_connection(("127.0.0.1", port), timeout=30) as sock:
        sock.sendall(head + payload)
        chunks = []
        while True:
            got = sock.recv(65536)
            if not got:
                break
            chunks.append(got)
    header, _, rest = b"".join(chunks).partition(b"\r\n\r\n")
    status = int(header.split()[1])
    return status, json.loads(rest.decode() or "{}")


def test_save_list_delete(server):
    status, resp = _request(server, "POST", "/funpack/api/shortcuts",
                             {"name": "Fox", "triggers": ["fox"], "replacements": ["red fox"]})
    assert status == 200, resp
    assert [s["name"] for s in resp["shortcuts"]] == ["Fox"]

    status, listed = _request(server, "GET", "/funpack/api/shortcuts")
    assert status == 200 and [s["name"] for s in listed["shortcuts"]] == ["Fox"]

    status, deleted = _request(server, "DELETE", "/funpack/api/shortcuts/Fox")
    assert status == 200 and deleted["shortcuts"] == []


def test_saving_a_shortcut_with_no_triggers_is_a_400(server):
    status, resp = _request(server, "POST", "/funpack/api/shortcuts",
                             {"name": "Nothing", "triggers": [], "replacements": ["x"]})
    assert status == 400
    assert resp["problems"]


def test_clear_empties_the_library(server):
    _request(server, "POST", "/funpack/api/shortcuts",
             {"name": "Fox", "triggers": ["fox"], "replacements": ["red fox"]})
    status, resp = _request(server, "POST", "/funpack/api/shortcuts/clear")
    assert status == 200 and resp["shortcuts"] == []


def test_prompt_expand_combines_anchor_scene_and_postfix(server):
    status, resp = _request(server, "POST", "/funpack/api/prompt/expand",
                             {"text": "a fox runs", "anchor": "cinematic", "postfix": "4k"})
    assert status == 200
    assert resp["text"] == "cinematic a fox runs 4k"


def test_prompt_expand_uses_the_saved_shortcut_library(server):
    _request(server, "POST", "/funpack/api/shortcuts",
             {"name": "Fox", "triggers": ["fox"], "replacements": ["red fox"]})
    status, resp = _request(server, "POST", "/funpack/api/prompt/expand", {"text": "a fox runs"})
    assert status == 200 and resp["text"] == "a red fox runs"


def test_prompt_expand_with_variables_and_postfix_disabled(server):
    status, resp = _request(server, "POST", "/funpack/api/prompt/expand", {
        "text": "a $animal runs", "anchor": "", "postfix": "ignored",
        "postfix_enabled": False, "variables": [{"name": "animal", "value": "fox"}],
    })
    assert status == 200 and resp["text"] == "a fox runs"


def test_prompt_expand_on_a_non_object_body_is_a_400(server):
    status, resp = _request(server, "POST", "/funpack/api/prompt/expand", ["nope"])
    assert status == 400 and resp["problems"]
