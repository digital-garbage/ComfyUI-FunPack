"""The settings card, over HTTP.

core/settings_card.py's own tests prove the rendering is correct in a vacuum;
these prove the route actually reaches it with a real request body and comes
back with real PNG bytes and the right content type -- reachability, the
same reason test_routes_pipeline.py exists.
"""

import socket
import threading

import pytest

pytest.importorskip("aiohttp")

from core import routes  # noqa: E402


@pytest.fixture(scope="module")
def server(comfyui):
    """The real route table on a throwaway app, on a real socket.

    No node registration needed here (unlike test_routes_pipeline.py) --
    this route never asks ComfyUI's schema registry anything; it only walks
    the slot list handed to it and renders host facts, both node-schema-free.
    """
    from aiohttp import web as aioweb

    app = aioweb.Application()
    table = aioweb.RouteTableDef()
    routes.register(table, prefix="/funpack")
    app.add_routes(table)

    holder = {}

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

    holder["ready"] = threading.Event()
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert holder["ready"].wait(20), "the test server did not start"
    yield holder["port"]
    holder["loop"].call_soon_threadsafe(holder["loop"].stop)


def _raw_request(port, method, path, body_bytes):
    head = (f"{method} {path} HTTP/1.1\r\nHost: localhost\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body_bytes)}\r\nConnection: close\r\n\r\n").encode()
    with socket.create_connection(("127.0.0.1", port), timeout=30) as sock:
        sock.sendall(head + body_bytes)
        chunks = []
        while True:
            got = sock.recv(65536)
            if not got:
                break
            chunks.append(got)
    raw = b"".join(chunks)
    header, _, body = raw.partition(b"\r\n\r\n")
    lines = header.split(b"\r\n")
    status = int(lines[0].split()[1])
    headers = {}
    for line in lines[1:]:
        if b":" in line:
            k, _, v = line.partition(b":")
            headers[k.strip().lower().decode()] = v.strip().decode()
    return status, headers, body


def test_a_png_comes_back_for_a_real_pipeline(server):
    import json
    body = json.dumps({
        "slots": [{"id": "model", "node": "FunPackDiffusionModelLoader",
                   "inputs": {"model_name": "h3.safetensors"}}],
        "project_name": "Integration test", "theme": "dark",
    }).encode()
    status, headers, png = _raw_request(server, "POST", "/funpack/api/settings-card", body)
    assert status == 200
    assert headers.get("content-type") == "image/png"
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_no_slots_falls_back_to_the_default_pipeline(server):
    """Same convention as /api/pipeline: an ABSENT slots key means 'use the
    live default', an explicitly empty list means 'this pipeline has none'."""
    status, headers, png = _raw_request(server, "POST", "/funpack/api/settings-card", b"{}")
    assert status == 200
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_a_malformed_pipeline_is_refused_not_rendered(server):
    import json
    body = json.dumps({"slots": [{"id": "dup"}, {"id": "dup"}]}).encode()
    status, headers, raw = _raw_request(server, "POST", "/funpack/api/settings-card", body)
    assert status == 400
    assert b"problems" in raw


def test_a_non_object_body_is_a_400_not_a_500(server):
    status, headers, raw = _raw_request(server, "POST", "/funpack/api/settings-card", b"[]")
    assert status == 400
