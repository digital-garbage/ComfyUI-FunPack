"""Model family detection, over HTTP -- against the REAL registry, so this
proves modules/models/minimax_h3's own detect() is actually reachable
through core/registry.py's provider lookup, not just correct in isolation
(core/tests/test_probe.py) or in the module's own tests.
"""

import json
import socket
import struct
import threading

import pytest

pytest.importorskip("aiohttp")

from aiohttp import web  # noqa: E402

from core import config, routes  # noqa: E402


@pytest.fixture(scope="module")
def registered(comfyui):
    """FunPack's modules importable the way the real registry scan needs --
    modules/models/minimax_h3 imports `comfy.*` at module level."""
    import asyncio
    import nodes as comfy_nodes

    async def load():
        await comfy_nodes.init_extra_nodes(init_custom_nodes=False)

    asyncio.run(load())
    return comfy_nodes


@pytest.fixture(scope="module")
def server(registered, tmp_path_factory):
    from aiohttp import web as aioweb

    models_dir = tmp_path_factory.mktemp("models")

    import folder_paths
    folder_paths.folder_names_and_paths["diffusion_models"] = ([str(models_dir)], {".safetensors"})

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
    holder["models_dir"] = models_dir
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert holder["ready"].wait(20), "the test server did not start"
    yield holder
    holder["loop"].call_soon_threadsafe(holder["loop"].stop)


def _request(port, method, path):
    head = (f"{method} {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n").encode()
    with socket.create_connection(("127.0.0.1", port), timeout=30) as sock:
        sock.sendall(head)
        chunks = []
        while True:
            got = sock.recv(65536)
            if not got:
                break
            chunks.append(got)
    header, _, rest = b"".join(chunks).partition(b"\r\n\r\n")
    status = int(header.split()[1])
    try:
        return status, json.loads(rest.decode() or "{}")
    except ValueError:
        return status, {"raw": rest.decode(errors="replace")}


def _write_safetensors(path, keys):
    header = {k: {"dtype": "F16", "shape": [1], "data_offsets": [0, 0]} for k in keys}
    body = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(body)) + body)


def test_a_real_h3_signature_is_recognised_through_the_real_registry(server):
    _write_safetensors(server["models_dir"] / "h3.safetensors",
                       ["video_patch_proj.weight", "audio_patch_proj.weight"])
    status, body = _request(server["port"], "GET", "/funpack/api/probe?file=h3.safetensors")
    assert status == 200, body
    assert body["module"] == "model_minimax_h3"
    assert body["detected"] is True


def test_a_file_matching_no_known_signature_says_so(server):
    _write_safetensors(server["models_dir"] / "unknown.safetensors", ["some.other.weight"])
    status, body = _request(server["port"], "GET", "/funpack/api/probe?file=unknown.safetensors")
    assert status == 200, body
    assert body["module"] is None
    assert body["detected"] is False


def test_a_file_that_does_not_exist_says_not_found(server):
    status, body = _request(server["port"], "GET", "/funpack/api/probe?file=nope.safetensors")
    assert status == 200
    assert body["detected"] is False
    assert "not found" in body["reason"]


def test_no_file_named_is_a_400_not_a_guess(server):
    status, body = _request(server["port"], "GET", "/funpack/api/probe")
    assert status == 400
    assert body["problems"]
