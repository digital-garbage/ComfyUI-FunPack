"""The pipeline, over HTTP.

core/graph.py was well tested and called from nowhere -- proven correct in a
vacuum while the claim it backs ("a built-in node can be replaced or removed")
was not something the running server could do. These tests are about
reachability as much as behaviour.
"""

import json
import socket
import threading

import pytest

pytest.importorskip("aiohttp")

from aiohttp import web  # noqa: E402

from core import routes  # noqa: E402


@pytest.fixture(scope="module")
def registered(comfyui):
    """FunPack's nodes registered the way ComfyUI registers them.

    Without this the pipeline's slots all look like missing nodes, because the
    schema lookup reads ComfyUI's registry and nothing had put anything in it.
    """
    import asyncio
    import nodes as comfy_nodes
    from pathlib import Path

    async def load():
        await comfy_nodes.init_extra_nodes(init_custom_nodes=False)
        await comfy_nodes.load_custom_node(
            str(Path(__file__).resolve().parents[2]), module_parent="custom_nodes")

    asyncio.run(load())
    return comfy_nodes


@pytest.fixture(scope="module")
def server(comfyui, registered):
    """The real route table on a throwaway app, on a real socket."""
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
    raw = b"".join(chunks)
    header, _, rest = raw.partition(b"\r\n\r\n")
    status = int(header.split()[1])
    try:
        return status, json.loads(rest.decode() or "{}")
    except ValueError:
        return status, {"raw": rest.decode(errors="replace")}


def test_the_default_pipeline_is_served(server):
    status, body = _request(server, "GET", "/funpack/api/pipeline")
    assert status == 200
    assert [slot["id"] for slot in body["slots"]], "no slots came back"
    assert "queueable" in body
    # A fresh install has no model picked, so the default pipeline is
    # incomplete rather than wrong -- and the two are reported separately.
    assert body["refused"] == []
    assert any("nothing fills it" in p for p in body["incomplete"])


def test_a_built_in_slot_can_be_removed_over_http(server):
    """The claim, exercised the way the app will: the modifiers node taken out,
    and what it fed rewired to what fed it."""
    status, body = _request(server, "POST", "/funpack/api/pipeline",
                            {"action": "remove", "slot": "modifiers"})
    assert status == 200, body
    assert body["refused"] == [], body["refused"]
    ids = [slot["id"] for slot in body["slots"]]
    assert "modifiers" not in ids

    sampler = next(s for s in body["slots"] if s["id"] == "sampler")
    assert sampler["inputs"]["model"] == ["model", 0], "the sampler was not rewired"


def test_a_removal_that_cannot_work_comes_back_as_a_reason(server):
    status, body = _request(server, "POST", "/funpack/api/pipeline",
                            {"action": "remove", "slot": "latent"})
    assert status == 200
    assert any("LATENT" in p for p in body["refused"])
    assert body["queueable"] is False
    assert body["prompt"] is None


def test_a_built_in_slot_can_be_replaced_over_http(server):
    status, body = _request(server, "POST", "/funpack/api/pipeline",
                            {"action": "replace", "slot": "vae", "node": "VAELoader"})
    assert status == 200, body
    vae = next(s for s in body["slots"] if s["id"] == "vae")
    assert vae["node"] == "VAELoader"


def test_a_replacement_that_gives_the_wrong_thing_is_refused(server):
    status, body = _request(server, "POST", "/funpack/api/pipeline",
                            {"action": "replace", "slot": "model", "node": "SaveImage"})
    assert status == 200
    assert body["refused"], "a nonsense replacement was accepted"


def test_an_unknown_action_is_refused(server):
    status, body = _request(server, "POST", "/funpack/api/pipeline",
                            {"action": "explode", "slot": "model"})
    assert status == 400 and body["problems"]  # a malformed request, not an edit


def test_an_explicitly_empty_pipeline_is_not_replaced_by_the_default(server):
    """`slots or default()` resurrects the default, because an empty list is
    falsy. A client that removed every slot is entitled to be told it has none
    rather than handed the built-ins back."""
    status, body = _request(server, "POST", "/funpack/api/pipeline",
                            {"action": "check", "slots": []})
    assert status == 200
    assert body["slots"] == [], "an empty pipeline came back full"


def test_omitting_slots_still_falls_back_to_the_default(server):
    """The distinction that matters: absent means "use the default", empty means
    "there is nothing". Only one of them is a fallback."""
    status, body = _request(server, "POST", "/funpack/api/pipeline",
                            {"action": "check"})
    assert status == 200
    assert body["slots"], "omitting slots should use the default pipeline"


def test_removing_slots_one_by_one_never_resurrects_the_default(server):
    """The failure this guards: each request carries the client's current slots,
    so a fallback that fires on an empty list would silently hand the built-ins
    back mid-way through and the count would jump UP."""
    _status, body = _request(server, "GET", "/funpack/api/pipeline")
    slots = body["slots"]
    started_with = len(slots)
    assert started_with > 1, "the default pipeline is too small to test this"

    counts = [started_with]
    # Consumers before sources: removing a source nothing can replace is refused,
    # which is correct and not what this test is about.
    for slot_id in [s["id"] for s in reversed(slots)]:
        _status, body = _request(server, "POST", "/funpack/api/pipeline",
                                 {"action": "remove", "slot": slot_id, "slots": slots})
        if body["refused"]:
            continue
        slots = body["slots"]
        counts.append(len(slots))

    assert len(slots) < started_with, "nothing was actually removed"
    assert counts == sorted(counts, reverse=True), (
        f"the pipeline grew back part way through: {counts}")
