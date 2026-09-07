"""Media: files the user brought in, not something a run produced.

Mirrors test_projects.py's shape (store-level, then over HTTP) because the
store itself mirrors projects.py's -- same atomic-index-write, same
id-is-the-only-safe-filename rule, same "a request is not the thing it
claims to be" refusals.
"""

import json
import socket
import threading

import pytest

from core import config, media


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MEDIA_DIR", tmp_path / "media")
    return tmp_path / "media"


# ── the model ────────────────────────────────────────────────────────────


def test_an_upload_round_trips(store):
    entry = media.save_upload("ref.png", b"\x89PNG fake bytes")
    assert entry["kind"] == "image"
    assert entry["name"] == "ref.png"

    back = media.get(entry["id"])
    assert back == entry
    assert media.path_for(entry["id"]).read_bytes() == b"\x89PNG fake bytes"


def test_kind_is_read_from_the_extension(store):
    assert media.save_upload("a.png", b"x")["kind"] == "image"
    assert media.save_upload("a.mp4", b"x")["kind"] == "video"
    assert media.save_upload("a.wav", b"x")["kind"] == "audio"


def test_an_extension_not_on_the_allowlist_is_refused(store):
    with pytest.raises(ValueError):
        media.save_upload("payload.exe", b"MZ")
    with pytest.raises(ValueError):
        media.save_upload("script.js", b"alert(1)")
    assert media.listing() == []


def test_an_empty_upload_is_refused(store):
    with pytest.raises(ValueError):
        media.save_upload("empty.png", b"")


def test_an_oversized_upload_is_refused(store, monkeypatch):
    monkeypatch.setattr(media, "MAX_BYTES", 4)
    with pytest.raises(ValueError):
        media.save_upload("big.png", b"way too much data")


def test_concurrent_uploads_all_survive_in_the_index(store):
    """The index is one shared index.json, read-modified-and-written by every
    upload. Without a lock around that, two threads racing it lose entries or
    crash on the shared tmp file -- found by the adversarial review, which
    reproduced both with real threads against the bare store."""
    errors = []

    def go(i):
        try:
            media.save_upload(f"f{i}.png", f"data{i}".encode())
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=go, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert errors == []
    assert len(media.listing()) == 20
    assert len({it["id"] for it in media.listing()}) == 20


def test_concurrent_upload_and_delete_do_not_corrupt_the_index(store):
    entries = [media.save_upload(f"f{i}.png", f"d{i}".encode()) for i in range(5)]

    def upload(i):
        media.save_upload(f"new{i}.png", f"n{i}".encode())

    def remove(mid):
        media.delete(mid)

    threads = ([threading.Thread(target=upload, args=(i,)) for i in range(10)]
               + [threading.Thread(target=remove, args=(e["id"],)) for e in entries])
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    listed = media.listing()
    assert len(listed) == 10, "an upload or a delete was lost to the race"
    assert len({it["id"] for it in listed}) == 10


def test_the_listing_is_newest_first(store):
    a = media.save_upload("a.png", b"1")
    b = media.save_upload("b.png", b"2")
    assert [it["id"] for it in media.listing()] == [b["id"], a["id"]]


def test_a_missing_file_drops_out_of_the_listing(store):
    entry = media.save_upload("gone.png", b"1")
    (store / entry["filename"]).unlink()
    assert media.listing() == []
    assert media.path_for(entry["id"]) is None


def test_delete_removes_the_file_and_the_entry(store):
    entry = media.save_upload("x.png", b"1")
    assert media.delete(entry["id"]) is True
    assert media.get(entry["id"]) is None
    assert not (store / entry["filename"]).exists()
    assert media.delete(entry["id"]) is False


def test_an_id_from_a_request_cannot_be_a_path(store):
    for bad in ("../escape", "a/b", "", "..", "x" * 64, None, 5):
        assert media.is_id(bad) is False
        assert media.get(bad) is None
        assert media.path_for(bad) is None


def test_a_long_name_is_cut_not_refused(store):
    entry = media.save_upload("x" * 500 + ".png", b"1")
    assert len(entry["name"]) == media.MAX_NAME


# ── over HTTP ────────────────────────────────────────────────────────────

pytest.importorskip("aiohttp")


@pytest.fixture
def server(store):
    from aiohttp import web as aioweb
    from core import routes

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


def _request(port, method, path, body=None, raw=None, content_type="application/json"):
    if raw is not None:
        payload = raw
    else:
        payload = b"" if body is None else json.dumps(body).encode()
    head = (f"{method} {path} HTTP/1.1\r\nHost: localhost\r\n"
            f"Content-Type: {content_type}\r\n"
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
    try:
        return status, json.loads(rest.decode() or "{}")
    except ValueError:
        return status, {"raw": rest}


def _upload(port, filename, data, field="file"):
    boundary = "----funpacktestboundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    return _request(port, "POST", "/funpack/api/media", raw=body,
                     content_type=f"multipart/form-data; boundary={boundary}")


def test_upload_list_file_delete(server):
    status, resp = _upload(server, "ref.png", b"fake png bytes")
    assert status == 200, resp
    assert len(resp["media"]) == 1
    mid = resp["media"][0]["id"]

    status, listed = _request(server, "GET", "/funpack/api/media")
    assert status == 200 and [it["id"] for it in listed["media"]] == [mid]

    status, body = _request(server, "GET", f"/funpack/api/media/{mid}/file")
    assert status == 200 and body["raw"] == b"fake png bytes"

    assert _request(server, "DELETE", f"/funpack/api/media/{mid}")[0] == 200
    assert _request(server, "GET", f"/funpack/api/media/{mid}/file")[0] == 404


def test_a_disallowed_extension_is_refused_over_http(server):
    status, resp = _upload(server, "payload.exe", b"MZ")
    assert status == 400
    assert "problems" in resp
    assert _request(server, "GET", "/funpack/api/media")[1]["media"] == []


def test_a_non_multipart_body_is_refused(server):
    status, resp = _request(server, "POST", "/funpack/api/media", {"not": "multipart"})
    assert status == 400


def test_media_for_a_bad_id_is_a_404(server):
    assert _request(server, "GET", "/funpack/api/media/../../etc/passwd/file")[0] in (400, 404)
    assert _request(server, "DELETE", "/funpack/api/media/0123456789ab")[0] == 404
