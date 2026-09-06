"""Keeping the install alive, over HTTP.

Every one of these routes ends by relaunching ComfyUI. The point of the tests is
that the ANSWER is written first and the restart is a scheduled callback -- so
they replace it, and a suite that did not would relaunch the test runner.
"""

import json
import socket
import threading

import pytest

pytest.importorskip("aiohttp")

from core import routes, update as update_mod  # noqa: E402


@pytest.fixture
def server(comfyui, monkeypatch):
    from aiohttp import web as aioweb

    restarts = []
    from core import restart as restart_mod
    monkeypatch.setattr(restart_mod, "restart", lambda: restarts.append(1))

    app = aioweb.Application()
    table = aioweb.RouteTableDef()
    routes.register(table, prefix="/funpack")
    app.add_routes(table)

    holder = {"ready": threading.Event(), "restarts": restarts}

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
    yield holder
    holder["loop"].call_soon_threadsafe(holder["loop"].stop)


def _request(port, method, path, body=None):
    payload = b"" if body is None else json.dumps(body).encode()
    head = (f"{method} {path} HTTP/1.1\r\nHost: localhost\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n").encode()
    with socket.create_connection(("127.0.0.1", port), timeout=60) as sock:
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


def test_the_app_can_see_which_branch_it_is_on(server):
    """The one question every update starts from."""
    status, body = _request(server["port"], "GET", "/funpack/api/git/status")
    assert status == 200
    assert body["ok"] is True, body
    assert body["branch"], "no branch reported"
    assert isinstance(body["branches"], list) and body["branches"]
    assert body["repo"].endswith("ComfyUI-FunPack-v5")
    assert server["restarts"] == [], "reading the status restarted ComfyUI"


def test_switching_to_a_branch_that_is_not_there_is_refused_by_name(server):
    """A refusal has to say which branch and that it exists nowhere -- otherwise
    the user cannot tell it from a typo in their own request."""
    status, body = _request(server["port"], "POST", "/funpack/api/git/checkout",
                            {"branch": "no-such-branch-here"})
    assert status == 400
    assert "no-such-branch-here" in body["detail"]
    assert server["restarts"] == [], "a refused switch still restarted ComfyUI"


def test_switching_to_nothing_is_refused_before_git_is_touched(server):
    status, body = _request(server["port"], "POST", "/funpack/api/git/checkout", {})
    assert status == 400
    assert "branch" in body["detail"].lower()
    assert server["restarts"] == []


def test_a_refusal_from_git_is_reported_as_a_reason_not_a_crash(server, monkeypatch):
    """Dirty tree, no git on PATH, not a checkout: the user has to be told which."""
    def refuse(*_a, **_k):
        raise update_mod.GitUpdateError("Working tree has local changes.")
    monkeypatch.setattr(update_mod, "pull", refuse)

    status, body = _request(server["port"], "POST", "/funpack/api/git/update", {})
    assert status == 400
    assert "local changes" in body["detail"]
    assert server["restarts"] == []


def test_an_update_that_worked_answers_before_it_restarts(server, monkeypatch):
    """The answer has to be written first: the process is about to go."""
    monkeypatch.setattr(update_mod, "pull",
                        lambda **_k: {"updated": True, "before": "a1", "after": "b2"})

    status, body = _request(server["port"], "POST", "/funpack/api/git/update",
                            {"branch": "dev"})
    assert status == 200, body
    assert body["restarting"] is True
    assert body["after"] == "b2"


def test_an_unexpected_failure_is_a_500_with_the_reason_not_a_hang(server, monkeypatch):
    def explode(*_a, **_k):
        raise OSError("disk went away")
    monkeypatch.setattr(update_mod, "rollback", explode)

    status, body = _request(server["port"], "POST", "/funpack/api/git/rollback")
    assert status == 500
    assert "disk went away" in body["detail"]
    assert server["restarts"] == [], "a failed rollback restarted ComfyUI anyway"


def test_the_status_can_be_had_without_touching_the_network(server):
    """The fetch is the only part that can hang. On a machine that cannot reach
    the remote it must still be possible to see which branch you are on."""
    status, body = _request(server["port"], "GET", "/funpack/api/git/status?remote=0")
    assert status == 200
    assert body["ok"] is True
    assert body["branch"]
    assert body["checked_remote"] is False
    assert body["behind"] == 0, "behind was reported without asking the remote"


def test_a_remote_that_cannot_be_reached_is_said_rather_than_guessed(server, monkeypatch):
    def no_network(*args, **kwargs):
        class Failed:
            returncode = 1
            stdout = ""
            stderr = "could not resolve host"
        return Failed() if args and args[0] == "fetch" else _real(*args, **kwargs)
    _real = update_mod._run_git
    monkeypatch.setattr(update_mod, "_run_git", no_network)

    status, body = _request(server["port"], "GET", "/funpack/api/git/status")
    assert status == 200
    assert body["fetch_ok"] is False
    assert body["checked_remote"] is True
    assert body["behind"] == 0, "a failed fetch was read as 'up to date'"
