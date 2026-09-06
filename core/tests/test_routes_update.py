"""Keeping the install alive, over HTTP.

Every one of these routes ends by relaunching ComfyUI. The point of the tests is
that the ANSWER is written first and the restart is a scheduled callback -- so
they replace it, and a suite that did not would relaunch the test runner.
"""

import json
import socket
import threading
import time

import pytest

pytest.importorskip("aiohttp")

from core import routes, update as update_mod  # noqa: E402


@pytest.fixture
def server(comfyui, monkeypatch):
    from aiohttp import web as aioweb

    restarts = []
    from core import restart as restart_mod
    monkeypatch.setattr(restart_mod, "restart", lambda: restarts.append(1))
    # `_pending_restart` is module state, not per-request -- a test that blocks a
    # restart and never finishes it would otherwise leak "an update is waiting"
    # into every test that runs after it in the same process.
    monkeypatch.setattr(routes, "_pending_restart", False)

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


# --- node packs --------------------------------------------------------------

def test_the_pack_list_says_where_it_is_looking(server):
    status, body = _request(server["port"], "GET", "/funpack/api/packs")
    assert status == 200, body
    assert body["root"].endswith("custom_nodes"), body["root"]
    assert isinstance(body["nodes"], list)


def test_removing_something_that_is_not_a_pack_name_is_refused(server):
    """The delete is the whole risk here: one segment, resolved, inside
    custom_nodes, and never FunPack itself."""
    for name in ["..", "../..", "a/b", ".git", ""]:
        status, body = _request(server["port"], "POST", "/funpack/api/packs/remove",
                                {"name": name})
        assert status == 400, (name, status, body)
        assert body["detail"], name


def test_removing_funpack_itself_is_refused_by_name(server):
    from core import nodes_manager
    me = nodes_manager.FUNPACK_ROOT.name
    status, body = _request(server["port"], "POST", "/funpack/api/packs/remove", {"name": me})
    assert status == 400
    # It has to say WHICH thing it refused to delete, or the user tries again.
    assert "FunPack" in body["detail"]


def test_a_pack_that_is_not_installed_is_refused_rather_than_created(server):
    status, body = _request(server["port"], "POST", "/funpack/api/packs/update",
                            {"name": "ComfyUI-DoesNotExist"})
    assert status == 400
    assert "ComfyUI-DoesNotExist" in body["detail"]


def test_installing_from_nothing_is_refused(server):
    status, body = _request(server["port"], "POST", "/funpack/api/packs/install", {"url": ""})
    assert status == 400
    assert body["detail"]


def test_an_install_outside_comfyui_says_it_cannot_tell_where_packs_live(server, monkeypatch):
    """A development checkout is not inside custom_nodes. v4 could assume it was,
    because v4 only ever ran from an install; here the assumption is checked and
    the answer is a refusal rather than somebody's home directory."""
    from core import nodes_manager
    import sys as _sys
    monkeypatch.setitem(_sys.modules, "folder_paths", None)
    monkeypatch.setattr(nodes_manager, "FUNPACK_ROOT",
                        nodes_manager.Path("/tmp/somewhere/ComfyUI-FunPack-v5"))

    status, body = _request(server["port"], "GET", "/funpack/api/packs")
    assert status == 400, body
    assert "custom_nodes" in body["detail"]
    assert "/tmp/somewhere" in body["detail"], "it did not say where it thinks it is"


def test_an_update_that_changed_nothing_does_not_restart(server, monkeypatch):
    """Pressing Update while already up to date is a normal thing to do -- there
    is no way to know until it has been asked. Restarting for it costs a boot,
    and a generation if one is running."""
    monkeypatch.setattr(update_mod, "pull",
                        lambda **_k: {"updated": False, "before": "a1", "after": "a1"})

    status, body = _request(server["port"], "POST", "/funpack/api/git/update", {})
    assert status == 200, body
    assert body["restarting"] is False
    assert server["restarts"] == [], "restarted for a checkout that had not moved"


def test_a_branch_switch_restarts_even_when_the_commit_is_the_same(server, monkeypatch):
    """Two branches can point at the same commit and still be different code
    once anything is committed to either."""
    monkeypatch.setattr(update_mod, "checkout",
                        lambda **_k: {"updated": False, "before": "a1", "after": "a1",
                                      "before_branch": "v5", "branch": "dev"})

    status, body = _request(server["port"], "POST", "/funpack/api/git/checkout",
                            {"branch": "dev"})
    assert status == 200, body
    assert body["restarting"] is True
    # Scheduled 0.7s AFTER the answer is written, which is the whole point: the
    # process is about to go and the client needs the reply first.
    time.sleep(1.0)
    assert server["restarts"] == [1]


def test_a_generation_started_during_the_07s_delay_still_stops_the_restart(server, monkeypatch):
    """`_generation_running()` was only ever checked once, at schedule time --
    the 0.7s gap before the process actually goes (so the response reaches the
    client first) was a window nothing re-checked. A run queued in that window
    used to be killed anyway, which is exactly what this whole mechanism
    exists to prevent."""
    monkeypatch.setattr(update_mod, "pull",
                        lambda **_k: {"updated": True, "before": "a1", "after": "b2"})
    running = False
    monkeypatch.setattr(routes, "_generation_running", lambda: running)

    status, body = _request(server["port"], "POST", "/funpack/api/git/update", {})
    assert status == 200, body
    assert body["restarting"] is True, "scheduled optimistically, before the delay"

    # A generation starts in the gap between the response and the actual
    # restart -- the moment the real bug happened.
    running = True
    time.sleep(1.0)

    assert server["restarts"] == [], "restarted anyway despite the generation"

    status, body = _request(server["port"], "GET", "/funpack/api/git/status")
    assert body["restart_pending"] is True, "the deferred restart is not visible anywhere"


def test_a_moved_checkout_does_not_restart_while_a_generation_is_running(server, monkeypatch):
    """The client disables the buttons while a run is in flight, but that check
    goes stale if the dialog was left open across one starting -- this is the
    guard that actually stops the restart, not just the appearance of one."""
    monkeypatch.setattr(update_mod, "pull",
                        lambda **_k: {"updated": True, "before": "a1", "after": "b2"})
    monkeypatch.setattr(routes, "_generation_running", lambda: True)

    status, body = _request(server["port"], "POST", "/funpack/api/git/update",
                            {"branch": "dev"})
    assert status == 200, body
    assert body["restarting"] is False
    assert body["blocked"], "did not say why it refused to restart"
    assert body["after"] == "b2", "the git operation itself should still have run"
    time.sleep(1.0)
    assert server["restarts"] == [], "restarted ComfyUI while a generation was running"


def test_a_blocked_restart_is_actually_finishable_once_the_run_ends(server, monkeypatch):
    """A blocked update used to be a dead end: HEAD had already moved, so the
    same button's next press saw nothing to update and never restarted --
    ComfyUI kept running stale code with no way back short of the terminal."""
    monkeypatch.setattr(update_mod, "pull",
                        lambda **_k: {"updated": True, "before": "a1", "after": "b2"})
    monkeypatch.setattr(routes, "_generation_running", lambda: True)

    status, body = _request(server["port"], "POST", "/funpack/api/git/update", {})
    assert status == 200, body
    assert body["restarting"] is False

    status, body = _request(server["port"], "GET", "/funpack/api/git/status")
    assert status == 200, body
    assert body["restart_pending"] is True, "the owed restart is not visible anywhere"

    # Nothing else here works while a restart is owed -- running another git
    # action on top of an unrestarted change is not a retry, it is a second
    # change stacked on the first.
    status, body = _request(server["port"], "POST", "/funpack/api/git/update", {})
    assert status == 409, body

    # Still running: the dedicated restart route refuses too, same as the guard.
    status, body = _request(server["port"], "POST", "/funpack/api/git/restart")
    assert status == 200, body
    assert body["restarting"] is False
    assert server["restarts"] == []

    # The run ends. The SAME deferred restart finishes -- `pull` is not called
    # again, so if it were, this would fail on Mock's default None return.
    monkeypatch.setattr(routes, "_generation_running", lambda: False)
    status, body = _request(server["port"], "POST", "/funpack/api/git/restart")
    assert status == 200, body
    assert body["restarting"] is True
    time.sleep(1.0)
    assert server["restarts"] == [1]

    status, body = _request(server["port"], "GET", "/funpack/api/git/status")
    assert body["restart_pending"] is False, "stayed pending after actually restarting"


def test_restarting_with_nothing_pending_is_refused(server):
    status, body = _request(server["port"], "POST", "/funpack/api/git/restart")
    assert status == 400
    assert "detail" in body


def test_a_rollback_pressed_twice_while_blocked_does_not_roll_back_twice(server, monkeypatch):
    """The dangerous version of the dead end: HEAD already moved once, so
    re-running `rollback` on a second press would target the commit BEFORE the
    blocked rollback and silently undo it instead of finishing the restart."""
    calls = []

    def rollback():
        calls.append(1)
        return {"branch": "v5", "before": "b2", "after": "a1"}
    monkeypatch.setattr(update_mod, "rollback", rollback)
    monkeypatch.setattr(routes, "_generation_running", lambda: True)

    status, body = _request(server["port"], "POST", "/funpack/api/git/rollback")
    assert status == 200, body
    assert body["restarting"] is False
    assert calls == [1]

    # Pressed again while still blocked: refused outright, `rollback` not called
    # a second time.
    status, body = _request(server["port"], "POST", "/funpack/api/git/rollback")
    assert status == 409, body
    assert calls == [1], "ran rollback a second time instead of refusing"


def test_two_git_actions_at_once_do_not_touch_the_checkout_at_the_same_time(server, monkeypatch):
    """Two tabs, or a double-click before the button disables -- both requests
    used to be able to pass the pending-restart check and then run two real git
    subprocesses against the same working tree at once."""
    events = []

    def slow_pull(**_k):
        events.append(("pull", "enter"))
        time.sleep(0.3)
        events.append(("pull", "exit"))
        return {"updated": False, "before": "a1", "after": "a1"}

    def slow_checkout(**_k):
        events.append(("checkout", "enter"))
        time.sleep(0.3)
        events.append(("checkout", "exit"))
        return {"updated": False, "before": "a1", "after": "a1",
                "before_branch": "v5", "branch": "v5"}

    monkeypatch.setattr(update_mod, "pull", slow_pull)
    monkeypatch.setattr(update_mod, "checkout", slow_checkout)

    results = {}

    def fire(name, path, body):
        results[name] = _request(server["port"], "POST", path, body)

    t1 = threading.Thread(target=fire, args=("update", "/funpack/api/git/update", {}))
    t2 = threading.Thread(target=fire, args=("checkout", "/funpack/api/git/checkout", {"branch": "v5"}))
    t1.start()
    time.sleep(0.05)  # let the first one actually reach the lock first
    t2.start()
    t1.join(5)
    t2.join(5)

    assert results["update"][0] == 200, results["update"]
    assert results["checkout"][0] == 200, results["checkout"]
    # Whichever ran first, it must have fully exited before the other entered --
    # never interleaved (enter, enter, exit, exit) and never reordered so an
    # exit is missing before the next enter.
    assert events in (
        [("pull", "enter"), ("pull", "exit"), ("checkout", "enter"), ("checkout", "exit")],
        [("checkout", "enter"), ("checkout", "exit"), ("pull", "enter"), ("pull", "exit")],
    ), events


def test_status_with_a_remote_fetch_does_not_race_a_git_action_either(server, monkeypatch):
    """`GET /api/git/status?remote=1` runs its own `git fetch` -- a real
    mutation of the checkout's refs, same as pull/checkout/rollback -- and the
    Updates window fires it on every open. Without sharing the lock, opening
    that window mid-update raced two fetches against the same repo."""
    events = []

    def slow_status(*, remote=True):
        events.append(("status", "enter"))
        time.sleep(0.3)
        events.append(("status", "exit"))
        return {"ok": True, "branch": "v5", "branches": ["v5"], "dirty": False,
                "ahead": 0, "behind": 0, "fetch_ok": True, "repo": "/x"}

    def slow_pull(**_k):
        events.append(("pull", "enter"))
        time.sleep(0.3)
        events.append(("pull", "exit"))
        return {"updated": False, "before": "a1", "after": "a1"}

    monkeypatch.setattr(update_mod, "status", slow_status)
    monkeypatch.setattr(update_mod, "pull", slow_pull)

    results = {}

    def fire(name, method, path, body=None):
        results[name] = _request(server["port"], method, path, body)

    t1 = threading.Thread(target=fire, args=("update", "POST", "/funpack/api/git/update", {}))
    t2 = threading.Thread(target=fire, args=("status", "GET", "/funpack/api/git/status"))
    t1.start()
    time.sleep(0.05)
    t2.start()
    t1.join(5)
    t2.join(5)

    assert results["update"][0] == 200, results["update"]
    assert results["status"][0] == 200, results["status"]
    assert events in (
        [("pull", "enter"), ("pull", "exit"), ("status", "enter"), ("status", "exit")],
        [("status", "enter"), ("status", "exit"), ("pull", "enter"), ("pull", "exit")],
    ), events


def test_a_rollback_restarts_because_the_commit_moved(server, monkeypatch):
    monkeypatch.setattr(update_mod, "rollback",
                        lambda: {"branch": "v5", "before": "b2", "after": "a1"})

    status, body = _request(server["port"], "POST", "/funpack/api/git/rollback")
    assert status == 200, body
    assert body["restarting"] is True
    time.sleep(1.0)
    assert server["restarts"] == [1]


def test_the_pack_check_answers_under_the_key_the_app_reads(server):
    """The app reads `checked`. It read `nodes` for a while: the request
    succeeded, the JSON was valid, and every pack reported nothing to update
    however far behind it was. Neither side's tests could see it alone."""
    from core import nodes_manager
    shape = nodes_manager.check_updates.__doc__
    assert shape, "check_updates lost its docstring"

    status, body = _request(server["port"], "POST", "/funpack/api/packs/check")
    assert status == 200, body
    assert "checked" in body, f"the key the app reads is not in {sorted(body)}"
    assert isinstance(body["checked"], dict)


# --- the backend log ---------------------------------------------------------

def test_the_log_comes_back_with_where_it_was_read_from(server):
    """A log panel showing lines from an unknown file is a panel nobody can
    check against the terminal they also have open."""
    status, body = _request(server["port"], "GET", "/funpack/api/log?limit=5")
    assert status == 200, body
    assert isinstance(body["lines"], list)
    assert len(body["lines"]) <= 5
    # This machine's ComfyUI does write one; either way the answer names itself.
    assert body["path"] or body["detail"]


def test_no_log_file_is_an_answer_with_a_reason(server, monkeypatch):
    """An empty list looks exactly like a quiet log. The difference matters when
    the reason somebody opened the panel is that something already broke."""
    from core import backend_log
    monkeypatch.setattr(backend_log, "log_file", lambda: None)

    status, body = _request(server["port"], "GET", "/funpack/api/log")
    assert status == 200
    assert body["lines"] == []
    assert "terminal" in body["detail"], body["detail"]


def test_a_log_that_cannot_be_read_says_so_rather_than_raising(server, monkeypatch, tmp_path):
    from core import backend_log
    missing = tmp_path / "gone.log"
    missing.write_text("x\n")
    monkeypatch.setattr(backend_log, "log_file", lambda: missing)
    missing.unlink()

    status, body = _request(server["port"], "GET", "/funpack/api/log")
    assert status == 200, body
    assert body["lines"] == []
    assert "gone.log" in body["detail"]


def test_a_silly_limit_does_not_read_a_whole_session_into_memory(server):
    for limit in ("999999", "-4", "0", "not-a-number"):
        status, body = _request(server["port"], "GET", f"/funpack/api/log?limit={limit}")
        assert status == 200, (limit, body)
        assert len(body["lines"]) <= 2000, limit


def test_funpacks_own_log_answers_at_its_own_path_not_the_backend_ones(server):
    """Both logs were once registered on the same path, one shadowing the
    other -- FunPack's severity/source log never answered, and its consumer
    crashed on `data.levels`. They live at different paths now."""
    status, body = _request(server["port"], "GET", "/funpack/api/log/funpack")
    assert status == 200, body
    assert "levels" in body and "records" in body
    assert isinstance(body["levels"], list)


# --- temp files --------------------------------------------------------------

def test_temp_files_come_back_newest_first_with_where_they_are(server, monkeypatch, tmp_path):
    """Newest first, because the file somebody is hunting for is the one just
    written."""
    import os
    from core import temp_files
    (tmp_path / "sub").mkdir()
    old = tmp_path / "old.png"
    new = tmp_path / "sub" / "new.mp4"
    old.write_bytes(b"x")
    new.write_bytes(b"yy")
    os.utime(old, (1, 1))
    os.utime(new, (2000, 2000))
    monkeypatch.setattr(temp_files, "temp_dir", lambda: str(tmp_path))

    status, body = _request(server["port"], "GET", "/funpack/api/temp")
    assert status == 200, body
    assert [f["filename"] for f in body["files"]] == ["new.mp4", "old.png"]
    assert body["files"][0]["subfolder"] == "sub"
    assert body["files"][0]["kind"] == "video"
    assert body["files"][1]["subfolder"] == "", "a file at the root grew a subfolder"
    assert body["path"] == str(tmp_path)


def test_only_media_is_listed(server, monkeypatch, tmp_path):
    """A temp directory fills with whatever any node felt like writing, and a
    browser listing 400 .pt files is not a media bin."""
    from core import temp_files
    for name in ("keep.png", "keep.wav", "weights.pt", "notes.txt", "no_extension"):
        (tmp_path / name).write_bytes(b"x")
    monkeypatch.setattr(temp_files, "temp_dir", lambda: str(tmp_path))

    status, body = _request(server["port"], "GET", "/funpack/api/temp")
    assert sorted(f["filename"] for f in body["files"]) == ["keep.png", "keep.wav"]


def test_an_empty_temp_directory_says_why_it_is_empty(server, monkeypatch, tmp_path):
    from core import temp_files
    monkeypatch.setattr(temp_files, "temp_dir", lambda: str(tmp_path))
    status, body = _request(server["port"], "GET", "/funpack/api/temp")
    assert body["files"] == []
    assert "wiped" in body["detail"]


def test_no_comfyui_is_a_different_answer_from_an_empty_directory(server, monkeypatch):
    from core import temp_files
    monkeypatch.setattr(temp_files, "temp_dir", lambda: None)
    status, body = _request(server["port"], "GET", "/funpack/api/temp")
    assert body["files"] == []
    assert "ComfyUI is not here" in body["detail"]
    assert body["path"] is None


def test_the_listing_is_bounded(server, monkeypatch, tmp_path):
    from core import temp_files
    for i in range(30):
        (tmp_path / f"f{i}.png").write_bytes(b"x")
    monkeypatch.setattr(temp_files, "temp_dir", lambda: str(tmp_path))

    status, body = _request(server["port"], "GET", "/funpack/api/temp?limit=10")
    assert len(body["files"]) == 10
    status, body = _request(server["port"], "GET", "/funpack/api/temp?limit=999999")
    assert len(body["files"]) <= temp_files.MAX_FILES
