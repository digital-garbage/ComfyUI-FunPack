"""Bridge import helpers and parse error formatting."""
import pytest

from movie_editor.backend import bridge


def test_format_funpack_error_with_message():
    assert bridge.format_funpack_error(ValueError("bad trigger")) == "ValueError: bad trigger"


def test_format_funpack_error_empty_message():
    assert bridge.format_funpack_error(KeyError()) == "KeyError (no message)"


def test_ensure_funpack_path_adds_repo_root():
    bridge._FUNPACK_PATH_ENSURED = False
    import sys
    root = str(bridge._FUNPACK_ROOT)
    sys.path[:] = [p for p in sys.path if p != root]
    bridge._ensure_funpack_path()
    assert root in sys.path


def _q_item(prompt_id, client_id, funpack=None):
    extra = {"client_id": client_id}
    if funpack is not None:
        extra["funpack"] = funpack
    return [1, prompt_id, {"graph": True}, extra, ["9"]]


def test_select_active_picks_editor_running_job_with_metadata():
    state = {
        "queue_running": [_q_item("p1", bridge.EDITOR_CLIENT_ID,
                                  {"pid": "proj-7", "scene_ids": ["a", "b"], "only_scene": None})],
        "queue_pending": [_q_item("p2", bridge.EDITOR_CLIENT_ID, {"pid": "proj-7"})],
    }
    out = bridge._select_active(state)
    assert out["running"] is True
    assert out["prompt_id"] == "p1"
    assert out["pid"] == "proj-7"
    assert out["scene_ids"] == ["a", "b"]
    assert out["pending"] == 1


def test_select_active_ignores_foreign_client_jobs():
    state = {
        "queue_running": [_q_item("other", "some-comfy-tab")],
        "queue_pending": [_q_item("other2", "some-comfy-tab")],
    }
    out = bridge._select_active(state)
    assert out["running"] is False
    assert out["prompt_id"] is None
    assert out["pending"] == 0


def test_select_active_empty_queue():
    out = bridge._select_active({})
    assert out == {"running": False, "prompt_id": None, "pid": None,
                   "scene_ids": [], "only_scene": None, "pending": 0}


def test_select_active_running_without_funpack_meta():
    state = {"queue_running": [_q_item("p1", bridge.EDITOR_CLIENT_ID)]}
    out = bridge._select_active(state)
    assert out["running"] is True
    assert out["prompt_id"] == "p1"
    assert out["pid"] is None
    assert out["scene_ids"] == []


def test_rating_labels_hide_internal_editor_values(monkeypatch):
    def _fake_attr(_mod, name):
        table = {
            "V2_RATING_LABELS": ["Perfect", "__funpack_continue__", "__funpack_fresh_prompt__"],
            "MOVIE_EDITOR_CONTINUE_RATING": "__funpack_continue__",
            "MOVIE_EDITOR_FRESH_PROMPT_RATING": "__funpack_fresh_prompt__",
        }
        return table[name]

    monkeypatch.setattr(bridge, "_funpack_attr", _fake_attr)
    labels = bridge.rating_labels().get("labels") or []
    assert "__funpack_continue__" not in labels
    assert "__funpack_fresh_prompt__" not in labels
    assert all(not str(l).startswith("__funpack_") for l in labels)
    assert "Perfect" in labels


# ── the log panel has to survive a restart ────────────────────────────────────
# The in-memory buffer only holds what THIS process printed, so a crash-and-restart left the
# panel empty — exactly when the lines before the crash are the ones worth reading.


@pytest.fixture
def logfile(tmp_path, monkeypatch):
    path = tmp_path / "comfyui.log"
    monkeypatch.setattr(bridge, "_comfy_log_file", lambda: path if path.is_file() else None)
    bridge._LOG_FILE_CACHE.update({"at": 0.0, "lines": []})
    with bridge._LOG_LOCK:
        bridge._LOG.clear()
    return path


def test_a_short_buffer_is_backfilled_from_comfyuis_own_log(logfile):
    logfile.write_text("older 1\nolder 2\nolder 3\n")
    assert bridge.recent_log(10) == ["older 1", "older 2", "older 3"]


def test_the_seam_is_not_shown_twice(logfile):
    """The file contains what the buffer holds. Without cutting the overlap the panel shows
    the same lines once from the file and once from the buffer."""
    logfile.write_text("old\nlive 1\nlive 2\n")
    with bridge._LOG_LOCK:
        bridge._LOG.extend(["live 1", "live 2"])
    assert bridge.recent_log(10) == ["old", "live 1", "live 2"]


def test_a_full_buffer_never_touches_the_file(logfile, monkeypatch):
    """Polled every 1.5s — once this process has enough of its own output, reading the file
    every time would be a file read per poll for nothing."""
    logfile.write_text("should not be read\n")
    monkeypatch.setattr(bridge, "_log_file_tail",
                        lambda n: pytest.fail("read the file with a full buffer"))
    with bridge._LOG_LOCK:
        bridge._LOG.extend(["a", "b", "c"])
    assert bridge.recent_log(3) == ["a", "b", "c"]


def test_no_log_file_is_not_an_error(logfile):
    with bridge._LOG_LOCK:
        bridge._LOG.extend(["only live"])
    assert bridge.recent_log(10) == ["only live"]


def test_an_unreadable_log_file_is_not_an_error(logfile, monkeypatch):
    logfile.write_text("x\n")
    monkeypatch.setattr(bridge, "_comfy_log_file",
                        lambda: (_ for _ in ()).throw(OSError("nope")))
    assert isinstance(bridge.recent_log(10), list)


def test_only_the_tail_of_a_huge_log_is_read(logfile):
    """A log can be hundreds of MB on a long-running box; the panel wants the last lines."""
    logfile.write_text("\n".join(f"line {i}" for i in range(200_000)) + "\n")
    out = bridge.recent_log(5)
    assert len(out) == 5 and out[-1] == "line 199999"
