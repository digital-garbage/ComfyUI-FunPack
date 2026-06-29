"""Bridge import helpers and parse error formatting."""

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
