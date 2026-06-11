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
