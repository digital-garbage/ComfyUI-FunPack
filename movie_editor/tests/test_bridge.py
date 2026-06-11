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
