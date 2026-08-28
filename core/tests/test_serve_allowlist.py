"""Rule 1, transport half: a module cannot ship anything but JavaScript.

These exist before there is any UI to enforce them against — the wall goes up
first, so nothing can be built that leans on a hole in it.
"""

import pytest

from core import config
from core.serve import content_type_for, serve


@pytest.fixture()
def tree(tmp_path):
    """A miniature app/ + modules/ tree with the files that matter."""
    app = tmp_path / "app"
    (app / "composer").mkdir(parents=True)
    (app / "index.html").write_text("<p>shell</p>")
    (app / "boot.js").write_text("export const ok = 1;")
    (app / "composer" / "composer.css").write_text(":root{}")
    (app / "assets").mkdir()
    (app / "assets" / "Inter.woff2").write_bytes(b"\x00woff2")

    mods = tmp_path / "modules"
    (mods / "timing" / "audio_clock").mkdir(parents=True)
    (mods / "timing" / "audio_clock" / "ui.js").write_text("export default {};")
    (mods / "timing" / "audio_clock" / "sneaky.css").write_text(".x{color:red}")
    (mods / "timing" / "audio_clock" / "panel.html").write_text("<b>no</b>")
    (mods / "timing" / "audio_clock" / "module.py").write_text("x = 1")

    (tmp_path / "secret.txt").write_text("not yours")
    return tmp_path


def app_get(tree, rel):
    return serve(tree / "app", rel, config.APP_EXTS)


def module_get(tree, rel):
    return serve(tree / "modules", rel, config.MODULE_EXTS)


# --- the allowlist ---------------------------------------------------------

def test_module_js_is_served(tree):
    r = module_get(tree, "timing/audio_clock/ui.js")
    assert r.status == 200
    assert b"export default" in r.body


def test_module_css_is_not_served(tree):
    assert module_get(tree, "timing/audio_clock/sneaky.css").status == 404


def test_module_html_is_not_served(tree):
    assert module_get(tree, "timing/audio_clock/panel.html").status == 404


def test_module_python_is_not_served(tree):
    assert module_get(tree, "timing/audio_clock/module.py").status == 404


def test_module_directory_is_not_an_index(tree):
    # modules/ has no .html in its allowlist, so a directory is nothing to serve.
    assert module_get(tree, "timing/audio_clock").status == 404


def test_app_serves_its_own_css_and_fonts(tree):
    assert app_get(tree, "composer/composer.css").status == 200
    assert app_get(tree, "assets/Inter.woff2").status == 200


def test_app_root_serves_index(tree):
    r = app_get(tree, "")
    assert r.status == 200 and b"shell" in r.body


# --- the traversal guard ---------------------------------------------------

@pytest.mark.parametrize("rel", [
    "../secret.txt",
    "../../etc/passwd",
    "composer/../../secret.txt",
    "/../secret.txt",
])
def test_traversal_is_forbidden(tree, rel):
    assert app_get(tree, rel).status == 403


def test_traversal_out_of_modules_is_forbidden(tree):
    assert module_get(tree, "../app/boot.js").status == 403


def test_null_byte_is_forbidden(tree):
    assert app_get(tree, "boot.js\x00.png").status == 403


def test_symlink_escape_is_forbidden(tree):
    link = tree / "app" / "escape.js"
    link.symlink_to(tree / "secret.txt")
    assert app_get(tree, "escape.js").status == 403


# --- content types ---------------------------------------------------------

def test_js_is_explicitly_text_javascript(tree):
    # With ES modules a text/plain guess makes the browser refuse the module.
    assert app_get(tree, "boot.js").content_type == "text/javascript"
    assert module_get(tree, "timing/audio_clock/ui.js").content_type == "text/javascript"


def test_known_types_are_explicit():
    from pathlib import Path
    assert content_type_for(Path("a.css")) == "text/css"
    assert content_type_for(Path("a.woff2")) == "font/woff2"
    assert content_type_for(Path("a.svg")) == "image/svg+xml"


def test_responses_are_never_cached(tree):
    assert app_get(tree, "boot.js").headers["Cache-Control"] == "no-store, max-age=0"


# --- missing files ---------------------------------------------------------

def test_missing_file_is_404(tree):
    assert app_get(tree, "nope.js").status == 404
