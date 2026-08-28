"""Bundled assets referenced by CSS must actually exist and be servable.

A missing font file does not raise anything -- the browser quietly renders the
fallback stack, which looks like a design choice rather than a broken build.
"""

import re
from pathlib import Path

import pytest

from core import config
from core.serve import serve

FONTS_CSS = config.APP_DIR / "composer" / "tokens" / "fonts.css"
COMPOSER_CSS = config.APP_DIR / "composer" / "composer.css"
URL_PREFIX = config.UI_PREFIX + "/app/"


def font_urls():
    text = FONTS_CSS.read_text()
    return re.findall(r"url\('([^']+)'\)", text)


def test_fonts_css_is_imported_first():
    # Faces have to be declared before any rule uses the family, and tokens.css
    # names them immediately.
    imports = re.findall(r'@import\s+"([^"]+)"', COMPOSER_CSS.read_text())
    assert imports and imports[0] == "./tokens/fonts.css", imports


def test_there_are_faces_at_all():
    assert len(font_urls()) >= 1


@pytest.mark.parametrize("url", font_urls())
def test_every_declared_face_resolves_to_a_servable_file(url):
    assert url.startswith(URL_PREFIX), f"{url} is not under {URL_PREFIX}"
    tail = url[len(URL_PREFIX):]
    served = serve(config.APP_DIR, tail, config.APP_EXTS)
    assert served.status == 200, f"{url} -> {served.status}"
    assert served.content_type == "font/woff2"
    assert served.body[:4] == b"wOF2", "not a woff2 file"


def test_every_bundled_font_is_actually_referenced():
    # The reverse direction: a font file nothing declares is dead weight in the
    # repo, and usually means a face was renamed and its file left behind.
    declared = {Path(u).name for u in font_urls()}
    on_disk = {p.name for p in (config.APP_DIR / "assets" / "fonts").glob("*.woff2")}
    assert on_disk - declared == set(), f"unreferenced: {sorted(on_disk - declared)}"


def declared_families():
    return set(re.findall(r"font-family:\s*'([^']+)'", FONTS_CSS.read_text()))


def test_each_family_ships_its_licence():
    # SIL OFL: the licence has to travel with the fonts we redistribute, so a
    # newly bundled family without its OFL.txt is a licence violation, not a
    # tidiness problem.
    fonts_dir = config.APP_DIR / "assets" / "fonts"
    for family in declared_families():
        licence = fonts_dir / f"OFL-{family.replace(' ', '')}.txt"
        assert licence.is_file(), f"{family} is bundled without {licence.name}"
        assert "SIL OPEN FONT LICENSE" in licence.read_text().upper()
