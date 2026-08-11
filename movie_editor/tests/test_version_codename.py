"""Version and release codename, as shown in Settings ▸ About.

The codename is cosmetic, so the only things worth pinning are the ones that would make it
look broken: a major with no name must render as ABSENCE (the panel omits the line), never
as an empty pair of quotes, and the pyproject version must stay parseable — About reads it
through the same helper the update check does.
"""
import pytest

from movie_editor.backend import git_update


def test_installed_version_is_parseable():
    version = git_update.funpack_version()
    assert version, "About and the update check both read this"
    parts = version.split(".")
    assert len(parts) >= 2 and all(p.isdigit() for p in parts[:2]), version


def test_current_major_has_a_codename():
    assert git_update.funpack_codename() == "Auspicious Asparagus"


def test_codename_is_keyed_by_major_not_by_release():
    """Every 3.x release ships under one name; only a new major earns a new letter."""
    for version in ("3.0.0", "3.5.1", "3.99.12"):
        assert git_update.funpack_codename(version) == "Auspicious Asparagus"


def test_unknown_major_has_no_codename():
    """Absence, not an empty string in quotes — the panel omits the line entirely."""
    assert git_update.funpack_codename("9.0.0") == ""
    assert git_update.funpack_codename("") != None  # noqa: E711 - never None, always str


@pytest.mark.parametrize("version", ["3", "3.5", "3.5.1", " 3.5.1 "])
def test_codename_tolerates_short_and_padded_versions(version):
    assert git_update.funpack_codename(version) == "Auspicious Asparagus"


def test_codenames_advance_alphabetically():
    """The Ubuntu rule this follows: adjective and vegetable share an initial, and the
    initial advances with the major. A future entry that breaks it should fail here."""
    for major, name in sorted(git_update.CODENAMES.items()):
        adjective, _, noun = name.partition(" ")
        assert noun, f"{name!r} should be two words"
        assert adjective[0].lower() == noun[0].lower(), f"{name!r} initials differ"
    majors = sorted(git_update.CODENAMES, key=int)
    initials = [git_update.CODENAMES[m][0].lower() for m in majors]
    assert initials == sorted(initials), f"codename initials must ascend: {initials}"


def test_status_carries_the_codename():
    """The About panel reads it off the git status payload, not a separate call."""
    assert "codename" in git_update.status()
