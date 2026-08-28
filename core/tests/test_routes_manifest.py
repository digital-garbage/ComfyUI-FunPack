"""The manifest: only what loaded and validated reaches the browser."""

from pathlib import Path

import pytest

from core import routes as routes_mod
from core import registry as registry_mod

FIXTURES = Path(__file__).parent / "fixtures" / "fixture_modules"
PACKAGE = "core.tests.fixtures.fixture_modules"


@pytest.fixture()
def manifest(monkeypatch):
    reg = registry_mod.scan(FIXTURES, package=PACKAGE)
    monkeypatch.setattr(routes_mod, "modules", lambda rescan=False: reg)
    return routes_mod.manifest


def test_only_valid_modules_are_listed(manifest):
    ids = [m["id"] for m in manifest()["modules"]]
    assert "audio_clock" in ids
    for broken in ("explodes", "bad_ui", "bad_label", "no_default"):
        assert broken not in ids


def test_a_cycle_is_reported_but_not_listed(manifest):
    result = manifest()
    ids = [m["id"] for m in result["modules"]]
    assert "cycle_first" not in ids and "cycle_second" not in ids
    assert any("cycle" in entry["why"] for entry in result["failed"])


def test_failures_are_carried_for_the_dump_not_for_rendering(manifest):
    # The UI renders `modules`. `failed` exists so a person can find out why
    # something is missing without reading the server log.
    result = manifest()
    assert result["failed"]
    assert all({"where", "why"} == set(entry) for entry in result["failed"])


def test_the_manifest_is_ordered_by_relation(manifest):
    ids = [m["id"] for m in manifest()["modules"]]
    assert ids.index("audio_clock") < ids.index("momentum")


def test_traits_filter_what_is_offered(manifest):
    with_audio = manifest(["audio_stream"])
    assert "audio_clock" in [m["id"] for m in with_audio["modules"]]
    assert "needs_trait" in [m["id"] for m in with_audio["incompatible"]]

    without = manifest([])
    assert "audio_clock" not in [m["id"] for m in without["modules"]]
    assert "momentum" in [m["id"] for m in without["modules"]], "no requirements, fits anything"


def test_no_traits_means_no_filtering(manifest):
    assert manifest()["incompatible"] == []


def test_the_contract_version_travels_with_it(manifest):
    from core.contract import CONTRACT_VERSION
    assert manifest()["contract"] == CONTRACT_VERSION


def test_the_import_path_is_not_published(manifest):
    # Where a module lives on disk is core's business.
    for entry in manifest()["modules"]:
        assert "source" not in entry
