"""Scanning a real tree of modules, most of them deliberately broken.

The fixtures under fixtures/fixture_modules are the point: one good module and
five ways of being wrong, so "a broken module is absent, not fatal" is a thing
the suite demonstrates rather than a claim in a comment.
"""

from pathlib import Path

import pytest

from core import registry as registry_mod
from core.relations import order
from core.traits import missing_for, split

FIXTURES = Path(__file__).parent / "fixtures" / "fixture_modules"
PACKAGE = "core.tests.fixtures.fixture_modules"


@pytest.fixture(scope="module")
def scanned():
    return registry_mod.scan(FIXTURES, package=PACKAGE)


def test_the_good_module_loads(scanned):
    assert "audio_clock" in scanned.specs
    spec = scanned.specs["audio_clock"]
    assert spec.title == "Audio clock"
    assert spec.stage == "conditioning"
    assert spec.status == "proven"


def test_a_module_that_raises_on_import_is_absent_not_fatal(scanned):
    # The scan completed and returned other modules, which is the guarantee.
    assert "explodes" not in scanned.specs
    assert any("raises" in where for where, _ in scanned.failed)
    assert scanned.specs, "one bad module did not take the rest down with it"


@pytest.mark.parametrize("bad_id", ["bad_ui", "bad_label", "no_default"])
def test_an_invalid_declaration_never_reaches_the_manifest(scanned, bad_id):
    assert bad_id not in scanned.specs


def test_every_failure_says_why(scanned):
    # A module that is silently missing is indistinguishable from one that was
    # never installed; the log line is what makes it debuggable.
    assert scanned.failed
    for where, why in scanned.failed:
        assert where and why, f"failure recorded with no reason: {where!r} {why!r}"


def test_a_module_with_a_ui_file_advertises_its_path(scanned):
    assert scanned.specs["audio_clock"].ui.endswith("/modules/timing/audio_clock/ui.js")


def test_a_module_without_a_ui_file_advertises_none(scanned):
    # The default is no JavaScript at all: a settings schema is a whole module.
    assert scanned.specs["momentum"].ui is None


def test_defaults_are_available_without_any_ui(scanned):
    assert scanned.specs["audio_clock"].defaults() == {
        "enabled": True, "strength": 0.65, "mode": "beat",
    }


def test_a_duplicate_id_is_refused_rather_than_decided_by_import_order():
    reg = registry_mod.Registry()
    from core.schema import validate
    a = validate({"id": "x", "title": "A", "mount": "m"}, source="one")
    b = validate({"id": "x", "title": "B", "mount": "m"}, source="two")
    reg.add(a)
    reg.add(b)
    assert reg.specs["x"].title == "A"
    assert any("duplicate id" in why for _, why in reg.failed)


# --- ordering --------------------------------------------------------------

def test_relations_decide_the_order(scanned):
    ordered, rejected = order([scanned.specs["audio_clock"], scanned.specs["momentum"]])
    ids = [s.id for s in ordered]
    assert ids.index("audio_clock") < ids.index("momentum")
    assert rejected == []


def test_before_is_the_same_edge_from_the_other_end():
    from core.schema import validate
    a = validate({"id": "a", "title": "A", "mount": "m", "before": ["b"]})
    b = validate({"id": "b", "title": "B", "mount": "m"})
    ordered, _ = order([b, a])
    assert [s.id for s in ordered] == ["a", "b"]


def test_a_relation_naming_an_absent_module_is_not_an_error():
    # Modules come and go; "after audio_clock" has nothing to wait for when
    # audio_clock was never installed.
    from core.schema import validate
    solo = validate({"id": "solo", "title": "S", "mount": "m", "after": ["not_installed"]})
    ordered, rejected = order([solo])
    assert [s.id for s in ordered] == ["solo"]
    assert rejected == []


def test_a_cycle_is_dropped_rather_than_ordered_arbitrarily(scanned):
    # An arbitrary order is one that works until it does not, and by then nobody
    # remembers it was never real.
    pair = [scanned.specs["cycle_first"], scanned.specs["cycle_second"]]
    ordered, rejected = order(pair)
    assert ordered == []
    assert {spec.id for spec, _ in rejected} == {"cycle_first", "cycle_second"}
    assert all("cycle" in why for _, why in rejected)


def test_a_cycle_does_not_take_the_healthy_modules_with_it(scanned):
    everything = list(scanned.specs.values())
    ordered, rejected = order(everything)
    assert "audio_clock" in [s.id for s in ordered]
    assert {s.id for s, _ in rejected} == {"cycle_first", "cycle_second"}


def test_stages_run_in_declared_order():
    from core.schema import validate
    late = validate({"id": "late", "title": "L", "mount": "m", "stage": "post"})
    early = validate({"id": "early", "title": "E", "mount": "m", "stage": "load"})
    ordered, _ = order([late, early])
    assert [s.id for s in ordered] == ["early", "late"]


def test_ordering_is_stable_without_any_relations():
    from core.schema import validate
    specs = [validate({"id": i, "title": i, "mount": "m"}) for i in ("c", "a", "b")]
    assert [s.id for s in order(specs)[0]] == ["a", "b", "c"]


# --- traits ----------------------------------------------------------------

def test_a_module_is_kept_only_when_the_model_has_what_it_needs(scanned):
    specs = list(scanned.specs.values())
    ok, no = split(specs, ["audio_stream", "flow_match"])
    assert "audio_clock" in [s.id for s in ok]
    assert "needs_trait" in [s.id for s in no], "cfg_free was not on offer"


def test_a_module_with_no_requirements_fits_anything(scanned):
    ok, _ = split([scanned.specs["momentum"]], [])
    assert [s.id for s in ok] == ["momentum"]


def test_the_missing_traits_are_nameable(scanned):
    assert missing_for(scanned.specs["needs_trait"], ["audio_stream"]) == ["cfg_free"]


def test_traits_are_names_the_model_offers_not_model_names(scanned):
    # A module listing model names has to be edited whenever a model ships, and
    # the one nobody edits is the one that silently stops appearing.
    for spec in scanned.specs.values():
        for trait in spec.requires:
            assert trait.islower() and " " not in trait
