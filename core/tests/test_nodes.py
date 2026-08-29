"""Announcing ComfyUI nodes.

Core registers whatever modules declared and nothing it names itself. These tests
use stub node classes rather than real comfy_api ones on purpose: `collect()` only
ever asks a node for its schema, so it stays provable without ComfyUI installed --
the same property the rest of the suite has.
"""

from pathlib import Path

import pytest

from core import nodes as nodes_mod
from core import registry as registry_mod
from core.schema import SchemaError, validate

FIXTURES = Path(__file__).parent / "fixtures" / "fixture_modules"
PACKAGE = "core.tests.fixtures.fixture_modules"


@pytest.fixture(scope="module")
def collected():
    registry = registry_mod.scan(FIXTURES, package=PACKAGE)
    return registry, nodes_mod.collect(registry)


def _ids(nodes):
    return [n.GET_SCHEMA().node_id for n in nodes]


def _good():
    """The real GoodNode class, by identity."""
    from core.tests.fixtures.fixture_modules.nodes.good import GoodNode
    return GoodNode


def test_an_announced_node_is_collected(collected):
    _, (nodes, _) = collected
    # By identity. Asserting on the id alone let a DIFFERENT fixture module that
    # claims the same id satisfy this, so the test passed without the module
    # under test contributing anything.
    assert _good() in nodes


def test_a_node_outside_the_funpack_namespace_is_refused(collected):
    _, (nodes, rejected) = collected
    # ComfyUI would let this silently replace core's own VAELoader.
    assert "VAELoader" not in _ids(nodes)
    assert any("VAELoader" in why for _, why in rejected)


def test_two_modules_cannot_claim_one_node_id(collected):
    _, (nodes, rejected) = collected
    assert _ids(nodes).count("FunPackGood") == 1
    # And the one that survived is the first module to claim it, not whichever
    # happened to be reached last.
    assert _good() in nodes
    assert any("duplicate node_id" in why and "nodes_zz_clashes" in where
               for where, why in rejected)


def test_something_that_is_not_a_comfy_node_is_refused(collected):
    _, (nodes, rejected) = collected
    assert any("not a comfy_api" in why for _, why in rejected)
    # And it did not stop the good one being collected.
    assert _good() in nodes


def test_a_node_whose_schema_raises_does_not_take_the_others_down():
    class Exploding:
        @classmethod
        def define_schema(cls):
            raise RuntimeError("boom")

        @classmethod
        def execute(cls):
            return None

        @classmethod
        def GET_SCHEMA(cls):
            raise RuntimeError("boom")

    registry = registry_mod.scan(FIXTURES, package=PACKAGE)
    spec = registry.specs["nodes_good"]
    registry.specs["nodes_good"] = type(spec)(
        **{**spec.__dict__, "nodes": [Exploding, *spec.nodes]}
    )

    nodes, rejected = nodes_mod.collect(registry)
    assert _good() in nodes, "a raising sibling took the good node with it"
    assert any("boom" in why for _, why in rejected)


def test_a_node_only_module_needs_no_mount(collected):
    registry, _ = collected
    assert registry.specs["nodes_good"].mount == ""


def test_a_module_with_settings_still_needs_a_mount():
    with pytest.raises(SchemaError, match="mount"):
        validate({
            "id": "m", "title": "M",
            "settings": {"x": {"type": "bool", "default": True, "label": "X"}},
        })


@pytest.mark.parametrize("bad", [
    "not a list",
    ["not a class"],
    [object],                       # a class, but with no define_schema/execute
])
def test_a_malformed_nodes_declaration_refuses_the_module(bad):
    with pytest.raises(SchemaError):
        validate({"id": "m", "title": "M", "nodes": bad})


def test_a_node_that_cannot_be_named_does_not_take_the_others_down():
    """`getattr(node, "__name__", repr(node))` evaluates repr() eagerly, so a
    class whose metaclass raises there threw outside the guard and killed the
    whole collection -- and ComfyUI reports that as the entire pack failing."""

    class Meta(type):
        def __repr__(cls):
            raise RuntimeError("unreprable")

        def __getattribute__(cls, item):
            if item == "__name__":
                raise RuntimeError("unnameable")
            return type.__getattribute__(cls, item)

    class Hostile(metaclass=Meta):
        @classmethod
        def define_schema(cls):
            return None

        @classmethod
        def execute(cls):
            return None

        @classmethod
        def GET_SCHEMA(cls):
            return None

    registry = registry_mod.scan(FIXTURES, package=PACKAGE)
    spec = registry.specs["nodes_good"]
    registry.specs["nodes_good"] = type(spec)(
        **{**spec.__dict__, "nodes": [Hostile, *spec.nodes]}
    )

    nodes, rejected = nodes_mod.collect(registry)
    assert _good() in nodes
    assert any("unnameable" in why or "unreprable" in why for _, why in rejected)


@pytest.mark.parametrize("falsy", [0, False, ""])
def test_a_falsy_but_wrong_nodes_declaration_is_refused_not_repaired(falsy):
    """`announcement.get("nodes") or []` quietly turned a typo into "this module
    ships no nodes". This file refuses rather than repairs, and 0 is not []."""
    with pytest.raises(SchemaError, match="must be a list"):
        validate({"id": "m", "title": "M", "nodes": falsy})


@pytest.mark.parametrize("field", ["requires", "after", "before"])
def test_a_falsy_but_wrong_id_list_is_refused_too(field):
    with pytest.raises(SchemaError, match="must be a list"):
        validate({"id": "m", "title": "M", field: 0})


def test_an_absent_declaration_is_still_fine():
    # Absent means "none", which is different from a typo.
    spec = validate({"id": "m", "title": "M"})
    assert spec.nodes == [] and spec.requires == []
