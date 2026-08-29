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


def test_an_announced_node_is_collected(collected):
    _, (nodes, _) = collected
    assert "FunPackGood" in _ids(nodes)


def test_a_node_outside_the_funpack_namespace_is_refused(collected):
    _, (nodes, rejected) = collected
    # ComfyUI would let this silently replace core's own VAELoader.
    assert "VAELoader" not in _ids(nodes)
    assert any("VAELoader" in why for _, why in rejected)


def test_two_modules_cannot_claim_one_node_id(collected):
    _, (nodes, rejected) = collected
    assert _ids(nodes).count("FunPackGood") == 1
    assert any("duplicate node_id" in why for _, why in rejected)


def test_something_that_is_not_a_comfy_node_is_refused(collected):
    _, (nodes, rejected) = collected
    assert any("not a comfy_api" in why for _, why in rejected)
    # And it did not stop the good one being collected.
    assert "FunPackGood" in _ids(nodes)


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
    assert "FunPackGood" in _ids(nodes)
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
