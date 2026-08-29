"""The nodes FunPack actually ships, checked against ComfyUI itself.

Needs a ComfyUI source tree, because a node's schema is only meaningful in terms
of comfy_api's own types. Skipped where there is none.
"""

import pytest


@pytest.fixture(scope="module")
def shipped(comfyui):
    from core import nodes as nodes_mod
    from core import registry as registry_mod

    registry = registry_mod.scan()
    nodes, rejected = nodes_mod.collect(registry)
    return registry, nodes, rejected


def _schema(node):
    return node.GET_SCHEMA()


def _input_ids(schema):
    return [getattr(i, "id", None) for i in (schema.inputs or [])]


def test_nothing_shipped_is_rejected(shipped):
    _, _, rejected = shipped
    assert rejected == [], f"a shipped node did not register: {rejected}"


def test_every_shipped_node_is_a_real_comfy_node(shipped, comfyui):
    from comfy_api.latest import io
    _, nodes, _ = shipped
    for node in nodes:
        assert issubclass(node, io.ComfyNode), f"{node.__name__} is not an io.ComfyNode"
        assert isinstance(_schema(node), io.Schema)


# Each entry is a promise to everyone's saved workflows: `node_id` is stored as
# `class_type`, and input ids are stored as key names. Changing either silently
# breaks graphs that already exist, so this test is meant to FAIL on a rename and
# make that a decision rather than an accident.
FROZEN = {
    "FunPackCLIPLoader": [
        "clip_name1", "clip_name2", "clip_name3", "clip_name4", "type", "device",
    ],
    "FunPackLoadModifiers": ["model", "settings"],
    "FunPackModifierSettings": ["settings"],
    "FunPackEmptyLatent": ["model", "width", "height", "length", "batch_size"],
    "FunPackDiffusionModelLoader": [
        "model_name", "weight_dtype", "compute_dtype", "attention", "fp16_accumulation",
    ],
    "FunPackLoraLoader": [
        "model", "lora_name", "strength_model", "clip", "strength_clip",
    ],
    "FunPackVAELoader": ["vae_name", "dtype"],
}


def test_shipped_node_ids_and_inputs_are_frozen(shipped):
    _, nodes, _ = shipped
    actual = {_schema(n).node_id: _input_ids(_schema(n)) for n in nodes}
    assert actual == FROZEN, (
        "a shipped node's id or input names changed. Every saved workflow refers "
        "to these by name; update FROZEN only if you intend to break them."
    )


def test_a_setting_never_becomes_a_node_socket(shipped):
    """The rule that keeps the sampler from growing 70 widgets, and keeps a new
    setting from rotting saved graphs: settings are a payload, not sockets."""
    registry, _, _ = shipped
    for spec in registry.specs.values():
        keys = set(spec.settings)
        for node in spec.nodes:
            ids = set(_input_ids(_schema(node))) - {None}
            overlap = keys & ids
            assert not overlap, (
                f"{spec.id}'s node {node.__name__} exposes settings as sockets: "
                f"{sorted(overlap)}"
            )


def test_the_extension_hands_comfyui_exactly_what_was_collected(shipped, comfyui):
    import asyncio
    from core import nodes as nodes_mod

    extension = nodes_mod.extension()
    assert extension is not None
    listed = asyncio.run(extension.get_node_list())
    _, nodes, _ = shipped
    assert {_schema(n).node_id for n in listed} == {_schema(n).node_id for n in nodes}
