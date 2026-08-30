"""A node's inputs, as a form.

Read against REAL nodes from ComfyUI's registry rather than hand-written
declarations: the shape of INPUT_TYPES is ComfyUI's to define, and a fixture
written from memory is a fixture that agrees with whatever was remembered.
"""

import pytest

from core import widgets


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports nodes."""


@pytest.fixture(autouse=True)
def _registry():
    """ComfyUI's own nodes, plus FunPack's.

    The same two steps the dev server takes, through the same function, because
    a pipeline made of FunPack nodes cannot be described by a registry that only
    holds ComfyUI's.
    """
    import nodes as comfy_nodes
    from core import nodes as funpack_nodes
    funpack_nodes.install_into(comfy_nodes.NODE_CLASS_MAPPINGS,
                               comfy_nodes.NODE_DISPLAY_NAME_MAPPINGS)


@pytest.mark.parametrize("node,widget_names,socket_names", [
    # A text box is a widget even though STRING is upper case -- deciding by
    # case classified every prompt in the pipeline as a socket nobody could
    # type in.
    ("CLIPTextEncode", ["text"], ["clip"]),
    ("EmptyLatentImage", ["width", "height", "batch_size"], []),
    ("CheckpointLoaderSimple", ["ckpt_name"], []),
    ("SaveImage", ["filename_prefix"], ["images"]),
])
def test_what_is_typed_and_what_is_wired(node, widget_names, socket_names):
    described = widgets.describe(node)
    assert described, f"{node} is not installed"
    assert [w["name"] for w in described["widgets"]] == widget_names
    assert [s["name"] for s in described["sockets"]] == socket_names


def test_a_number_carries_the_bounds_a_form_needs():
    described = widgets.describe("KSampler")
    steps = next(w for w in described["widgets"] if w["name"] == "steps")
    assert steps["type"] == "INT"
    assert steps["default"] == 20
    assert steps["min"] >= 1 and steps["max"] > steps["min"]
    assert steps["tooltip"], "the node explains itself and the form dropped it"


def test_a_combo_carries_its_choices():
    described = widgets.describe("KSampler")
    sampler = next(w for w in described["widgets"] if w["name"] == "sampler_name")
    assert sampler["type"] == widgets.COMBO
    assert "euler" in sampler["choices"]


def test_an_empty_combo_is_still_a_combo():
    """A file picker on a machine with no files. Rendering a select with nothing
    in it is right; calling it a socket, or dropping it, is not."""
    described = widgets.describe("CheckpointLoaderSimple")
    ckpt = next(w for w in described["widgets"] if w["name"] == "ckpt_name")
    assert ckpt["type"] == widgets.COMBO
    assert isinstance(ckpt["choices"], list)


def test_required_and_optional_are_both_described_and_told_apart():
    described = widgets.describe("KSampler")
    every = described["widgets"] + described["sockets"]
    assert all("required" in item for item in every)
    assert any(item["required"] for item in every)


def test_a_node_nobody_installed_is_null_rather_than_an_error():
    assert widgets.describe("NoSuchNodeAnywhere") is None


def test_asking_about_several_says_which_are_absent():
    found = widgets.describe_all(["KSampler", "NoSuchNodeAnywhere", "KSampler"])
    assert set(found) == {"KSampler", "NoSuchNodeAnywhere"}
    assert found["KSampler"] and found["NoSuchNodeAnywhere"] is None


def test_every_node_in_the_default_pipeline_can_be_described():
    """The pipeline is what the window edits, so a slot nothing can describe is
    a slot nobody can fill in."""
    from modules.system.pipeline import DEFAULT
    for slot in DEFAULT:
        described = widgets.describe(slot["node"])
        assert described, f"{slot['id']} points at {slot['node']}, which cannot be described"


def test_a_node_whose_declaration_raises_is_absent_not_fatal(monkeypatch):
    import nodes as comfy_nodes

    class Explodes:
        @classmethod
        def INPUT_TYPES(cls):
            raise RuntimeError("the declaration is upside down")

    monkeypatch.setitem(comfy_nodes.NODE_CLASS_MAPPINGS, "Explodes", Explodes)
    assert widgets.describe("Explodes") is None
