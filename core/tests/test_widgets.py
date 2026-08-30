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
    """ComfyUI's own nodes, plus FunPack's, and PUT BACK afterwards.

    The same two steps the dev server takes, through the same function, because
    a pipeline made of FunPack nodes cannot be described by a registry that only
    holds ComfyUI's.

    Restored because NODE_CLASS_MAPPINGS is a module-level dict shared by the
    whole session: installing into it and walking away left every later test
    running against a registry this file had changed, and twenty-three of them
    failed somewhere else entirely. This project has been bitten by exactly that
    before.
    """
    import nodes as comfy_nodes
    from core import nodes as funpack_nodes

    before = dict(comfy_nodes.NODE_CLASS_MAPPINGS)
    before_names = dict(comfy_nodes.NODE_DISPLAY_NAME_MAPPINGS)
    funpack_nodes.install_into(comfy_nodes.NODE_CLASS_MAPPINGS,
                               comfy_nodes.NODE_DISPLAY_NAME_MAPPINGS)
    try:
        yield
    finally:
        comfy_nodes.NODE_CLASS_MAPPINGS.clear()
        comfy_nodes.NODE_CLASS_MAPPINGS.update(before)
        comfy_nodes.NODE_DISPLAY_NAME_MAPPINGS.clear()
        comfy_nodes.NODE_DISPLAY_NAME_MAPPINGS.update(before_names)


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


# --- searching for a node to put in a slot ---------------------------------

def test_searching_finds_a_node_by_its_class_name():
    found = widgets.search("KSampler")
    names = [entry["node"] for entry in found["nodes"]]
    assert "KSampler" in names
    # Exact first. Typing the whole name and being handed KSamplerAdvanced is
    # the picker being unhelpful about the one thing it was told.
    assert names[0] == "KSampler"


def test_searching_finds_a_node_by_its_category():
    found = widgets.search("latent")
    assert found["nodes"], "nothing matched a category every install has"
    assert all("latent" in f"{e['node']}{e['title']}{e['category']}".lower()
               for e in found["nodes"])


def test_a_search_says_how_many_it_did_not_show():
    """A picker that shows forty of four hundred and says nothing implies forty
    is all there is."""
    everything = widgets.search("", limit=10)
    assert len(everything["nodes"]) == 10
    assert everything["total"] > 10


def test_nothing_matching_is_an_empty_answer_rather_than_everything():
    found = widgets.search("zzz-no-node-is-called-this")
    assert found == {"nodes": [], "total": 0}


def test_a_found_node_carries_what_a_picker_shows():
    entry = widgets.search("KSampler")["nodes"][0]
    assert set(entry) == {"node", "title", "category", "outputs"}
    assert entry["outputs"] == ["LATENT"]


def test_funpacks_own_nodes_are_findable_too():
    """The registry is one registry: a picker that could not offer FunPack's own
    nodes would make the built-in pipeline unrepairable from the window."""
    names = [e["node"] for e in widgets.search("FunPack", limit=200)["nodes"]]
    assert "FunPackSampler" in names


# --- the two spellings of a combo ------------------------------------------

def test_a_v3_combo_carries_its_choices():
    """ComfyUI spells a combo two ways and both are current.

    The old one puts the choices where the type goes -- (["a", "b"], {...}) --
    and the V3 schema puts the string "COMBO" there with the choices under
    `options`. Reading only the first found no choices on ANY V3 node, and the
    window said "no files of this kind were found" over dropdowns that had
    seven entries. It was believable, which is what made it bad.
    """
    described = widgets.describe("FunPackDiffusionModelLoader")
    weights = next(w for w in described["widgets"] if w["name"] == "weight_dtype")
    assert "fp16" in weights["choices"]
    assert weights["default"] == "default"


def test_an_old_style_combo_still_carries_its_choices():
    described = widgets.describe("KSampler")
    scheduler = next(w for w in described["widgets"] if w["name"] == "scheduler")
    assert "normal" in scheduler["choices"]


def test_every_combo_in_the_default_pipeline_offers_something_or_is_genuinely_empty():
    """A combo with no choices is a real state -- a file picker on a machine
    with no files -- so this asserts the ones that cannot be empty are not."""
    from modules.system.pipeline import DEFAULT
    for slot in DEFAULT:
        for widget in widgets.describe(slot["node"])["widgets"]:
            if widget["type"] != widgets.COMBO:
                continue
            assert "choices" in widget, f"{slot['id']}.{widget['name']} has no choices key"
            if "name" in widget["name"] or widget["name"].endswith("_name"):
                continue                          # a file picker; may be empty
            assert widget["choices"], f"{slot['id']}.{widget['name']} offers nothing"
