"""The pipeline: built, replaced, removed.

Schemas are injected, so all of this runs without ComfyUI. The point under test
is not any particular node -- it is that no node is privileged.
"""

import pytest

from core import graph

NODES = {
    "LoadModel":  {"inputs": {"name": "COMBO"}, "outputs": ["MODEL"]},
    "OtherModel": {"inputs": {"path": "COMBO"}, "outputs": ["MODEL"]},
    "AddLora":    {"inputs": {"model": "MODEL", "strength": "FLOAT"}, "outputs": ["MODEL"]},
    "Sampler":    {"inputs": {"model": "MODEL", "latent": "LATENT"}, "outputs": ["LATENT"]},
    "Empty":      {"inputs": {"width": "INT"}, "outputs": ["LATENT"]},
    "Save":       {"inputs": {"latent": "LATENT"}, "outputs": []},
    "TwoIn":      {"inputs": {"a": "MODEL", "b": "MODEL"}, "outputs": ["MODEL"]},
    "Wrong":      {"inputs": {"model": "MODEL"}, "outputs": ["IMAGE"]},
}
SCHEMAS = graph.Schemas(NODES.get)


def pipeline():
    return [
        {"id": "model", "node": "LoadModel", "inputs": {"name": "a.safetensors"}},
        {"id": "lora", "node": "AddLora", "inputs": {"model": ["model", 0], "strength": 1.0}},
        {"id": "latent", "node": "Empty", "inputs": {"width": 512}},
        {"id": "sample", "node": "Sampler", "inputs": {"model": ["lora", 0], "latent": ["latent", 0]}},
        {"id": "save", "node": "Save", "inputs": {"latent": ["sample", 0]}},
    ]


# --- building --------------------------------------------------------------

def test_slots_become_the_prompt_comfyui_queues():
    prompt, problems = graph.build(pipeline(), SCHEMAS)
    assert problems == []
    assert prompt["sample"] == {"class_type": "Sampler",
                                "inputs": {"model": ["lora", 0], "latent": ["latent", 0]}}
    assert prompt["model"]["inputs"]["name"] == "a.safetensors"


@pytest.mark.parametrize("break_it,expected", [
    (lambda s: s[0].update(node="NoSuchNode"), "no node called"),
    (lambda s: s[1]["inputs"].update(nonsense=1), "has no input"),
    (lambda s: s[1]["inputs"].update(model=["ghost", 0]), "not in the pipeline"),
    (lambda s: s[1]["inputs"].update(model=["model", 7]), "reads output 7"),
    (lambda s: s[3]["inputs"].update(model=["latent", 0]), "wants MODEL but"),
])
def test_a_graph_that_cannot_work_is_refused_with_the_reason(break_it, expected):
    slots = pipeline()
    break_it(slots)
    _prompt, problems = graph.build(slots, SCHEMAS)
    assert any(expected in p for p in problems), problems


# --- replacing -------------------------------------------------------------

def test_a_slot_can_point_at_a_different_node_producing_the_same_thing():
    """The built-in nodes have no special standing. Anything giving the same
    result is as good, which is what makes them replaceable rather than fixed."""
    slots, problems = graph.replace(pipeline(), "model", "OtherModel", SCHEMAS)
    assert problems == []
    assert graph.slots_by_id(slots)["model"]["node"] == "OtherModel"
    _prompt, build_problems = graph.build(slots, SCHEMAS)
    assert build_problems == []


def test_a_replacement_that_produces_the_wrong_thing_is_refused():
    slots, problems = graph.replace(pipeline(), "model", "Wrong", SCHEMAS)
    assert any("needs MODEL but Wrong gives IMAGE" in p for p in problems)
    assert graph.slots_by_id(slots)["model"]["node"] == "LoadModel", "it changed anyway"


def test_replacing_does_not_carry_inputs_across():
    """A different node has different inputs, and keeping the ones whose names
    happen to match is how a value quietly comes to mean something else."""
    slots, _ = graph.replace(pipeline(), "model", "OtherModel", SCHEMAS)
    assert graph.slots_by_id(slots)["model"]["inputs"] == {}


# --- removing --------------------------------------------------------------

def test_a_built_in_slot_can_be_removed_and_what_it_fed_is_rewired():
    """The thing v4 could not do. Removing the LoRA leaves the sampler reading
    the model directly."""
    slots, problems = graph.remove(pipeline(), "lora", SCHEMAS)
    assert problems == []
    assert "lora" not in graph.slots_by_id(slots)
    assert graph.slots_by_id(slots)["sample"]["inputs"]["model"] == ["model", 0]
    _prompt, build_problems = graph.build(slots, SCHEMAS)
    assert build_problems == []


def test_removing_something_nothing_reads_just_removes_it():
    slots = pipeline() + [{"id": "spare", "node": "Empty", "inputs": {"width": 64}}]
    slots, problems = graph.remove(slots, "spare", SCHEMAS)
    assert problems == [] and "spare" not in graph.slots_by_id(slots)


def test_removing_a_source_with_nothing_to_inherit_is_refused():
    """`Empty` produces a LATENT out of nothing, so there is no upstream latent
    for the sampler to fall back to."""
    _slots, problems = graph.remove(pipeline(), "latent", SCHEMAS)
    assert any("no wired LATENT input" in p for p in problems)


def test_an_ambiguous_removal_is_refused_and_names_the_ambiguity():
    slots = [
        {"id": "a", "node": "LoadModel", "inputs": {}},
        {"id": "b", "node": "LoadModel", "inputs": {}},
        {"id": "merge", "node": "TwoIn", "inputs": {"a": ["a", 0], "b": ["b", 0]}},
        {"id": "sample", "node": "Sampler", "inputs": {"model": ["merge", 0]}},
    ]
    _slots, problems = graph.remove(slots, "merge", SCHEMAS)
    assert any("ambiguous" in p and "a, b" in p for p in problems)


def test_a_refused_removal_changes_nothing():
    before = pipeline()
    after, problems = graph.remove(before, "latent", SCHEMAS)
    assert problems and after == before


def test_an_output_nobody_wired_does_not_block_removal():
    """v4 required a passthrough for every output a class could produce, so a
    node with one spare output nobody consumed became unremovable."""
    NODES["Spare"] = {"inputs": {"model": "MODEL"}, "outputs": ["MODEL", "FLOAT"]}
    slots = [
        {"id": "model", "node": "LoadModel", "inputs": {}},
        {"id": "spare", "node": "Spare", "inputs": {"model": ["model", 0]}},
        {"id": "sample", "node": "Sampler", "inputs": {"model": ["spare", 0]}},
    ]
    slots, problems = graph.remove(slots, "spare", SCHEMAS)
    assert problems == []
    assert graph.slots_by_id(slots)["sample"]["inputs"]["model"] == ["model", 0]
