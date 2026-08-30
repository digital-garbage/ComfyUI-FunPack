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


def test_two_slots_cannot_share_an_id():
    """A set of ids cannot see a duplicate, so writing the prompt by id used to
    overwrite one slot with the other: a node silently missing from the queued
    graph, or one left reading its own output."""
    slots = [
        {"id": "n1", "node": "LoadModel", "inputs": {}},
        {"id": "n1", "node": "AddLora", "inputs": {"model": ["n1", 0]}},
    ]
    prompt, problems = graph.build(slots, SCHEMAS)
    assert prompt == {}
    assert any("more than one slot" in p for p in problems)


def test_a_duplicate_id_is_refused_before_anything_else_is_reported():
    """Ids are how a link names what feeds it, so every other check downstream
    is meaningless while two slots answer to one name."""
    slots = [
        {"id": "a", "node": "LoadModel", "inputs": {}},
        {"id": "a", "node": "LoadModel", "inputs": {}},
        {"id": "b", "node": "AddLora", "inputs": {"nonsense": 1}},
    ]
    _prompt, problems = graph.build(slots, SCHEMAS)
    assert len(problems) == 1 and "more than one slot" in problems[0]


# --- cycles ----------------------------------------------------------------
#
# Every other refusal is local: one link, one type, one name. A cycle is correct
# at every link and impossible as a whole, so it is the one wrong graph that can
# pass a per-link check and be queued.

def test_two_slots_feeding_each_other_are_refused():
    prompt, problems = graph.build([
        {"id": "a", "node": "AddLora", "inputs": {"model": ["b", 0], "strength": 1.0}},
        {"id": "b", "node": "AddLora", "inputs": {"model": ["a", 0], "strength": 1.0}},
    ], SCHEMAS)
    assert len(problems) == 1
    assert "a" in problems[0] and "b" in problems[0]
    assert prompt  # the graph is still returned; it is `problems` that stops it


def test_a_slot_feeding_itself_is_refused():
    _, problems = graph.build([
        {"id": "loop", "node": "AddLora", "inputs": {"model": ["loop", 0], "strength": 1.0}},
    ], SCHEMAS)
    assert problems and "loop" in problems[0]


def test_one_loop_is_reported_once_however_many_links_close_it():
    _, problems = graph.build([
        {"id": "a", "node": "LoadModel", "inputs": {"name": "x"}},
        {"id": "b", "node": "TwoIn", "inputs": {"a": ["a", 0], "b": ["c", 0]}},
        {"id": "c", "node": "TwoIn", "inputs": {"a": ["b", 0], "b": ["b", 0]}},
    ], SCHEMAS)
    assert len(problems) == 1


def test_a_long_chain_is_walked_without_running_out_of_stack():
    # Slots come from a request, so their number is the caller's choice: a chain
    # deeper than Python's recursion limit must be a refusal or a graph, never a
    # RecursionError from inside the check.
    depth = 3000
    slots = [{"id": "s0", "node": "LoadModel", "inputs": {"name": "x"}}]
    slots += [{"id": f"s{i}", "node": "AddLora",
               "inputs": {"model": [f"s{i - 1}", 0], "strength": 1.0}}
              for i in range(1, depth)]
    _, problems = graph.build(slots, SCHEMAS)
    assert problems == []

    slots[0] = {"id": "s0", "node": "AddLora",
                "inputs": {"model": [f"s{depth - 1}", 0], "strength": 1.0}}
    _, problems = graph.build(slots, SCHEMAS)
    assert len(problems) == 1


def test_checking_a_big_pipeline_does_not_cost_more_than_reading_it():
    # Slots arrive from a request and nothing caps how many, and build() runs on
    # the event loop -- so the cost of checking one pipeline is paid by every
    # other request the server is holding. Looking the source of each link up in
    # a dict rebuilt per link made this quadratic: 20000 slots took 21 seconds.
    # The budget is deliberately loose; it is there to catch a return to N^2,
    # not to measure this machine.
    import time
    slots = [{"id": "s0", "node": "LoadModel", "inputs": {"name": "x"}}]
    slots += [{"id": f"s{i}", "node": "AddLora",
               "inputs": {"model": [f"s{i - 1}", 0], "strength": 1.0}}
              for i in range(1, 20000)]
    started = time.perf_counter()
    _, problems = graph.build(slots, SCHEMAS)
    assert problems == []
    assert time.perf_counter() - started < 2.0


def test_a_diamond_is_not_a_cycle():
    # Two paths reaching one source is ordinary wiring; a walk that marks a slot
    # seen without unmarking it would call this a loop.
    _, problems = graph.build([
        {"id": "model", "node": "LoadModel", "inputs": {"name": "x"}},
        {"id": "left", "node": "AddLora", "inputs": {"model": ["model", 0], "strength": 1.0}},
        {"id": "right", "node": "AddLora", "inputs": {"model": ["model", 0], "strength": 1.0}},
        {"id": "join", "node": "TwoIn", "inputs": {"a": ["left", 0], "b": ["right", 0]}},
    ], SCHEMAS)
    assert problems == []


# --- a value ComfyUI would refuse -------------------------------------------
#
# ComfyUI checks these too, at /prompt, and it refuses the WHOLE graph when one
# value is wrong. v4 hit exactly that with a LoRA file that had been deleted: a
# feature that was switched OFF still stopped every generation, and the message
# was "Prompt outputs failed validation" over a graph of a dozen loaders.

def _schemas_with_limits():
    return graph.Schemas(lambda ct: {
        "Loader": {
            "inputs": {"ckpt_name": "COMBO", "steps": "INT", "on": "BOOLEAN"},
            "outputs": ["MODEL"], "required": ["ckpt_name"],
            "limits": {"ckpt_name": {"choices": ["a.safetensors", "b.safetensors"]},
                       "steps": {"min": 1, "max": 100}},
        },
        "Empty": {"inputs": {"pick": "COMBO"}, "outputs": [], "required": [],
                  "limits": {"pick": {"choices": []}}},
    }.get(ct))


def test_a_combo_value_that_is_no_longer_on_disk_is_refused_before_queueing():
    _prompt, problems = graph.build(
        [{"id": "model", "node": "Loader", "inputs": {"ckpt_name": "gone.safetensors"}}],
        _schemas_with_limits())
    assert any("gone.safetensors" in p and "not one of" in p for p in problems), problems
    assert any("model.ckpt_name" in p for p in problems), problems


def test_a_combo_value_that_is_on_disk_is_accepted():
    prompt, problems = graph.build(
        [{"id": "model", "node": "Loader", "inputs": {"ckpt_name": "b.safetensors"}}],
        _schemas_with_limits())
    assert problems == []
    assert prompt["model"]["inputs"]["ckpt_name"] == "b.safetensors"


def test_a_number_outside_what_the_node_takes_is_refused():
    _prompt, problems = graph.build(
        [{"id": "model", "node": "Loader",
          "inputs": {"ckpt_name": "a.safetensors", "steps": 0}}],
        _schemas_with_limits())
    assert any("below the smallest" in p for p in problems), problems

    _prompt, problems = graph.build(
        [{"id": "model", "node": "Loader",
          "inputs": {"ckpt_name": "a.safetensors", "steps": 1000}}],
        _schemas_with_limits())
    assert any("above the largest" in p for p in problems), problems


def test_a_bool_is_not_measured_against_a_number_bound():
    """True is 1 in Python, so a naive comparison lets a checkbox be judged
    against a step count -- and quietly passes it."""
    prompt, problems = graph.build(
        [{"id": "model", "node": "Loader",
          "inputs": {"ckpt_name": "a.safetensors", "on": True}}],
        _schemas_with_limits())
    assert problems == []
    assert prompt["model"]["inputs"]["on"] is True


def test_an_empty_choice_list_is_reported_as_unfilled_not_as_a_wrong_value():
    """A machine with no files of that kind has not rejected anything. Saying
    "x is not one of: " helps nobody, and the slot being unfilled is already
    reported."""
    _prompt, problems = graph.build(
        [{"id": "e", "node": "Empty", "inputs": {"pick": "whatever"}}],
        _schemas_with_limits())
    assert not any("not one of" in p for p in problems), problems


def test_a_schema_that_says_nothing_about_a_value_does_not_refuse_it():
    """Most injected schemas carry no limits at all. Unknown is not a refusal:
    a schema that does not say is not a schema that says no."""
    schemas = graph.Schemas(lambda ct: {"inputs": {"anything": "STRING"},
                                  "outputs": [], "required": []})
    _prompt, problems = graph.build(
        [{"id": "x", "node": "Whatever", "inputs": {"anything": "hello"}}], schemas)
    assert problems == []


# --- place: what the UI holds, into the graph ---------------------------------

SINK = [{"node": "Sink", "input": "settings"}]


def test_place_writes_into_every_slot_a_sink_names():
    slots = [{"id": "a", "node": "Sink", "inputs": {"settings": "{}"}},
             {"id": "b", "node": "Other", "inputs": {"settings": "{}"}},
             {"id": "c", "node": "Sink", "inputs": {}}]
    out, placed = graph.place(slots, '{"m": {"on": true}}', SINK)

    assert placed == 2
    assert out[0]["inputs"]["settings"] == '{"m": {"on": true}}'
    assert out[1]["inputs"]["settings"] == "{}", "a node nobody named was written to"
    assert out[2]["inputs"]["settings"] == '{"m": {"on": true}}'


def test_place_reports_nothing_placed_rather_than_pretending():
    """The count is the whole point: a value with nowhere to go has to be said."""
    slots = [{"id": "a", "node": "Other", "inputs": {}}]
    out, placed = graph.place(slots, "{}", SINK)
    assert placed == 0
    assert out[0]["inputs"] == {}


def test_place_does_not_edit_the_pipeline_it_was_given():
    """The default pipeline is handed out by a provider; editing it in place
    would edit it for every later request in the process."""
    slots = [{"id": "a", "node": "Sink", "inputs": {"settings": "{}"}}]
    out, _ = graph.place(slots, "PAYLOAD", SINK)
    assert slots[0]["inputs"]["settings"] == "{}"
    assert out[0]["inputs"]["settings"] == "PAYLOAD"


def test_a_sink_missing_half_its_answer_is_ignored():
    slots = [{"id": "a", "node": "Sink", "inputs": {}}]
    for sink in ({"node": "Sink"}, {"input": "settings"}, {}, "Sink", None):
        out, placed = graph.place(slots, "PAYLOAD", [sink])
        assert placed == 0, sink
        assert out[0]["inputs"] == {}, sink
