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
    # A Primitive-shaped node: a WIDGET input, an output of the same type. This
    # is what "linked inputs" turns out to be -- one of these fed into several
    # widget inputs elsewhere, no separate grouping mechanism required.
    "PrimitiveInt": {"inputs": {"value": "INT"}, "outputs": ["INT"]},
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
    assert not any("nothing fills it" in p for p in problems), problems       # refused, not missing: said once, truly


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


def test_a_value_the_app_writes_is_refused_by_the_name_the_person_sees():
    _prompt, problems = graph.build(
        [{"id": "model", "node": "Loader", "roles": [{"at": "project.video", "input": "steps", "label": "Length"}],
          "inputs": {"ckpt_name": "a.safetensors", "steps": 1000}}],
        _schemas_with_limits())
    assert any(p.startswith("Length is 1000, above the largest") for p in problems), problems
    assert not any("nothing fills it" in p for p in problems), problems       # the refused value was said; it is not "missing"


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


# --- override: a value the app is holding, into a named input -----------------

PAIR = [{"id": "positive", "node": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": ""}},
        {"id": "latent", "node": "Empty", "inputs": {"width": 512}}]


def test_override_writes_a_value_into_the_input_it_names():
    out, problems = graph.override(PAIR, {"positive": {"text": "a cat"}})
    assert problems == []
    assert out[0]["inputs"]["text"] == "a cat"
    assert out[0]["inputs"]["clip"] == ["clip", 0], "the wiring was disturbed"
    assert out[1]["inputs"]["width"] == 512


def test_override_refuses_a_slot_that_is_not_there():
    """A prompt typed into a box for a slot somebody removed has gone nowhere,
    and a run that starts anyway is a run without the prompt in it."""
    out, problems = graph.override(PAIR, {"gone": {"text": "a cat"}})
    assert any("no such slot" in p for p in problems)
    assert out[0]["inputs"]["text"] == ""


def test_override_refuses_to_write_over_a_link():
    out, problems = graph.override(PAIR, {"positive": {"clip": "a string"}})
    assert any("fed by another node" in p for p in problems)
    assert out[0]["inputs"]["clip"] == ["clip", 0]


def test_override_refuses_a_shape_that_is_not_an_address():
    for edits in ("text", ["positive"], 7):
        _out, problems = graph.override(PAIR, edits)
        assert problems, edits
    _out, problems = graph.override(PAIR, {"positive": "a cat"})
    assert any("keyed by input name" in p for p in problems)


def test_override_does_not_edit_the_pipeline_it_was_given():
    graph.override(PAIR, {"positive": {"text": "a cat"}})
    assert PAIR[0]["inputs"]["text"] == ""


# --- wiring ------------------------------------------------------------

def test_wire_connects_one_slots_output_to_anothers_input():
    slots = [
        {"id": "model", "node": "LoadModel", "inputs": {"name": "a.safetensors"}},
        {"id": "lora", "node": "AddLora", "inputs": {"strength": 1.0}},
    ]
    out, problems = graph.wire(slots, "lora", "model", "model", 0, SCHEMAS)
    assert problems == []
    assert out[1]["inputs"]["model"] == ["model", 0]
    assert out[1]["inputs"]["strength"] == 1.0, "an unrelated input was disturbed"


def test_wire_refuses_a_slot_that_does_not_exist_on_either_side():
    slots = pipeline()
    _out, problems = graph.wire(slots, "gone", "model", "model", 0, SCHEMAS)
    assert any("gone" in p for p in problems)
    _out, problems = graph.wire(slots, "lora", "model", "gone", 0, SCHEMAS)
    assert any("gone" in p for p in problems)


def test_wire_refuses_an_input_the_node_does_not_declare():
    slots = pipeline()
    _out, problems = graph.wire(slots, "lora", "nope", "model", 0, SCHEMAS)
    assert any("no input" in p for p in problems)


def test_wire_refuses_an_output_index_that_does_not_exist():
    slots = pipeline()
    _out, problems = graph.wire(slots, "lora", "model", "model", 3, SCHEMAS)
    assert any("does not exist" in p for p in problems)


def test_wire_refuses_a_type_mismatch():
    slots = [
        {"id": "model", "node": "LoadModel", "inputs": {"name": "a.safetensors"}},
        {"id": "latent", "node": "Empty", "inputs": {"width": 512}},
    ]
    # latent's "width" wants INT; model's output 0 is MODEL.
    _out, problems = graph.wire(slots, "latent", "width", "model", 0, SCHEMAS)
    assert any("wants INT but" in p for p in problems)


def test_wire_refuses_a_link_that_would_close_a_cycle():
    slots = [
        {"id": "a", "node": "AddLora", "inputs": {"model": ["b", 0], "strength": 1.0}},
        {"id": "b", "node": "AddLora", "inputs": {"strength": 1.0}},
    ]
    _out, problems = graph.wire(slots, "b", "model", "a", 0, SCHEMAS)
    assert any("feeding itself" in p or "feeds" in p for p in problems)


def test_a_refused_wire_changes_nothing():
    slots = pipeline()
    graph.wire(slots, "lora", "nope", "model", 0, SCHEMAS)
    assert "nope" not in slots[1]["inputs"]


def test_wire_does_not_edit_the_pipeline_it_was_given():
    slots = [
        {"id": "model", "node": "LoadModel", "inputs": {"name": "a.safetensors"}},
        {"id": "lora", "node": "AddLora", "inputs": {"strength": 1.0}},
    ]
    graph.wire(slots, "lora", "model", "model", 0, SCHEMAS)
    assert "model" not in slots[1]["inputs"]


def test_wiring_a_widget_input_is_how_a_linked_input_works():
    """A Primitive node's output feeding several widget inputs -- v4's "linked
    inputs" was exactly this, minus the freezing: nothing here stops a FOURTH
    slot being wired to the same Primitive later, because there is no group
    object that has to already know about it."""
    slots = [
        {"id": "shared", "node": "PrimitiveInt", "inputs": {"value": 512}},
        {"id": "a", "node": "Empty", "inputs": {}},
        {"id": "b", "node": "Empty", "inputs": {}},
    ]
    slots, problems = graph.wire(slots, "a", "width", "shared", 0, SCHEMAS)
    assert problems == []
    slots, problems = graph.wire(slots, "b", "width", "shared", 0, SCHEMAS)
    assert problems == []
    assert slots[1]["inputs"]["width"] == slots[2]["inputs"]["width"] == ["shared", 0]

    # A third consumer, added on afterward -- not a group that had to exist
    # ahead of time.
    slots.append({"id": "c", "node": "Empty", "inputs": {}})
    slots, problems = graph.wire(slots, "c", "width", "shared", 0, SCHEMAS)
    assert problems == []
    assert slots[3]["inputs"]["width"] == ["shared", 0]


def test_wire_re_wires_an_input_that_already_had_something():
    slots = pipeline()  # lora.model is already wired to model.0
    out, problems = graph.wire(slots, "lora", "model", "model", 0, SCHEMAS)
    assert problems == []
    assert out[1]["inputs"]["model"] == ["model", 0]


# --- unwiring ------------------------------------------------------------

def test_unwire_removes_the_link():
    slots = pipeline()
    out, problems = graph.unwire(slots, "lora", "model")
    assert problems == []
    assert "model" not in out[1]["inputs"]


def test_unwire_refuses_a_slot_that_is_not_there():
    slots = pipeline()
    _out, problems = graph.unwire(slots, "gone", "model")
    assert any("gone" in p for p in problems)


def test_unwiring_an_input_that_was_never_wired_is_not_an_error():
    slots = pipeline()
    out, problems = graph.unwire(slots, "lora", "strength")
    assert problems == []
    assert "strength" not in out[1]["inputs"]


def test_unwire_does_not_edit_the_pipeline_it_was_given():
    slots = pipeline()
    graph.unwire(slots, "lora", "model")
    assert slots[1]["inputs"]["model"] == ["model", 0]


def test_a_role_is_checked_for_shape_and_nothing_else():
    """Core does not know what any of these names MEAN. It knows they are names."""
    ok = [{"id": "a", "node": "N", "roles": [{"at": "anywhere.at.all", "input": "text",
                                              "label": "Whatever"}]}]
    assert graph.shape_problems(ok) == []

    assert graph.shape_problems([{"id": "a", "node": "N", "roles": {}}])
    assert graph.shape_problems([{"id": "a", "node": "N", "roles": ["text"]}])
    assert any("has no at" in p for p in
               graph.shape_problems([{"id": "a", "node": "N", "roles": [{"input": "text"}]}]))
    assert any("has no input" in p for p in
               graph.shape_problems([{"id": "a", "node": "N", "roles": [{"at": "x"}]}]))
    assert any("label must be text" in p for p in
               graph.shape_problems([{"id": "a", "node": "N",
                                      "roles": [{"at": "x", "input": "t", "label": 5}]}]))


# --- adding ----------------------------------------------------------------

def test_a_node_can_be_added_to_a_group_and_gets_its_own_id():
    slots, problems = graph.add(pipeline(), "PrimitiveInt", "My group", SCHEMAS)
    assert problems == [] and slots[-1] == {"id": "primitiveint", "node": "PrimitiveInt", "inputs": {}, "group": "My group"}
    slots, _ = graph.add(slots, "PrimitiveInt", None, SCHEMAS)
    assert [s["id"] for s in slots[-2:]] == ["primitiveint", "primitiveint_2"] and "group" not in slots[-1]


def test_adding_what_is_not_installed_or_to_a_blank_group_is_refused():
    slots, problems = graph.add(pipeline(), "Nope", None, SCHEMAS)
    assert problems and len(slots) == len(pipeline())
    assert graph.add(pipeline(), "Empty", "  ", SCHEMAS)[1]


def test_a_new_node_can_be_wired_into_several_inputs_and_later_ones_added_to_the_same_link():
    """The v4 'linked input' that froze once made: here it is one Primitive and ordinary wires."""
    slots, _ = graph.add(pipeline(), "PrimitiveInt", None, SCHEMAS)
    slots, _ = graph.add(slots, "Empty", None, SCHEMAS)
    slots, problems = graph.wire(slots, "empty", "width", "primitiveint", 0, SCHEMAS)
    assert problems == []
    slots, _ = graph.add(slots, "Empty", None, SCHEMAS)
    slots, problems = graph.wire(slots, "empty_2", "width", "primitiveint", 0, SCHEMAS)
    assert problems == [] and graph.slots_by_id(slots)["empty_2"]["inputs"]["width"] == ["primitiveint", 0]

def test_a_swap_keeps_only_the_roles_the_new_node_can_still_take():
    slots = [{"id": "empty", "node": "Empty", "inputs": {"width": 8},
              "roles": [{"at": "project.video", "input": "width"}, {"at": "project.video", "input": "length"}]},
             {"id": "save", "node": "Save", "inputs": {"latent": ["empty", 0]}}]
    swapped, problems = graph.replace(slots, "empty", "Empty", SCHEMAS)
    assert problems == []
    assert graph.slots_by_id(swapped)["empty"]["roles"] == [{"at": "project.video", "input": "width"}]


def test_a_numeric_combo_choice_saved_as_text_is_sent_as_the_number():
    bounds = {"choices": ["auto", 8, 10]}
    assert graph.as_choice("8", bounds) == 8 and graph.unacceptable("bit_depth", 8, bounds) is None
    assert graph.as_choice("auto", bounds) == "auto"
    assert graph.as_choice("9", bounds) == "9", "not a choice: left as it is, and refused by name"


# --- splicing an added pass-through node ------------------------------------

def test_an_added_model_patch_goes_last_before_the_sampler():
    """Added beside the chain it feeds nothing, and ComfyUI never runs a node nothing reads:
    a sparse-attention patch wired only from the LoRA was silently never run."""
    slots, _ = graph.add(pipeline(), "AddLora", None, SCHEMAS)
    slots, note = graph.splice(slots, "addlora", SCHEMAS)
    by = graph.slots_by_id(slots)
    assert note is None
    assert by["addlora"]["inputs"]["model"] == ["lora", 0] and by["sample"]["inputs"]["model"] == ["addlora", 0]
    assert graph.build(slots, SCHEMAS)[1] == []


def test_splice_never_guesses():
    two_samplers = pipeline() + [{"id": "sample2", "node": "Sampler", "inputs": {"model": ["lora", 0], "latent": ["latent", 0]}}]
    slots, _ = graph.add(two_samplers, "AddLora", None, SCHEMAS)
    spliced, note = graph.splice(slots, "addlora", SCHEMAS)
    assert spliced == slots and "not connected" in note and "sample.model" in note and "sample2.model" in note
    for node in ("TwoIn", "PrimitiveInt", "Wrong", "Empty"):      # two MODEL inputs; a widget value; not a pass-through
        slots, _ = graph.add(pipeline(), node, None, SCHEMAS)
        assert graph.splice(slots, slots[-1]["id"], SCHEMAS) == (slots, None)


def test_wiring_a_loose_pass_through_from_x_puts_it_after_x():
    """Dex wired a sparse-attention node's model from the LoRA; the sampler still read the LoRA,
    so the node fed nothing and never ran."""
    slots = pipeline() + [{"id": "patch", "node": "AddLora", "inputs": {}}]
    slots, problems = graph.wire(slots, "patch", "model", "lora", 0, SCHEMAS)
    slots, note = graph.forward(slots, "patch", "model", SCHEMAS)
    assert problems == [] and graph.slots_by_id(slots)["sample"]["inputs"]["model"] == ["patch", 0]
    assert note == "sample.model now reads AddLora"
    # Already read by something: the person wired its output; nothing moves.
    again, note = graph.forward(slots, "patch", "model", SCHEMAS)
    assert again == slots and note is None
    # Not its pass-through input, or not a pass-through at all: nothing moves.
    loose = pipeline() + [{"id": "s2", "node": "Sampler", "inputs": {"model": ["lora", 0]}}]
    assert graph.forward(loose, "s2", "model", SCHEMAS) == (loose, None)


def test_a_dynamic_dropdowns_fields_are_inputs_and_needed_only_for_their_choice():
    """Model Sparse Attention: picking vsa brings selection.keep_percent; ComfyUI refused the
    prompt ("Required input is missing: keep_percent") because v5 never drew or sent it."""
    from core import comfy_types
    opts = {"options": [{"key": "sol-attn", "inputs": {"required": {"tau": ("FLOAT", {"default": 1.3})}}},
                        {"key": "sla", "inputs": {"required": {"keep_percent": ("FLOAT", {"min": 0.5})}}},
                        {"key": "vsa", "inputs": {"required": {"keep_percent": ("FLOAT", {"min": 0.5})}}}]}
    got = {d: (req, parent, keys) for d, _, req, parent, keys in comfy_types.revealed("selection", opts)}
    assert got == {"selection.tau": (True, "selection", ["sol-attn"]),
                   "selection.keep_percent": (True, "selection", ["sla", "vsa"])}
    schemas = graph.Schemas({"Sparse": {"inputs": {"model": "MODEL", "selection": "COMBO", "selection.tau": "FLOAT",
                                                    "selection.keep_percent": "FLOAT"},
                                         "outputs": ["MODEL"], "required": ["model", "selection"],
                                         "required_when": {"selection.tau": ("selection", ["sol-attn"]),
                                                           "selection.keep_percent": ("selection", ["sla", "vsa"])}}}.get)
    assert schemas.required("Sparse", {"selection": "vsa"}) == ["model", "selection", "selection.keep_percent"]
    assert schemas.required("Sparse", {"selection": "sol-attn"}) == ["model", "selection", "selection.tau"]


def test_a_lora_loader_added_after_the_first_goes_into_the_model_chain():
    """Its CLIP in/out is optional, so it is still a MODEL patch: left loose it never ran, and the
    sampler read only the first LoRA."""
    nodes = {"Lora": {"inputs": {"model": "MODEL", "lora_name": "COMBO", "clip": "CLIP"}, "outputs": ["MODEL", "CLIP", "STRING"], "required": ["model", "lora_name"]},
             "Loader": {"inputs": {}, "outputs": ["MODEL"]}, "Sampler": {"inputs": {"model": "MODEL"}, "outputs": ["LATENT"], "required": ["model"]}}
    schemas = graph.Schemas(nodes.get)
    slots = [{"id": "model", "node": "Loader", "inputs": {}}, {"id": "lora", "node": "Lora", "inputs": {"model": ["model", 0], "lora_name": "a"}},
             {"id": "sample", "node": "Sampler", "inputs": {"model": ["lora", 0]}}]
    for _ in range(3):
        slots, _p = graph.add(slots, "Lora", None, schemas)
        slots, note = graph.splice(slots, slots[-1]["id"], schemas)
        assert note is None
    by = graph.slots_by_id(slots)
    assert [by[i]["inputs"]["model"][0] for i in ("lora_2", "lora_3", "lora_4")] == ["lora", "lora_2", "lora_3"]
    assert by["sample"]["inputs"]["model"] == ["lora_4", 0]


# --- optional slots: a pipeline that runs without a node whose module did not load ---------------

OPT_NODES = {
    "Encode": {"inputs": {"prompt": "STRING", "model": "MODEL"}, "outputs": ["LATENT"], "required": ["prompt"]},
    "Rewrite": {"inputs": {"text": "STRING", "enabled": "BOOLEAN"}, "outputs": ["STRING"]},
    "Markup": {"inputs": {"text": "STRING"}, "outputs": ["STRING"]},
    "Out": {"inputs": {"latent": "LATENT"}, "outputs": []},
    "LoadModel": NODES["LoadModel"],
    "Tune": {"inputs": {"model": "MODEL"}, "outputs": ["MODEL"]},
}


def _optional_pipeline():
    return [
        {"id": "model", "node": "LoadModel", "inputs": {"name": "a"}},
        {"id": "tune", "node": "Tune", "inputs": {"model": ["model", 0]}, "optional": {"pass": {"0": "model"}}},
        {"id": "rewrite", "node": "Rewrite", "inputs": {"text": "a typed prompt", "enabled": False}, "optional": {"pass": {"0": "text"}}},
        {"id": "markup", "node": "Markup", "inputs": {"text": ["rewrite", 0]}, "optional": {"pass": {"0": "text"}}},
        {"id": "encode", "node": "Encode", "inputs": {"prompt": ["markup", 0], "model": ["tune", 0]}},
        {"id": "out", "node": "Out", "inputs": {"latent": ["encode", 0]}},
    ]


def _without(*gone):
    return graph.Schemas({k: v for k, v in OPT_NODES.items() if k not in gone}.get)


def test_a_skipped_slot_hands_on_the_text_that_was_typed_into_it():
    """The prompt box writes into the Enhance slot's text input; skipping the slot must not lose it."""
    prompt, problems = graph.build(_optional_pipeline(), _without("Rewrite", "Markup"))
    assert problems == []
    assert prompt["encode"]["inputs"]["prompt"] == "a typed prompt"
    assert "rewrite" not in prompt and "markup" not in prompt


def test_a_skipped_slot_hands_on_the_wire_that_fed_it():
    prompt, problems = graph.build(_optional_pipeline(), _without("Tune"))
    assert problems == [] and prompt["encode"]["inputs"]["model"] == ["model", 0] and "tune" not in prompt


def test_nothing_is_skipped_when_the_node_is_there():
    prompt, problems = graph.build(_optional_pipeline(), _without())
    assert problems == [] and prompt["encode"]["inputs"]["prompt"] == ["markup", 0] and "rewrite" in prompt


def test_the_slots_that_were_skipped_are_named_and_the_saved_pipeline_is_untouched():
    slots = _optional_pipeline()
    kept, skipped = graph.bypass_missing(slots, _without("Rewrite", "Markup", "Tune"))
    assert sorted(s["slot"] for s in skipped) == ["markup", "rewrite", "tune"]
    assert [s["id"] for s in kept] == ["model", "encode", "out"]
    assert [s["id"] for s in slots] == ["model", "tune", "rewrite", "markup", "encode", "out"]       # the input is not edited
    assert slots[3]["inputs"]["text"] == ["rewrite", 0]


def test_an_optional_slot_with_no_pass_for_a_read_output_is_still_reported_missing():
    slots = _optional_pipeline()
    slots[2]["optional"] = {"pass": {"5": "text"}}
    _prompt, problems = graph.build(slots, _without("Rewrite"))
    assert any("no node called 'Rewrite'" in p for p in problems)


def test_an_optional_slot_whose_pass_input_is_empty_is_still_reported_missing():
    slots = _optional_pipeline()
    del slots[2]["inputs"]["text"]
    _prompt, problems = graph.build(slots, _without("Rewrite"))
    assert any("no node called 'Rewrite'" in p for p in problems)


def test_a_slot_that_is_not_marked_optional_still_refuses_when_missing():
    slots = _optional_pipeline()
    del slots[1]["optional"]
    _prompt, problems = graph.build(slots, _without("Tune"))
    assert any("no node called 'Tune'" in p for p in problems)


def test_a_malformed_optional_is_refused_as_a_shape_problem():
    slots = _optional_pipeline()
    slots[1]["optional"] = {"pass": "model"}
    assert any("optional must be" in p for p in graph.shape_problems(slots))
    slots[1]["optional"] = {"pass": {"zero": "model"}}
    assert any("optional must be" in p for p in graph.shape_problems(slots))


def test_two_skipped_slots_reading_each_other_do_not_hang():
    slots = [
        {"id": "a", "node": "Rewrite", "inputs": {"text": ["b", 0]}, "optional": {"pass": {"0": "text"}}},
        {"id": "b", "node": "Markup", "inputs": {"text": ["a", 0]}, "optional": {"pass": {"0": "text"}}},
        {"id": "encode", "node": "Encode", "inputs": {"prompt": ["a", 0]}},
    ]
    prompt, problems = graph.build(slots, _without("Rewrite", "Markup"))
    assert any("no node called" in p for p in problems)             # nothing can be passed through a loop: both stay, and are reported missing


def test_a_chain_that_cannot_be_passed_through_keeps_the_slot_that_breaks_it_and_does_not_crash():
    """A skipped slot whose pass input is another skipped slot's output the second does not pass: both used to raise KeyError."""
    nodes = {k: OPT_NODES[k] for k in ("Encode", "Out", "LoadModel")}          # A and B are not installed
    slots = [
        {"id": "b", "node": "B", "inputs": {"y": "typed"}, "optional": {"pass": {"0": "y"}}},
        {"id": "a", "node": "A", "inputs": {"x": ["b", 1]}, "optional": {"pass": {"0": "x"}}},
        {"id": "encode", "node": "Encode", "inputs": {"prompt": ["a", 0]}},
    ]
    schemas = graph.Schemas({k: v for k, v in nodes.items()}.get)
    prompt, problems = graph.build(slots, schemas)                  # B has no pass for output 1: A cannot be skipped, and neither can B
    assert any("no node called" in p for p in problems)
    # a pass input that is not filled at all
    slots[0]["optional"] = {"pass": {"0": "y", "1": "missing"}}
    _prompt, problems = graph.build(slots, schemas)
    assert any("no node called" in p for p in problems)


def test_the_table_marks_nodes_by_class_for_pipelines_saved_without_any_marker():
    slots = _optional_pipeline()
    for s in slots:
        s.pop("optional", None)
    table = {"Rewrite": {"pass": {"0": "text"}}}
    prompt, problems = graph.build(slots, _without("Rewrite", "Markup"), table)
    assert any("Markup" in p for p in problems) and not any("Rewrite" in p for p in problems)      # Markup is not in the table: still required


def test_a_slots_own_marker_wins_over_the_table_and_a_null_pass_value_is_not_passed_on():
    slots = _optional_pipeline()
    slots[2]["optional"] = {"pass": {"0": "text"}}
    prompt, _ = graph.build(slots, _without("Rewrite", "Markup"), {"Rewrite": {"pass": {"0": "enabled"}}})
    assert prompt["encode"]["inputs"]["prompt"] == "a typed prompt"               # the slot said "text"; the table said "enabled"
    slots[2]["inputs"]["text"] = None
    _prompt, problems = graph.build(slots, _without("Rewrite", "Markup"))
    assert any("no node called" in p for p in problems)                            # nothing typed: refused, not sent as null
