"""The pipeline, over HTTP.

core/graph.py was well tested and called from nowhere -- proven correct in a
vacuum while the claim it backs ("a built-in node can be replaced or removed")
was not something the running server could do. These tests are about
reachability as much as behaviour.
"""

import json
import socket
import threading

import pytest

pytest.importorskip("aiohttp")

from aiohttp import web  # noqa: E402

from core import routes  # noqa: E402


@pytest.fixture(scope="module")
def registered(comfyui):
    """FunPack's nodes registered the way ComfyUI registers them.

    Without this the pipeline's slots all look like missing nodes, because the
    schema lookup reads ComfyUI's registry and nothing had put anything in it.
    """
    import asyncio
    import nodes as comfy_nodes
    from pathlib import Path

    async def load():
        await comfy_nodes.init_extra_nodes(init_custom_nodes=False)
        await comfy_nodes.load_custom_node(
            str(Path(__file__).resolve().parents[2]), module_parent="custom_nodes")

    asyncio.run(load())
    return comfy_nodes


@pytest.fixture(scope="module")
def server(comfyui, registered):
    """The real route table on a throwaway app, on a real socket."""
    from aiohttp import web as aioweb

    app = aioweb.Application()
    table = aioweb.RouteTableDef()
    routes.register(table, prefix="/funpack")
    app.add_routes(table)

    holder = {}

    def run():
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        runner = aioweb.AppRunner(app)
        loop.run_until_complete(runner.setup())
        site = aioweb.TCPSite(runner, "127.0.0.1", 0)
        loop.run_until_complete(site.start())
        holder["port"] = site._server.sockets[0].getsockname()[1]
        holder["loop"] = loop
        holder["ready"].set()
        loop.run_forever()

    holder["ready"] = threading.Event()
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert holder["ready"].wait(20), "the test server did not start"
    yield holder["port"]
    holder["loop"].call_soon_threadsafe(holder["loop"].stop)


def _request(port, method, path, body=None):
    payload = b"" if body is None else json.dumps(body).encode()
    head = (f"{method} {path} HTTP/1.1\r\nHost: localhost\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n").encode()
    with socket.create_connection(("127.0.0.1", port), timeout=30) as sock:
        sock.sendall(head + payload)
        chunks = []
        while True:
            got = sock.recv(65536)
            if not got:
                break
            chunks.append(got)
    raw = b"".join(chunks)
    header, _, rest = raw.partition(b"\r\n\r\n")
    status = int(header.split()[1])
    try:
        return status, json.loads(rest.decode() or "{}")
    except ValueError:
        return status, {"raw": rest.decode(errors="replace")}


def test_the_default_pipeline_is_served(server):
    status, body = _request(server, "GET", "/funpack/api/pipeline")
    assert status == 200
    assert [slot["id"] for slot in body["slots"]], "no slots came back"
    assert "queueable" in body
    # A fresh install has no model picked, so the default pipeline is
    # incomplete rather than wrong -- and the two are reported separately.
    assert body["refused"] == []
    assert any("nothing fills it" in p for p in body["incomplete"])


def test_a_built_in_slot_can_be_removed_over_http(server):
    """The claim, exercised the way the app will: the modifiers node taken out,
    and what it fed rewired to what fed it."""
    status, body = _request(server, "POST", "/funpack/api/pipeline",
                            {"action": "remove", "slot": "modifiers"})
    assert status == 200, body
    assert body["refused"] == [], body["refused"]
    ids = [slot["id"] for slot in body["slots"]]
    assert "modifiers" not in ids

    sampler = next(s for s in body["slots"] if s["id"] == "sampler")
    assert sampler["inputs"]["model"] == ["model", 0], "the sampler was not rewired"


def test_a_removal_that_cannot_work_comes_back_as_a_reason(server):
    status, body = _request(server, "POST", "/funpack/api/pipeline",
                            {"action": "remove", "slot": "latent"})
    assert status == 200
    assert any("LATENT" in p for p in body["refused"])
    assert body["queueable"] is False
    assert body["prompt"] is None


def test_a_built_in_slot_can_be_replaced_over_http(server):
    status, body = _request(server, "POST", "/funpack/api/pipeline",
                            {"action": "replace", "slot": "vae", "node": "VAELoader"})
    assert status == 200, body
    vae = next(s for s in body["slots"] if s["id"] == "vae")
    assert vae["node"] == "VAELoader"


def test_a_replacement_that_gives_the_wrong_thing_is_refused(server):
    status, body = _request(server, "POST", "/funpack/api/pipeline",
                            {"action": "replace", "slot": "model", "node": "SaveImage"})
    assert status == 200
    assert body["refused"], "a nonsense replacement was accepted"


def test_an_unknown_action_is_refused(server):
    status, body = _request(server, "POST", "/funpack/api/pipeline",
                            {"action": "explode", "slot": "model"})
    assert status == 400 and body["problems"]  # a malformed request, not an edit


def test_an_explicitly_empty_pipeline_is_not_replaced_by_the_default(server):
    """`slots or default()` resurrects the default, because an empty list is
    falsy. A client that removed every slot is entitled to be told it has none
    rather than handed the built-ins back."""
    status, body = _request(server, "POST", "/funpack/api/pipeline",
                            {"action": "check", "slots": []})
    assert status == 200
    assert body["slots"] == [], "an empty pipeline came back full"


def test_omitting_slots_still_falls_back_to_the_default(server):
    """The distinction that matters: absent means "use the default", empty means
    "there is nothing". Only one of them is a fallback."""
    status, body = _request(server, "POST", "/funpack/api/pipeline",
                            {"action": "check"})
    assert status == 200
    assert body["slots"], "omitting slots should use the default pipeline"


def test_removing_slots_one_by_one_never_resurrects_the_default(server):
    """The failure this guards: each request carries the client's current slots,
    so a fallback that fires on an empty list would silently hand the built-ins
    back mid-way through and the count would jump UP."""
    _status, body = _request(server, "GET", "/funpack/api/pipeline")
    slots = body["slots"]
    started_with = len(slots)
    assert started_with > 1, "the default pipeline is too small to test this"

    counts = [started_with]
    # Consumers before sources: removing a source nothing can replace is refused,
    # which is correct and not what this test is about.
    for slot_id in [s["id"] for s in reversed(slots)]:
        _status, body = _request(server, "POST", "/funpack/api/pipeline",
                                 {"action": "remove", "slot": slot_id, "slots": slots})
        if body["refused"]:
            continue
        slots = body["slots"]
        counts.append(len(slots))

    assert len(slots) < started_with, "nothing was actually removed"
    assert counts == sorted(counts, reverse=True), (
        f"the pipeline grew back part way through: {counts}")


@pytest.mark.parametrize("payload,expected", [
    ({"action": "check", "slots": ["not-a-dict"]}, "not an object"),
    ({"action": "check", "slots": [{"node": "LoadModel"}]}, "has no id"),
    ({"action": "check", "slots": [{"id": "a"}]}, "has no node"),
    ({"action": "check", "slots": [{"id": "a", "node": "X", "inputs": "nope"}]}, "must be an object"),
    ({"action": "check", "slots": "a pipeline"}, "is a list of slots"),
    # The body itself. Every case above is already a well-formed object, so the
    # layer that reads named fields off it had never been handed something with
    # no names: `body.get` on a list is an AttributeError and a 500.
    ([], "an object, not a list"),
    ("just a string", "an object, not a str"),
    (42, "an object, not a int"),
    # The fields naming what to edit. The slots array is shape-checked; these
    # come from the same body and were not, and both reach an `in` against a
    # dict, where an unhashable value is a TypeError rather than a miss.
    ({"action": "replace", "slot": ["a"], "node": "X", "slots": []}, "named by a string"),
    ({"action": "remove", "slot": {"x": 1}, "slots": []}, "named by a string"),
    ({"action": "replace", "slot": "a", "node": ["X"], "slots": []}, "named by a string"),
    # A group is the pipeline's own arrangement and core knows nothing about
    # what the names mean -- only that they are names. A number here becomes a
    # card titled `5`, and `5` and `"5"` become two cards reading the same.
    ({"action": "check", "slots": [{"id": "a", "node": "X", "group": 5}]}, "must be a name"),
    ({"action": "check", "slots": [{"id": "a", "node": "X", "group": "  "}]}, "is blank"),
])
def test_a_malformed_pipeline_comes_back_as_a_reason_not_a_500(server, payload, expected):
    """A payload arrives from HTTP, so "it is a list" is all that has been
    established. Reading slot["id"] off a string raised a TypeError and the
    route answered 500 with plain text, which the app cannot parse into
    anything to show."""
    status, body = _request(server, "POST", "/funpack/api/pipeline", payload)
    assert status in (200, 400), f"a malformed payload crashed the route: {status}"
    reasons = body.get("refused", []) + body.get("problems", [])
    assert any(expected in r for r in reasons), reasons
    assert body.get("queueable") is False


def test_the_default_pipeline_arrives_already_arranged(server):
    """The window draws a card per group, so a pipeline with no groups is one
    card holding everything -- which is the arrangement it exists to avoid."""
    _status, body = _request(server, "GET", "/funpack/api/pipeline")
    groups = [slot.get("group") for slot in body["slots"]]
    assert all(groups), f"slots with no group: {[s['id'] for s in body['slots'] if not s.get('group')]}"
    assert len(set(groups)) > 1, "everything landed in one group"


def test_a_group_survives_an_edit(server):
    """Replacing a node keeps its arrangement. `replace` deliberately drops the
    slot's inputs -- a different node has different inputs -- and dropping the
    group with them would scatter the pipeline every time a node was swapped."""
    _status, before = _request(server, "GET", "/funpack/api/pipeline")
    was = {slot["id"]: slot.get("group") for slot in before["slots"]}

    _status, after = _request(
        server, "POST", "/funpack/api/pipeline",
        {"action": "replace", "slot": "save", "node": "PreviewImage",
         "slots": before["slots"]})
    assert not after["refused"], after["refused"]
    now = {slot["id"]: slot.get("group") for slot in after["slots"]}
    assert now == was


# --- the whole registry, which is the only place the odd shapes live --------
#
# This is the one test file that loads ComfyUI's extras, so it is the only place
# a sweep like this means anything. These were written after a v4 bug report
# about a node "showing only its LoRA selection"; the shapes themselves are
# covered in test_comfy_types.py.

def test_no_installed_node_is_described_with_a_type_nothing_can_edit(registered):
    """A widget's type is what decides which control is drawn. A union like
    "FLOAT,INT" reaching the form asks for a control for a type that does not
    exist, and the whole node then renders as nothing."""
    from core import widgets

    for class_type in registered.NODE_CLASS_MAPPINGS:
        described = widgets.describe(class_type)
        if not described:
            continue
        for widget in described["widgets"]:
            assert "," not in widget["type"], f"{class_type}.{widget['name']} is {widget['type']}"
            assert widget["type"] in widgets.PRIMITIVE, (
                f"{class_type}.{widget['name']} is {widget['type']}, which has no control")


def test_every_dropdown_in_the_registry_is_offered_as_one(registered):
    """A dropdown classified as a socket is an input the user cannot set and the
    window claims is wired.

    The check is spelled out here rather than asked of `comfy_types.is_combo`.
    It was written that way first, and reintroducing the bug did not fail this
    test: the sweep was asking the broken function whether the function was
    broken, and got the answer the bug implies. A test that consults the code
    under test for its own verdict cannot fail.
    """
    from core import comfy_types, widgets

    dropdowns = 0
    for class_type in registered.NODE_CLASS_MAPPINGS:
        described = widgets.describe(class_type)
        if not described:
            continue
        for socket in described["sockets"]:
            assert "COMBO" not in str(socket["type"]).upper(), (
                f"{class_type}.{socket['name']} is a dropdown ({socket['type']}) "
                f"reported as a wire")
        dropdowns += sum(1 for w in described["widgets"]
                         if w["type"] == comfy_types.COMBO)

    assert dropdowns > 100, f"only {dropdowns} dropdowns found; the sweep is not seeing them"


def test_a_file_that_is_no_longer_there_is_refused_here_not_at_the_queue(server, registered):
    """v4's failure, reproduced against real nodes.

    A LoRA file that had been deleted made ComfyUI refuse the WHOLE prompt --
    a feature that was switched off still stopped every generation, and the
    message was "Prompt outputs failed validation" over a graph of a dozen
    loaders. ComfyUI checks this at /prompt, which is one step after our own
    check says the pipeline is queueable.
    """
    _status, body = _request(server, "GET", "/funpack/api/pipeline")
    slots = body["slots"]
    for slot in slots:
        if slot["id"] == "model":
            slot["inputs"]["model_name"] = "deleted-yesterday.safetensors"

    _status, checked = _request(server, "POST", "/funpack/api/pipeline",
                                {"action": "check", "slots": slots})
    assert checked["queueable"] is False
    assert any("deleted-yesterday.safetensors" in problem and "not one of" in problem
               for problem in checked["incomplete"]), checked["incomplete"]
    assert checked["prompt"] is None, "a graph the queue would refuse was handed over"


def test_a_value_the_node_does_offer_still_builds(server, registered):
    """The control. Without it the test above passes on a check that refuses
    everything."""
    from core import widgets

    _status, body = _request(server, "GET", "/funpack/api/pipeline")
    slots = body["slots"]
    described = widgets.describe("FunPackDiffusionModelLoader")
    choices = next(w for w in described["widgets"] if w["name"] == "model_name")["choices"]
    if not choices:
        pytest.skip("no model files on this machine, so there is no valid value to use")

    for slot in slots:
        if slot["id"] == "model":
            slot["inputs"]["model_name"] = choices[0]

    _status, checked = _request(server, "POST", "/funpack/api/pipeline",
                                {"action": "check", "slots": slots})
    assert not any("not one of" in problem for problem in checked["incomplete"]), \
        checked["incomplete"]


def test_what_the_ui_holds_reaches_the_graph(server):
    """The settings a person picked, on the node that carries them.

    Until this existed every panel in the app was decoration: the values were
    kept in the browser and nothing ever sent them, so a modifier could be
    switched on and the run would not know.
    """
    values = {"sampling_alg": {"enabled": True, "strength": 0.4}}
    status, body = _request(server, "POST", "/funpack/api/pipeline", {"values": values})
    assert status == 200, body
    assert body["refused"] == []

    settings = next(s for s in body["slots"] if s["node"] == "FunPackModifierSettings")
    assert json.loads(settings["inputs"]["settings"]) == values
    assert body["notes"] == []


def test_settings_with_nowhere_to_go_are_said_and_not_swallowed(server):
    """A pipeline with nothing to accept them is legitimate -- and has to say so.

    A run that silently ignores every switch in the app is the exact fault this
    project keeps finding: a knob present and inert.
    """
    status, body = _request(server, "GET", "/funpack/api/pipeline")
    kept = [s for s in body["slots"] if s["node"] != "FunPackModifierSettings"]
    for slot in kept:                                # nothing may still link to it
        slot["inputs"] = {k: v for k, v in slot["inputs"].items()
                          if not (isinstance(v, list) and v and v[0] == "settings")}

    status, body = _request(server, "POST", "/funpack/api/pipeline",
                            {"slots": kept, "values": {"sampling_alg": {"enabled": True}}})
    assert status == 200, body
    assert any("will not be applied" in note for note in body["notes"]), body["notes"]
    # Said, not blocked: the pipeline is otherwise exactly as valid as it was.
    assert body["refused"] == []


def test_no_settings_sent_means_the_slot_is_left_alone(server):
    status, body = _request(server, "POST", "/funpack/api/pipeline", {})
    settings = next(s for s in body["slots"] if s["node"] == "FunPackModifierSettings")
    assert settings["inputs"]["settings"] == "{}"
    assert body["notes"] == []


def test_settings_that_are_not_an_object_are_refused(server):
    for payload in (["a"], "everything on", 7):
        status, body = _request(server, "POST", "/funpack/api/pipeline", {"values": payload})
        assert status == 400, (payload, status, body)
        assert body["queueable"] is False
        assert any("object keyed by module id" in p for p in body["problems"]), body


def test_empty_settings_are_not_reported_as_having_nowhere_to_go(server):
    """Nothing set is not a warning. Every page load starts here."""
    status, body = _request(server, "POST", "/funpack/api/pipeline", {"values": {}})
    assert status == 200
    assert body["notes"] == []


def test_a_prompt_typed_in_the_app_reaches_the_node_that_encodes_it(server):
    status, body = _request(server, "POST", "/funpack/api/pipeline",
                            {"inputs": {"positive": {"text": "a cat on a roof"}}})
    assert status == 200, body
    assert body["refused"] == []
    positive = next(s for s in body["slots"] if s["id"] == "positive")
    assert positive["inputs"]["text"] == "a cat on a roof"
    assert positive["inputs"]["clip"] == ["clip", 0]


def test_a_value_for_a_slot_that_is_gone_stops_the_run(server):
    """Not a note: the prompt went nowhere, and a run without it is not the run
    that was asked for."""
    status, body = _request(server, "POST", "/funpack/api/pipeline",
                            {"inputs": {"positive_2": {"text": "a cat"}}})
    assert status == 200
    assert any("no such slot" in p for p in body["refused"]), body["refused"]
    assert body["queueable"] is False
    assert body["prompt"] is None


def test_the_default_pipeline_says_where_its_prompt_belongs(server):
    """The app has no list of what a prompt is: the pipeline says which input
    goes on the surface, and the app offers a place with that name or does not."""
    status, body = _request(server, "GET", "/funpack/api/pipeline")
    assert status == 200
    roles = [(s["id"], r) for s in body["slots"] for r in (s.get("roles") or [])]
    assert roles, "no slot offers anything to the main window"
    places = {r["at"] for _id, r in roles}
    assert places == {"generation.prompt", "project.video", "generation.sampling"}
    assert all(r["input"] and r["label"] for _id, r in roles)
