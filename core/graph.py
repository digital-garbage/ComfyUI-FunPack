"""The pipeline, as slots rather than as a core.

The pipeline is DATA: a list of slots, each naming a node and what feeds it.
Nothing here is a fixed chain, and that is the whole point.

v4 could not remove a built-in node from the pipeline, and the reason was
structural rather than an oversight: the wiring lived in code as a `CORE` dict
the user could not reach, so there was no entry to delete. Anything that only
exists as code cannot be replaced by something that is not that code.

So three operations, and each has to hold for a foundation nobody can break:

* **build** -- turn slots into the API prompt ComfyUI queues.
* **replace** -- point a slot at a different node. Allowed when the replacement
  still produces what its consumers read; refused, with the reason, when it does
  not. Anything producing the same result is as good as the built-in one.
* **remove** -- take a slot out and rewire what it fed to what fed it. Allowed
  when the removed node has exactly one input matching each output somebody
  consumes; refused otherwise, naming the ambiguity.

Refused and not silently skipped. v4 shipped a bypass that reported a problem
into a field nothing rendered, so toggling it did nothing and said nothing --
the user's word for it was "terrifying".
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import comfy_types

Link = list          # [slot_id, output_index]


def is_link(value: Any) -> bool:
    """A reference to another slot's output, as ComfyUI spells it."""
    return (isinstance(value, (list, tuple)) and len(value) == 2
            and isinstance(value[0], str) and isinstance(value[1], int))


class Schemas:
    """What core is allowed to know about a node: its input and output types.

    Injected rather than imported so the whole of this file is testable without
    ComfyUI, and so nothing here depends on how node classes happen to be built.
    """

    def __init__(self, lookup: Callable[[str], Optional[dict]]):
        self._lookup = lookup

    def of(self, class_type: str) -> Optional[dict]:
        return self._lookup(class_type)

    def outputs(self, class_type: str) -> List[str]:
        found = self.of(class_type)
        return list(found["outputs"]) if found else []

    def inputs(self, class_type: str) -> Dict[str, str]:
        found = self.of(class_type)
        return dict(found["inputs"]) if found else {}

    def required(self, class_type: str) -> List[str]:
        """Inputs that must be filled. Absent from a schema means none are."""
        found = self.of(class_type)
        return list(found.get("required", [])) if found else []

    def limits(self, class_type: str) -> Dict[str, dict]:
        """What each input will ACCEPT: a combo's choices, a number's bounds.

        Absent from a schema means unknown, which is checked as no limit --
        a schema that does not say is not a schema that says no.
        """
        found = self.of(class_type)
        return dict(found.get("limits", {})) if found else {}


def from_comfyui() -> Schemas:
    """Schemas read from whatever ComfyUI has registered, FunPack's and everyone
    else's alike -- a slot may point at any node, not only ours."""
    def lookup(class_type: str) -> Optional[dict]:
        import nodes as comfy_nodes
        node = comfy_nodes.NODE_CLASS_MAPPINGS.get(class_type)
        if node is None:
            return None
        spec = node.INPUT_TYPES()
        inputs, required, limits = {}, [], {}
        for section in ("required", "optional"):
            for name, declared in (spec.get(section) or {}).items():
                kind, options = comfy_types.declared(declared)
                # The type as ComfyUI would WIRE it. A list of choices is a
                # combo; so is the string "COMBO" and every V3 dynamic combo,
                # and none of them can be fed by a wire.
                edited = comfy_types.widget_type(kind, options)
                inputs[name] = edited if edited else kind
                if section == "required":
                    # Kept, because flattening the two sections loses the only
                    # thing that says a slot is incomplete -- and a slot missing
                    # a required input built clean and failed once queued.
                    required.append(name)

                bounds = {}
                if inputs[name] == comfy_types.COMBO:
                    bounds["choices"] = (list(kind) if isinstance(kind, (list, tuple))
                                         else comfy_types.choices(options))
                for edge in ("min", "max"):
                    if edge in options:
                        bounds[edge] = options[edge]
                if bounds:
                    limits[name] = bounds

        return {"inputs": inputs, "outputs": list(node.RETURN_TYPES),
                "required": required, "limits": limits}
    return Schemas(lookup)


def shape_problems(slots: Any) -> List[str]:
    """Whether this is even a pipeline, before anything indexes into it.

    A payload arrives from HTTP, so "it parsed as JSON" is all that has been
    established. Reading slot["id"] off a string raised a TypeError and the
    route answered 500 in plain text -- which the app cannot turn into a reason
    to show, and a refusal nobody can read is the failure this file exists to
    avoid.

    Separate from build() so a caller can refuse a malformed REQUEST before
    reporting on an incomplete PIPELINE: those are different things and an app
    showing them together says the wrong thing about both.
    """
    if not isinstance(slots, (list, tuple)):
        return [f"a pipeline is a list of slots, not {type(slots).__name__}"]

    problems: List[str] = []
    for index, slot in enumerate(slots):
        where = f"slot {index}"
        if not isinstance(slot, dict):
            problems.append(f"{where} is {type(slot).__name__}, not an object")
            continue
        for key in ("id", "node"):
            value = slot.get(key)
            if not isinstance(value, str) or not value.strip():
                problems.append(f"{where} has no {key}")
        inputs = slot.get("inputs")
        if inputs is not None and not isinstance(inputs, dict):
            problems.append(f"{where}'s inputs must be an object, not {type(inputs).__name__}")
        # Core does not know what a group MEANS -- it is the pipeline's own way
        # of arranging itself, and any name is a valid one. It does know a name
        # is text: a number here becomes a card titled `5`, and two slots
        # grouped under `5` and `"5"` land in two cards reading the same.
        group = slot.get("group")
        if group is not None:
            if not isinstance(group, str):
                problems.append(f"{where}'s group must be a name, not {type(group).__name__}")
            elif not group.strip():
                problems.append(f"{where}'s group is blank; leave it out to leave the "
                                f"slot ungrouped")
    return problems


def build(slots: Sequence[dict], schemas: Optional[Schemas] = None) -> Tuple[dict, List[str]]:
    """(prompt, problems) -- the graph ComfyUI queues, and why it should not.

    Nothing is queued while `problems` is non-empty, so a wrong graph is refused
    here rather than surfacing as a traceback from inside sampling.
    """
    schemas = schemas or from_comfyui()
    problems: List[str] = []
    prompt: Dict[str, dict] = {}

    # The shape first, because everything below indexes into a slot.
    problems = shape_problems(slots)
    if problems:
        return {}, problems

    # A set of ids cannot see a duplicate, and writing the prompt by id would
    # then silently overwrite one slot with the other -- losing a node, or
    # leaving one reading its own output. Refused, because a graph quietly
    # missing a node is the shape of fault this whole file exists to prevent.
    seen = set()
    for slot in slots:
        if slot["id"] in seen:
            problems.append(f"{slot['id']!r} is used by more than one slot; ids are how "
                            f"a link names what feeds it, so they have to be unique")
        seen.add(slot["id"])
    if problems:
        return {}, problems

    known = seen
    # Built once. Rebuilding it per link made validating a chain of N slots do N
    # dict rebuilds, and this runs on the event loop: the cost of checking a
    # pipeline is paid by every other request the server is holding.
    by_id = slots_by_id(slots)

    for slot in slots:
        slot_id, class_type = slot["id"], slot["node"]
        if schemas.of(class_type) is None:
            problems.append(f"{slot_id}: there is no node called {class_type!r} installed")
            continue

        declared = schemas.inputs(class_type)
        limits = schemas.limits(class_type)
        inputs = {}
        for name, value in (slot.get("inputs") or {}).items():
            if name not in declared:
                problems.append(f"{slot_id}: {class_type} has no input {name!r}")
                continue
            if is_link(value):
                source, index = value
                if source not in known:
                    problems.append(f"{slot_id}.{name} is fed by {source!r}, which is not in the pipeline")
                    continue
                produced = schemas.outputs(by_id[source]["node"])
                if index >= len(produced):
                    problems.append(f"{slot_id}.{name} reads output {index} of {source!r}, "
                                    f"which has {len(produced)}")
                    continue
                # A union on either side, not an exact string. `"IMAGE,MASK"`
                # takes either, and comparing the whole string refused a MASK
                # feeding it -- a legal wire refused, which is as bad a failure
                # as an illegal one accepted, and harder to argue with.
                if not comfy_types.accepts(declared[name], produced[index]):
                    problems.append(f"{slot_id}.{name} wants {declared[name]} but {source!r} "
                                    f"gives {produced[index]}")
                    continue
                inputs[name] = [source, index]
            else:
                problem = unacceptable(name, value, limits.get(name))
                if problem:
                    problems.append(f"{slot_id}.{problem}")
                    continue
                inputs[name] = value
        for name in schemas.required(class_type):
            if name not in inputs:
                problems.append(f"{slot_id}: {class_type} needs {name!r} and nothing "
                                f"fills it")

        prompt[slot_id] = {"class_type": class_type, "inputs": inputs}

    problems.extend(cycles(prompt))
    return prompt, problems


def unacceptable(name: str, value: Any, bounds: Optional[dict]) -> Optional[str]:
    """Why this literal will not do, or None.

    ComfyUI checks these too, at /prompt -- and it refuses the WHOLE graph when
    one value is wrong, which v4 hit with a LoRA file that had been deleted: a
    feature that was switched off still stopped every generation. Refusing here
    means the reason arrives naming the slot and the input, before anything is
    queued, instead of as "Prompt outputs failed validation".

    No bounds means unknown, and unknown is not a refusal.
    """
    if not bounds:
        return None

    choices = bounds.get("choices")
    if choices is not None:
        # An empty list is a machine with no files of that kind, not a value
        # that was checked and rejected -- the slot being unfilled is what gets
        # reported, and saying "x is not one of: " helps nobody.
        if choices and value not in choices:
            offered = ", ".join(repr(c) for c in choices[:6])
            more = "" if len(choices) <= 6 else f", and {len(choices) - 6} more"
            return (f"{name} is {value!r}, which is not one of: {offered}{more}")
        return None

    # Only a number has bounds, and only a number is compared to them. A bool is
    # an int in Python and would sail through a naive comparison as 0 or 1.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if "min" in bounds and value < bounds["min"]:
        return f"{name} is {value}, below the smallest {bounds['min']} it takes"
    if "max" in bounds and value > bounds["max"]:
        return f"{name} is {value}, above the largest {bounds['max']} it takes"
    return None


def place(slots: Sequence[dict], value: Any,
          sinks: Sequence[dict]) -> Tuple[List[dict], int]:
    """Write one value into every slot a sink names. Returns (slots, how many).

    This is how what the UI holds reaches the graph, and it is deliberately
    ignorant of what the value MEANS: a module says "a node of this class takes
    it, under this input name" and core does exactly that. Core naming the node
    itself would make one module's node privileged, which is the thing the whole
    announcement contract exists to prevent.

    The count is returned rather than a message, because only the caller knows
    what the value was -- and a value with nowhere to go has to be SAID. Writing
    it nowhere and reporting success is how a settings panel becomes decoration.
    """
    out = [dict(slot, inputs=dict(slot.get("inputs") or {})) for slot in slots]
    wanted = [(sink.get("node"), sink.get("input")) for sink in sinks
              if isinstance(sink, dict) and sink.get("node") and sink.get("input")]

    placed = 0
    for slot in out:
        for class_type, name in wanted:
            if slot.get("node") == class_type:
                slot["inputs"][name] = value
                placed += 1
    return out, placed


def cycles(prompt: Dict[str, dict]) -> List[str]:
    """Slots that feed each other, named in the order the loop runs.

    Every other refusal here is local -- a type, a name, an index -- and a cycle
    is the one wrong graph that is correct at every link and impossible as a
    whole. ComfyUI does catch it, but at execution: the run is queued, accepted,
    and dies inside the executor with a message the app has no reason to be
    reading, which is the "refused somewhere nobody looks" shape this file
    exists to prevent.

    Walked with an explicit stack rather than recursion. Slots arrive from a
    request, so how many there are is the caller's choice, and a long enough
    chain would turn a refusal into a RecursionError -- a 500 instead of a
    reason.
    """
    feeds = {slot_id: [value[0] for value in node["inputs"].values() if is_link(value)]
             for slot_id, node in prompt.items()}
    cleared: set = set()
    reported: set = set()
    found: List[str] = []

    for root in prompt:
        if root in cleared:
            continue
        stack: List[List] = [[root, 0]]     # [slot, index of its next feed]
        path: List[str] = [root]            # what we are currently inside
        on_path = {root}
        while stack:
            slot_id, i = stack[-1]
            sources = feeds.get(slot_id, ())
            if i >= len(sources):
                cleared.add(slot_id)
                on_path.discard(slot_id)
                path.pop()
                stack.pop()
                continue
            stack[-1][1] = i + 1
            source = sources[i]
            if source in on_path:
                loop = path[path.index(source):]
                # One loop, reported once, however many links close it.
                key = frozenset(loop)
                if key not in reported:
                    reported.add(key)
                    found.append(f"{' feeds '.join(reversed(loop))} feeds {loop[-1]}: "
                                 f"a slot cannot end up feeding itself")
            elif source not in cleared:
                stack.append([source, 0])
                path.append(source)
                on_path.add(source)
    return found


def slots_by_id(slots: Sequence[dict]) -> Dict[str, dict]:
    return {slot["id"]: slot for slot in slots}


def consumers(slots: Sequence[dict], slot_id: str) -> List[Tuple[str, str, int]]:
    """Every (consumer_id, input_name, output_index) reading this slot."""
    found = []
    for slot in slots:
        for name, value in (slot.get("inputs") or {}).items():
            if is_link(value) and value[0] == slot_id:
                found.append((slot["id"], name, value[1]))
    return found


def replace(slots: Sequence[dict], slot_id: str, class_type: str,
            schemas: Optional[Schemas] = None) -> Tuple[List[dict], List[str]]:
    """Point a slot at a different node.

    Allowed when the replacement still produces what its consumers read. The
    built-in nodes have no special standing: anything producing the same result
    is as good, which is what makes them replaceable rather than merely present.
    """
    schemas = schemas or from_comfyui()
    by_id = slots_by_id(slots)
    if slot_id not in by_id:
        return list(slots), [f"there is no slot called {slot_id!r}"]
    if schemas.of(class_type) is None:
        return list(slots), [f"there is no node called {class_type!r} installed"]

    was = schemas.outputs(by_id[slot_id]["node"])
    now = schemas.outputs(class_type)
    problems = []
    for consumer, name, index in consumers(slots, slot_id):
        wanted = was[index] if index < len(was) else None
        if index >= len(now):
            problems.append(f"{consumer}.{name} reads output {index}, which {class_type} "
                            f"does not have")
        elif wanted is not None and not comfy_types.accepts(wanted, now[index]):
            problems.append(f"{consumer}.{name} needs {wanted} but {class_type} gives "
                            f"{now[index]} there")
    if problems:
        return list(slots), problems

    # Inputs are NOT carried across: a different node has different inputs, and
    # silently keeping the ones whose names happen to match is how a value ends
    # up meaning something else.
    changed = [dict(slot, node=class_type, inputs={}) if slot["id"] == slot_id else dict(slot)
               for slot in slots]
    return changed, []


def remove(slots: Sequence[dict], slot_id: str,
           schemas: Optional[Schemas] = None) -> Tuple[List[dict], List[str]]:
    """Take a slot out, and rewire what it fed to what fed it.

    The passthrough rule is ComfyUI's own for a bypassed node: for each output
    somebody actually consumes, the removed node must have exactly one input of
    that type to inherit from. Only CONSUMED outputs are checked -- v4 required
    one for every output a class could theoretically produce, and a node with a
    spare FLOAT output nobody wired became unremovable.
    """
    schemas = schemas or from_comfyui()
    by_id = slots_by_id(slots)
    if slot_id not in by_id:
        return list(slots), [f"there is no slot called {slot_id!r}"]

    going = by_id[slot_id]
    produced = schemas.outputs(going["node"])
    declared = schemas.inputs(going["node"])
    wired = going.get("inputs") or {}

    replacement: Dict[int, Any] = {}
    problems: List[str] = []

    for index in sorted({index for _c, _n, index in consumers(slots, slot_id)}):
        if index >= len(produced):
            problems.append(f"something reads output {index} of {slot_id!r}, which does not exist")
            continue
        kind = produced[index]
        matching = [name for name, wants in declared.items()
                    if comfy_types.accepts(wants, kind)
                    and name in wired and is_link(wired[name])]
        if len(matching) == 1:
            replacement[index] = wired[matching[0]]
        elif not matching:
            problems.append(f"removing {slot_id!r} would leave its {kind} output unfed: it has "
                            f"no wired {kind} input for whatever reads it to inherit")
        else:
            problems.append(f"removing {slot_id!r} is ambiguous: {kind} could come from "
                            f"{', '.join(sorted(matching))}")

    if problems:
        return list(slots), problems

    changed = []
    for slot in slots:
        if slot["id"] == slot_id:
            continue
        inputs = dict(slot.get("inputs") or {})
        for name, value in list(inputs.items()):
            if is_link(value) and value[0] == slot_id:
                inputs[name] = replacement[value[1]]
        changed.append(dict(slot, inputs=inputs))
    return changed, []
