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


def from_comfyui() -> Schemas:
    """Schemas read from whatever ComfyUI has registered, FunPack's and everyone
    else's alike -- a slot may point at any node, not only ours."""
    def lookup(class_type: str) -> Optional[dict]:
        import nodes as comfy_nodes
        node = comfy_nodes.NODE_CLASS_MAPPINGS.get(class_type)
        if node is None:
            return None
        spec = node.INPUT_TYPES()
        inputs = {}
        for section in ("required", "optional"):
            for name, declared in (spec.get(section) or {}).items():
                kind = declared[0] if isinstance(declared, (list, tuple)) else declared
                # A list of choices IS the type for a combo; what matters here is
                # only that it is not a socket, so it cannot be wired.
                inputs[name] = kind if isinstance(kind, str) else "COMBO"
        return {"inputs": inputs, "outputs": list(node.RETURN_TYPES)}
    return Schemas(lookup)


def build(slots: Sequence[dict], schemas: Optional[Schemas] = None) -> Tuple[dict, List[str]]:
    """(prompt, problems) -- the graph ComfyUI queues, and why it should not.

    Nothing is queued while `problems` is non-empty, so a wrong graph is refused
    here rather than surfacing as a traceback from inside sampling.
    """
    schemas = schemas or from_comfyui()
    known = {slot["id"] for slot in slots}
    problems: List[str] = []
    prompt: Dict[str, dict] = {}

    for slot in slots:
        slot_id, class_type = slot["id"], slot["node"]
        if schemas.of(class_type) is None:
            problems.append(f"{slot_id}: there is no node called {class_type!r} installed")
            continue

        declared = schemas.inputs(class_type)
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
                produced = schemas.outputs(slots_by_id(slots)[source]["node"])
                if index >= len(produced):
                    problems.append(f"{slot_id}.{name} reads output {index} of {source!r}, "
                                    f"which has {len(produced)}")
                    continue
                if produced[index] != declared[name] and declared[name] != "*":
                    problems.append(f"{slot_id}.{name} wants {declared[name]} but {source!r} "
                                    f"gives {produced[index]}")
                    continue
                inputs[name] = [source, index]
            else:
                inputs[name] = value
        prompt[slot_id] = {"class_type": class_type, "inputs": inputs}

    return prompt, problems


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
        elif wanted is not None and now[index] != wanted:
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
                    if wants == kind and name in wired and is_link(wired[name])]
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
