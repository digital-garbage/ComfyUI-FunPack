"""A ComfyUI workflow file, as pipeline slots.

Reads both exports: the API format ({id: {class_type, inputs}}) and the UI format (nodes + links). The UI
format is first made plain: subgraphs are opened, Set/Get pairs and Reroutes are stepped over, Primitive nodes
hand their value to what they feed, bypassed nodes pass their input through, muted ones are dropped.

Then `bind` gives the app's own controls (prompt, seed, size, length, FPS, start picture) a home on the nodes
that look like they take them. It guesses from names and wiring; what it did is returned so the person can see
it, and anything it could not place is simply not bound -- never bound to a wrong node quietly.

Pure data in, data out. `schemas` is core.graph.Schemas (what each node's inputs and outputs are).
"""

import copy
from typing import Any, Dict, List, Tuple

from . import comfy_types
from .graph import is_link

SKIP = {"Note", "MarkdownNote"}
VIRTUAL = {"SetNode", "GetNode"}
PASS = {"Reroute"}
CONTROL = {"fixed", "increment", "decrement", "randomize"}     # the value the UI stores after a seed widget
SG_IN, SG_OUT = -10, -20                                        # a subgraph definition's own boundary nodes
MAX_DEPTH = 8

TEXT_NAMES = ("text", "prompt", "text_g", "string", "value")
SIZE_NAMES = {"width": ("width",), "height": ("height",),
              "frames": ("length", "num_frames", "frame_count", "frames", "video_frames"),
              "fps": ("frame_rate", "fps")}


def is_api(wf: Any) -> bool:
    return isinstance(wf, dict) and not isinstance(wf.get("nodes"), list) and any(
        isinstance(v, dict) and "class_type" in v for v in wf.values())


def sid(node_id: Any) -> str:
    return f"w{node_id}"


# ── UI format, made plain ────────────────────────────────────────────────────

def _rows(wf):
    return [r for r in (wf.get("links") or []) if isinstance(r, list) and len(r) >= 6]


def _group_of(node, groups):
    """Title of the smallest group whose box holds the node's centre, or ""."""
    pos, size = node.get("pos") or [], node.get("size") or []
    if len(pos) < 2:
        return ""
    try:
        cx = float(pos[0]) + float(size[0] if size else 0) / 2
        cy = float(pos[1]) + float(size[1] if len(size) > 1 else 0) / 2
        best = None
        for g in groups or []:
            x, y, w, h = (float(v) for v in (g.get("bounding") or [])[:4])
            if x <= cx <= x + w and y <= cy <= y + h and (best is None or w * h < best[0]):
                best = (w * h, str(g.get("title") or ""))
        return best[1] if best else ""
    except (TypeError, ValueError):
        return ""


def flatten_subgraphs(wf: dict, schemas) -> dict:
    """Every subgraph instance replaced by the nodes inside it, rewired; same shape back."""
    defs = {str(d.get("id")): d for d in ((wf.get("definitions") or {}).get("subgraphs") or []) if isinstance(d, dict) and d.get("id")}
    if not defs:
        return wf
    wf = copy.deepcopy(wf)
    counter = [max([int(r[0]) for r in _rows(wf)] or [0]) + 1]

    def promoted(defn):
        """Which of the definition's inputs stand for an inner widget (what an instance's widgets_values lists)."""
        by_id = {n.get("id"): n for n in defn.get("nodes") or []}
        found = []
        for i in range(len(defn.get("inputs") or [])):
            for link in defn.get("links") or []:
                if link.get("origin_id") == SG_IN and link.get("origin_slot") == i:
                    sockets = (by_id.get(link.get("target_id")) or {}).get("inputs") or []
                    ts = link.get("target_slot")
                    if ts is not None and ts < len(sockets) and sockets[ts].get("widget"):
                        found.append(i)
                    break
        return found

    def expand(nodes, rows, depth):
        if depth > MAX_DEPTH:
            return nodes, rows
        out_nodes, out_rows, again = [], list(rows), False
        for inst in nodes:
            defn = defs.get(str(inst.get("type")))
            if not defn or inst.get("mode") in (2, 4):          # a bypassed or muted instance stays one node: origin() passes it through or drops it
                out_nodes.append(inst)
                continue
            again = True
            pfx = f"sg{inst.get('id')}_"
            inner = copy.deepcopy(defn.get("nodes") or [])
            by_old = {n.get("id"): n for n in inner}
            title = str(inst.get("title") or defn.get("name") or "Subgraph")
            for n in inner:
                n["id"] = f"{pfx}{n.get('id')}"
                n.setdefault("_group", _group_of(n, defn.get("groups")) or title)
            # the instance's own widget values beat the inner node's stale copy
            values = inst.get("widgets_values")
            if isinstance(values, list):
                for idx, val in zip(promoted(defn), values):
                    for link in defn.get("links") or []:
                        if link.get("origin_id") == SG_IN and link.get("origin_slot") == idx:
                            tgt = by_old.get(link.get("target_id"))
                            sockets = (tgt or {}).get("inputs") or []
                            ts = link.get("target_slot")
                            name = ((sockets[ts].get("widget") or {}).get("name")) if tgt and ts is not None and ts < len(sockets) else None
                            if name:
                                tgt.setdefault("_values", {})[name] = val
            feeds = {i.get("name"): i.get("link") for i in inst.get("inputs") or []}
            by_row = {int(r[0]): r for r in out_rows}
            names = [i.get("name") for i in defn.get("inputs") or []]
            remap, out_src = {}, {}
            for link in defn.get("links") or []:
                src, ss, dst, ds = link.get("origin_id"), link.get("origin_slot"), link.get("target_id"), link.get("target_slot")
                if dst == SG_OUT:
                    out_src[int(ds or 0)] = (f"{pfx}{src}", int(ss or 0))
                    remap[link.get("id")] = None
                    continue
                if src == SG_IN:
                    feed = by_row.get(int(feeds.get(names[ss] if ss is not None and ss < len(names) else None) or -1))
                    if not feed:
                        remap[link.get("id")] = None
                        continue
                    src, ss = feed[1], feed[2]
                else:
                    src = f"{pfx}{src}"
                n = counter[0]; counter[0] += 1
                remap[link.get("id")] = n
                out_rows.append([n, src, int(ss or 0), f"{pfx}{dst}", int(ds or 0), link.get("type")])
            for n in inner:
                for sock in n.get("inputs") or []:
                    if sock.get("link") is not None:
                        sock["link"] = remap.get(sock["link"])
            for row in out_rows:
                if row[1] == inst.get("id") and int(row[2] or 0) in out_src:
                    row[1], row[2] = out_src[int(row[2] or 0)]
            out_nodes.extend(inner)
        return expand(out_nodes, out_rows, depth + 1) if again else (out_nodes, out_rows)

    wf["nodes"], wf["links"] = expand(wf.get("nodes") or [], _rows(wf), 0)
    live = {n.get("id") for n in wf["nodes"]}
    wf["links"] = [r for r in wf["links"] if r[1] in live and r[3] in live]
    return wf


def _scope(node_id):
    """The subgraph instance a flattened node came from ("sg2_7" -> "sg2"), "" for the top level."""
    text = str(node_id)
    return text.rsplit("_", 1)[0] if text.startswith("sg") and "_" in text else ""


def _widgets(schemas, node):
    """(name, type) of the node's widgets, in the order widgets_values lists them. The export's own sockets say
    which inputs are widgets (they carry a "widget" key); the node's schema is the fallback for older exports."""
    types = schemas.inputs(node.get("type"))
    own = [i.get("name") for i in node.get("inputs") or [] if isinstance(i, dict) and i.get("widget")]
    if own:
        return [(n, types.get(n)) for n in own]
    return [(n, t) for n, t in types.items() if t in comfy_types.PRIMITIVE]


def _values(schemas, node):
    """The node's typed-in values by input name."""
    raw = node.get("widgets_values")
    if isinstance(raw, dict):
        out = {k: v for k, v in raw.items()}
    elif isinstance(raw, list):
        out, i = {}, 0
        for name, kind in _widgets(schemas, node):
            if i >= len(raw):
                break
            out[name] = raw[i]; i += 1
            if i < len(raw) and kind in ("INT", "FLOAT") and raw[i] in CONTROL:
                i += 1                                       # control_after_generate rides after a number
    else:
        out = {}
    out.update(node.get("_values") or {})
    return out


def _from_ui(wf, schemas, notes):
    wf = flatten_subgraphs(wf, schemas)
    nodes = {n.get("id"): n for n in wf.get("nodes") or []}
    rows = _rows(wf)
    into = {(r[3], r[4]): r for r in rows}                   # (node, input index) -> link row
    by_in_name = {}
    for r in rows:
        node = nodes.get(r[3]) or {}
        socks = node.get("inputs") or []
        if 0 <= r[4] < len(socks):
            by_in_name[(r[3], socks[r[4]].get("name"))] = r

    def origin(nid, slot, seen=()):
        """Follow Reroutes, Set/Get pairs and bypassed nodes to the node that really makes this output."""
        for _ in range(64):
            n = nodes.get(nid)
            if n is None or (nid, slot) in seen:
                return None
            kind, mode = n.get("type"), n.get("mode", 0)
            if mode == 2:
                return None                                  # muted: nothing comes out of it
            if kind in PASS or kind == "SetNode" or (mode == 4 and kind not in VIRTUAL):
                socks = n.get("inputs") or []
                pick = 0
                if mode == 4 and kind not in PASS:           # bypass: the input of the same type as this output
                    outs = n.get("outputs") or []
                    want = (outs[slot] or {}).get("type") if slot < len(outs) else None
                    pick = next((i for i, s in enumerate(socks) if s.get("type") == want and s.get("link") is not None), None)
                    if pick is None:
                        return None
                row = next((r for r in rows if r[3] == nid and r[4] == pick), None)
                if row is None:
                    return None
                seen += ((nid, slot),)
                nid, slot = row[1], row[2]
            elif kind == "GetNode":
                name = (n.get("widgets_values") or [""])[0]
                scope = _scope(nid)           # a Set inside the same subgraph instance wins over one elsewhere
                setters = [m for m in nodes.values() if m.get("type") == "SetNode" and m.get("mode", 0) != 2 and (m.get("widgets_values") or [""])[0] == name]
                setter = next((m for m in setters if _scope(m.get("id")) == scope), setters[0] if setters else None)
                if setter is None:
                    return None
                row = next((r for r in rows if r[3] == setter.get("id") and r[4] == 0), None)
                if row is None:
                    return None
                seen += ((nid, slot),)
                nid, slot = row[1], row[2]
            else:
                return (nid, slot) if mode != 2 else None
        return None

    groups = wf.get("groups") or []
    slots, node_of = [], {}
    for nid, n in nodes.items():
        cls = n.get("type")
        if cls in SKIP or cls in PASS or cls in VIRTUAL or cls == "PrimitiveNode" or n.get("mode") in (2, 4):
            continue
        inputs = _values(schemas, n)
        slots.append({"id": sid(nid), "group": n.get("_group") or _group_of(n, groups) or "Workflow", "node": cls, "inputs": inputs})
        node_of[sid(nid)] = n
    keep = {s["id"]: s for s in slots}
    for s in slots:                                          # links
        node = node_of[s["id"]]
        for i, sock in enumerate(node.get("inputs") or []):
            name = sock.get("name")
            row = into.get((node.get("id"), i))
            if row is None:
                continue
            hit = origin(row[1], row[2])
            src = nodes.get(hit[0]) if hit else None
            if src is not None and src.get("type") == "PrimitiveNode":       # a Primitive hands over its value, even through a Reroute or Set/Get
                vals = src.get("widgets_values") or []
                if vals:
                    s["inputs"][name] = vals[0]
                continue
            if hit and sid(hit[0]) in keep:
                s["inputs"][name] = [sid(hit[0]), hit[1]]
            else:
                s["inputs"].pop(name, None)
                notes.append(f"{s['node']} #{node.get('id')}: its “{name}” input comes from a node that is muted, missing or unresolved, so it is left unconnected.")
    return slots


def _from_api(wf, notes):
    slots = []
    for nid, n in wf.items():
        if not isinstance(n, dict) or "class_type" not in n:
            continue
        inputs = {k: ([sid(v[0]), v[1]] if is_link(v) else v) for k, v in (n.get("inputs") or {}).items()}
        slots.append({"id": sid(nid), "group": "Workflow", "node": n["class_type"], "inputs": inputs})
    return slots


# ── giving the app's controls a home ─────────────────────────────────────────

def _upstream(slots_by_id, start, prefer=None):
    """Slots feeding `start`, nearest first. Through a node that takes both a positive and a negative, only
    the side named `prefer` is followed, so one prompt is never read off the other's encoder."""
    seen, order, queue = {start}, [], [start]
    while queue:
        cur = queue.pop(0)
        ins = slots_by_id[cur]["inputs"] or {}
        if prefer and "positive" in ins and "negative" in ins and prefer in ins:
            ins = {prefer: ins[prefer]}
        for v in ins.values():
            if is_link(v) and v[0] in slots_by_id and v[0] not in seen:
                seen.add(v[0]); order.append(v[0]); queue.append(v[0])
    return order


def bind(slots: List[dict], schemas, notes: List[str] = None) -> Tuple[List[dict], Dict[str, List[str]]]:
    """(slots with roles added, what was bound: control -> ['node #id.input', ...]).

    A control goes to EVERY node of the kind that first takes it (two samplers both get the seed), but only when
    that kind's input is typed in: if the obvious node takes it from another node, the control is left unbound and
    a note says so, rather than landing on some other node that happens to share the input name."""
    notes = notes if notes is not None else []
    by_id = {s["id"]: s for s in slots}
    bound: Dict[str, List[str]] = {}

    def give(control, slot, name, role):
        slot.setdefault("roles", []).append(dict(role, input=name))
        bound.setdefault(control, []).append(f"{slot['node']} #{slot['id'][1:]}.{name}")

    def text_home(slot_id, prefer):
        taken = {x for v in bound.values() for x in v}
        for sid_ in [slot_id, *_upstream(by_id, slot_id, prefer)]:
            s = by_id[sid_]
            for name in TEXT_NAMES:
                if schemas.inputs(s["node"]).get(name) == "STRING" and not is_link(s["inputs"].get(name)) and f"{s['node']} #{sid_[1:]}.{name}" not in taken:
                    return s, name
        return None

    for want, key, role in (("positive", "prompt", {"at": "generation.prompt", "label": "Prompt"}),
                            ("negative", "negative", {"at": "project.negative", "label": "Negative prompt"})):
        for s in slots:
            link = (s["inputs"] or {}).get(want)
            found = text_home(link[0], want) if is_link(link) and link[0] in by_id else None
            if found:
                give(key, found[0], found[1], role)

    def numeric(control, label, names, kinds, role, match):
        """Bind `control` to every unlinked input of the first matching node class; say when that class takes it from a wire."""
        first = next(((s, n) for s in slots for n in s["inputs"] if match(n, names) and schemas.inputs(s["node"]).get(n) in kinds), None)
        if first is None:
            return
        cls = first[0]["node"]
        mine = [(s, n) for s in slots if s["node"] == cls for n in s["inputs"] if match(n, names) and schemas.inputs(cls).get(n) in kinds]
        free = [(s, n) for s, n in mine if not is_link(s["inputs"][n])]
        for s, n in free:
            give(control, s, n, dict(role, **({"drives": control} if control in ("frames", "fps") else {})))
        if not free:
            notes.append(f"{label}: the nodes that take it ({cls}) get it from other nodes, so the app's {label.lower()} control is not connected. Set it where it comes from.")

    numeric("seed", "Seed", None, ("INT",), {"at": "generation.seed", "label": "Seed"}, lambda n, _: "seed" in n.lower())
    for control, names in SIZE_NAMES.items():
        label = {"width": "Width", "height": "Height", "frames": "Length", "fps": "FPS"}[control]
        numeric(control, label, names, ("INT", "FLOAT"), {"at": "project.video", "label": label}, lambda n, ns: n in ns)
    for s in slots:                                           # a LoadImage becomes the scene's start picture
        if s["node"] == "LoadImage" and "image" not in bound:
            if any(is_link(v) and v == [s["id"], 1] for o in slots for v in o["inputs"].values()):
                continue                                      # its mask is used: leave it as it is
            notes.append(f"LoadImage #{s['id'][1:]} now takes the scene's start picture; the file it named is not kept. With no picture on the scene it cannot run.")
            s["node"], s["inputs"] = "FunPackLoadMedia", {"media_id": ""}
            give("image", s, "media_id", {"at": "assets.source_image"})
    return slots, bound


def convert(workflow: Any, schemas) -> dict:
    """{"slots", "bound", "notes"} for a workflow in either export format."""
    notes: List[str] = []
    api = is_api(workflow)
    if api:
        slots = _from_api(workflow, notes)
    elif isinstance(workflow, dict) and isinstance(workflow.get("nodes"), list):
        slots = _from_ui(workflow, schemas, notes)
    else:
        raise ValueError("this is not a ComfyUI workflow: expected the UI export (nodes and links) or the API export")
    if not slots:
        raise ValueError("the workflow has no nodes that run")
    missing = sorted({s["node"] for s in slots if not schemas.of(s["node"])})
    if missing:
        notes.append("Not installed here: " + ", ".join(missing) + ". Those nodes are kept but cannot run until their pack is installed" + ("" if api else "; their typed-in values could not be read from this export") + ".")
    slots, bound = bind(slots, schemas, notes)
    return {"slots": slots, "bound": bound, "notes": notes}
