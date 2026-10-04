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
            if not defn:
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


def _widget_names(schemas, cls):
    return [n for n, t in schemas.inputs(cls).items() if t in comfy_types.PRIMITIVE]


def _values(schemas, node):
    """The node's typed-in values by input name."""
    raw = node.get("widgets_values")
    if isinstance(raw, dict):
        out = {k: v for k, v in raw.items()}
    elif isinstance(raw, list):
        out, i = {}, 0
        for name in _widget_names(schemas, node.get("type")):
            if i >= len(raw):
                break
            out[name] = raw[i]; i += 1
            if i < len(raw) and isinstance(out[name], (int, float)) and not isinstance(out[name], bool) and raw[i] in CONTROL:
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
                setter = next((m for m in nodes.values() if m.get("type") == "SetNode" and (m.get("widgets_values") or [""])[0] == name), None)
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
            src = nodes.get(row[1])
            if src is not None and src.get("type") == "PrimitiveNode":       # a Primitive hands over its value
                vals = src.get("widgets_values") or []
                if vals:
                    s["inputs"][name] = vals[0]
                continue
            hit = origin(row[1], row[2])
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

def _upstream(slots_by_id, start):
    """Slots feeding `start`, nearest first."""
    seen, order, queue = {start}, [], [start]
    while queue:
        cur = queue.pop(0)
        for v in (slots_by_id[cur]["inputs"] or {}).values():
            if is_link(v) and v[0] in slots_by_id and v[0] not in seen:
                seen.add(v[0]); order.append(v[0]); queue.append(v[0])
    return order


def bind(slots: List[dict], schemas) -> Tuple[List[dict], Dict[str, str]]:
    """(slots with roles added, what was bound: control -> 'node #id.input')."""
    by_id = {s["id"]: s for s in slots}
    bound: Dict[str, str] = {}

    def give(control, slot, name, role):
        slot.setdefault("roles", []).append(dict(role, input=name))
        bound[control] = f"{slot['node']} #{slot['id'][1:]}.{name}"

    def text_home(slot_id):
        """The first unlinked text input on the node a sampler's conditioning comes from, or upstream of it."""
        taken = {b.split("#")[-1] for b in bound.values()}
        for sid_ in [slot_id, *_upstream(by_id, slot_id)]:
            s = by_id[sid_]
            for name in TEXT_NAMES:
                if schemas.inputs(s["node"]).get(name) == "STRING" and not is_link(s["inputs"].get(name)) and f"{sid_[1:]}.{name}" not in taken:
                    return s, name
        return None

    for want, key, role in (("positive", "prompt", {"at": "generation.prompt", "label": "Prompt"}),
                            ("negative", "negative", {"at": "project.negative", "label": "Negative prompt"})):
        for s in slots:
            link = (s["inputs"] or {}).get(want)
            found = text_home(link[0]) if is_link(link) and link[0] in by_id else None
            if found:
                give(key, found[0], found[1], role)
                break
    seeds = [(s, n) for s in slots for n in s["inputs"] if "seed" in n.lower() and schemas.inputs(s["node"]).get(n) == "INT" and not is_link(s["inputs"][n])]
    if seeds:
        give("seed", seeds[0][0], seeds[0][1], {"at": "generation.seed", "label": "Seed"})
    for control, names in SIZE_NAMES.items():
        for s in slots:
            name = next((n for n in names if schemas.inputs(s["node"]).get(n) in ("INT", "FLOAT") and not is_link(s["inputs"].get(n))), None)
            if name:
                role = {"at": "project.video", "label": {"width": "Width", "height": "Height", "frames": "Length", "fps": "FPS"}[control]}
                if control in ("frames", "fps"):
                    role["drives"] = control
                give(control, s, name, role)
                break
    for s in slots:                                           # a LoadImage becomes the scene's start picture
        if s["node"] == "LoadImage" and "image" not in bound:
            if any(is_link(v) and v == [s["id"], 1] for o in slots for v in o["inputs"].values()):
                continue                                      # its mask is used: leave it as it is
            s["node"], s["inputs"] = "FunPackLoadMedia", {"media_id": ""}
            give("image", s, "media_id", {"at": "assets.source_image"})
    return slots, bound


def convert(workflow: Any, schemas) -> dict:
    """{"slots", "bound", "notes"} for a workflow in either export format."""
    notes: List[str] = []
    if is_api(workflow):
        slots = _from_api(workflow, notes)
    elif isinstance(workflow, dict) and isinstance(workflow.get("nodes"), list):
        slots = _from_ui(workflow, schemas, notes)
    else:
        raise ValueError("this is not a ComfyUI workflow: expected the UI export (nodes and links) or the API export")
    if not slots:
        raise ValueError("the workflow has no nodes that run")
    missing = sorted({s["node"] for s in slots if not schemas.of(s["node"])})
    if missing:
        notes.append("Not installed here: " + ", ".join(missing) + ". Those nodes are kept but cannot run until their pack is installed.")
    slots, bound = bind(slots, schemas)
    return {"slots": slots, "bound": bound, "notes": notes}
