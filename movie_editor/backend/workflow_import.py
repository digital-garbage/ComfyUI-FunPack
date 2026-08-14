"""Parse a ComfyUI workflow JSON and convert it into Movie Editor models config.

Supports the UI workflow format (nodes + links arrays) and the API /prompt format
({node_id: {class_type, inputs}}). Editor bindings map project settings (prompt,
timeline image, FPS, etc.) onto imported node inputs; built-in pipeline is disabled.
"""
from __future__ import annotations

import json
from typing import Any

from . import builder
from .nodes import connection_inputs, describe_node, node_outputs, widget_inputs

# Editor settings the wizard can wire into an imported workflow.
EDITOR_BINDINGS: list[dict[str, str]] = [
    {"key": "prompt", "label": "Positive prompt", "kind": "string"},
    {"key": "negative_prompt", "label": "Negative prompt", "kind": "string"},
    {"key": "seed", "label": "Seed", "kind": "int"},
    {"key": "timeline_image", "label": "Input image (scene)", "kind": "image_input"},
    {"key": "num_frames_per_scene", "label": "Frames per scene", "kind": "int"},
    {"key": "frame_rate", "label": "Frame rate (FPS)", "kind": "float"},
    {"key": "width", "label": "Width", "kind": "int"},
    {"key": "height", "label": "Height", "kind": "int"},
    {"key": "video_output", "label": "Saved video comes from", "kind": "output"},
]

_SKIP_NODE_TYPES = frozenset({"Reroute", "Note", "MarkdownNote", "PrimitiveNode"})

_KIND_MAP = {"string": "string", "int": "int", "float": "float", "boolean": "boolean", "combo": "string"}


def _slot_id(node_id: Any) -> str:
    return f"w{node_id}"


def _uid() -> str:
    import random
    return random.randbytes(4).hex()


def _is_api_format(workflow: Any) -> bool:
    if not isinstance(workflow, dict):
        return False
    if isinstance(workflow.get("nodes"), list):
        return False
    for v in workflow.values():
        if isinstance(v, dict) and "class_type" in v:
            return True
    return False


def _output_name_at(nd: dict | None, slot: int) -> str:
    outs = node_outputs(nd or {})
    if 0 <= slot < len(outs):
        return outs[slot]["name"]
    return str(slot)


def _node_label(node_id: Any, cls: str, props: dict | None = None) -> str:
    props = props or {}
    for k in ("Node name for S&R", "title", "name"):
        if props.get(k):
            return str(props[k])
    return f"{cls} #{node_id}"


def _parse_ui_links(links_raw: list) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for link in links_raw or []:
        if not isinstance(link, list) or len(link) < 6:
            continue
        lid, origin_id, origin_slot, target_id, target_slot, typ = link[:6]
        out[int(lid)] = {
            "origin_id": origin_id,
            "origin_slot": int(origin_slot),
            "target_id": target_id,
            "target_slot": int(target_slot),
            "type": typ,
        }
    return out


def _ui_input_links(nodes_raw: list) -> dict[tuple[Any, str], int]:
    """Map (target_node_id, input_name) -> link_id from node input sockets."""
    m: dict[tuple[Any, str], int] = {}
    for node in nodes_raw or []:
        nid = node.get("id")
        for inp in node.get("inputs") or []:
            if inp.get("link") is not None:
                m[(nid, inp.get("name", ""))] = int(inp["link"])
    return m




def _extract_slot_inputs(cls: str, widgets_values: Any, object_info: dict) -> dict[str, Any]:
    nd = object_info.get(cls)
    if not nd:
        return {}
    return builder.extract_widgets(nd, widgets_values)


def _build_slots_from_nodes(
    node_entries: list[dict],
    edges: list[dict],
    object_info: dict,
) -> tuple[list[dict], list[dict]]:
    """Return (slots, internal_edges) where internal_edges have resolved names."""
    slots: list[dict] = []
    resolved_edges: list[dict] = []

    by_wf_id: dict[Any, dict] = {}
    for ent in node_entries:
        wf_id = ent["wf_id"]
        cls = ent["class_type"]
        if cls in _SKIP_NODE_TYPES:
            continue
        sid = _slot_id(wf_id)
        nd = object_info.get(cls)
        inputs = _extract_slot_inputs(cls, ent.get("widgets_values"), object_info)
        slot = {
            "id": sid,
            "role": "custom",
            "role_label": "Workflow",
            "node_class": cls,
            "label": ent.get("label") or _node_label(wf_id, cls, ent.get("properties")),
            "inputs": inputs,
            "wires": {},
            "input_sources": {},
            "_wf_node_id": wf_id,
        }
        if ent.get("group"):
            slot["group"] = ent["group"]
        if cls not in object_info:
            slot["_missing_class"] = True
        slots.append(slot)
        by_wf_id[wf_id] = slot

    for edge in edges:
        src_wf = edge["origin_id"]
        tgt_wf = edge["target_id"]
        src_slot = by_wf_id.get(src_wf)
        tgt_slot = by_wf_id.get(tgt_wf)
        if not src_slot or not tgt_slot:
            continue
        src_nd = object_info.get(src_slot["node_class"])
        tgt_nd = object_info.get(tgt_slot["node_class"])
        out_name = edge.get("origin_input") or _output_name_at(src_nd, edge["origin_slot"])
        in_name = edge.get("target_input")
        if not in_name:
            in_name = _input_name_for_slot(tgt_nd, edge["target_slot"], edge.get("type"))
        if not in_name:
            continue
        tgt_slot["input_sources"][in_name] = f"out:{src_slot['id']}:{out_name}"
        resolved_edges.append({
            "from": f"{src_slot['label']}.{out_name}",
            "to": f"{tgt_slot['label']}.{in_name}",
            "type": edge.get("type", ""),
        })

    for s in slots:
        s.pop("_wf_node_id", None)
    return slots, resolved_edges


def _input_name_for_slot(nd: dict | None, slot_idx: int, typ: str | None) -> str | None:
    """Best-effort map of UI input slot index -> input name."""
    if not nd:
        return None
    cis = connection_inputs(nd)
    if 0 <= slot_idx < len(cis):
        return cis[slot_idx]["name"]
    # Some nodes interleave widgets; fall back to typed connection_inputs only.
    if typ:
        typed = [c for c in cis if c["type"] == typ]
        if 0 <= slot_idx < len(typed):
            return typed[slot_idx]["name"]
    return None


# ── subgraphs ─────────────────────────────────────────────────────────────────
# A subgraph instance is a node whose `type` is a uuid listed in
# definitions.subgraphs; the real nodes live inside that definition. Without
# expanding it the importer takes the uuid for a class name, so the whole pipeline
# inside the subgraph is lost and the graph is queued with a class ComfyUI has never
# heard of. ComfyUI's own MiniMax H3 templates ship this way.
_SG_IN, _SG_OUT = -10, -20      # the definition's own input / output boundary nodes
_SG_MAX_DEPTH = 8               # a subgraph may contain subgraphs; this bounds a cycle


def _subgraph_defs(workflow: dict) -> dict[str, dict]:
    defs = ((workflow.get("definitions") or {}).get("subgraphs")) or []
    return {str(d.get("id")): d for d in defs if isinstance(d, dict) and d.get("id")}


def _promoted_widget_inputs(defn: dict) -> list[int]:
    """Indices of the definition's inputs that stand for an inner WIDGET.

    An instance's widgets_values lists exactly these, in this order — the socket-only
    inputs (an IMAGE to wire in) are not among them. Which is which is not stated
    anywhere; it is read off the inner node the boundary link lands on, whose input
    socket carries a "widget" key when it is a promoted widget.
    """
    by_id = {n.get("id"): n for n in defn.get("nodes") or []}
    promoted: list[int] = []
    for idx in range(len(defn.get("inputs") or [])):
        for link in defn.get("links") or []:
            if link.get("origin_id") != _SG_IN or link.get("origin_slot") != idx:
                continue
            tgt = by_id.get(link.get("target_id")) or {}
            socket = (tgt.get("inputs") or [])[link.get("target_slot")] \
                if link.get("target_slot") is not None \
                and link.get("target_slot") < len(tgt.get("inputs") or []) else None
            if socket and socket.get("widget"):
                promoted.append(idx)
            break
    return promoted


def _flatten_subgraphs(workflow: dict, object_info: dict) -> dict:
    """Replace every subgraph instance with the nodes inside it, rewired.

    Returns a workflow in the same shape, so everything downstream is unchanged.
    """
    defs = _subgraph_defs(workflow)
    if not defs:
        return workflow

    wf = json.loads(json.dumps(workflow))          # never mutate the caller's dict
    next_link = [max((int(l[0]) for l in (wf.get("links") or [])
                      if isinstance(l, list) and l), default=0) + 1]

    def expand(nodes: list, links: list, depth: int) -> tuple[list, list]:
        if depth > _SG_MAX_DEPTH:
            return nodes, links
        out_nodes, out_links, expanded = [], list(links), False
        for inst in nodes:
            defn = defs.get(str(inst.get("type")))
            if not defn:
                out_nodes.append(inst)
                continue
            expanded = True
            pfx = f"sg{inst.get('id')}_"
            inner_nodes = json.loads(json.dumps(defn.get("nodes") or []))
            by_old = {n.get("id"): n for n in inner_nodes}
            # The subgraph becomes a group, so what it organised survives being flattened.
            # Its own inner groups are more specific, so they win; the instance's title (or
            # the definition name) is the fallback.
            outer_name = str(inst.get("title") or defn.get("name") or "Subgraph")
            for n in inner_nodes:
                n["id"] = f"{pfx}{n.get('id')}"
                if not n.get(_GROUP_KEY):
                    n[_GROUP_KEY] = _group_of(n, defn.get("groups") or []) or outer_name

            # 1. promoted widget values: the instance's, not the inner node's stale copy
            values = inst.get("widgets_values")
            if isinstance(values, list):
                for slot, val in zip(_promoted_widget_inputs(defn), values):
                    for link in defn.get("links") or []:
                        if link.get("origin_id") != _SG_IN or link.get("origin_slot") != slot:
                            continue
                        tgt = by_old.get(link.get("target_id"))
                        sockets = (tgt or {}).get("inputs") or []
                        ts = link.get("target_slot")
                        if tgt is None or ts is None or ts >= len(sockets):
                            continue
                        name = (sockets[ts].get("widget") or {}).get("name")
                        if not name:
                            continue
                        cur = builder.extract_widgets(
                            object_info.get(tgt.get("type")), tgt.get("widgets_values"))
                        cur[name] = val
                        tgt["widgets_values"] = cur   # dict form: extract_widgets passes it through

            # 2. what the outer graph feeds each instance input, by name
            outer_in = {i.get("name"): i.get("link") for i in inst.get("inputs") or []}
            outer_by_id = {int(l[0]): l for l in out_links if isinstance(l, list) and l}
            in_names = [i.get("name") for i in defn.get("inputs") or []]

            # 3. inner links -> outer link rows, resolving both boundaries
            remap: dict[Any, Any] = {}                 # inner link id -> new id or None
            out_slot_src: dict[int, tuple[Any, int]] = {}
            for link in defn.get("links") or []:
                src, sslot = link.get("origin_id"), link.get("origin_slot")
                dst, dslot = link.get("target_id"), link.get("target_slot")
                if dst == _SG_OUT:
                    out_slot_src[int(dslot or 0)] = (f"{pfx}{src}", int(sslot or 0))
                    remap[link.get("id")] = None
                    continue
                if src == _SG_IN:
                    name = in_names[sslot] if sslot is not None and sslot < len(in_names) else None
                    feed = outer_by_id.get(int(outer_in.get(name) or -1)) if name else None
                    if not feed:
                        remap[link.get("id")] = None   # unlinked: the widget value stands
                        continue
                    src, sslot = feed[1], feed[2]
                else:
                    src = f"{pfx}{src}"
                nid = next_link[0]; next_link[0] += 1
                remap[link.get("id")] = nid
                out_links.append([nid, src, int(sslot or 0), f"{pfx}{dst}", int(dslot or 0),
                                  link.get("type")])

            # 4. inner sockets point at inner link ids — repoint or clear them
            for n in inner_nodes:
                for socket in n.get("inputs") or []:
                    if socket.get("link") is not None:
                        socket["link"] = remap.get(socket["link"])

            # 5. consumers of the instance's outputs now read the inner producer
            for row in out_links:
                if isinstance(row, list) and len(row) >= 6 and row[1] == inst.get("id"):
                    hit = out_slot_src.get(int(row[2] or 0))
                    if hit:
                        row[1], row[2] = hit[0], hit[1]
            out_nodes.extend(inner_nodes)
        if expanded:
            return expand(out_nodes, out_links, depth + 1)
        return out_nodes, out_links

    wf["nodes"], wf["links"] = expand(wf.get("nodes") or [], wf.get("links") or [], 0)
    # Links into the vanished instance nodes are dead; drop them rather than leave rows
    # whose target no longer exists.
    live = {n.get("id") for n in wf["nodes"]}
    wf["links"] = [l for l in wf["links"]
                   if isinstance(l, list) and len(l) >= 6 and l[3] in live and l[1] in live]
    return wf


# ── groups ────────────────────────────────────────────────────────────────────
# Workflow authors already organise their graphs — "MODELS", "Prompt", "Sampler - First
# Pass". That grouping is in the file and was being thrown away, leaving one flat wall of
# cards. A group here is a VIEW, never a container: the nodes stay flat and first-class and
# every wire keeps its real endpoints, so there is no inside to navigate into. That is the
# whole difference from a ComfyUI subgraph, which is the part people dislike.
_GROUP_KEY = "_fp_group"


def _group_of(node: dict, groups: list) -> str:
    """Title of the smallest group whose box contains the node's centre, or "".

    Centre rather than corner because nodes routinely overhang a group's edge by a few
    pixels; smallest-wins so a group nested inside another takes the node.
    """
    pos, size = node.get("pos") or [], node.get("size") or []
    if len(pos) < 2:
        return ""
    try:
        cx = float(pos[0]) + float(size[0] if len(size) > 0 else 0) / 2
        cy = float(pos[1]) + float(size[1] if len(size) > 1 else 0) / 2
    except (TypeError, ValueError):
        return ""
    best, best_area = "", None
    for g in groups or []:
        box = g.get("bounding") or []
        if len(box) < 4:
            continue
        try:
            x, y, w, h = (float(v) for v in box[:4])
        except (TypeError, ValueError):
            continue
        if not (x <= cx <= x + w and y <= cy <= y + h):
            continue
        area = w * h
        if best_area is None or area < best_area:
            best, best_area = str(g.get("title") or ""), area
    return best


def _apply_groups(workflow: dict) -> dict:
    """Stamp each node with its group title. Nodes that came from a subgraph already carry
    one and keep it — the subgraph's own name is a better label than whatever outer box the
    instance happened to sit in."""
    groups = workflow.get("groups") or []
    for node in workflow.get("nodes") or []:
        if not node.get(_GROUP_KEY):
            title = _group_of(node, groups)
            if title:
                node[_GROUP_KEY] = title
    return workflow


# ── virtual link nodes ────────────────────────────────────────────────────────
# KJNodes' Set/Get exist ONLY in JavaScript (web/js/setgetnodes.js — there is no Python
# class), so ComfyUI's frontend resolves them away when it builds the prompt. Emitting one
# queues a node type the backend has never heard of. They are common in big LTX workflows:
# one shipped 53 of them across 117 nodes.
_VIRTUAL_LINK_NODES = frozenset({"SetNode", "GetNode"})
_VIRTUAL_MAX_CHASE = 32          # Set -> Get -> Set chains; bounds a cycle


def _virtual_name(node: dict) -> str:
    values = node.get("widgets_values")
    if isinstance(values, list) and values and isinstance(values[0], str):
        return values[0]
    return ""


def _resolve_virtual_links(workflow: dict) -> dict:
    """Rewire past Set/Get nodes, then drop them.

    Link rows are re-pointed IN PLACE, keeping their id, so the `link` ids recorded on
    every consumer's input socket stay valid and only the origin changes.
    """
    nodes = workflow.get("nodes") or []
    if not any(n.get("type") in _VIRTUAL_LINK_NODES for n in nodes):
        return workflow

    wf = json.loads(json.dumps(workflow))
    nodes = wf.get("nodes") or []
    links = [l for l in (wf.get("links") or []) if isinstance(l, list) and len(l) >= 6]
    by_id = {n.get("id"): n for n in nodes}
    by_target = {}                                   # (node id, slot) -> link row
    for row in links:
        by_target[(row[3], row[4])] = row

    # What each name was Set from.
    produced: dict[str, tuple] = {}
    for n in nodes:
        if n.get("type") != "SetNode":
            continue
        row = by_target.get((n.get("id"), 0))
        if row:
            produced[_virtual_name(n)] = (row[1], row[2])

    def source_of(nid, slot, seen=None):
        """Follow Set/Get hops to the real producing node, or None if the chain is broken."""
        seen = seen or set()
        for _ in range(_VIRTUAL_MAX_CHASE):
            node = by_id.get(nid)
            kind = (node or {}).get("type")
            if kind not in _VIRTUAL_LINK_NODES:
                return (nid, slot)
            if nid in seen:
                return None
            seen.add(nid)
            if kind == "GetNode":
                hit = produced.get(_virtual_name(node))
                if not hit:
                    return None                      # Get with no matching Set
                nid, slot = hit
            else:                                    # SetNode passthrough output
                row = by_target.get((nid, 0))
                if not row:
                    return None
                nid, slot = row[1], row[2]
        return None

    kept, dropped = [], set()
    for row in links:
        if (by_id.get(row[3]) or {}).get("type") in _VIRTUAL_LINK_NODES:
            dropped.add(int(row[0]))                 # feeds a Set/Get: it goes with them
            continue
        src = source_of(row[1], row[2])
        if src is None:
            dropped.add(int(row[0]))                 # unresolvable — leave the input unwired
            continue
        row[1], row[2] = src
        kept.append(row)

    wf["links"] = kept
    wf["nodes"] = [n for n in nodes if n.get("type") not in _VIRTUAL_LINK_NODES]
    for n in wf["nodes"]:
        for socket in n.get("inputs") or []:
            if socket.get("link") is not None and int(socket["link"]) in dropped:
                socket["link"] = None
    return wf


def _parse_ui_workflow(workflow: dict, object_info: dict) -> dict:
    workflow = _flatten_subgraphs(workflow, object_info)
    # After flattening: a subgraph can contain Set/Get, and its nodes are outer nodes now.
    workflow = _resolve_virtual_links(workflow)
    workflow = _apply_groups(workflow)
    nodes_raw = workflow.get("nodes") or []
    links_map = _parse_ui_links(workflow.get("links") or [])
    inp_links = _ui_input_links(nodes_raw)

    node_entries: list[dict] = []
    edges: list[dict] = []

    for node in nodes_raw:
        cls = node.get("type") or node.get("class_type")
        if not cls:
            continue
        wf_id = node.get("id")
        node_entries.append({
            "wf_id": wf_id,
            "class_type": cls,
            "widgets_values": node.get("widgets_values"),
            "label": None,
            "properties": node.get("properties") or {},
            "group": node.get(_GROUP_KEY) or "",
        })

    for (tgt_wf, in_name), lid in inp_links.items():
        link = links_map.get(lid)
        if not link:
            continue
        edges.append({
            "origin_id": link["origin_id"],
            "origin_slot": link["origin_slot"],
            "target_id": link["target_id"],
            "target_slot": link["target_slot"],
            "target_input": in_name,
            "type": link.get("type"),
        })

    slots, resolved = _build_slots_from_nodes(node_entries, edges, object_info)
    name = (
        (workflow.get("extra") or {}).get("workflow_name")
        or workflow.get("id")
        or "Imported workflow"
    )
    return {
        "format": "ui",
        "name": str(name),
        "node_count": len(slots),
        "link_count": len(resolved),
        "slots": slots,
        "links": resolved,
        "bindings": EDITOR_BINDINGS,
        "targets": binding_targets(slots, object_info),
        "suggestions": suggest_bindings(slots, object_info),
        "warnings": _collect_warnings(slots, object_info),
    }


def _parse_api_workflow(workflow: dict, object_info: dict) -> dict:
    node_entries: list[dict] = []
    edges: list[dict] = []

    for wf_id, node in workflow.items():
        if not isinstance(node, dict) or "class_type" not in node:
            continue
        cls = node["class_type"]
        node_entries.append({
            "wf_id": wf_id,
            "class_type": cls,
            "widgets_values": _widgets_from_api_inputs(node.get("inputs") or {}, cls, object_info),
            "label": None,
            "properties": {},
        })

    for wf_id, node in workflow.items():
        if not isinstance(node, dict) or "class_type" not in node:
            continue
        cls = node["class_type"]
        tgt_nd = object_info.get(cls)
        for in_name, val in (node.get("inputs") or {}).items():
            if not isinstance(val, list) or len(val) != 2:
                continue
            src_wf, src_slot = val[0], int(val[1])
            src_cls = (workflow.get(str(src_wf)) or workflow.get(src_wf) or {}).get("class_type")
            src_nd = object_info.get(src_cls or "")
            edges.append({
                "origin_id": src_wf,
                "origin_slot": src_slot,
                "target_id": wf_id,
                "target_slot": 0,
                "target_input": in_name,
                "origin_input": _output_name_at(src_nd, src_slot),
                "type": None,
            })

    slots, resolved = _build_slots_from_nodes(node_entries, edges, object_info)
    return {
        "format": "api",
        "name": "Imported workflow",
        "node_count": len(slots),
        "link_count": len(resolved),
        "slots": slots,
        "links": resolved,
        "bindings": EDITOR_BINDINGS,
        "targets": binding_targets(slots, object_info),
        "suggestions": suggest_bindings(slots, object_info),
        "warnings": _collect_warnings(slots, object_info),
    }


def _widgets_from_api_inputs(inputs: dict, cls: str, object_info: dict) -> dict:
    """Pull scalar widget values out of API-format inputs (non-link values)."""
    nd = object_info.get(cls)
    if not nd:
        return {k: v for k, v in inputs.items() if not isinstance(v, list)}
    widgets = {w["name"] for w in widget_inputs(nd)}
    out = {}
    for k, v in inputs.items():
        if isinstance(v, list):
            continue
        if k in widgets:
            out[k] = v
    return out


def _collect_warnings(slots: list[dict], object_info: dict) -> list[str]:
    warns = []
    missing = [s for s in slots if s.get("_missing_class")]
    if missing:
        names = ", ".join(sorted({s["node_class"] for s in missing})[:8])
        extra = f" (+{len(missing) - 8} more)" if len(missing) > 8 else ""
        warns.append(f"{len(missing)} node class(es) not installed in ComfyUI: {names}{extra}")
    for s in slots:
        s.pop("_missing_class", None)
    return warns


def binding_targets(slots: list[dict], object_info: dict) -> dict[str, list[dict]]:
    """Dropdown options per editor binding key."""
    out: dict[str, list[dict]] = {b["key"]: [] for b in EDITOR_BINDINGS}
    for slot in slots:
        spec = describe_node(object_info, slot["node_class"])
        if not spec:
            continue
        label_base = slot.get("label") or slot["node_class"]
        for b in EDITOR_BINDINGS:
            kind = b["kind"]
            if kind in ("string", "int", "float"):
                for inp in spec.get("inputs") or []:
                    ik = _KIND_MAP.get(inp.get("kind", ""), "")
                    if kind == "int" and ik not in ("int",):
                        continue
                    if kind == "float" and ik not in ("float",):
                        continue
                    if kind == "string" and ik not in ("string", "combo"):
                        continue
                    out[b["key"]].append({
                        "value": f"link:{slot['id']}:{inp['name']}",
                        "label": f"{label_base} · {inp['name']}",
                    })
            elif kind == "image_input":
                for ci in spec.get("connection_inputs") or []:
                    if ci["type"] == "IMAGE":
                        out[b["key"]].append({
                            "value": f"source:{slot['id']}:{ci['name']}",
                            "label": f"{label_base} · {ci['name']} ({ci['type']})",
                        })
            elif kind == "output":
                for o in spec.get("outputs") or []:
                    if o["type"] == "IMAGE":
                        out[b["key"]].append({
                            "value": f"wire:{slot['id']}:{o['name']}",
                            "label": f"{label_base} → {o['name']} ({o['type']})",
                        })
    return out


def suggest_bindings(slots: list[dict], object_info: dict) -> dict[str, str]:
    """Heuristic auto-suggestions for common node patterns."""
    sug: dict[str, str] = {}
    text_nodes: list[tuple[str, str, str]] = []  # slot_id, input_name, cls
    clip_encodes: list[tuple[str, str]] = []
    image_inputs: list[tuple[str, str]] = []
    image_outputs: list[tuple[str, str, str]] = []
    seed_inputs: list[tuple[str, str]] = []
    frame_inputs: list[tuple[str, str]] = []
    fps_inputs: list[tuple[str, str]] = []
    wh_inputs: list[tuple[str, str, str]] = []
    load_image_ids: set[str] = set()

    for slot in slots:
        spec = describe_node(object_info, slot["node_class"])
        if not spec:
            continue
        cls = slot["node_class"]
        sid = slot["id"]
        if cls == "LoadImage":
            load_image_ids.add(sid)
        if cls == "CLIPTextEncode":
            for inp in spec.get("inputs") or []:
                if inp.get("kind") == "string":
                    clip_encodes.append((sid, inp["name"]))
        for inp in spec.get("inputs") or []:
            nm = inp["name"].lower()
            if inp.get("kind") == "string" and cls != "CLIPTextEncode" and any(x in nm for x in ("text", "prompt", "positive", "negative")):
                text_nodes.append((sid, inp["name"], cls))
            if inp.get("kind") == "int" and "seed" in nm:
                seed_inputs.append((sid, inp["name"]))
            if inp.get("kind") == "int" and any(x in nm for x in ("frame", "length", "num_frames")):
                frame_inputs.append((sid, inp["name"]))
            if inp.get("kind") == "float" and any(x in nm for x in ("fps", "frame_rate", "framerate")):
                fps_inputs.append((sid, inp["name"]))
            if inp.get("kind") == "int" and nm in ("width", "height"):
                wh_inputs.append((sid, inp["name"], nm))
        for ci in spec.get("connection_inputs") or []:
            if ci["type"] == "IMAGE":
                image_inputs.append((sid, ci["name"]))
        for o in spec.get("outputs") or []:
            if o["type"] == "IMAGE":
                image_outputs.append((sid, o["name"], cls))

    if clip_encodes:
        sug["prompt"] = f"link:{clip_encodes[0][0]}:{clip_encodes[0][1]}"
        if len(clip_encodes) > 1:
            sug["negative_prompt"] = f"link:{clip_encodes[1][0]}:{clip_encodes[1][1]}"
    else:
        pos = [(s, n, c) for s, n, c in text_nodes if "negative" not in n.lower()]
        neg = [(s, n, c) for s, n, c in text_nodes if "negative" in n.lower()]
        if pos:
            s, n, _ = pos[0]
            sug["prompt"] = f"link:{s}:{n}"
        if neg:
            s, n, _ = neg[0]
            sug["negative_prompt"] = f"link:{s}:{n}"

    if seed_inputs:
        s, n = seed_inputs[0]
        sug["seed"] = f"link:{s}:{n}"
    # Prefer an IMAGE input fed from LoadImage; else first consumer IMAGE input.
    timeline = None
    for slot in slots:
        for inp_name, src in (slot.get("input_sources") or {}).items():
            if not src.startswith("out:"):
                continue
            src_id = src.split(":", 2)[1]
            if src_id in load_image_ids:
                timeline = (slot["id"], inp_name)
                break
        if timeline:
            break
    if not timeline and image_inputs:
        timeline = image_inputs[0]
    if timeline:
        sug["timeline_image"] = f"source:{timeline[0]}:{timeline[1]}"
    if frame_inputs:
        s, n = frame_inputs[0]
        sug["num_frames_per_scene"] = f"link:{s}:{n}"
    if fps_inputs:
        s, n = fps_inputs[0]
        sug["frame_rate"] = f"link:{s}:{n}"
    for s, n, nm in wh_inputs:
        if nm == "width" and "width" not in sug:
            sug["width"] = f"link:{s}:{n}"
        if nm == "height" and "height" not in sug:
            sug["height"] = f"link:{s}:{n}"
    # Prefer combine / sampler / decode outputs for global video
    priority = ("VHS_VideoCombine", "SaveVideo", "PreviewImage", "VAEDecode")
    chosen = None
    for pref in priority:
        for s, n, cls in image_outputs:
            if pref in cls:
                chosen = (s, n)
                break
        if chosen:
            break
    if not chosen and image_outputs:
        chosen = (image_outputs[-1][0], image_outputs[-1][1])
    if chosen:
        sug["video_output"] = f"wire:{chosen[0]}:{chosen[1]}"
    return sug


def parse_workflow(workflow: Any, object_info: dict | None = None) -> dict:
    """Parse workflow JSON into slots, internal links, and binding target options."""
    object_info = object_info or {}
    if isinstance(workflow, str):
        try:
            workflow = json.loads(workflow)
        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON: {e}"}
    if not isinstance(workflow, dict):
        return {"error": "Workflow must be a JSON object."}
    if _is_api_format(workflow):
        return _parse_api_workflow(workflow, object_info)
    if isinstance(workflow.get("nodes"), list):
        return _parse_ui_workflow(workflow, object_info)
    return {"error": "Unrecognized workflow format (expected UI nodes[] or API prompt dict)."}


def _slot_by_id(slots: list[dict], sid: str) -> dict | None:
    return next((s for s in slots if s["id"] == sid), None)


def _binding_label(key: str) -> str:
    for b in EDITOR_BINDINGS:
        if b["key"] == key:
            return b["label"]
    return key.replace("_", " ").title()


def _add_editor_link(links: list[dict], editor_key: str, slot_id: str, inp_name: str, kind: str = "string") -> None:
    link = next(
        (l for l in links if l.get("source") == "editor" and l.get("editor_key") == editor_key),
        None,
    )
    if not link:
        link = {
            "id": _uid(),
            "name": _binding_label(editor_key),
            "source": "editor",
            "editor_key": editor_key,
            "kind": kind,
            "members": [],
        }
        links.append(link)
    members = link.setdefault("members", [])
    if not any(m.get("slotId") == slot_id and m.get("input") == inp_name for m in members):
        members.append({"slotId": slot_id, "input": inp_name})


def apply_bindings(parsed: dict, bindings: dict[str, str], object_info: dict | None = None) -> dict:
    """Build models config from parsed workflow + user binding choices."""
    object_info = object_info or {}
    slots = [dict(s) for s in (parsed.get("slots") or [])]
    for s in slots:
        s["inputs"] = dict(s.get("inputs") or {})
        s["input_sources"] = dict(s.get("input_sources") or {})
        s["wires"] = dict(s.get("wires") or {})

    links: list[dict] = []
    for key, target in (bindings or {}).items():
        if not target:
            continue
        if target.startswith("link:"):
            rest = target[5:]
            slot_id, _, inp_name = rest.partition(":")
            spec = describe_node(object_info, (_slot_by_id(slots, slot_id) or {}).get("node_class", ""))
            kind = "string"
            if spec:
                for inp in spec.get("inputs") or []:
                    if inp["name"] == inp_name:
                        kind = inp.get("kind", "string")
                        break
            _add_editor_link(links, key, slot_id, inp_name, kind)
        elif target.startswith("source:"):
            rest = target[7:]
            slot_id, _, inp_name = rest.partition(":")
            slot = _slot_by_id(slots, slot_id)
            if slot and key == "timeline_image":
                slot["input_sources"][inp_name] = "timeline"
        elif target.startswith("wire:"):
            rest = target[5:]
            slot_id, _, out_name = rest.partition(":")
            slot = _slot_by_id(slots, slot_id)
            if slot:
                slot["wires"][out_name] = "global:video"

    config = {
        "slots": slots,
        "links": links,
        "disable_core": True,
        "workflow_import": {
            "name": parsed.get("name") or "Imported workflow",
            "node_count": parsed.get("node_count", len(slots)),
            "link_count": parsed.get("link_count", 0),
            "format": parsed.get("format"),
        },
    }
    return config


def apply_workflow(workflow: Any, bindings: dict[str, str], object_info: dict | None = None) -> dict:
    """Parse + apply in one step."""
    parsed = parse_workflow(workflow, object_info)
    if parsed.get("error"):
        raise ValueError(parsed["error"])
    return apply_bindings(parsed, bindings, object_info)
