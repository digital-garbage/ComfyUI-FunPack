"""Load the user's API-format workflow and inject per-run parameters.

The exact graph (node ids/classes) is the user's — exported via ComfyUI
"Save (API Format)" to config.TEMPLATE_PATH (workflow-first: we don't fabricate
wiring). This module is graph-agnostic: it binds logical params to nodes by
class_type or explicit node id, and writes their widget inputs.

API format: { "<node_id>": {"class_type": str, "inputs": {...}, ...}, ... }

Bindings can be overridden by a JSON file next to the template
(`<template>.bindings.json`). Until that exists, the defaults target the known
FunPack node classes; `prompt` defaults to the first text-like node feeding Studio.
"""
from __future__ import annotations

import copy
import json
from typing import Any, Optional

from . import config

# Logical param -> list of candidate (class_type, input_key). First match wins per
# param; a param may bind to multiple nodes (e.g. seed on both Studio and Sampler).
DEFAULT_BINDINGS: dict[str, list[dict[str, str]]] = {
    "prompt": [
        # Studio reads positive_prompt from a connected text node; common text nodes:
        {"class_type": "FunPackPromptCombiner", "input_key": "text"},
        {"class_type": "CLIPTextEncode", "input_key": "text"},
        {"class_type": "PrimitiveString", "input_key": "value"},
        {"class_type": "String", "input_key": "value"},
        {"class_type": "Text Multiline", "input_key": "text"},
    ],
    "seed": [
        {"class_type": "FunPackLTXAVSceneChainSampler", "input_key": "seed"},
        {"class_type": "FunPackStudio", "input_key": "seed"},
    ],
    "num_frames_per_scene": [
        {"class_type": "FunPackLTXAVSceneChainSampler", "input_key": "num_frames_per_scene"},
    ],
    "max_scenes": [
        {"class_type": "FunPackLTXAVSceneChainSampler", "input_key": "max_scenes"},
    ],
    "frame_rate": [
        {"class_type": "EmptyLTXVLatentVideo", "input_key": "frame_rate"},
        {"class_type": "LTXVConditioning", "input_key": "frame_rate"},
    ],
}


class WorkflowError(RuntimeError):
    pass


def load_template(path=None) -> dict:
    path = path or config.TEMPLATE_PATH
    try:
        graph = json.loads(open(path).read())
    except FileNotFoundError:
        raise WorkflowError(
            f"Workflow template not found at {path}. Export your Studio -> Chain Sampler "
            f"graph from ComfyUI via 'Save (API Format)' and place it there (plan step 0)."
        )
    except json.JSONDecodeError as e:
        raise WorkflowError(f"Workflow template is not valid JSON: {e}")
    if not isinstance(graph, dict) or not graph:
        raise WorkflowError("Workflow template must be a non-empty API-format object.")
    return graph


def load_bindings(path=None) -> dict[str, list[dict[str, str]]]:
    path = path or config.TEMPLATE_PATH
    override = str(path) + ".bindings.json"
    try:
        data = json.loads(open(override).read())
        merged = copy.deepcopy(DEFAULT_BINDINGS)
        merged.update(data)
        return merged
    except FileNotFoundError:
        return copy.deepcopy(DEFAULT_BINDINGS)


def describe(graph: dict) -> list[dict]:
    """List nodes (id, class_type, title) — helps configure bindings for a new template."""
    out = []
    for nid, node in graph.items():
        if not isinstance(node, dict):
            continue
        out.append({
            "id": nid,
            "class_type": node.get("class_type"),
            "title": (node.get("_meta") or {}).get("title"),
        })
    return out


def _candidates(graph: dict, spec: dict) -> list[str]:
    """Node ids matching a binding spec (by explicit id or class_type)."""
    if "node_id" in spec:
        return [spec["node_id"]] if spec["node_id"] in graph else []
    ct = spec.get("class_type")
    return [nid for nid, node in graph.items()
            if isinstance(node, dict) and node.get("class_type") == ct]


def inject(graph: dict, values: dict[str, Any], bindings=None) -> tuple[dict, list[str]]:
    """Return (new_graph, applied_log). Only params present in `values` are written.

    A param with no matching node is reported in the log but is not fatal — except
    `prompt`, which is required for a meaningful run.
    """
    graph = copy.deepcopy(graph)
    bindings = bindings or load_bindings()
    applied: list[str] = []

    for param, value in values.items():
        if value is None:
            continue
        specs = bindings.get(param, [])
        hit = False
        for spec in specs:
            key = spec.get("input_key")
            for nid in _candidates(graph, spec):
                node = graph[nid]
                inputs = node.setdefault("inputs", {})
                # Don't clobber a wired link ([node_id, slot]) unless it's the prompt
                # text sink — for widgets we set the literal value.
                if isinstance(inputs.get(key), list) and param != "prompt":
                    continue
                inputs[key] = value
                applied.append(f"{param} -> node {nid} ({node.get('class_type')}).{key}")
                hit = True
            if hit:
                break  # first matching class wins per param (except multi-node seed below)
        if not hit:
            applied.append(f"{param}: no matching node (skipped)")

    if "prompt" in values and not any(a.startswith("prompt ->") for a in applied):
        raise WorkflowError(
            "Could not find a text node to inject the prompt into. Add a `prompt` entry to "
            f"{config.TEMPLATE_PATH}.bindings.json mapping to the text node feeding Studio."
        )
    return graph, applied
