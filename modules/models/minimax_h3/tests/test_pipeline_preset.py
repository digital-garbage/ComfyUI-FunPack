"""The MiniMax H3 Reference-to-Video preset: does it hang together as a real
ComfyUI graph, the way a person would load it and run it?

Registers ComfyUI's own extra nodes (MiniMaxH3SigmaShift, MiniMaxH3Reference-
ToVideo, CreateVideo, SaveVideo all live in comfy_extras), FunPack's own nodes
the way ComfyUI itself loads a pack, and the installed KJNodes pack (for
ImageTransformKJ) -- then runs the exact preset through core/graph.py's real
build(), the same function a queued run is checked with.
"""

import asyncio
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def registered(comfyui):
    import nodes as comfy_nodes

    async def load():
        await comfy_nodes.init_extra_nodes(init_custom_nodes=False)
        await comfy_nodes.load_custom_node(
            str(Path(__file__).resolve().parents[4]), module_parent="custom_nodes")
        kjnodes = Path(comfyui) / "custom_nodes" / "ComfyUI-KJNodes"
        if kjnodes.is_dir():
            await comfy_nodes.load_custom_node(str(kjnodes), module_parent="custom_nodes")

    asyncio.run(load())
    return comfy_nodes


def test_the_preset_is_offered(registered):
    from modules.models.minimax_h3.pipeline import presets
    offered = presets()
    assert len(offered) == 1
    assert offered[0]["id"] == "minimax_h3_reference_to_video"
    assert offered[0]["slots"]


def test_the_route_serves_it(registered):
    from core import routes as routes_mod

    async def _run():
        found = []
        for spec, make in routes_mod.modules().providers("pipeline_presets"):
            for preset in make():
                found.append((spec.id, preset["id"]))
        return found

    found = asyncio.run(_run())
    assert ("model_minimax_h3", "minimax_h3_reference_to_video") in found


def test_the_preset_builds_a_real_graph_with_no_structural_problems(registered):
    """Every reason build() could refuse this graph, other than "nothing has
    been picked in a combo yet" (choices exist and this ran with none set) --
    that one is a real state (a fresh install, nothing configured), not a bug
    in how this preset wires itself together."""
    from core import graph

    if "ImageTransformKJ" not in registered.NODE_CLASS_MAPPINGS:
        pytest.skip("ComfyUI-KJNodes is not installed on this machine")

    from modules.models.minimax_h3.pipeline import h3_reference_to_video

    _prompt, problems = graph.build(h3_reference_to_video())
    structural = [p for p in problems if "is not one of" not in p
                  and "and nothing fills it" not in p]
    assert structural == [], f"unexpected graph problems: {structural}"


def test_every_slot_id_is_unique_and_every_link_points_somewhere_real():
    """A self-consistency check on the preset's own data, independent of
    whether ComfyUI happens to have every node installed on this machine."""
    from modules.models.minimax_h3.pipeline import h3_reference_to_video

    slots = h3_reference_to_video()
    ids = [s["id"] for s in slots]
    assert len(ids) == len(set(ids)), "duplicate slot id"

    known = set(ids)
    for slot in slots:
        for name, value in (slot.get("inputs") or {}).items():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                assert value[0] in known, (
                    f"{slot['id']}.{name} is wired to {value[0]!r}, which is not a slot here")


def test_no_reference_slot_is_pre_wired_into_r2v():
    """References only enter the graph when boot.js actually adds the wire for
    a reference the user picked -- see the preset module's own docstring. A
    static wire here would mean R2V always sees MAX_REFERENCES entries, most
    of them empty, corrupting its <Picture i> numbering for everyone."""
    from modules.models.minimax_h3.pipeline import h3_reference_to_video

    slots = h3_reference_to_video()
    r2v = next(s for s in slots if s["id"] == "r2v")
    assert not any(k.startswith("ref_images") for k in (r2v.get("inputs") or {}))
