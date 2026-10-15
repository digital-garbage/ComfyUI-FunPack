"""Every optional node of every shipped preset can be missing and the preset still builds the same graph, minus that node.

Real node schemas (ComfyUI's own plus this pack's), one node hidden at a time. "Same graph" is judged against the same
preset built with everything present: hiding an optional node must add no new problem (a third-party pack that is absent
on this machine, or an unset model file, is a problem in both and cancels out).
"""
import asyncio
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def registered(comfyui):
    import nodes as comfy_nodes

    async def load():
        await comfy_nodes.init_extra_nodes(init_custom_nodes=False)
        await comfy_nodes.load_custom_node(str(Path(__file__).resolve().parents[3]), module_parent="custom_nodes")
    asyncio.run(load())
    return comfy_nodes


def _presets():
    from modules.models.minimax_h3.pipeline import presets as h3
    from modules.models.ltx.pipeline import presets as ltx
    from modules.models.sd15 import presets as sd15
    return [p for make in (h3, ltx, sd15) for p in make()]


#: Nodes that come from one module go missing together.
TOGETHER = {"FunPackPromptMarkup": {"FunPackPromptMarkup", "FunPackApplyPromptMarkup"},
            "FunPackApplyPromptMarkup": {"FunPackPromptMarkup", "FunPackApplyPromptMarkup"}}


def _cases():
    out = []
    for preset in _presets():
        for slot in preset["slots"]:
            if slot.get("optional"):
                out.append((preset["id"], slot["id"]))
    return out


def test_the_six_optional_nodes_are_declared():
    nodes = {s["node"] for p in _presets() for s in p["slots"] if s.get("optional")}
    assert nodes == {"FunPackEnhancePrompt", "FunPackShotCamera", "FunPackPromptMarkup", "FunPackApplyPromptMarkup",
                     "FunPackNegativeErase", "FunPackCutOpening", "FunPackIdentityTransfer"}


@pytest.mark.parametrize("preset_id,slot_id", _cases())
def test_a_preset_builds_without_its_optional_node(registered, preset_id, slot_id):
    from core import graph
    preset = next(p for p in _presets() if p["id"] == preset_id)
    node = next(s["node"] for s in preset["slots"] if s["id"] == slot_id)
    real = graph.from_comfyui()
    gone = TOGETHER.get(node, {node})
    hidden = graph.Schemas(lambda name: None if name in gone else real.of(name))
    base_prompt, base_problems = graph.build(preset["slots"], real)
    prompt, problems = graph.build(preset["slots"], hidden)
    assert [p for p in problems if p not in base_problems] == [], f"hiding {node} breaks {preset_id}"
    away = {s["id"] for s in preset["slots"] if s["node"] in gone}
    assert slot_id not in prompt and slot_id in base_prompt
    assert set(prompt) == set(base_prompt) - away


def test_the_enhance_slots_typed_text_reaches_the_encoder_when_it_is_skipped(registered):
    from core import graph
    from modules.models.minimax_h3.pipeline import presets
    slots = [dict(s) for s in presets()[0]["slots"]]
    for s in slots:
        if s["id"] == "enhance":
            s["inputs"] = dict(s["inputs"], text="a cat on a roof")
    real = graph.from_comfyui()
    hidden = graph.Schemas(lambda name: None if name in ("FunPackEnhancePrompt", "FunPackShotCamera", "FunPackPromptMarkup", "FunPackApplyPromptMarkup") else real.of(name))
    prompt, _problems = graph.build(slots, hidden)
    assert prompt["r2v"]["inputs"]["prompt"] == "a cat on a roof"
