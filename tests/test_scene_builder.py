import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakeRoutes:
    def get(self, _path):
        return lambda func: func

    def post(self, _path):
        return lambda func: func


sys.modules.setdefault(
    "folder_paths",
    types.SimpleNamespace(get_user_directory=lambda: ""),
)
sys.modules.setdefault(
    "server",
    types.SimpleNamespace(PromptServer=types.SimpleNamespace(instance=types.SimpleNamespace(routes=FakeRoutes()))),
)

import templates
from conditioning import FunPackVideoRefinerV2
from templates import FunPackSceneBuilder, load_scene_db


def use_tmp_scene_store(monkeypatch, tmp_path):
    monkeypatch.setattr(templates, "template_store_dir", lambda: str(tmp_path))
    monkeypatch.setattr(
        templates,
        "refinement_state_path",
        lambda key, namespace, prefix="refine": str(tmp_path / "refinements" / f"{prefix}_{namespace}_{key}.json"),
    )
    monkeypatch.setattr(
        FunPackVideoRefinerV2,
        "_v2_state_path",
        lambda self, key: str(tmp_path / "refinements" / f"refine_v2_clip_{key}.json"),
    )


def test_scene_builder_collects_inputs_as_memory_but_outputs_selection(monkeypatch, tmp_path):
    use_tmp_scene_store(monkeypatch, tmp_path)
    builder = FunPackSceneBuilder()
    payload = {
        "positive_phrases": ["red dress", "walking pose"],
        "negative_phrases": ["bad hands"],
    }
    input_lora_stack = {
        "version": 2,
        "mode": "ltx2",
        "per_block": True,
        "loras": [{"name": "vampire.safetensors", "model_weight": 0.8}],
    }

    positive, negative, scene_name, _, lora_stack, refinement_key, status = builder.build_scene(
        scene="-None-",
        scene_name="Vampire Courtyard",
        aliases="gothic vampire",
        action="save",
        output_mode="Manual",
        intent_prompt="",
        positive_prompt="woman in courtyard, red dress, moonlight",
        negative_prompt="blurry, bad hands",
        refinement_key="scene_key",
        scene_payload=json.dumps(payload),
        lora_stack=input_lora_stack,
    )

    assert positive == "red dress, walking pose"
    assert negative == "bad hands"
    assert scene_name == "Vampire Courtyard"
    assert refinement_key == "scene_key"
    assert "moonlight" not in positive
    assert lora_stack is input_lora_stack
    assert "Manual saved" in status

    data = load_scene_db("scene_key")
    assert "woman in courtyard" in data["universal_memory"]
    assert "blurry" in data["universal_memory"]
    assert data["scenes"]["Vampire Courtyard"]["positive_phrases"] == ["red dress", "walking pose"]


def test_scene_builder_auto_exact_and_fuzzy_match_saved_scene(monkeypatch, tmp_path):
    use_tmp_scene_store(monkeypatch, tmp_path)
    builder = FunPackSceneBuilder()
    saved_payload = {
        "positive_phrases": ["anthro dragon", "mossy scales"],
        "negative_phrases": ["extra limbs"],
    }
    manual_payload = {
        "positive_phrases": ["manual fallback"],
        "negative_phrases": [],
    }

    builder.build_scene(
        scene="-None-",
        scene_name="Forest Dragon",
        aliases="woodland beast",
        action="save",
        output_mode="Manual",
        intent_prompt="",
        positive_prompt="anthro dragon, mossy scales",
        negative_prompt="extra limbs",
        refinement_key="dragon_key",
        scene_payload=json.dumps(saved_payload),
    )

    exact = builder.build_scene(
        scene="-None-",
        scene_name="",
        aliases="",
        action="load",
        output_mode="Auto",
        intent_prompt="please apply woodland beast now",
        positive_prompt="ignored prompt",
        negative_prompt="ignored negative",
        refinement_key="dragon_key",
        scene_payload=json.dumps(manual_payload),
    )
    assert exact[0] == "anthro dragon, mossy scales"
    assert exact[1] == "extra limbs"
    assert exact[2] == "Forest Dragon"
    assert "Auto exact" in exact[6]

    fuzzy = builder.build_scene(
        scene="-None-",
        scene_name="",
        aliases="",
        action="load",
        output_mode="Auto",
        intent_prompt="make it forest and dragon",
        positive_prompt="ignored prompt",
        negative_prompt="",
        refinement_key="dragon_key",
        scene_payload=json.dumps(manual_payload),
    )
    assert fuzzy[0] == "anthro dragon, mossy scales"
    assert "Auto fuzzy" in fuzzy[6]


def test_scene_builder_auto_no_match_falls_back_to_manual(monkeypatch, tmp_path):
    use_tmp_scene_store(monkeypatch, tmp_path)
    builder = FunPackSceneBuilder()
    payload = {
        "positive_phrases": ["manual only"],
        "negative_phrases": ["manual negative"],
    }

    outputs = builder.build_scene(
        scene="-None-",
        scene_name="",
        aliases="",
        action="load",
        output_mode="Auto",
        intent_prompt="no known scene here",
        positive_prompt="source memory only",
        negative_prompt="bad source memory",
        refinement_key="",
        scene_payload=json.dumps(payload),
    )

    assert outputs[0] == "manual only"
    assert outputs[1] == "manual negative"
    assert "Manual" in outputs[6]


def test_scene_builder_learning_mode_collects_memory_and_passes_inputs_through(monkeypatch, tmp_path):
    use_tmp_scene_store(monkeypatch, tmp_path)
    builder = FunPackSceneBuilder()
    lora_stack = {"loras": [{"name": "active.safetensors"}]}
    positive_prompt = "  first prompt,\nsecond prompt  "
    negative_prompt = "bad anatomy, blur"

    outputs = builder.build_scene(
        scene="-None-",
        scene_name="",
        aliases="",
        action="load",
        output_mode="Learning",
        intent_prompt="ignored scene trigger",
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        refinement_key="learn_key",
        scene_payload=json.dumps({
            "positive_phrases": ["manual phrase"],
            "negative_phrases": ["manual negative"],
        }),
        lora_stack=lora_stack,
    )

    assert outputs[0] == positive_prompt
    assert outputs[1] == negative_prompt
    assert outputs[2] == ""
    assert outputs[4] is lora_stack
    assert outputs[5] == "learn_key"
    assert "Learning" in outputs[6]

    data = load_scene_db("learn_key")
    assert "first prompt" in data["universal_memory"]
    assert "second prompt" in data["universal_memory"]
    assert "bad anatomy" in data["universal_memory"]
    assert data["scenes"] == {}


def test_refiner_reset_preserves_scene_builder_memory_in_refinement_key(monkeypatch, tmp_path):
    use_tmp_scene_store(monkeypatch, tmp_path)
    builder = FunPackSceneBuilder()

    builder.build_scene(
        scene="-None-",
        scene_name="Memory Scene",
        aliases="",
        action="save",
        output_mode="Manual",
        positive_prompt="silver hair, rain street",
        negative_prompt="blur",
        refinement_key="shared_key",
        scene_payload=json.dumps({
            "positive_phrases": ["silver hair"],
            "negative_phrases": ["blur"],
        }),
    )

    refiner = FunPackVideoRefinerV2()
    reset_state, reset_status = refiner._v2_load_state("shared_key", reset_session=True)

    assert reset_status == "fresh"
    assert reset_state["prompt_histories"] == {}
    assert reset_state["global"]["phrase_memory"] == {}
    assert reset_state["scene_builder"]["scenes"]["Memory Scene"]["positive_phrases"] == ["silver hair"]
    assert "silver hair" in reset_state["scene_builder"]["universal_memory"]
