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
from templates import FunPackSceneBuilder, load_scene_db, normalize_scene_memory_items, save_scene_db


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

    positive, negative, lora_stack, status = builder.build_scene(
        scene="-None-",
        scene_name="Vampire Courtyard",
        aliases="gothic vampire",
        action="save",
        mode="Manual",
        intent_prompt="",
        positive_prompt="woman in courtyard, red dress, moonlight",
        negative_prompt="blurry, bad hands",
        refinement_key="scene_key",
        scene_payload=json.dumps(payload),
        lora_stack=input_lora_stack,
    )

    assert positive == "red dress, walking pose"
    assert negative == "bad hands"
    assert "moonlight" not in positive
    assert lora_stack is input_lora_stack
    assert "Manual saved" in status

    data = load_scene_db("scene_key")
    assert "woman in courtyard" in data["universal_memory"]
    assert "blurry" in data["universal_memory"]
    assert data["scenes"]["Vampire Courtyard"]["positive_phrases"] == ["red dress", "walking pose"]


def test_scene_builder_public_interface_is_prompt_stack_status_only():
    assert FunPackSceneBuilder.RETURN_NAMES == (
        "positive_prompt",
        "negative_prompt",
        "lora_stack",
        "status",
    )
    assert FunPackSceneBuilder.RETURN_TYPES == (
        "STRING",
        "STRING",
        "FUNPACK_LORA_STACK",
        "STRING",
    )

    inputs = FunPackSceneBuilder.INPUT_TYPES()
    required = inputs["required"]
    optional = inputs["optional"]
    assert "sigmas" not in required
    assert "sigmas" not in optional
    assert "positive_conditioning" not in required
    assert "positive_conditioning" not in optional
    assert "refinement_key" not in FunPackSceneBuilder.RETURN_NAMES
    assert "scene_name" not in FunPackSceneBuilder.RETURN_NAMES


def test_scene_builder_manual_outputs_exact_composed_text(monkeypatch, tmp_path):
    use_tmp_scene_store(monkeypatch, tmp_path)
    builder = FunPackSceneBuilder()

    outputs = builder.build_scene(
        scene="-None-",
        scene_name="Editable Scene",
        aliases="",
        action="save",
        mode="Manual",
        scene_positive="red dress. walking pose, custom order",
        scene_negative="bad hands. blur",
        positive_prompt="source memory only, should not output",
        negative_prompt="source negative only",
        refinement_key="scene_key",
        scene_payload=json.dumps({
            "positive_phrases": ["ignored payload phrase"],
            "negative_phrases": ["ignored payload negative"],
        }),
    )

    assert outputs[0] == "red dress. walking pose, custom order"
    assert outputs[1] == "bad hands. blur"
    assert "source memory only" not in outputs[0]
    data = load_scene_db("scene_key")
    scene = data["scenes"]["Editable Scene"]
    assert scene["positive_text"] == "red dress. walking pose, custom order"
    assert scene["negative_text"] == "bad hands. blur"
    assert "source memory only" in data["universal_memory"]


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
        mode="Manual",
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
        mode="Auto",
        intent_prompt="please apply woodland beast now",
        positive_prompt="ignored prompt",
        negative_prompt="ignored negative",
        refinement_key="dragon_key",
        scene_payload=json.dumps(manual_payload),
    )
    assert exact[0] == "anthro dragon, mossy scales"
    assert exact[1] == "extra limbs"
    assert exact[2] is None
    assert "Auto exact" in exact[3]

    fuzzy = builder.build_scene(
        scene="-None-",
        scene_name="",
        aliases="",
        action="load",
        mode="Auto",
        intent_prompt="make it forest and dragon",
        positive_prompt="ignored prompt",
        negative_prompt="",
        refinement_key="dragon_key",
        scene_payload=json.dumps(manual_payload),
    )
    assert fuzzy[0] == "anthro dragon, mossy scales"
    assert "Auto fuzzy" in fuzzy[3]


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
        mode="Auto",
        intent_prompt="no known scene here",
        positive_prompt="source memory only",
        negative_prompt="bad source memory",
        refinement_key="",
        scene_payload=json.dumps(payload),
    )

    assert outputs[0] == "manual only"
    assert outputs[1] == "manual negative"
    assert "Manual" in outputs[3]


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
        mode="Learning",
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
    assert outputs[2] is lora_stack
    assert "Learning" in outputs[3]

    data = load_scene_db("learn_key")
    assert "first prompt" in data["universal_memory"]
    assert "second prompt" in data["universal_memory"]
    assert "bad anatomy" in data["universal_memory"]
    assert data["scenes"] == {}


def test_scene_builder_collects_prompt_words_from_sentence_chunks(monkeypatch, tmp_path):
    use_tmp_scene_store(monkeypatch, tmp_path)
    builder = FunPackSceneBuilder()

    builder.build_scene(
        scene="-None-",
        scene_name="",
        aliases="",
        action="load",
        mode="Learning",
        positive_prompt="person smoking in the rain",
        negative_prompt="",
        refinement_key="word_key",
    )

    data = load_scene_db("word_key")
    assert "person smoking in the rain" in data["universal_memory"]
    assert "person" in data["universal_memory"]
    assert "smoking" in data["universal_memory"]
    assert "the" not in data["universal_memory"]


def test_refiner_reset_preserves_scene_builder_memory_in_refinement_key(monkeypatch, tmp_path):
    use_tmp_scene_store(monkeypatch, tmp_path)
    builder = FunPackSceneBuilder()

    builder.build_scene(
        scene="-None-",
        scene_name="Memory Scene",
        aliases="",
        action="save",
        mode="Manual",
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


def test_scene_builder_database_items_keep_categories_and_wildcards():
    memory = normalize_scene_memory_items([
        {"text": "red dress", "category": "appearance", "wildcard": True},
        {"text": "blue dress", "category": "appearance", "wildcard": True},
        {"text": "bad hands", "category": "negative"},
    ])

    assert memory["red dress"]["category"] == "appearance"
    assert memory["red dress"]["wildcard"] is True
    assert memory["blue dress"]["wildcard"] is True
    assert memory["bad hands"]["wildcard"] is False
    assert memory["bad hands"]["source"] == "negative"


def test_scene_builder_wildcard_outputs_one_adjacent_matching_phrase(monkeypatch, tmp_path):
    use_tmp_scene_store(monkeypatch, tmp_path)
    builder = FunPackSceneBuilder()
    data = load_scene_db("wild_key")
    data["universal_memory"] = normalize_scene_memory_items([
        {"text": "red dress", "category": "appearance", "wildcard": True},
        {"text": "blue dress", "category": "appearance", "wildcard": True},
        {"text": "green dress", "category": "appearance", "wildcard": True},
    ])
    save_scene_db(data, "wild_key")
    monkeypatch.setattr(templates.random, "choice", lambda choices: "blue dress")

    outputs = builder.build_scene(
        scene="-None-",
        scene_name="Wildcard Scene",
        aliases="",
        action="load",
        mode="Manual",
        scene_positive="red dress, blue dress, walking pose, green dress",
        scene_negative="",
        refinement_key="wild_key",
    )

    assert outputs[0] == "blue dress, walking pose, green dress"
    assert "red dress" not in outputs[0]


def test_scene_builder_legacy_wildcard_group_migrates_to_checkbox():
    memory = normalize_scene_memory_items([
        {"text": "red dress", "category": "appearance", "wildcard_group": "outfit"},
    ])

    assert memory["red dress"]["wildcard"] is True
