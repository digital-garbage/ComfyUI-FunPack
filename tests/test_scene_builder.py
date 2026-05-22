import json
import sys
import types
from pathlib import Path

import torch

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
from templates import (
    FunPackSceneBuilder,
    apply_prompt_shortcuts,
    apply_scene_database_authority,
    extract_scene_phrases,
    load_shortcut_db,
    load_scene_db,
    load_transition_db,
    load_custom_transition_triggers,
    normalize_scene_text_spacing,
    normalize_scene_memory_items,
    normalize_shortcut_db,
    save_shortcut_db,
    remember_scene_phrases,
    save_shortcut_item,
    save_scene_db,
    save_transition_item,
    delete_transition_item,
    transition_items,
)


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
    assert required["scene_positive"][1]["multiline"] is False
    assert required["scene_negative"][1]["multiline"] is False


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


def test_scene_builder_save_load_preserves_raw_prompt_order(monkeypatch, tmp_path):
    use_tmp_scene_store(monkeypatch, tmp_path)
    builder = FunPackSceneBuilder()
    positive = "middle detail, first action. close-up camera, last environment, subject returns"
    negative = "blur, bad hands, wrong action"

    saved = builder.build_scene(
        scene="-None-",
        scene_name="Order Scene",
        aliases="",
        action="save",
        mode="Manual",
        scene_positive=positive,
        scene_negative=negative,
        positive_prompt="memory source only",
        negative_prompt="negative memory source only",
        refinement_key="order_key",
    )

    assert saved[0] == positive
    assert saved[1] == negative
    data = load_scene_db("order_key")
    scene = data["scenes"]["Order Scene"]
    assert scene["positive_text"] == positive
    assert scene["negative_text"] == negative
    assert scene["positive_phrases"] == [
        "middle detail",
        "first action",
        "close-up camera",
        "last environment",
        "subject returns",
    ]

    loaded = builder.build_scene(
        scene="Order Scene",
        scene_name="",
        aliases="",
        action="load",
        mode="Manual",
        scene_positive="",
        scene_negative="",
        refinement_key="order_key",
    )

    assert loaded[0] == positive
    assert loaded[1] == negative
    assert "Manual loaded" in loaded[3]


def test_scene_builder_empty_saved_text_does_not_rebuild_from_phrase_arrays(monkeypatch, tmp_path):
    use_tmp_scene_store(monkeypatch, tmp_path)
    data = load_scene_db("empty_text_key")
    data["scenes"]["Empty Positive"] = {
        "name": "Empty Positive",
        "aliases": [],
        "output_mode": "Manual",
        "positive_text": "",
        "negative_text": "",
        "positive_phrases": ["should not appear"],
        "negative_phrases": ["also should not appear"],
    }
    save_scene_db(data, "empty_text_key")

    outputs = FunPackSceneBuilder().build_scene(
        scene="Empty Positive",
        scene_name="",
        aliases="",
        action="load",
        mode="Manual",
        scene_positive="",
        scene_negative="",
        refinement_key="empty_text_key",
    )

    assert outputs[0] == ""
    assert outputs[1] == ""


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


def test_scene_builder_auto_still_collects_provided_scene_knowledge(monkeypatch, tmp_path):
    use_tmp_scene_store(monkeypatch, tmp_path)
    builder = FunPackSceneBuilder()

    builder.build_scene(
        scene="-None-",
        scene_name="",
        aliases="",
        action="load",
        mode="Auto",
        intent_prompt="no matching saved scene",
        scene_positive="glass rain, neon reflection",
        scene_negative="bad fingers",
        positive_prompt="",
        negative_prompt="",
        refinement_key="auto_memory_key",
    )

    data = load_scene_db("auto_memory_key")
    assert "glass rain" in data["universal_memory"]
    assert "neon reflection" in data["universal_memory"]
    assert "bad fingers" in data["universal_memory"]


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


def test_scene_builder_learning_mode_ignores_selected_scene_output(monkeypatch, tmp_path):
    use_tmp_scene_store(monkeypatch, tmp_path)
    builder = FunPackSceneBuilder()

    builder.build_scene(
        scene="-None-",
        scene_name="Saved Scene",
        aliases="",
        action="save",
        mode="Manual",
        scene_positive="saved positive",
        scene_negative="saved negative",
        positive_prompt="memory source",
        negative_prompt="",
        refinement_key="learn_selected_key",
    )

    outputs = builder.build_scene(
        scene="Saved Scene",
        scene_name="",
        aliases="",
        action="load",
        mode="Learning",
        positive_prompt="live positive",
        negative_prompt="live negative",
        refinement_key="learn_selected_key",
    )

    assert outputs[0] == "live positive"
    assert outputs[1] == "live negative"
    assert "saved positive" not in outputs[0]
    assert "Learning" in outputs[3]


def test_scene_builder_auto_mode_ignores_selected_scene_when_intent_matches_other(monkeypatch, tmp_path):
    use_tmp_scene_store(monkeypatch, tmp_path)
    builder = FunPackSceneBuilder()

    builder.build_scene(
        scene="-None-",
        scene_name="Manual Scene",
        aliases="manual scene",
        action="save",
        mode="Manual",
        scene_positive="manual selected",
        scene_negative="",
        refinement_key="auto_selected_key",
    )
    builder.build_scene(
        scene="-None-",
        scene_name="Auto Target",
        aliases="dragon alley",
        action="save",
        mode="Manual",
        scene_positive="auto matched",
        scene_negative="",
        refinement_key="auto_selected_key",
    )

    outputs = builder.build_scene(
        scene="Manual Scene",
        scene_name="",
        aliases="",
        action="load",
        mode="Auto",
        intent_prompt="show dragon alley",
        refinement_key="auto_selected_key",
    )

    assert outputs[0] == "auto matched"
    assert "Auto" in outputs[3]


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


def test_scene_builder_treats_space_before_punctuation_as_normal_punctuation():
    assert normalize_scene_text_spacing("red dress . walking pose , smoke") == "red dress. walking pose, smoke"
    phrases = extract_scene_phrases("red dress . walking pose , smoke")
    assert [item["text"] for item in phrases] == ["red dress", "walking pose", "smoke"]


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


def test_shortcuts_are_global_and_independent_from_refinement_keys(monkeypatch, tmp_path):
    use_tmp_scene_store(monkeypatch, tmp_path)

    save_shortcut_item({
        "name": "Run",
        "triggers": ["running"],
        "replacements": ["running freely"],
    })

    data = load_scene_db("some_key")
    assert data["scenes"] == {}
    shortcuts = load_shortcut_db()["shortcuts"]
    assert "run" in shortcuts
    assert shortcuts["run"]["triggers"] == ["running"]


def test_refiner_reset_session_does_not_clear_global_shortcuts(monkeypatch, tmp_path):
    use_tmp_scene_store(monkeypatch, tmp_path)
    save_shortcut_item({
        "name": "Run",
        "triggers": ["running"],
        "replacements": ["running freely"],
    })

    FunPackVideoRefinerV2()._v2_load_state("reset_key", reset_session=True)

    assert "run" in load_shortcut_db()["shortcuts"]


def test_shortcuts_match_boundaries_and_preserve_surrounding_spacing(monkeypatch, tmp_path):
    use_tmp_scene_store(monkeypatch, tmp_path)
    save_shortcut_item({
        "name": "Run",
        "triggers": ["running"],
        "replacements": ["running freely"],
    })

    expanded, applied = apply_prompt_shortcuts(
        "A video of man running in the park, running.",
        seed=1,
    )

    assert expanded == "A video of man running freely in the park, running freely."
    assert "manrunning" not in expanded
    assert "parkfreely" not in expanded
    assert len(applied) == 2

    unchanged, applied = apply_prompt_shortcuts("A video of over-running shoes", seed=1)
    assert unchanged == "A video of over-running shoes"
    assert applied == []


def test_shortcuts_use_longest_trigger_and_seeded_replacement(monkeypatch, tmp_path):
    use_tmp_scene_store(monkeypatch, tmp_path)
    save_shortcut_item({
        "name": "Look",
        "triggers": ["camera", "looks into camera"],
        "replacements": [
            "a man is facing the camera and looking into it",
            "a man's gaze is directed at the viewer",
        ],
    })

    first, first_applied = apply_prompt_shortcuts("He looks into camera.", seed=77)
    second, second_applied = apply_prompt_shortcuts("He looks into camera.", seed=77)

    assert first == second
    assert first_applied == second_applied
    assert first_applied[0]["trigger"] == "looks into camera"
    assert "He " in first
    assert first.endswith(".")


def test_longer_phrase_wins_across_different_shortcuts(monkeypatch, tmp_path):
    use_tmp_scene_store(monkeypatch, tmp_path)
    save_shortcut_item({
        "name": "Smoke",
        "triggers": ["smoking"],
        "replacements": ["pipe smoking"],
    })
    save_shortcut_item({
        "name": "Smoking Fish",
        "triggers": ["smoking fish"],
        "replacements": ["cold smoked salmon"],
    })
    save_shortcut_item({
        "name": "Chef Smoking Fish",
        "triggers": ["chef smoking fish"],
        "replacements": ["a chef preparing cold smoked salmon"],
    })

    # "smoking fish" wins over "smoking" alone
    result, applied = apply_prompt_shortcuts("I want smoking fish for dinner.", seed=1)
    assert result == "I want cold smoked salmon for dinner."
    assert len(applied) == 1
    assert applied[0]["trigger"] == "smoking fish"

    # "chef smoking fish" wins over both "smoking" and "smoking fish"
    result, applied = apply_prompt_shortcuts("chef smoking fish on a plate.", seed=1)
    assert result == "a chef preparing cold smoked salmon on a plate."
    assert len(applied) == 1
    assert applied[0]["trigger"] == "chef smoking fish"

    # standalone "smoking" not part of a longer phrase still applies
    result, applied = apply_prompt_shortcuts("smoking is shown on screen.", seed=1)
    assert result == "pipe smoking is shown on screen."
    assert len(applied) == 1
    assert applied[0]["trigger"] == "smoking"

    # mixed: standalone "smoking" and "smoking fish" each match their own shortcut
    result, applied = apply_prompt_shortcuts("smoking and smoking fish.", seed=1)
    assert result == "pipe smoking and cold smoked salmon."
    assert len(applied) == 2
    triggers = {a["trigger"] for a in applied}
    assert triggers == {"smoking", "smoking fish"}


def test_empty_replacement_removes_phrase_and_cleans_artifacts(monkeypatch, tmp_path):
    use_tmp_scene_store(monkeypatch, tmp_path)
    save_shortcut_item({
        "name": "Strip Game Tag",
        "triggers": ["from Zenless Zone Zero"],
        "replacements": [""],
    })

    # trailing removal - space+comma artifact cleaned
    result, applied = apply_prompt_shortcuts("Nicole from Zenless Zone Zero, fighting stance.", seed=1)
    assert result == "Nicole, fighting stance."
    assert len(applied) == 1
    assert applied[0]["replacement"] == ""

    # leading removal - leading comma+space cleaned
    result, applied = apply_prompt_shortcuts("from Zenless Zone Zero, Nicole in fighting stance.", seed=1)
    assert result == "Nicole in fighting stance."

    # middle removal - double space cleaned
    result, applied = apply_prompt_shortcuts("Nicole from Zenless Zone Zero and a sword.", seed=1)
    assert result == "Nicole and a sword."

    # end of string - trailing space cleaned
    result, applied = apply_prompt_shortcuts("Nicole from Zenless Zone Zero", seed=1)
    assert result == "Nicole"


def test_disabled_shortcuts_do_not_expand(monkeypatch, tmp_path):
    use_tmp_scene_store(monkeypatch, tmp_path)
    save_shortcut_item({
        "name": "Run",
        "enabled": False,
        "triggers": ["running"],
        "replacements": ["running freely"],
    })

    expanded, applied = apply_prompt_shortcuts("running", seed=1)

    assert expanded == "running"
    assert applied == []


def test_shortcut_only_import_export_shape_round_trips(monkeypatch, tmp_path):
    use_tmp_scene_store(monkeypatch, tmp_path)
    exported = normalize_shortcut_db({
        "shortcuts": {
            "look": {
                "name": "Look",
                "enabled": True,
                "triggers": ["looks into camera"],
                "replacements": ["a man is facing the camera and looking into it"],
            }
        }
    })

    save_shortcut_db(exported)

    loaded = load_shortcut_db()
    assert loaded["shortcuts"]["look"]["name"] == "Look"
    assert loaded["shortcuts"]["look"]["triggers"] == ["looks into camera"]


def test_scene_builder_expands_shortcuts_only_in_positive_prompt(monkeypatch, tmp_path):
    use_tmp_scene_store(monkeypatch, tmp_path)
    save_shortcut_item({
        "name": "Run",
        "triggers": ["running"],
        "replacements": ["running freely"],
    })

    positive, negative, _, status = FunPackSceneBuilder().build_scene(
        scene="-None-",
        scene_name="Shortcut Scene",
        aliases="",
        action="load",
        mode="Manual",
        scene_positive="A video of man running in the park",
        scene_negative="running",
        refinement_key="shortcut_key",
    )

    assert positive == "A video of man running freely in the park"
    assert negative == "running"
    assert "Shortcuts: expanded" in status


def test_refiner_expands_positive_and_intent_shortcuts_before_encoding(monkeypatch, tmp_path):
    use_tmp_scene_store(monkeypatch, tmp_path)
    save_shortcut_item({
        "name": "Run",
        "triggers": ["running"],
        "replacements": ["running freely"],
    })

    cond = [(torch.ones(1, 2, 4), {"pooled_output": torch.ones(1, 4)})]
    _, status, _, _, encoded_prompts, _, _ = FunPackVideoRefinerV2().refine_v2(
        "A video of man running",
        clip=None,
        rating="Missing action",
        refinement_key="refiner_shortcut_key",
        positive_conditioning=cond,
        user_intent_prompt="running",
        reset_session=True,
    )

    assert "A video of man running freely" in encoded_prompts
    assert "Shortcuts: expanded" in status


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


def test_scene_builder_database_authority_preserves_unchanged_rows():
    previous = normalize_scene_memory_items([
        {"text": "red dress", "category": "appearance", "category_source": "auto", "category_locked": False},
    ])
    incoming = normalize_scene_memory_items([
        {"text": "red dress", "category": "appearance"},
    ])

    resolved = apply_scene_database_authority(previous, incoming)

    assert resolved["red dress"]["category_source"] == "auto"
    assert resolved["red dress"]["category_locked"] is False


def test_scene_builder_database_authority_locks_new_and_changed_rows():
    previous = normalize_scene_memory_items([
        {"text": "red dress", "category": "appearance"},
        {"text": "walking pose", "category": "action"},
    ])
    incoming = normalize_scene_memory_items([
        {"text": "red dress", "category": "style"},
        {"text": "blue dress", "category": "appearance"},
    ])

    resolved = apply_scene_database_authority(previous, incoming)

    assert resolved["red dress"]["category_source"] == "user"
    assert resolved["red dress"]["category_locked"] is True
    assert resolved["blue dress"]["category_source"] == "user"
    assert resolved["blue dress"]["category_locked"] is True


def test_scene_builder_prompt_learning_keeps_locked_user_category():
    data = {
        "universal_memory": normalize_scene_memory_items([
            {"text": "running pose", "category": "appearance", "category_source": "user", "category_locked": True},
        ]),
        "scenes": {},
    }

    remember_scene_phrases(data, positive_prompt="running pose")

    item = data["universal_memory"]["running pose"]
    assert item["category"] == "appearance"
    assert item["category_source"] == "user"
    assert item["category_locked"] is True
    assert item["count"] == 1


def test_refiner_scene_builder_sync_skips_locked_rows():
    refiner = FunPackVideoRefinerV2()
    state = {
        "scene_builder": {
            "universal_memory": normalize_scene_memory_items([
                {"text": "running pose", "category": "appearance", "category_source": "user", "category_locked": True},
            ]),
            "scenes": {},
        }
    }
    global_state = {"phrase_memory": {}}
    phrase = refiner._v2_classify_phrases(None, [{"text": "running pose", "tokens": ["running", "pose"]}])[0]
    last_run = {"prompt": "running pose", "phrases": [phrase]}

    refiner._v2_update_phrase_memory(
        global_state,
        last_run,
        {"key": "like", "reward": 1.0, "label": "Perfect"},
        1,
        {"missing_axes": [], "satisfied_axes": ["action"], "resolved_axes": [], "regressed_axes": [], "wrong_axes": []},
    )
    refiner._v2_sync_scene_builder_memory(state, global_state, last_run, 1)

    item = state["scene_builder"]["universal_memory"]["running pose"]
    assert item["category"] == "appearance"
    assert item["category_source"] == "user"
    assert item["category_locked"] is True


def test_custom_transition_save_load_delete(monkeypatch, tmp_path):
    use_tmp_scene_store(monkeypatch, tmp_path)
    save_transition_item({"name": "Number One", "trigger": "1"})
    save_transition_item({"name": "Next Shot", "trigger": "NEXT"})

    db = load_transition_db()
    assert "number one" in db["transitions"]
    assert db["transitions"]["number one"]["trigger"] == "1"
    assert "replacement" not in db["transitions"]["number one"]

    items = transition_items()
    assert any(i["trigger"] == "1" for i in items)

    delete_transition_item("Number One")
    assert "number one" not in load_transition_db()["transitions"]


def test_custom_transition_triggers_returns_enabled_only(monkeypatch, tmp_path):
    use_tmp_scene_store(monkeypatch, tmp_path)
    save_transition_item({"name": "Cut", "trigger": "CUT", "enabled": True})
    save_transition_item({"name": "Disabled", "trigger": "SKIP", "enabled": False})

    triggers = load_custom_transition_triggers()
    assert "CUT" in triggers
    assert "SKIP" not in triggers


def test_split_prompt_recognizes_custom_trigger(monkeypatch, tmp_path):
    use_tmp_scene_store(monkeypatch, tmp_path)
    save_transition_item({"name": "1", "trigger": "1"})

    refiner = FunPackVideoRefinerV2()
    segments = refiner._v2_split_prompt_by_transitions("beach waves, 1, mountain snow")

    assert len(segments) == 2
    assert segments[0].strip() == "beach waves"
    assert "mountain snow" in segments[1]


def test_split_prompt_custom_trigger_kept_at_segment_start(monkeypatch, tmp_path):
    use_tmp_scene_store(monkeypatch, tmp_path)
    save_transition_item({"name": "NEXT", "trigger": "NEXT"})

    refiner = FunPackVideoRefinerV2()
    segments = refiner._v2_split_prompt_by_transitions("beach waves, NEXT, mountain snow")

    assert len(segments) == 2
    assert "beach waves" in segments[0]
    assert segments[1].startswith("NEXT")
    assert "mountain snow" in segments[1]


def test_split_prompt_drops_dangling_trailing_transition(monkeypatch, tmp_path):
    use_tmp_scene_store(monkeypatch, tmp_path)
    refiner = FunPackVideoRefinerV2()

    # transition at the end with no real content after it
    segments = refiner._v2_split_prompt_by_transitions(
        "female character Kai'Sa, the scene then cuts to"
    )
    assert len(segments) == 1
    assert "Kai'Sa" in segments[0]

    # transition + lone article is still dangling - should also be dropped
    segments = refiner._v2_split_prompt_by_transitions(
        "female character Kai'Sa, the scene then cuts to the"
    )
    assert len(segments) == 1
    assert "Kai'Sa" in segments[0]

    # transition + real content after it must NOT be dropped
    segments = refiner._v2_split_prompt_by_transitions(
        "female character Kai'Sa, the scene then cuts to the beach"
    )
    assert len(segments) == 2
    assert "beach" in segments[1]


def test_split_prompt_no_stray_comma_after_article(monkeypatch, tmp_path):
    use_tmp_scene_store(monkeypatch, tmp_path)
    refiner = FunPackVideoRefinerV2()

    # "the scene then cuts to" matches, leaving "the, next view" as following - comma must be gone
    segments = refiner._v2_split_prompt_by_transitions(
        "Kai'Sa, the scene then cuts to the, next view"
    )
    assert len(segments) == 2
    assert "the, next" not in segments[1]
    assert "the next view" in segments[1]
