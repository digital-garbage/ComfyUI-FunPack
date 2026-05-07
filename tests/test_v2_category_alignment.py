import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.modules.setdefault("folder_paths", types.SimpleNamespace(models_dir=""))

import torch

from conditioning import FunPackVideoRefinerV2, normalize_refiner_v2_rating, tensor_to_serializable


def primary_category(phrase):
    refiner = FunPackVideoRefinerV2()
    scores = refiner._v2_heuristic_scores(phrase)
    primary, confidence = refiner._v2_scores_primary(scores)
    return primary, confidence, scores


def test_environment_descriptions_do_not_become_actions():
    primary, confidence, scores = primary_category("detailed background")

    assert primary == "environment"
    assert confidence >= 0.60
    assert scores["action"] == 0.0


def test_weathered_environment_descriptions_do_not_become_actions():
    primary, confidence, scores = primary_category("weathered stone room")

    assert primary == "environment"
    assert confidence >= 0.60
    assert scores["action"] == 0.0


def test_appearance_descriptions_stay_appearance():
    primary, confidence, scores = primary_category("flowing hair and blue eyes")

    assert primary == "appearance"
    assert confidence >= 0.60
    assert scores["action"] == 0.0


def test_clothing_descriptions_stay_appearance():
    primary, confidence, scores = primary_category("wearing red dress")

    assert primary == "appearance"
    assert confidence >= 0.60
    assert scores["action"] == 0.0


def test_motion_descriptions_stay_action():
    primary, confidence, scores = primary_category("running through the street")

    assert primary == "action"
    assert confidence >= 0.60
    assert scores["action"] >= scores["environment"]


def test_axis_feedback_treats_unmentioned_axes_as_satisfied():
    refiner = FunPackVideoRefinerV2()
    profile = normalize_refiner_v2_rating("Missing quality")

    feedback = refiner._v2_axis_feedback(profile, ["details", "action"])

    assert feedback["missing_axes"] == ["quality"]
    assert feedback["satisfied_axes"] == ["details", "action"]
    assert feedback["resolved_axes"] == ["details", "action"]
    assert feedback["regressed_axes"] == ["quality"]


def test_axis_feedback_does_not_infer_regression_without_previous_rating():
    refiner = FunPackVideoRefinerV2()
    profile = normalize_refiner_v2_rating("Missing quality")

    feedback = refiner._v2_axis_feedback(profile, None)

    assert feedback["missing_axes"] == ["quality"]
    assert feedback["satisfied_axes"] == ["details", "action"]
    assert feedback["resolved_axes"] == []
    assert feedback["regressed_axes"] == []


def test_conditioning_memory_records_missing_and_satisfied_axes():
    refiner = FunPackVideoRefinerV2()
    global_state = {"axis_conditioning_memory": {}}
    quality_good = tensor_to_serializable(torch.ones(1, 3, 2))
    action_good = tensor_to_serializable(torch.zeros(1, 3, 2))

    first_profile = normalize_refiner_v2_rating("Missing details + action")
    first_feedback = refiner._v2_axis_feedback(first_profile, None)
    refiner._v2_update_conditioning_memory(
        global_state,
        {"conditioning": quality_good},
        first_profile,
        first_feedback,
    )

    second_profile = normalize_refiner_v2_rating("Missing quality")
    second_feedback = refiner._v2_axis_feedback(second_profile, ["details", "action"])
    refiner._v2_update_conditioning_memory(
        global_state,
        {"conditioning": action_good},
        second_profile,
        second_feedback,
    )

    memory = global_state["axis_conditioning_memory"]
    assert memory["quality"]["positive"]["count"] == 1
    assert memory["quality"]["negative"]["count"] == 1
    assert memory["action"]["positive"]["count"] >= 1
    assert memory["details"]["positive"]["count"] >= 1
    assert memory["action"]["negative"]["count"] == 1
    assert memory["details"]["negative"]["count"] == 1


def test_awful_lora_feedback_reduces_before_missing_axis_boosts():
    refiner = FunPackVideoRefinerV2()
    prompt_history = {}
    global_state = {"lora_weight_memory": {}}
    profile = normalize_refiner_v2_rating("Awful")
    feedback = refiner._v2_axis_feedback(profile, [])

    refiner._v2_update_lora_suggestions(
        {
            "loras": [
                {
                    "id": "motion",
                    "name": "motion_lora.safetensors",
                    "type": "action",
                    "base_model_weight": 1.0,
                }
            ]
        },
        prompt_history,
        global_state,
        [{"text": "running", "primary": "action"}],
        profile,
        feedback,
    )

    suggestion = prompt_history["lora_weight_suggestions"]["motion"]
    assert suggestion["model_weight"] < 1.0
