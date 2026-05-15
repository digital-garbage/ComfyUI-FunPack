import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.modules.setdefault("folder_paths", types.SimpleNamespace(models_dir=""))

import torch

from conditioning import (
    FunPackVideoRefinerV2,
    V2_RATING_LABELS,
    normalize_refiner_v2_rating,
    tensor_to_serializable,
)


class FakeClip:
    def tokenize(self, text):
        return text

    def encode_from_tokens_scheduled(self, tokens):
        return [(torch.ones(1, 4, 3), {"pooled_output": torch.ones(1, 3)})]


class CountingClip(FakeClip):
    def __init__(self):
        self.calls = 0

    def encode_from_tokens_scheduled(self, tokens):
        self.calls += 1
        return super().encode_from_tokens_scheduled(tokens)


class GeneratingClip(FakeClip):
    def __init__(self, generated_text):
        self.generated_text = generated_text
        self.tokenize_calls = []
        self.generate_kwargs = {}

    def tokenize(self, text, **kwargs):
        self.tokenize_calls.append((text, kwargs))
        return text

    def generate(self, tokens, **kwargs):
        self.generate_kwargs = kwargs
        return [1, 2, 3]

    def decode(self, token_ids, skip_special_tokens=True):
        return self.generated_text


def primary_category(phrase):
    refiner = FunPackVideoRefinerV2()
    scores = refiner._v2_heuristic_scores(phrase)
    primary, confidence = refiner._v2_scores_primary(scores)
    return primary, confidence, scores


def classified_phrase(refiner, text):
    return refiner._v2_classify_phrases(
        None,
        [{"text": text, "tokens": refiner._v2_phrase_words(text)}],
    )[0]


def train_phrase(refiner, text, rating_label, global_state=None, previous_missing_axes=None, iterations=1):
    global_state = global_state or {"phrase_memory": {}}
    phrase = classified_phrase(refiner, text)
    last_run = {
        "prompt": text,
        "phrases": [phrase],
    }
    profile = normalize_refiner_v2_rating(rating_label)
    feedback = refiner._v2_axis_feedback(profile, previous_missing_axes)
    for iteration in range(iterations):
        refiner._v2_update_phrase_memory(global_state, last_run, profile, iteration + 1, feedback)
    return global_state["phrase_memory"][text.lower()], global_state, feedback


def prompt_items(refiner, words):
    return [
        {"text": word, "tokens": refiner._v2_phrase_words(word)}
        for word in words
    ]


def prompt_phrases(refiner, prompt, global_state=None):
    return refiner._v2_classify_phrases(
        None,
        refiner._ordered_prompt_phrases(prompt),
        global_state or {"phrase_memory": {}},
    )


def train_prompt_context(refiner, words, rating_label, global_state=None, iterations=1):
    global_state = global_state or {"phrase_memory": {}}
    phrases = refiner._v2_classify_phrases(None, prompt_items(refiner, words), global_state)
    last_run = {
        "prompt": " ".join(words),
        "phrases": phrases,
    }
    profile = normalize_refiner_v2_rating(rating_label)
    feedback = refiner._v2_axis_feedback(profile, None)
    for iteration in range(iterations):
        refiner._v2_update_phrase_memory(global_state, last_run, profile, iteration + 1, feedback)
    return global_state, feedback


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


def test_category_weights_are_recorded_for_every_learning_rating():
    refiner = FunPackVideoRefinerV2()
    learning_labels = [
        label for label in V2_RATING_LABELS
        if label not in {"-Just forget it-", "Wrong appearance"}
    ]

    for label in learning_labels:
        entry, _, _ = train_phrase(refiner, f"running test {label}", label)

        assert entry["category_evidence_count"] == 1
        assert set(entry["category_weights"]) == set(refiner.CATEGORY_DESCRIPTIONS)
        assert set(entry["clip_heuristic_scores"]) == set(refiner.CATEGORY_DESCRIPTIONS)
        assert set(entry["effective_category_scores"]) == set(refiner.CATEGORY_DESCRIPTIONS)


def test_wrong_appearance_rating_is_available():
    profile = normalize_refiner_v2_rating("Wrong appearance")

    assert "Wrong appearance" in V2_RATING_LABELS
    assert profile["key"] == "wrong_appearance"
    assert profile["wrong_categories"] == ["appearance", "subject", "environment"]


def test_prompt_repair_blocks_appearance_subject_and_environment():
    refiner = FunPackVideoRefinerV2()
    global_state = {
        "phrase_memory": {
            "white tights": {
                "text": "white tights",
                "primary": "appearance",
                "effective_category_scores": refiner._v2_heuristic_scores("white tights"),
                "wanted_axes": {"details": 3},
                "score": 6.0,
                "liked_count": 4,
            },
            "detailed background": {
                "text": "detailed background",
                "primary": "environment",
                "effective_category_scores": refiner._v2_heuristic_scores("detailed background"),
                "wanted_axes": {"details": 3},
                "score": 6.0,
                "liked_count": 4,
            },
            "female character": {
                "text": "female character",
                "primary": "subject",
                "effective_category_scores": refiner._v2_heuristic_scores("female character"),
                "wanted_axes": {"details": 3},
                "score": 6.0,
                "liked_count": 4,
            },
            "tiny smoke curls": {
                "text": "tiny smoke curls",
                "primary": "details",
                "effective_category_scores": refiner._v2_heuristic_scores("tiny smoke curls"),
                "wanted_axes": {"details": 3},
                "score": 4.0,
                "liked_count": 2,
            },
        }
    }
    profile = normalize_refiner_v2_rating("Missing details")
    feedback = refiner._v2_axis_feedback(profile, None)

    repaired, _, candidates = refiner._v2_repair_prompt_for_missing_axes(
        "person smoking",
        refiner._v2_classify_phrases(None, prompt_items(refiner, ["person smoking"]), global_state),
        global_state,
        None,
        feedback,
    )

    assert "tiny smoke curls" in repaired
    assert "white tights" not in repaired
    assert "detailed background" not in repaired
    assert "female character" not in repaired
    assert all(candidate["text"] != "white tights" for candidate in candidates)


def test_scene_builder_user_category_overrides_refiner_classification():
    refiner = FunPackVideoRefinerV2()
    scene_db = {
        "universal_memory": {
            "running pose": {
                "text": "running pose",
                "category": "appearance",
                "category_source": "user",
                "category_locked": True,
            }
        }
    }

    phrase = refiner._v2_classify_phrases(
        None,
        [{"text": "running pose", "tokens": refiner._v2_phrase_words("running pose")}],
        {"phrase_memory": {}},
        scene_db=scene_db,
    )[0]

    assert phrase["primary"] == "appearance"
    assert phrase["source"] == "scene_builder_user"
    assert phrase["scene_category_locked"] is True


def test_scene_builder_locked_category_is_not_trained_by_refiner():
    refiner = FunPackVideoRefinerV2()
    scene_db = {
        "universal_memory": {
            "running pose": {
                "text": "running pose",
                "category": "appearance",
                "category_source": "user",
                "category_locked": True,
            }
        }
    }
    global_state = {"phrase_memory": {}}
    phrase = refiner._v2_classify_phrases(
        None,
        [{"text": "running pose", "tokens": refiner._v2_phrase_words("running pose")}],
        global_state,
        scene_db=scene_db,
    )[0]

    refiner._v2_update_phrase_memory(
        global_state,
        {"prompt": "running pose", "phrases": [phrase]},
        normalize_refiner_v2_rating("Perfect"),
        1,
        refiner._v2_axis_feedback(normalize_refiner_v2_rating("Perfect"), None),
    )

    entry = global_state["phrase_memory"]["running pose"]
    assert entry["primary"] == "appearance"
    assert entry["category_locked"] is True
    assert entry["category_evidence_count"] == 0
    assert entry["liked_count"] == 0


def test_scene_builder_wildcard_cleanup_runs_after_refiner_processing():
    refiner = FunPackVideoRefinerV2()
    scene_db = {
        "universal_memory": {
            "red dress": {"text": "red dress", "category": "appearance", "wildcard": True},
            "blue dress": {"text": "blue dress", "category": "appearance", "wildcard": True},
        }
    }

    prompt, status = refiner._v2_resolve_scene_builder_wildcards(
        "person walking, red dress, blue dress",
        scene_db,
    )

    assert prompt == "person walking, red dress"
    assert "removed 1 duplicate" in status


def test_lucky_composition_gates_unrelated_old_action_by_current_intent():
    refiner = FunPackVideoRefinerV2()
    global_state = {
        "phrase_memory": {
            "dancing in rain": {
                "text": "dancing in rain",
                "primary": "action",
                "effective_category_scores": refiner._v2_heuristic_scores("dancing in rain"),
                "score": 6.0,
                "liked_count": 6,
                "wanted_axes": {"action": 4},
            },
            "walking on beach": {
                "text": "walking on beach",
                "primary": "action",
                "effective_category_scores": refiner._v2_heuristic_scores("walking on beach"),
                "score": 2.0,
                "liked_count": 1,
                "wanted_axes": {"action": 1},
            },
        }
    }

    lucky, status = refiner._v2_compose_lucky_prompt(
        "walking on beach",
        prompt_phrases(refiner, "walking on beach", global_state),
        global_state,
        intent_prompt="walking on beach",
    )

    assert "walking on beach" in lucky
    assert "dancing in rain" not in lucky
    assert "Lucky: on" in status


def test_perfect_repair_does_not_inject_after_scene_change():
    refiner = FunPackVideoRefinerV2()
    slot = {
        "perfect_repairs": {
            "dancing in rain": {"text": "dancing in rain", "axes": {"action": 1}, "count": 3}
        }
    }

    repaired, status, adjustments = refiner._v2_apply_perfect_repair_phrases(
        "walking on beach",
        slot,
        intent_prompt="walking on beach",
        intent_phrases=prompt_phrases(refiner, "walking on beach"),
    )

    assert repaired == "walking on beach"
    assert adjustments == []
    assert "already represented" in status


def test_intent_alignment_learns_missing_original_intent_from_enhancer_variant():
    refiner = FunPackVideoRefinerV2()
    global_state = {"phrase_memory": {}, "intent_alignment_memory": {}}
    intent_prompt = "woman walking through neon rain"
    enhanced_prompt = "woman smiling, cinematic studio portrait"
    profile = normalize_refiner_v2_rating("Missing action")
    feedback = refiner._v2_axis_feedback(profile, None)

    status = refiner._v2_update_intent_alignment_memory(
        global_state,
        {
            "prompt": enhanced_prompt,
            "phrases": prompt_phrases(refiner, enhanced_prompt, global_state),
            "intent_prompt": intent_prompt,
            "intent_phrases": prompt_phrases(refiner, intent_prompt, global_state),
        },
        profile,
        1,
        feedback,
    )

    slot = next(iter(global_state["intent_alignment_memory"].values()))
    missing = slot["missing_intent_phrases"]["woman walking through neon rain"]
    variant = next(iter(slot["variants"].values()))
    assert "Intent alignment learned" in status
    assert missing["score"] > 0.5
    assert missing["missing_count"] == 1
    assert variant["missing_intent_count"] == 1


def test_intent_alignment_restores_learned_missing_original_phrase():
    refiner = FunPackVideoRefinerV2()
    global_state = {"phrase_memory": {}, "intent_alignment_memory": {}}
    intent_prompt = "woman walking through neon rain"
    profile = normalize_refiner_v2_rating("Missing action")
    feedback = refiner._v2_axis_feedback(profile, None)
    refiner._v2_update_intent_alignment_memory(
        global_state,
        {
            "prompt": "woman smiling, cinematic studio portrait",
            "phrases": prompt_phrases(refiner, "woman smiling, cinematic studio portrait", global_state),
            "intent_prompt": intent_prompt,
            "intent_phrases": prompt_phrases(refiner, intent_prompt, global_state),
        },
        profile,
        1,
        feedback,
    )

    aligned, status, adjustments = refiner._v2_apply_intent_alignment_memory(
        "woman smiling, moody closeup",
        prompt_phrases(refiner, "woman smiling, moody closeup", global_state),
        intent_prompt,
        prompt_phrases(refiner, intent_prompt, global_state),
        global_state,
    )

    assert "woman walking through neon rain" in aligned
    assert "restored 1 original phrase" in status
    assert adjustments == [
        {"text": "woman walking through neon rain", "source": "intent_missing", "action": "added"}
    ]


def test_intent_alignment_removes_rejected_enhancer_only_extra():
    refiner = FunPackVideoRefinerV2()
    global_state = {"phrase_memory": {}, "intent_alignment_memory": {}}
    intent_prompt = "woman walking through neon rain"
    enhanced_prompt = "woman walking through neon rain, white tights"
    profile = normalize_refiner_v2_rating("Wrong appearance")
    feedback = refiner._v2_axis_feedback(profile, None)
    refiner._v2_update_intent_alignment_memory(
        global_state,
        {
            "prompt": enhanced_prompt,
            "phrases": prompt_phrases(refiner, enhanced_prompt, global_state),
            "intent_prompt": intent_prompt,
            "intent_phrases": prompt_phrases(refiner, intent_prompt, global_state),
        },
        profile,
        1,
        feedback,
    )

    aligned, status, adjustments = refiner._v2_apply_intent_alignment_memory(
        enhanced_prompt,
        prompt_phrases(refiner, enhanced_prompt, global_state),
        intent_prompt,
        prompt_phrases(refiner, intent_prompt, global_state),
        global_state,
    )

    assert "white tights" not in aligned
    assert "woman walking through neon rain" in aligned
    assert "removed 1 rejected enhancer-only phrase" in status
    assert adjustments == [
        {"text": "white tights", "source": "enhancer_extra", "action": "removed"}
    ]


def test_intent_alignment_stores_pairs_and_bad_tokens_to_omit():
    refiner = FunPackVideoRefinerV2()
    global_state = {"phrase_memory": {}, "intent_alignment_memory": {}}
    intent_prompt = "woman walking through neon rain"
    enhanced_prompt = "woman walking through neon rain, white tights"
    profile = normalize_refiner_v2_rating("Wrong appearance")
    feedback = refiner._v2_axis_feedback(profile, None)

    refiner._v2_update_intent_alignment_memory(
        global_state,
        {
            "prompt": enhanced_prompt,
            "phrases": prompt_phrases(refiner, enhanced_prompt, global_state),
            "intent_prompt": intent_prompt,
            "intent_phrases": prompt_phrases(refiner, intent_prompt, global_state),
        },
        profile,
        1,
        feedback,
    )

    slot = next(iter(global_state["intent_alignment_memory"].values()))
    pair = next(iter(slot["intent_enhance_pairs"].values()))
    assert pair["intent_prompt"] == intent_prompt
    assert pair["positive_prompt"] == enhanced_prompt
    assert "woman" in slot["provided_tokens"]
    assert slot["provided_tokens"]["woman"]["omit"] is False
    assert slot["provided_tokens"]["white"]["omit"] is True
    assert slot["provided_tokens"]["tights"]["omit"] is True
    assert slot["provided_tokens"]["white tights"]["kind"] == "pair"
    assert slot["provided_tokens"]["white tights"]["omit"] is True
    assert set(slot["bad_tokens"]) >= {"white", "tights", "white tights"}


def test_intent_alignment_omits_bad_token_in_new_enhancer_phrase():
    refiner = FunPackVideoRefinerV2()
    global_state = {"phrase_memory": {}, "intent_alignment_memory": {}}
    intent_prompt = "woman walking through neon rain"
    profile = normalize_refiner_v2_rating("Wrong appearance")
    feedback = refiner._v2_axis_feedback(profile, None)
    refiner._v2_update_intent_alignment_memory(
        global_state,
        {
            "prompt": "woman walking through neon rain, white tights",
            "phrases": prompt_phrases(refiner, "woman walking through neon rain, white tights", global_state),
            "intent_prompt": intent_prompt,
            "intent_phrases": prompt_phrases(refiner, intent_prompt, global_state),
        },
        profile,
        1,
        feedback,
    )

    aligned, _, adjustments = refiner._v2_apply_intent_alignment_memory(
        "woman walking through neon rain, red tights",
        prompt_phrases(refiner, "woman walking through neon rain, red tights", global_state),
        intent_prompt,
        prompt_phrases(refiner, intent_prompt, global_state),
        global_state,
    )

    assert "red tights" not in aligned
    assert "woman walking through neon rain" in aligned
    assert adjustments == [
        {"text": "red tights", "source": "enhancer_extra", "action": "removed"}
    ]


def test_repeated_intent_family_phrase_becomes_conservative_repair_candidate():
    refiner = FunPackVideoRefinerV2()
    global_state = {
        "phrase_memory": {},
        "intent_family_memory": {},
        "perfect_anchors": {},
        "variant_evidence": {},
        "intent_preference_phrases": {},
        "conditioning_deltas": {},
    }
    intent_prompt = "yellow car riding down the road"
    enhanced_prompt = "yellow car in a desert"
    profile = normalize_refiner_v2_rating("Missing action")
    feedback = refiner._v2_axis_feedback(profile, None)

    for iteration in range(2):
        refiner._v2_update_intent_family_memory(
            global_state,
            {
                "prompt": enhanced_prompt,
                "phrases": prompt_phrases(refiner, enhanced_prompt, global_state),
                "intent_prompt": intent_prompt,
                "intent_phrases": prompt_phrases(refiner, intent_prompt, global_state),
            },
            profile,
            iteration + 1,
            feedback,
        )

    current_intent = "yellow car down the road"
    _, family_slot, _ = refiner._v2_intent_family_slot(global_state, current_intent, create=False)
    repaired, status, candidates = refiner._v2_repair_prompt_for_missing_axes(
        enhanced_prompt,
        prompt_phrases(refiner, enhanced_prompt, global_state),
        global_state,
        None,
        feedback,
        intent_phrases=prompt_phrases(refiner, current_intent, global_state),
        intent_family_slot=family_slot,
    )

    assert "yellow car riding down the road" in repaired
    assert "intent_preference" in status
    assert any(candidate["source"] == "intent_preference" for candidate in candidates)


def test_repeated_intent_preference_does_not_repair_unrelated_family():
    refiner = FunPackVideoRefinerV2()
    global_state = {
        "phrase_memory": {},
        "intent_family_memory": {},
        "perfect_anchors": {},
        "variant_evidence": {},
        "intent_preference_phrases": {},
        "conditioning_deltas": {},
    }
    yellow_intent = "yellow car riding down the road"
    dragon_intent = "dragon sleeping in a cave"
    profile = normalize_refiner_v2_rating("Missing action")
    feedback = refiner._v2_axis_feedback(profile, None)

    for iteration in range(2):
        refiner._v2_update_intent_family_memory(
            global_state,
            {
                "prompt": "yellow car in a desert",
                "phrases": prompt_phrases(refiner, "yellow car in a desert", global_state),
                "intent_prompt": yellow_intent,
                "intent_phrases": prompt_phrases(refiner, yellow_intent, global_state),
            },
            profile,
            iteration + 1,
            feedback,
        )

    _, dragon_slot, _ = refiner._v2_intent_family_slot(global_state, dragon_intent, create=True)
    repaired, _, candidates = refiner._v2_repair_prompt_for_missing_axes(
        "dragon in a cave",
        prompt_phrases(refiner, "dragon in a cave", global_state),
        global_state,
        None,
        feedback,
        intent_phrases=prompt_phrases(refiner, dragon_intent, global_state),
        intent_family_slot=dragon_slot,
    )

    assert "yellow car riding down the road" not in repaired
    assert all(candidate["text"] != "yellow car riding down the road" for candidate in candidates)


def test_active_repair_axes_persist_until_perfect():
    refiner = FunPackVideoRefinerV2()
    global_state = {"active_repair_axes": []}

    missing_action = normalize_refiner_v2_rating("Missing action")
    action_feedback = refiner._v2_axis_feedback(missing_action, None)
    repair_feedback, status = refiner._v2_active_repair_feedback(global_state, action_feedback, missing_action)
    assert repair_feedback["missing_axes"] == ["action"]
    assert global_state["active_repair_axes"] == ["action"]
    assert "active until Perfect" in status

    missing_quality = normalize_refiner_v2_rating("Missing quality")
    quality_feedback = refiner._v2_axis_feedback(missing_quality, ["action"])
    repair_feedback, _ = refiner._v2_active_repair_feedback(global_state, quality_feedback, missing_quality)
    assert repair_feedback["missing_axes"] == ["action", "quality"]
    assert global_state["active_repair_axes"] == ["action", "quality"]

    perfect = normalize_refiner_v2_rating("Perfect")
    perfect_feedback = refiner._v2_axis_feedback(perfect, ["action", "quality"])
    repair_feedback, status = refiner._v2_active_repair_feedback(global_state, perfect_feedback, perfect)
    assert repair_feedback["missing_axes"] == []
    assert global_state["active_repair_axes"] == []
    assert "cleared by Perfect" in status


def test_perfect_rating_preserves_successful_repair_phrases():
    refiner = FunPackVideoRefinerV2()
    global_state = {
        "phrase_memory": {},
        "intent_family_memory": {},
        "perfect_anchors": {},
        "variant_evidence": {},
        "intent_preference_phrases": {},
        "conditioning_deltas": {},
    }
    intent_prompt = "person smoking"
    profile = normalize_refiner_v2_rating("Perfect")
    feedback = refiner._v2_axis_feedback(profile, None)
    source = tensor_to_serializable(torch.zeros(1, 3, 2))
    refined = tensor_to_serializable(torch.ones(1, 3, 2))

    refiner._v2_update_intent_family_memory(
        global_state,
        {
            "prompt": "person smoking",
            "encoded_prompt": "person smoking, tiny smoke curls",
            "phrases": prompt_phrases(refiner, "person smoking", global_state),
            "intent_prompt": intent_prompt,
            "intent_phrases": prompt_phrases(refiner, intent_prompt, global_state),
            "source_conditioning": source,
            "conditioning": refined,
            "repair_candidates": [{"text": "tiny smoke curls", "axes": ["details"], "score": 2.0, "source": "memory"}],
        },
        profile,
        1,
        feedback,
    )
    _, slot, _ = refiner._v2_intent_family_slot(global_state, intent_prompt, create=False)
    repaired, status, adjustments = refiner._v2_apply_perfect_repair_phrases("person smoking", slot)

    assert "tiny smoke curls" in repaired
    assert "Perfect-proven" in status
    assert adjustments == [{"text": "tiny smoke curls", "source": "perfect_repair", "action": "added"}]


def test_prompt_repair_falls_back_to_any_axis_candidates_when_requested_axis_is_empty():
    refiner = FunPackVideoRefinerV2()
    global_state = {
        "phrase_memory": {
            "portrait quality": {
                "text": "portrait quality",
                "primary": "quality",
                "effective_category_scores": refiner._v2_heuristic_scores("portrait quality"),
                "wanted_axes": {"quality": 3},
                "score": 4.0,
                "liked_count": 2,
                "category_evidence_count": 2,
            }
        }
    }
    profile = normalize_refiner_v2_rating("Missing action")
    feedback = refiner._v2_axis_feedback(profile, None)

    repaired, status, candidates = refiner._v2_repair_prompt_for_missing_axes(
        "static portrait",
        prompt_phrases(refiner, "static portrait", global_state),
        global_state,
        None,
        feedback,
    )

    assert "portrait quality" in repaired
    assert "showing other-axis repair candidates" in status
    assert candidates[0]["text"] == "portrait quality"
    assert candidates[0]["source"] == "memory"


def test_intent_family_perfect_anchor_keeps_loved_variant():
    refiner = FunPackVideoRefinerV2()
    global_state = {
        "phrase_memory": {},
        "intent_family_memory": {},
        "perfect_anchors": {},
        "variant_evidence": {},
        "intent_preference_phrases": {},
        "conditioning_deltas": {},
    }
    intent_prompt = "yellow car riding down the road"
    profile = normalize_refiner_v2_rating("Perfect")
    feedback = refiner._v2_axis_feedback(profile, None)
    source = tensor_to_serializable(torch.zeros(1, 3, 2))
    first = tensor_to_serializable(torch.ones(1, 3, 2))
    second = tensor_to_serializable(torch.ones(1, 3, 2) * 2.0)

    refiner._v2_update_intent_family_memory(
        global_state,
        {
            "prompt": "yellow car riding down the road",
            "encoded_prompt": "yellow car riding down the road",
            "phrases": prompt_phrases(refiner, "yellow car riding down the road", global_state),
            "intent_prompt": intent_prompt,
            "intent_phrases": prompt_phrases(refiner, intent_prompt, global_state),
            "source_conditioning": source,
            "conditioning": first,
        },
        profile,
        1,
        feedback,
    )
    refiner._v2_update_intent_family_memory(
        global_state,
        {
            "prompt": "yellow car riding down the road, camera focused on wheels",
            "encoded_prompt": "yellow car riding down the road, camera focused on wheels",
            "phrases": prompt_phrases(refiner, "yellow car riding down the road, camera focused on wheels", global_state),
            "intent_prompt": intent_prompt,
            "intent_phrases": prompt_phrases(refiner, intent_prompt, global_state),
            "source_conditioning": source,
            "conditioning": second,
        },
        profile,
        2,
        feedback,
    )

    _, slot, _ = refiner._v2_intent_family_slot(global_state, intent_prompt, create=False)
    assert slot["perfect_anchors"]["base"]["positive_prompt"] == "yellow car riding down the road"
    assert len(slot["loved_variants"]) == 1
    assert global_state["perfect_anchors"][slot["family_key"]]["base"]["positive_prompt"] == "yellow car riding down the road"
    assert slot["conditioning_deltas"]["positive"]["count"] == 2


def test_pre_perfect_missing_intent_learning_is_conservative():
    refiner = FunPackVideoRefinerV2()
    global_state = {"phrase_memory": {}, "intent_alignment_memory": {}, "intent_family_memory": {}}
    intent_prompt = "woman walking through neon rain"
    profile = normalize_refiner_v2_rating("Missing action")
    feedback = refiner._v2_axis_feedback(profile, None)

    refiner._v2_update_intent_family_memory(
        global_state,
        {
            "prompt": "cinematic studio portrait",
            "phrases": prompt_phrases(refiner, "cinematic studio portrait", global_state),
            "intent_prompt": intent_prompt,
            "intent_phrases": prompt_phrases(refiner, intent_prompt, global_state),
        },
        profile,
        1,
        feedback,
    )
    refiner._v2_update_intent_alignment_memory(
        global_state,
        {
            "prompt": "cinematic studio portrait",
            "phrases": prompt_phrases(refiner, "cinematic studio portrait", global_state),
            "intent_prompt": intent_prompt,
            "intent_phrases": prompt_phrases(refiner, intent_prompt, global_state),
        },
        profile,
        1,
        feedback,
    )

    slot = next(iter(global_state["intent_alignment_memory"].values()))
    missing = slot["missing_intent_phrases"]["woman walking through neon rain"]
    assert 0.5 <= missing["score"] < 0.8


def test_negative_repair_never_adds_current_intent_phrase():
    refiner = FunPackVideoRefinerV2()
    global_state = {"phrase_memory": {}, "negative_prompt_memory": {}, "intent_family_memory": {}}
    intent_prompt = "Nicole is shooting"
    enhanced_prompt = "Nicole from Zenless Zone Zero holding a gun and shooting"
    profile = normalize_refiner_v2_rating("Wrong action")
    feedback = refiner._v2_axis_feedback(profile, None)
    intent_phrases = prompt_phrases(refiner, intent_prompt, global_state)
    phrases = prompt_phrases(refiner, enhanced_prompt, global_state)

    refiner._v2_update_intent_family_memory(
        global_state,
        {
            "prompt": enhanced_prompt,
            "phrases": phrases,
            "intent_prompt": intent_prompt,
            "intent_phrases": intent_phrases,
        },
        profile,
        1,
        feedback,
    )
    status = refiner._v2_update_negative_prompt_memory(
        global_state,
        {
            "prompt": enhanced_prompt,
            "phrases": phrases,
            "intent_prompt": intent_prompt,
            "intent_phrases": intent_phrases,
        },
        profile,
        feedback,
    )
    _, family_slot, _ = refiner._v2_intent_family_slot(global_state, intent_prompt, create=False)
    repaired, repair_status = refiner._v2_repair_negative_prompt(
        "",
        global_state,
        feedback,
        current_prompt=enhanced_prompt,
        intent_prompt=intent_prompt,
        intent_phrases=intent_phrases,
        intent_family_slot=family_slot,
    )

    assert repaired == ""
    assert "Skipped 1 intent-locked" in status
    assert "intent/current" in repair_status or "no stored poor-rated tags" in repair_status


def test_negative_repair_blocks_requested_action_overlap_without_exact_phrase():
    refiner = FunPackVideoRefinerV2()
    global_state = {
        "negative_prompt_memory": {
            "tags": {
                "holding a gun and shooting": {
                    "text": "holding a gun and shooting",
                    "count": 4,
                    "axes": {"action": 2},
                    "last_seen_iter": 1,
                }
            }
        }
    }
    profile = normalize_refiner_v2_rating("Missing action")
    feedback = refiner._v2_axis_feedback(profile, None)

    repaired, status = refiner._v2_repair_negative_prompt(
        "",
        global_state,
        feedback,
        current_prompt="Nicole is shooting",
    )

    assert repaired == ""
    assert "intent/current 1" in status


def test_negative_memory_skips_partial_requested_action_with_explicit_intent():
    refiner = FunPackVideoRefinerV2()
    global_state = {"phrase_memory": {}, "negative_prompt_memory": {}, "intent_family_memory": {}}
    profile = normalize_refiner_v2_rating("Wrong action")
    feedback = refiner._v2_axis_feedback(profile, None)
    intent_prompt = "Nicole is shooting"

    status = refiner._v2_update_negative_prompt_memory(
        global_state,
        {
            "prompt": "Nicole from Zenless Zone Zero holding a gun and shooting",
            "phrases": prompt_phrases(refiner, "holding a gun and shooting", global_state),
            "intent_prompt": intent_prompt,
            "intent_phrases": prompt_phrases(refiner, intent_prompt, global_state),
        },
        profile,
        feedback,
    )

    assert global_state["negative_prompt_memory"]["tags"] == {}
    assert "Skipped 1 intent-locked" in status


def test_one_word_intent_matches_expanded_enhancer_phrase():
    refiner = FunPackVideoRefinerV2()
    global_state = {"phrase_memory": {}, "intent_alignment_memory": {}, "intent_family_memory": {}}
    intent_prompt = "shooting"
    enhanced_prompt = "Nicole from Zenless Zone Zero holding a gun and shooting"
    profile = normalize_refiner_v2_rating("Wrong action")
    feedback = refiner._v2_axis_feedback(profile, None)

    refiner._v2_update_intent_alignment_memory(
        global_state,
        {
            "prompt": enhanced_prompt,
            "phrases": prompt_phrases(refiner, enhanced_prompt, global_state),
            "intent_prompt": intent_prompt,
            "intent_phrases": prompt_phrases(refiner, intent_prompt, global_state),
        },
        profile,
        1,
        feedback,
    )

    slot = next(iter(global_state["intent_alignment_memory"].values()))
    assert slot["extra_positive_phrases"] == {}
    assert slot["bad_tokens"] == {}
    assert refiner._v2_phrase_texts_match(enhanced_prompt, intent_prompt)


def test_one_word_intent_blocks_expanded_action_negative_repair():
    refiner = FunPackVideoRefinerV2()
    global_state = {
        "negative_prompt_memory": {
            "tags": {
                "holding a gun and shooting": {
                    "text": "holding a gun and shooting",
                    "count": 4,
                    "axes": {"action": 2},
                    "last_seen_iter": 1,
                }
            }
        }
    }
    profile = normalize_refiner_v2_rating("Missing action")
    feedback = refiner._v2_axis_feedback(profile, None)

    repaired, status = refiner._v2_repair_negative_prompt(
        "",
        global_state,
        feedback,
        current_prompt="Nicole holding a gun and shooting",
        intent_prompt="shooting",
        intent_phrases=prompt_phrases(refiner, "shooting", global_state),
    )

    assert repaired == ""
    assert "intent/current 1" in status


def test_short_semantic_intent_locks_enhancer_expansion_without_shared_word():
    refiner = FunPackVideoRefinerV2()
    global_state = {"phrase_memory": {}, "intent_alignment_memory": {}, "intent_family_memory": {}}
    intent_prompt = "defenstration"
    enhanced_prompt = "a human is thrown out of window"
    intent_phrases = prompt_phrases(refiner, intent_prompt, global_state)
    phrases = prompt_phrases(refiner, enhanced_prompt, global_state)
    refiner._v2_mark_semantic_intent_locks(
        FakeClip(),
        phrases,
        intent_prompt,
        intent_phrases,
        encode_cache={},
    )
    profile = normalize_refiner_v2_rating("Wrong action")
    feedback = refiner._v2_axis_feedback(profile, None)

    refiner._v2_update_intent_alignment_memory(
        global_state,
        {
            "prompt": enhanced_prompt,
            "phrases": phrases,
            "intent_prompt": intent_prompt,
            "intent_phrases": intent_phrases,
        },
        profile,
        1,
        feedback,
    )

    slot = next(iter(global_state["intent_alignment_memory"].values()))
    assert phrases[0]["semantic_intent_locked"] is True
    assert slot["missing_intent_phrases"] == {}
    assert slot["extra_positive_phrases"] == {}
    assert slot["bad_tokens"] == {}


def test_short_semantic_intent_blocks_expansion_from_negative_memory():
    refiner = FunPackVideoRefinerV2()
    global_state = {"phrase_memory": {}, "negative_prompt_memory": {}, "intent_family_memory": {}}
    intent_prompt = "defenstration"
    enhanced_prompt = "a human is thrown out of window"
    intent_phrases = prompt_phrases(refiner, intent_prompt, global_state)
    phrases = prompt_phrases(refiner, enhanced_prompt, global_state)
    refiner._v2_mark_semantic_intent_locks(
        FakeClip(),
        phrases,
        intent_prompt,
        intent_phrases,
        encode_cache={},
    )
    profile = normalize_refiner_v2_rating("Wrong action")
    feedback = refiner._v2_axis_feedback(profile, None)

    status = refiner._v2_update_negative_prompt_memory(
        global_state,
        {
            "prompt": enhanced_prompt,
            "phrases": phrases,
            "intent_prompt": intent_prompt,
            "intent_phrases": intent_phrases,
        },
        profile,
        feedback,
    )

    assert global_state["negative_prompt_memory"]["tags"] == {}
    assert "Skipped 1 intent-locked" in status


def test_rejected_repair_candidate_can_be_penalized_as_negative_memory():
    refiner = FunPackVideoRefinerV2()
    global_state = {"phrase_memory": {}, "negative_prompt_memory": {}, "intent_family_memory": {}}
    profile = normalize_refiner_v2_rating("Wrong details")
    feedback = refiner._v2_axis_feedback(profile, None)

    refiner._v2_update_negative_prompt_memory(
        global_state,
        {
            "prompt": "person smoking",
            "phrases": prompt_phrases(refiner, "person smoking", global_state),
            "intent_prompt": "person smoking",
            "intent_phrases": prompt_phrases(refiner, "person smoking", global_state),
            "repair_candidates": [{"text": "tiny smoke curls", "axes": ["details"], "score": 2.0, "source": "memory"}],
        },
        profile,
        feedback,
    )

    assert "tiny smoke curls" in global_state["negative_prompt_memory"]["tags"]
    assert global_state["negative_prompt_memory"]["tags"]["tiny smoke curls"]["source"] == "repair_candidate"


def test_intent_family_delta_ignores_incompatible_shapes_and_caps_strength():
    refiner = FunPackVideoRefinerV2()
    conditioning = torch.ones(1, 3, 2)
    slot = {
        "conditioning_deltas": {
            "positive": {
                "count": 1,
                "delta": tensor_to_serializable(torch.ones(1, 4, 2)),
            }
        }
    }

    unchanged, status = refiner._v2_apply_intent_family_delta(conditioning, slot, 0.05)
    assert torch.equal(unchanged, conditioning)
    assert status == "intent-family idle"

    slot["conditioning_deltas"]["positive"]["delta"] = tensor_to_serializable(torch.ones(1, 3, 2) * 100.0)
    changed, status = refiner._v2_apply_intent_family_delta(conditioning, slot, 0.05)
    assert status.startswith("intent-family positive delta")
    assert torch.max(torch.abs(changed - conditioning)).item() < 1.0


def test_lucky_skips_appearance_memory_unless_prompt_mentions_it():
    refiner = FunPackVideoRefinerV2()
    global_state = {"phrase_memory": {}, "preferred_context_memory": {}}
    phrases = refiner._v2_classify_phrases(
        None,
        prompt_items(refiner, ["white tights", "walking"]),
        global_state,
    )
    profile = normalize_refiner_v2_rating("Perfect")
    feedback = refiner._v2_axis_feedback(profile, None)
    refiner._v2_update_phrase_memory(
        global_state,
        {"prompt": "white tights walking", "phrases": phrases},
        profile,
        1,
        feedback,
    )

    lucky_prompt, _ = refiner._v2_compose_lucky_prompt("", [], global_state)
    explicit_prompt, _ = refiner._v2_compose_lucky_prompt(
        "white tights",
        refiner._v2_classify_phrases(None, prompt_items(refiner, ["white tights"]), global_state),
        global_state,
    )

    assert "walking" in lucky_prompt
    assert "white tights" not in lucky_prompt
    assert "tights" not in lucky_prompt
    assert "white tights" in explicit_prompt


def test_wrong_appearance_suppresses_only_auto_inserted_appearance_memory():
    refiner = FunPackVideoRefinerV2()
    global_state = {"phrase_memory": {}, "preferred_context_memory": {}}
    phrases = refiner._v2_classify_phrases(None, prompt_items(refiner, ["walking"]), global_state)
    profile = normalize_refiner_v2_rating("Wrong appearance")
    feedback = refiner._v2_axis_feedback(profile, None)

    refiner._v2_update_phrase_memory(
        global_state,
        {
            "prompt": "walking",
            "encoded_prompt": "walking, white tights",
            "phrases": phrases,
            "repair_candidates": [{"text": "white tights", "axes": ["details"], "score": 3.0, "source": "memory"}],
        },
        profile,
        1,
        feedback,
    )

    appearance = global_state["phrase_memory"]["white tights"]
    motion = global_state["phrase_memory"]["walking"]
    assert appearance["auto_inject_suppressed"] is True
    assert appearance["wrong_appearance_count"] >= 1
    assert motion.get("wrong_appearance_count", 0) == 0
    assert motion["category_evidence_count"] == 0


def test_repeated_wrong_appearance_keeps_entry_out_of_lucky():
    refiner = FunPackVideoRefinerV2()
    global_state = {"phrase_memory": {}, "preferred_context_memory": {}}
    phrases = refiner._v2_classify_phrases(None, prompt_items(refiner, ["walking"]), global_state)
    profile = normalize_refiner_v2_rating("Wrong appearance")
    feedback = refiner._v2_axis_feedback(profile, None)

    for iteration in range(2):
        refiner._v2_update_phrase_memory(
            global_state,
            {
                "prompt": "walking",
                "encoded_prompt": "walking, white tights",
                "phrases": phrases,
                "repair_candidates": [{"text": "white tights", "axes": ["details"], "score": 3.0, "source": "memory"}],
            },
            profile,
            iteration + 1,
            feedback,
        )

    global_state["phrase_memory"]["white tights"]["score"] = 6.0
    global_state["phrase_memory"]["white tights"]["liked_count"] = 8
    lucky_prompt, _ = refiner._v2_compose_lucky_prompt("", [], global_state)

    assert global_state["phrase_memory"]["white tights"]["auto_inject_blocked_count"] >= 2
    assert "white tights" not in lucky_prompt


def test_forget_rating_skips_category_weight_learning():
    refiner = FunPackVideoRefinerV2()
    global_state = {"phrase_memory": {}}
    phrase = classified_phrase(refiner, "running")
    profile = normalize_refiner_v2_rating("-Just forget it-")

    status = refiner._v2_update_phrase_memory(
        global_state,
        {"prompt": "running", "phrases": [phrase]},
        profile,
        1,
        refiner._v2_axis_feedback(profile, None),
    )

    assert status == "Lucky memory: no learning update."
    assert global_state["phrase_memory"] == {}


def test_missing_details_trains_micro_movement_as_detail_without_losing_action_context():
    refiner = FunPackVideoRefinerV2()

    entry, global_state, _ = train_phrase(
        refiner,
        "patting head",
        "Missing details",
        iterations=5,
    )

    assert entry["category_weights"]["details"] > entry["category_weights"]["action"]
    assert entry["category_weights"]["action"] > 0.0
    learned = refiner._v2_classify_phrases(
        None,
        [{"text": "patting head", "tokens": ["patting", "head"]}],
        global_state,
    )[0]
    assert learned["primary"] == "details"
    assert "details" in refiner._v2_axes_for_scores(learned["effective_category_scores"])


def test_missing_quality_trains_quality_while_preserving_satisfied_motion():
    refiner = FunPackVideoRefinerV2()

    entry, _, feedback = train_phrase(
        refiner,
        "running",
        "Missing quality",
        previous_missing_axes=["details", "action"],
    )

    assert feedback["missing_axes"] == ["quality"]
    assert feedback["resolved_axes"] == ["details", "action"]
    assert entry["category_weights"]["quality"] > 0.0
    assert entry["category_weights"]["action"] > 0.0
    assert entry["resolved_axes"]["action"] == 1


def test_paired_missing_ratings_train_both_requested_axes():
    refiner = FunPackVideoRefinerV2()

    details_action, _, _ = train_phrase(refiner, "hugging", "Missing details + action")
    details_quality, _, _ = train_phrase(refiner, "hands in frame", "Missing details + quality")
    action_quality, _, _ = train_phrase(refiner, "smoking", "Missing action + quality")

    assert details_action["category_weights"]["details"] > 0.0
    assert details_action["category_weights"]["action"] > 0.0
    assert details_quality["category_weights"]["details"] > 0.0
    assert details_quality["category_weights"]["quality"] > 0.0
    assert action_quality["category_weights"]["action"] > 0.0
    assert action_quality["category_weights"]["quality"] > 0.0


def test_repeated_user_feedback_can_override_initial_machine_category():
    refiner = FunPackVideoRefinerV2()
    machine_primary, _, _ = primary_category("detailed background")
    assert machine_primary == "environment"

    entry, global_state, _ = train_phrase(
        refiner,
        "detailed background",
        "Missing action",
        iterations=9,
    )
    learned = refiner._v2_classify_phrases(
        None,
        [{"text": "detailed background", "tokens": ["detailed", "background"]}],
        global_state,
    )[0]

    assert entry["category_weights"]["action"] > entry["category_weights"]["environment"]
    assert learned["machine_primary"] == "environment"
    assert learned["primary"] == "action"
    assert learned["source"] == "rating_weighted"


def test_old_phrase_memory_entries_receive_category_weight_defaults():
    refiner = FunPackVideoRefinerV2()
    memory = {
        "running": {
            "text": "running",
            "primary": "action",
            "category_scores": {"action": 0.7},
        }
    }

    entry = refiner._v2_ensure_phrase_memory_entry(memory, "running")

    assert set(entry["category_weights"]) == set(refiner.CATEGORY_DESCRIPTIONS)
    assert set(entry["clip_heuristic_scores"]) == set(refiner.CATEGORY_DESCRIPTIONS)
    assert set(entry["effective_category_scores"]) == set(refiner.CATEGORY_DESCRIPTIONS)


def test_context_senses_keep_polysemous_token_categories_separate():
    refiner = FunPackVideoRefinerV2()
    global_state = {"phrase_memory": {}}

    train_prompt_context(refiner, ["smoke", "rising", "lips"], "Missing action", global_state, iterations=8)
    train_prompt_context(refiner, ["smoke", "makeup", "eyes"], "Missing details", global_state, iterations=8)

    action_smoke = refiner._v2_classify_phrases(
        None,
        prompt_items(refiner, ["smoke", "rising", "lips"]),
        global_state,
    )[0]
    detail_smoke = refiner._v2_classify_phrases(
        None,
        prompt_items(refiner, ["smoke", "makeup", "eyes"]),
        global_state,
    )[0]

    assert action_smoke["primary"] == "action"
    assert detail_smoke["primary"] == "details"
    assert action_smoke["context_source"] == "exact"
    assert detail_smoke["context_source"] == "exact"
    assert len(global_state["phrase_memory"]["smoke"]["context_senses"]) >= 2


def test_context_senses_are_pruned_by_evidence_and_recency():
    refiner = FunPackVideoRefinerV2()
    entry = {"context_senses": {}}
    for index in range(30):
        entry["context_senses"][f"mid|context-{index}"] = {
            "category_evidence_count": index,
            "occurrence_count": index,
            "last_seen_iter": index,
        }

    refiner._v2_prune_context_senses(entry, limit=24)

    assert len(entry["context_senses"]) == 24
    assert "mid|context-29" in entry["context_senses"]
    assert "mid|context-0" not in entry["context_senses"]


def test_training_diagnostics_explain_learning_state_and_guidance():
    refiner = FunPackVideoRefinerV2()
    phrases = refiner._v2_classify_phrases(None, prompt_items(refiner, ["smoke", "rising"]))
    profile = normalize_refiner_v2_rating("Missing action")
    feedback = refiner._v2_axis_feedback(profile, None)

    first_run_guidance = refiner._v2_training_guidance(
        False,
        profile,
        feedback,
        phrases,
        "Category memory trained: 0 concept unit(s).",
        "LoRA suggestions: no FunPack LoRA stack connected.",
    )
    diagnostics = refiner._v2_category_diagnostics(phrases)

    assert "first V2 run only seeds" in first_run_guidance
    assert "Category diagnostics:" in diagnostics
    assert "smoke:" in diagnostics
    assert "ctx=" in diagnostics


def test_refiner_training_info_uses_readable_sections(tmp_path):
    refiner = FunPackVideoRefinerV2()
    state_path = tmp_path / "state.json"
    refiner._v2_state_path = lambda refinement_key: str(state_path)

    _, _, training_info, _, _, _ = refiner.refine_v2(
        "woman walking through neon rain",
        FakeClip(),
        "Perfect",
        "readable-test",
    )

    assert "\n\nLearning\n" in training_info
    assert "\n\nPrompt Analysis\n" in training_info
    assert "\n\nAdaptation\n" in training_info
    assert "\n\nLoRA\n" in training_info
    assert "Category compact view:\n-" in training_info


def test_refiner_v2_exposes_clip_and_conditioning_as_optional_inputs():
    inputs = FunPackVideoRefinerV2.INPUT_TYPES()

    assert "clip" not in inputs["required"]
    assert "positive_conditioning" not in inputs["required"]
    assert inputs["required"]["mode"][0] == ["Refine", "Learning"]
    assert inputs["required"]["advisor_mode"][0] == ["Off", "Diagnostics", "Repair prompt"]
    assert FunPackVideoRefinerV2.RETURN_NAMES[-1] == "encoded_prompts"
    assert FunPackVideoRefinerV2.RETURN_TYPES[-1] == "STRING"
    assert "clip" in inputs["optional"]
    assert "advisor_clip" in inputs["optional"]
    assert "positive_conditioning" in inputs["optional"]
    assert inputs["optional"]["advisor_thinking"][1]["default"] is True


def test_refiner_v2_advisor_uses_explicit_system_prompt_previous_prompt_thinking_and_image():
    refiner = FunPackVideoRefinerV2()
    image = torch.zeros(1, 8, 8, 3)
    clip = GeneratingClip(
        "DIAGNOSTIC: add clearer smoke motion.\n"
        "REPAIRED_PROMPT: person smoking, smoke trails drifting upward"
    )

    prompt, status, diagnostic, applied = refiner._v2_prompt_advisor(
        clip,
        "Diagnostics",
        "person smoking",
        "person smoking",
        "Missing details",
        {"missing_axes": ["details"], "wrong_axes": []},
        prompt_phrases(refiner, "person smoking"),
        [],
        previous_run={"prompt": "old prompt", "encoded_prompt": "old encoded prompt"},
        image=image,
        thinking=True,
        seed=123,
    )

    advisor_prompt, kwargs = clip.tokenize_calls[0]
    assert prompt == "person smoking"
    assert applied is False
    assert "diagnostics only" in status
    assert diagnostic == "add clearer smoke motion."
    assert "FunPack Refiner V2 Prompt Advisor" in advisor_prompt
    assert "Prompt that caused the rating: old prompt" in advisor_prompt
    assert "Encoded prompt that caused the rating: old encoded prompt" in advisor_prompt
    assert "Source image available to advisor: yes" in advisor_prompt
    assert kwargs["image"] is image
    assert kwargs["thinking"] is True
    assert clip.generate_kwargs["seed"] == 123


def test_refiner_v2_advisor_repair_applies_validated_generated_prompt(tmp_path):
    refiner = FunPackVideoRefinerV2()
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({
        "version": 2,
        "refinement_key": "advisor-repair-test",
        "state_namespace": "clip",
        "global": {
            "total_iterations": 1,
            "avg_reward_ema": 0.0,
            "good_streak": 0,
            "bad_streak": 1,
            "last_rating_label": "Missing action",
            "last_missing_axes": ["action"],
            "phrase_memory": {},
            "axis_conditioning_memory": {},
            "lora_weight_memory": {},
            "preferred_context_memory": {},
            "intent_alignment_memory": {},
            "intent_family_memory": {},
            "perfect_anchors": {},
            "variant_evidence": {},
            "intent_preference_phrases": {},
            "conditioning_deltas": {},
            "active_repair_axes": [],
            "negative_prompt_memory": {},
            "vision_memory": {},
            "loss_history": [],
        },
        "prompt_histories": {},
        "last_run": {
            "prompt": "person smoking",
            "encoded_prompt": "person smoking",
            "source_conditioning": tensor_to_serializable(torch.zeros(1, 4, 3)),
            "conditioning": tensor_to_serializable(torch.zeros(1, 4, 3)),
            "phrases": prompt_phrases(refiner, "person smoking"),
            "rating_label": "Unrated",
            "iteration": 1,
        },
    }), encoding="utf-8")
    refiner._v2_state_path = lambda refinement_key: str(state_path)
    clip = GeneratingClip(
        "DIAGNOSTIC: add visible smoke motion.\n"
        "REPAIRED_PROMPT: person smoking, smoke trails drifting upward"
    )

    _, status, training_info, _, _, encoded_prompts = refiner.refine_v2(
        "person smoking",
        clip,
        "Missing details",
        "advisor-repair-test",
        user_intent_prompt="person smoking",
        advisor_mode="Repair prompt",
        advisor_thinking=True,
    )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert "Advisor: applied generated repair" in status
    assert "Encoded: advisor repaired prompt" in training_info
    assert encoded_prompts == "Positive prompt: person smoking, smoke trails drifting upward\n\nNegative prompt: "
    assert state["last_run"]["encoded_prompt"] == "person smoking, smoke trails drifting upward"
    assert state["last_run"]["advisor"]["applied"] is True


def test_refiner_v2_advisor_uses_separate_advisor_clip_when_connected(tmp_path):
    refiner = FunPackVideoRefinerV2()
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({
        "version": 2,
        "refinement_key": "advisor-clip-test",
        "state_namespace": "clip",
        "global": {
            "total_iterations": 1,
            "avg_reward_ema": 0.0,
            "good_streak": 0,
            "bad_streak": 1,
            "last_rating_label": "Missing action",
            "last_missing_axes": ["action"],
            "phrase_memory": {},
            "axis_conditioning_memory": {},
            "lora_weight_memory": {},
            "preferred_context_memory": {},
            "intent_alignment_memory": {},
            "intent_family_memory": {},
            "perfect_anchors": {},
            "variant_evidence": {},
            "intent_preference_phrases": {},
            "conditioning_deltas": {},
            "active_repair_axes": [],
            "negative_prompt_memory": {},
            "vision_memory": {},
            "loss_history": [],
        },
        "prompt_histories": {},
        "last_run": {
            "prompt": "person smoking",
            "encoded_prompt": "person smoking",
            "source_conditioning": tensor_to_serializable(torch.zeros(1, 4, 3)),
            "conditioning": tensor_to_serializable(torch.zeros(1, 4, 3)),
            "phrases": prompt_phrases(refiner, "person smoking"),
            "rating_label": "Unrated",
            "iteration": 1,
        },
    }), encoding="utf-8")
    refiner._v2_state_path = lambda refinement_key: str(state_path)
    main_clip = CountingClip()
    advisor_clip = GeneratingClip(
        "DIAGNOSTIC: add visible smoke motion.\n"
        "REPAIRED_PROMPT: person smoking, smoke trails drifting upward"
    )

    refiner.refine_v2(
        "person smoking",
        main_clip,
        "Missing details",
        "advisor-clip-test",
        user_intent_prompt="person smoking",
        advisor_mode="Repair prompt",
        advisor_clip=advisor_clip,
    )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert advisor_clip.tokenize_calls
    assert state["last_run"]["encoded_prompt"] == "person smoking, smoke trails drifting upward"
    assert state["last_run"]["advisor"]["applied"] is True


def test_refiner_v2_negative_advisor_repairs_negative_prompt_and_exports_one_string(tmp_path):
    refiner = FunPackVideoRefinerV2()
    state_path = tmp_path / "state.json"
    refiner._v2_state_path = lambda refinement_key: str(state_path)

    refiner.refine_v2("person smoking", FakeClip(), "Perfect", "negative-advisor-test")
    advisor_clip = GeneratingClip(
        "DIAGNOSTIC: suppress bad detail artifacts.\n"
        "NEGATIVE_PROMPT: bad anatomy, low quality, blurry motion"
    )

    _, status, training_info, _, negative, encoded_prompts = refiner.refine_v2(
        "person smoking",
        FakeClip(),
        "Wrong details",
        "negative-advisor-test",
        negative_prompt="bad anatomy",
        advisor_mode="Repair prompt",
        advisor_clip=advisor_clip,
    )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert "Negative advisor: applied generated negative prompt" in status
    assert "Negative advisor: applied generated negative prompt" in training_info
    assert negative
    assert encoded_prompts == (
        "Positive prompt: person smoking\n\n"
        "Negative prompt: bad anatomy, low quality, blurry motion"
    )
    assert state["last_run"]["negative_prompt"] == "bad anatomy, low quality, blurry motion"
    assert state["last_run"]["advisor"]["negative_applied"] is True
    assert any("Additional negative prompt task rules" in call[0] for call in advisor_clip.tokenize_calls)


def test_refiner_v2_advisor_skips_when_no_generation_clip_is_available(tmp_path):
    refiner = FunPackVideoRefinerV2()
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({
        "version": 2,
        "refinement_key": "advisor-no-generation-test",
        "state_namespace": "clip",
        "global": {
            "total_iterations": 1,
            "avg_reward_ema": 0.0,
            "good_streak": 0,
            "bad_streak": 1,
            "last_rating_label": "Missing action",
            "last_missing_axes": ["action"],
            "phrase_memory": {},
            "axis_conditioning_memory": {},
            "lora_weight_memory": {},
            "preferred_context_memory": {},
            "intent_alignment_memory": {},
            "intent_family_memory": {},
            "perfect_anchors": {},
            "variant_evidence": {},
            "intent_preference_phrases": {},
            "conditioning_deltas": {},
            "active_repair_axes": [],
            "negative_prompt_memory": {},
            "vision_memory": {},
            "loss_history": [],
        },
        "prompt_histories": {},
        "last_run": {
            "prompt": "person smoking",
            "encoded_prompt": "person smoking",
            "source_conditioning": tensor_to_serializable(torch.zeros(1, 4, 3)),
            "conditioning": tensor_to_serializable(torch.zeros(1, 4, 3)),
            "phrases": prompt_phrases(refiner, "person smoking"),
            "rating_label": "Unrated",
            "iteration": 1,
        },
    }), encoding="utf-8")
    refiner._v2_state_path = lambda refinement_key: str(state_path)

    _, status, training_info, _, _, _ = refiner.refine_v2(
        "person smoking",
        FakeClip(),
        "Missing details",
        "advisor-no-generation-test",
        advisor_mode="Repair prompt",
    )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert "Advisor: unavailable; connected CLIP does not expose text generation" in status
    assert "Advisor: unavailable; connected CLIP does not expose text generation" in training_info
    assert state["last_run"]["encoded_prompt"] == "person smoking"
    assert state["last_run"]["advisor"]["applied"] is False


def test_refiner_v2_learning_mode_passes_prompt_and_conditioning_through(tmp_path):
    refiner = FunPackVideoRefinerV2()
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({
        "version": 2,
        "refinement_key": "learning-mode-test",
        "state_namespace": "clip",
        "global": {
            "total_iterations": 1,
            "avg_reward_ema": 0.0,
            "good_streak": 0,
            "bad_streak": 1,
            "last_rating_label": "Missing action",
            "last_missing_axes": ["action"],
            "phrase_memory": {
                "tiny smoke curls": {
                    "text": "tiny smoke curls",
                    "primary": "details",
                    "effective_category_scores": refiner._v2_heuristic_scores("tiny smoke curls"),
                    "wanted_axes": {"details": 4},
                    "score": 8.0,
                    "liked_count": 6,
                }
            },
            "axis_conditioning_memory": {},
            "lora_weight_memory": {},
            "preferred_context_memory": {},
            "intent_alignment_memory": {},
            "intent_family_memory": {},
            "perfect_anchors": {},
            "variant_evidence": {},
            "intent_preference_phrases": {},
            "conditioning_deltas": {},
            "active_repair_axes": ["details"],
            "negative_prompt_memory": {
                "tags": {
                    "bad repaired detail": {
                        "text": "bad repaired detail",
                        "count": 3,
                        "axes": {"details": 2},
                        "last_seen_iter": 1,
                    }
                }
            },
            "vision_memory": {},
            "loss_history": [],
            "liked_conditioning": tensor_to_serializable(torch.flip(torch.arange(12, dtype=torch.float32).reshape(1, 4, 3), dims=[-1])),
        },
        "prompt_histories": {},
        "last_run": {
            "prompt": "person smoking",
            "encoded_prompt": "person smoking, tiny smoke curls",
            "source_conditioning": tensor_to_serializable(torch.zeros(1, 4, 3)),
            "conditioning": tensor_to_serializable(torch.ones(1, 4, 3)),
            "phrases": [],
            "rating_label": "Unrated",
            "iteration": 1,
        },
    }), encoding="utf-8")
    refiner._v2_state_path = lambda refinement_key: str(state_path)
    positive_conditioning = [(torch.arange(12, dtype=torch.float32).reshape(1, 4, 3), {"pooled_output": torch.ones(1, 3)})]

    modified, status, training_info, _, negative, encoded_prompts = refiner.refine_v2(
        "person smoking",
        None,
        "Missing details",
        "learning-mode-test",
        positive_conditioning=positive_conditioning,
        negative_prompt="blur",
        mode="Learning",
    )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert "Mode Learning" in status
    assert "Learning mode observation only" in status
    assert "Mode: Learning" in training_info
    assert torch.equal(modified[0][0], positive_conditioning[0][0])
    assert state["last_run"]["encoded_prompt"] == "person smoking"
    assert state["last_run"]["negative_prompt"] == "blur"
    assert encoded_prompts == "Positive prompt: person smoking\n\nNegative prompt: blur"
    assert "tiny smoke curls" not in state["last_run"]["encoded_prompt"]
    assert negative == []


def test_refiner_v2_accepts_conditioning_without_clip_and_loads_gemma3_tokenizer(tmp_path, monkeypatch):
    refiner = FunPackVideoRefinerV2()
    state_path = tmp_path / "state.json"
    refiner._v2_state_path = lambda refinement_key: str(state_path)
    tokenizer_modes = []

    class FakeTokenizer:
        name_or_path = "DreamFast/gemma-3-12b-it-heretic-v2"

    def fake_get_tokenizer(mode="ltx2"):
        tokenizer_modes.append(mode)
        return FakeTokenizer()

    monkeypatch.setattr(refiner, "_get_tokenizer", fake_get_tokenizer)
    positive_conditioning = [(torch.full((1, 4, 3), 2.0), {"pooled_output": torch.ones(1, 3)})]

    modified, status, training_info, _, negative, _ = refiner.refine_v2(
        "woman walking through neon rain",
        None,
        "Perfect",
        "conditioning-input-test",
        positive_conditioning=positive_conditioning,
    )

    assert tokenizer_modes == ["ltx2"]
    assert "CONDITIONING-owned" in status
    assert "Gemma3 tokenizer loaded: DreamFast/gemma-3-12b-it-heretic-v2" in training_info
    assert modified[0][0].shape == positive_conditioning[0][0].shape
    assert negative == []


def test_refiner_v2_prefers_clip_when_both_clip_and_conditioning_are_connected(tmp_path, monkeypatch):
    refiner = FunPackVideoRefinerV2()
    state_path = tmp_path / "state.json"
    refiner._v2_state_path = lambda refinement_key: str(state_path)
    monkeypatch.setattr(refiner, "_get_tokenizer", lambda mode="ltx2": (_ for _ in ()).throw(AssertionError("unexpected tokenizer load")))
    positive_conditioning = [(torch.full((1, 4, 3), 9.0), {"pooled_output": torch.ones(1, 3)})]

    modified, status, training_info, _, negative, _ = refiner.refine_v2(
        "woman walking through neon rain",
        FakeClip(),
        "Perfect",
        "clip-priority-test",
        positive_conditioning=positive_conditioning,
    )

    assert "CLIP-owned" in status
    assert "accepted connected positive CONDITIONING" not in training_info
    assert torch.allclose(modified[0][0], torch.ones(1, 4, 3))
    assert negative == []


def test_refiner_v2_errors_without_clip_or_conditioning(tmp_path):
    refiner = FunPackVideoRefinerV2()
    state_path = tmp_path / "state.json"
    refiner._v2_state_path = lambda refinement_key: str(state_path)

    modified, status, training_info, _, negative, encoded_prompts = refiner.refine_v2(
        "woman walking through neon rain",
        None,
        "Perfect",
        "missing-conditioning-test",
    )

    assert modified == []
    assert negative == []
    assert encoded_prompts == "Positive prompt: woman walking through neon rain\n\nNegative prompt: "
    assert "ERROR: V2 could not prepare conditioning" in status
    assert "CLIP missing and no positive CONDITIONING connected" in training_info


def test_prompt_enhancer_refusal_is_not_stored_as_last_run(tmp_path):
    refiner = FunPackVideoRefinerV2()
    state_path = tmp_path / "state.json"
    refiner._v2_state_path = lambda refinement_key: str(state_path)

    refiner.refine_v2(
        "I'm sorry, I cannot help you with this request.",
        FakeClip(),
        "Perfect",
        "refusal-test",
    )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["last_run"] is None
    assert state["prompt_histories"] == {}
    assert state["global"]["phrase_memory"] == {}
    assert state["global"]["total_iterations"] == 0


def test_saved_refusal_last_run_does_not_train_memory(tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({
        "version": 2,
        "refinement_key": "refusal-test",
        "state_namespace": "clip",
        "global": {
            "total_iterations": 1,
            "avg_reward_ema": 0.0,
            "good_streak": 0,
            "bad_streak": 0,
            "last_rating_label": "Initial discovery",
            "last_missing_axes": [],
            "phrase_memory": {},
            "axis_conditioning_memory": {},
            "lora_weight_memory": {},
            "loss_history": [],
        },
        "prompt_histories": {},
        "last_run": {
            "prompt": "I'm sorry, I cannot help you with this request.",
            "encoded_prompt": "I'm sorry, I cannot help you with this request.",
            "conditioning": {},
            "phrases": [{"text": "i'm sorry", "tokens": ["sorry"], "primary": "details"}],
            "rating_label": "Unrated",
            "iteration": 1,
        },
    }), encoding="utf-8")
    refiner = FunPackVideoRefinerV2()
    refiner._v2_state_path = lambda refinement_key: str(state_path)

    refiner.refine_v2("wide cinematic shot", FakeClip(), "Perfect", "refusal-test")

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["global"]["phrase_memory"] == {}
    assert state["last_run"]["prompt"] == "wide cinematic shot"


def test_normal_previous_run_can_train_before_current_refusal_is_discarded(tmp_path):
    refiner = FunPackVideoRefinerV2()
    state_path = tmp_path / "state.json"
    refiner._v2_state_path = lambda refinement_key: str(state_path)

    refiner.refine_v2("woman walking through neon rain", FakeClip(), "Perfect", "refusal-test")
    refiner.refine_v2(
        "Sorry, but I can't assist with that request.",
        FakeClip(),
        "Perfect",
        "refusal-test",
    )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert "woman walking through neon rain" in state["global"]["phrase_memory"]
    assert state["last_run"] is None
    assert "sorry, but i can't assist with that request." not in state["prompt_histories"]


def test_liked_action_detail_context_clusters_repair_missing_axes():
    refiner = FunPackVideoRefinerV2()
    global_state = {"phrase_memory": {}, "preferred_context_memory": {}}
    rich_prompt = "woman walking reaching hand tiny particles rain reflections"
    liked_phrases = refiner._v2_classify_phrases(
        None,
        prompt_items(refiner, [
            "woman",
            "walking",
            "reaching hand",
            "tiny particles",
            "rain reflections",
        ]),
        global_state,
    )
    profile = normalize_refiner_v2_rating("Perfect")
    feedback = refiner._v2_axis_feedback(profile, None)

    refiner._v2_update_phrase_memory(
        global_state,
        {"prompt": rich_prompt, "phrases": liked_phrases},
        profile,
        1,
        feedback,
    )
    assert "Preferred context stored" in refiner._v2_update_preferred_context_memory(
        global_state,
        {"phrases": liked_phrases},
        profile,
        2,
        feedback,
    )

    missing_profile = normalize_refiner_v2_rating("Missing details + action")
    missing_feedback = refiner._v2_axis_feedback(missing_profile, None)
    current_phrases = refiner._v2_classify_phrases(
        None,
        prompt_items(refiner, ["woman", "rain"]),
        global_state,
    )
    intent_phrases = refiner._v2_classify_phrases(
        None,
        refiner._ordered_prompt_phrases(rich_prompt),
        global_state,
    )

    repaired, status, candidates = refiner._v2_repair_prompt_for_missing_axes(
        "woman, rain",
        current_phrases,
        global_state,
        None,
        missing_feedback,
        intent_phrases=intent_phrases,
    )

    assert "walking" in repaired
    assert "tiny particles" in repaired
    assert "intent" in status or "preferred_context" in status
    assert len(candidates) >= 2


def test_prompt_repair_does_not_pull_unrequested_memory_for_any_missing_axis():
    refiner = FunPackVideoRefinerV2()
    global_state = {
        "phrase_memory": {
            "walking": {
                "text": "walking",
                "primary": "action",
                "effective_category_scores": refiner._v2_heuristic_scores("walking"),
                "wanted_axes": {"action": 5},
                "score": 5.0,
                "liked_count": 5,
            },
            "tiny particles": {
                "text": "tiny particles",
                "primary": "details",
                "effective_category_scores": refiner._v2_heuristic_scores("tiny particles"),
                "wanted_axes": {"details": 5},
                "score": 5.0,
                "liked_count": 5,
            },
            "cinematic lighting": {
                "text": "cinematic lighting",
                "primary": "style",
                "effective_category_scores": refiner._v2_heuristic_scores("cinematic lighting"),
                "wanted_axes": {"quality": 5},
                "score": 5.0,
                "liked_count": 5,
            },
            "glass reflections": {
                "text": "glass reflections",
                "primary": "details",
                "effective_category_scores": refiner._v2_heuristic_scores("glass reflections"),
                "wanted_axes": {"details": 2},
                "score": 2.0,
                "liked_count": 1,
            },
        },
    }
    profile = normalize_refiner_v2_rating("Awful")
    feedback = refiner._v2_axis_feedback(profile, None)
    current_phrases = refiner._v2_classify_phrases(
        None,
        prompt_items(refiner, ["woman", "glass"]),
        global_state,
    )

    repaired, _, candidates = refiner._v2_repair_prompt_for_missing_axes(
        "woman, glass",
        current_phrases,
        global_state,
        None,
        feedback,
    )

    candidate_texts = {candidate["text"] for candidate in candidates}
    assert "glass reflections" in repaired
    assert "walking" not in repaired
    assert "tiny particles" not in repaired
    assert "cinematic lighting" not in repaired
    assert "walking" not in candidate_texts
    assert "tiny particles" not in candidate_texts
    assert "cinematic lighting" not in candidate_texts


def test_prompt_repair_treats_same_word_with_different_neighbours_as_different():
    refiner = FunPackVideoRefinerV2()
    global_state = {
        "phrase_memory": {
            "turning head": {
                "text": "turning head",
                "primary": "action",
                "effective_category_scores": refiner._v2_heuristic_scores("turning head"),
                "wanted_axes": {"action": 4},
                "score": 4.0,
                "liked_count": 4,
                "context_senses": {
                    "mid|eyes,portrait": {
                        "context": {
                            "anchor_words": ["head", "turning"],
                            "context_words": ["portrait", "eyes"],
                            "position_bucket": "mid",
                            "window": 3,
                        },
                        "category_weights": refiner._v2_category_template(0.0),
                        "effective_category_scores": refiner._v2_heuristic_scores("turning head"),
                        "category_evidence_count": 4,
                        "occurrence_count": 4,
                    },
                },
            },
        },
    }
    profile = normalize_refiner_v2_rating("Missing action")
    feedback = refiner._v2_axis_feedback(profile, None)
    current_phrases = refiner._v2_classify_phrases(
        None,
        prompt_items(refiner, ["car", "turning wheel", "street"]),
        global_state,
    )

    repaired, _, candidates = refiner._v2_repair_prompt_for_missing_axes(
        "car, turning wheel, street",
        current_phrases,
        global_state,
        None,
        feedback,
    )

    assert "turning head" not in repaired
    assert all(candidate["text"] != "turning head" for candidate in candidates)


def test_vague_user_intent_lets_enhanced_positive_prompt_anchor_repair():
    refiner = FunPackVideoRefinerV2()
    global_state = {
        "phrase_memory": {
            "walking slowly": {
                "text": "walking slowly",
                "primary": "action",
                "effective_category_scores": refiner._v2_heuristic_scores("walking slowly"),
                "wanted_axes": {"action": 3},
                "score": 3.0,
                "liked_count": 3,
            },
        },
    }
    assert refiner._v2_user_intent_prompt_is_vague("Figure it out")
    assert not refiner._v2_user_intent_prompt_is_vague("person walking")

    profile = normalize_refiner_v2_rating("Missing action")
    feedback = refiner._v2_axis_feedback(profile, None)
    current_phrases = refiner._v2_classify_phrases(
        None,
        prompt_items(refiner, ["woman", "walking through glass hallway"]),
        global_state,
    )

    repaired, _, candidates = refiner._v2_repair_prompt_for_missing_axes(
        "woman, walking through glass hallway",
        current_phrases,
        global_state,
        None,
        feedback,
        intent_phrases=[],
    )

    assert "walking slowly" in repaired
    assert any(candidate["text"] == "walking slowly" for candidate in candidates)


def test_complex_user_intent_ranks_before_matching_preferred_memory():
    refiner = FunPackVideoRefinerV2()
    global_state = {
        "phrase_memory": {},
        "preferred_context_memory": {
            "reach-memory": {
                "anchor": "reaching hand slowly",
                "axes": {"action": 4},
                "context": {},
                "phrases": ["reaching hand slowly"],
                "score": 10.0,
                "liked_count": 8,
            },
        },
    }
    profile = normalize_refiner_v2_rating("Missing action")
    feedback = refiner._v2_axis_feedback(profile, None)
    current_phrases = refiner._v2_classify_phrases(
        None,
        prompt_items(refiner, ["woman"]),
        global_state,
    )
    intent_phrases = refiner._v2_classify_phrases(
        None,
        refiner._ordered_prompt_phrases("woman reaching hand toward glass"),
        global_state,
    )

    repaired, _, candidates = refiner._v2_repair_prompt_for_missing_axes(
        "woman",
        current_phrases,
        global_state,
        None,
        feedback,
        intent_phrases=intent_phrases,
    )

    assert "reaching hand toward glass" in repaired
    assert candidates[0]["source"] == "intent"


def test_phrase_clusters_train_more_strongly_than_ngrams_and_tokens():
    refiner = FunPackVideoRefinerV2()
    global_state = {"phrase_memory": {}, "preferred_context_memory": {}}
    phrases = refiner._v2_classify_phrases(
        None,
        [{"text": "reaching hand slowly", "tokens": ["reaching", "hand", "slowly"]}],
        global_state,
    )
    profile = normalize_refiner_v2_rating("Perfect")
    feedback = refiner._v2_axis_feedback(profile, None)

    refiner._v2_update_phrase_memory(
        global_state,
        {"prompt": "reaching hand slowly", "phrases": phrases},
        profile,
        1,
        feedback,
    )

    phrase_score = global_state["phrase_memory"]["reaching hand slowly"]["score"]
    ngram_score = global_state["phrase_memory"]["reaching hand"]["score"]
    token_score = global_state["phrase_memory"]["reaching"]["score"]

    assert phrase_score > ngram_score > token_score


def test_repair_candidates_saved_as_json_lists(tmp_path):
    refiner = FunPackVideoRefinerV2()
    state_path = tmp_path / "state.json"
    refiner._v2_state_path = lambda refinement_key: str(state_path)
    rich_prompt = "woman, walking, reaching hand, tiny particles, rain reflections"

    refiner.refine_v2(rich_prompt, FakeClip(), "Perfect", "json-repair-test")
    refiner.refine_v2(rich_prompt, FakeClip(), "Perfect", "json-repair-test")
    refiner.refine_v2(
        "woman, rain",
        FakeClip(),
        "Missing details + action",
        "json-repair-test",
        user_intent_prompt=rich_prompt,
    )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    candidates = state["last_run"]["repair_candidates"]
    assert candidates
    assert isinstance(candidates[0]["axes"], list)
    assert "walking" in state["last_run"]["encoded_prompt"]


def test_wrong_action_rating_preserves_quality_but_marks_action_wrong():
    refiner = FunPackVideoRefinerV2()
    global_state = {"phrase_memory": {}, "preferred_context_memory": {}}
    phrases = refiner._v2_classify_phrases(
        None,
        prompt_items(refiner, ["walking", "cinematic lighting"]),
        global_state,
    )
    profile = normalize_refiner_v2_rating("Wrong action")
    feedback = refiner._v2_axis_feedback(profile, None)

    refiner._v2_update_phrase_memory(
        global_state,
        {"prompt": "walking cinematic lighting", "phrases": phrases},
        profile,
        1,
        feedback,
    )

    assert feedback["missing_axes"] == ["action"]
    assert feedback["wrong_axes"] == ["action"]
    assert feedback["satisfied_axes"] == ["details", "quality"]
    assert global_state["phrase_memory"]["walking"]["wrong_count"] == 1
    assert global_state["phrase_memory"]["cinematic lighting"]["satisfied_count"] == 1


def test_ordered_prompt_phrases_preserve_stopwords_for_repair_text():
    refiner = FunPackVideoRefinerV2()

    phrases = refiner._ordered_prompt_phrases("running through the street, hands in the frame")

    assert phrases[0]["text"] == "running through the street"
    assert phrases[0]["tokens"] == ["running", "through", "street"]
    assert phrases[1]["text"] == "hands in the frame"


def test_v2_image_metadata_detects_aspect_bucket_and_changed_fingerprint():
    refiner = FunPackVideoRefinerV2()
    first = torch.zeros(1, 32, 64, 3)
    second = torch.ones(1, 32, 64, 3)

    metadata, status = refiner._v2_image_metadata(first)
    changed, _ = refiner._v2_image_metadata(second, metadata)

    assert metadata["width"] == 64
    assert metadata["height"] == 32
    assert metadata["aspect_bucket"] == "ultrawide"
    assert "64x32" in status
    assert changed["changed_from_previous"] is True


def test_refiner_v2_returns_empty_negative_conditioning_when_negative_blank(tmp_path):
    refiner = FunPackVideoRefinerV2()
    state_path = tmp_path / "state.json"
    refiner._v2_state_path = lambda refinement_key: str(state_path)

    _, _, training_info, _, negative, encoded_prompts = refiner.refine_v2(
        "woman walking through neon rain",
        FakeClip(),
        "Perfect",
        "negative-empty-test",
        negative_prompt="",
    )

    assert negative == []
    assert encoded_prompts == "Positive prompt: woman walking through neon rain\n\nNegative prompt: "
    assert "Negative repair:" in training_info


def test_negative_prompt_repair_adds_poorly_rated_tags_persistently(tmp_path):
    refiner = FunPackVideoRefinerV2()
    state_path = tmp_path / "state.json"
    refiner._v2_state_path = lambda refinement_key: str(state_path)

    refiner.refine_v2("woman, walking through the street", FakeClip(), "Perfect", "negative-repair-test")
    refiner.refine_v2("woman, walking through the street", FakeClip(), "Wrong action", "negative-repair-test")
    _, _, training_info, _, negative, encoded_prompts = refiner.refine_v2(
        "woman portrait",
        FakeClip(),
        "Missing action",
        "negative-repair-test",
        negative_prompt="bad anatomy",
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))

    assert negative
    assert encoded_prompts.startswith("Positive prompt: woman portrait\n\nNegative prompt: bad anatomy")
    assert "walking through the street" in state["global"]["negative_prompt_memory"]["tags"]
    assert "walking through the street" in state["last_run"]["negative_prompt"]
    assert "persistent poor-rated tag" in training_info


def test_refiner_v2_caches_repeated_clip_category_encodes():
    refiner = FunPackVideoRefinerV2()
    clip = CountingClip()
    phrases = [
        {"text": "soft glow", "tokens": ["soft", "glow"]},
        {"text": "soft glow", "tokens": ["soft", "glow"]},
    ]

    refiner._v2_classify_phrases(clip, phrases, {"phrase_memory": {}}, encode_cache={})

    assert clip.calls <= len(refiner.CATEGORY_DESCRIPTIONS) + 1
