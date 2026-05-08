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

    _, _, training_info, _ = refiner.refine_v2(
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
    assert "clip" in inputs["optional"]
    assert "positive_conditioning" in inputs["optional"]


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

    modified, status, training_info, _ = refiner.refine_v2(
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


def test_refiner_v2_prefers_clip_when_both_clip_and_conditioning_are_connected(tmp_path, monkeypatch):
    refiner = FunPackVideoRefinerV2()
    state_path = tmp_path / "state.json"
    refiner._v2_state_path = lambda refinement_key: str(state_path)
    monkeypatch.setattr(refiner, "_get_tokenizer", lambda mode="ltx2": (_ for _ in ()).throw(AssertionError("unexpected tokenizer load")))
    positive_conditioning = [(torch.full((1, 4, 3), 9.0), {"pooled_output": torch.ones(1, 3)})]

    modified, status, training_info, _ = refiner.refine_v2(
        "woman walking through neon rain",
        FakeClip(),
        "Perfect",
        "clip-priority-test",
        positive_conditioning=positive_conditioning,
    )

    assert "CLIP-owned" in status
    assert "accepted connected positive CONDITIONING" not in training_info
    assert torch.allclose(modified[0][0], torch.ones(1, 4, 3))


def test_refiner_v2_errors_without_clip_or_conditioning(tmp_path):
    refiner = FunPackVideoRefinerV2()
    state_path = tmp_path / "state.json"
    refiner._v2_state_path = lambda refinement_key: str(state_path)

    modified, status, training_info, _ = refiner.refine_v2(
        "woman walking through neon rain",
        None,
        "Perfect",
        "missing-conditioning-test",
    )

    assert modified == []
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
