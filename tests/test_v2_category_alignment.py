import json
import random
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.modules.setdefault("folder_paths", types.SimpleNamespace(models_dir=""))

import torch

from conditioning import (
    FunPackStudio,
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


# Ratings that deliberately do NOT train phrase memory, by profile key rather than by
# label text — the label list grows (the "|loved" variants, the control sentinels) and a
# hardcoded exclusion list silently starts asserting the wrong thing about the new ones.
#   forget / continue / fresh_prompt — control actions, not ratings at all
#   wrong_appearance                 — a character-consistency signal, handled by the
#                                      appearance anchor, never by phrase categories
NON_LEARNING_RATING_KEYS = {"forget", "wrong_appearance", "continue", "fresh_prompt"}


def test_category_weights_are_recorded_for_every_learning_rating():
    refiner = FunPackVideoRefinerV2()
    learning_labels = [
        label for label in V2_RATING_LABELS
        if normalize_refiner_v2_rating(label).get("key") not in NON_LEARNING_RATING_KEYS
    ]
    assert len(learning_labels) > 10   # the filter must not quietly empty the loop

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
    # Wrong appearance now drives the consistency anchor, not prompt repair (the dead
    # wrong_categories field was dropped in Stage 2 Part B). It stays quality-neutral.
    assert "wrong_categories" not in profile
    assert profile["skip_value_function"] is True


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


def _path_run(refiner, prompt, global_state, *, seed, family="fam", carry=True):
    return {
        "prompt": prompt,
        "phrases": prompt_phrases(refiner, prompt, global_state),
        "seed": seed,
        "intent_family_key": family,
        "gen_context": {
            "loras": [],
            "carry_i2v_guides": carry,
            "frame_overlap": 8,
            "transitions_enabled": False,
            "image_fp": "",
        },
    }


def test_path_outcomes_records_every_rating_not_just_good_ones():
    refiner = FunPackVideoRefinerV2()
    global_state = {"phrase_memory": {}, "path_outcomes": {}}
    prompt = "woman walking through neon rain"

    awful = normalize_refiner_v2_rating("Awful")
    refiner._v2_update_path_outcomes(global_state, _path_run(refiner, prompt, global_state, seed=11), awful, 1)
    refiner._v2_update_path_outcomes(global_state, _path_run(refiner, prompt, global_state, seed=22), awful, 2)

    arms = global_state["path_outcomes"]
    assert len(arms) == 1, "same config differing only by seed must collapse to one arm"
    arm = next(iter(arms.values()))
    assert arm["n_pulls"] == 2
    assert arm["outcomes"].get(awful["key"], 0) > 0
    # both seeds recorded as samples within the arm
    assert sorted(s["seed"] for s in arm["seeds"]) == [11, 22]


def test_path_outcomes_separates_arms_by_config():
    refiner = FunPackVideoRefinerV2()
    global_state = {"phrase_memory": {}, "path_outcomes": {}}
    prompt = "woman walking through neon rain"
    profile = normalize_refiner_v2_rating("Awful")

    refiner._v2_update_path_outcomes(global_state, _path_run(refiner, prompt, global_state, seed=1, carry=True), profile, 1)
    refiner._v2_update_path_outcomes(global_state, _path_run(refiner, prompt, global_state, seed=1, carry=False), profile, 2)

    assert len(global_state["path_outcomes"]) == 2, "differing guidance config must be different arms"


def test_path_outcomes_skips_runs_with_no_identifiable_path():
    refiner = FunPackVideoRefinerV2()
    global_state = {"path_outcomes": {}}
    profile = normalize_refiner_v2_rating("Awful")
    status = refiner._v2_update_path_outcomes(global_state, {"seed": 5}, profile, 1)
    assert global_state["path_outcomes"] == {}
    assert "no identifiable path" in status


def test_path_outcomes_ignores_just_forget_it():
    refiner = FunPackVideoRefinerV2()
    global_state = {"phrase_memory": {}, "path_outcomes": {}}
    run = _path_run(refiner, "woman walking through neon rain", global_state, seed=7)
    status = refiner._v2_update_path_outcomes(
        global_state, run, normalize_refiner_v2_rating("-Just forget it-"), 1
    )
    assert global_state["path_outcomes"] == {}
    assert "skips learning" in status


def test_rating_is_discard_targets_forget_and_refusal_only():
    refiner = FunPackVideoRefinerV2()
    assert refiner._v2_rating_is_discard(normalize_refiner_v2_rating("-Just forget it-")) is True
    refusal = dict(normalize_refiner_v2_rating("Perfect"))
    refusal["refusal_filtered"] = True
    assert refiner._v2_rating_is_discard(refusal) is True
    # Normal ratings and the other skip_learning workflow modes must NOT be treated as discards.
    assert refiner._v2_rating_is_discard(normalize_refiner_v2_rating("Awful")) is False
    assert refiner._v2_rating_is_discard(normalize_refiner_v2_rating("Perfect")) is False


def test_just_forget_it_does_not_store_feedback_history_or_intent_expansion():
    refiner = FunPackVideoRefinerV2()
    global_state = {"advisor_feedback_history": [], "intent_expansion_memory": {}}
    forget = normalize_refiner_v2_rating("-Just forget it-")

    # The guard the refine_v2 flow uses before either feedback sink fires.
    if not refiner._v2_rating_is_discard(forget):
        refiner._v2_update_advisor_feedback_history(global_state, "make the dress red", forget["label"], 1)
        refiner._v2_update_intent_expansion(global_state, "a woman in a dress", "make the dress red")

    assert global_state["advisor_feedback_history"] == []
    assert global_state["intent_expansion_memory"] == {}

    # Sanity: a normal rating with the same feedback DOES store it.
    normal = normalize_refiner_v2_rating("Missing details")
    if not refiner._v2_rating_is_discard(normal):
        refiner._v2_update_advisor_feedback_history(global_state, "make the dress red", normal["label"], 2)
        refiner._v2_update_intent_expansion(global_state, "a woman in a dress", "make the dress red")
    assert len(global_state["advisor_feedback_history"]) == 1
    assert global_state["intent_expansion_memory"]


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

    _, _, training_info, _, _, _, _ = refiner.refine_v2(
        "woman walking through neon rain",
        FakeClip(),
        "Perfect",
        "readable-test",
    )

    assert "\n\nMemory\n" in training_info
    assert "\n\nPrompt Analysis\n" in training_info
    assert "\n\nAdaptation\n" in training_info
    assert "\n\nLoRA\n" in training_info


def test_refiner_v2_exposes_clip_and_conditioning_as_optional_inputs():
    inputs = FunPackVideoRefinerV2.INPUT_TYPES()

    assert "clip" not in inputs["required"]
    assert "positive_conditioning" not in inputs["required"]
    assert inputs["required"]["mode"][0] == ["Refine", "Prompt only", "Learning"]
    assert inputs["required"]["advisor_mode"][0] == ["Off", "Only diagnostics", "Only prompt", "Full"]
    assert "encoded_prompts" in FunPackVideoRefinerV2.RETURN_NAMES
    assert "model" in FunPackVideoRefinerV2.RETURN_NAMES
    assert FunPackVideoRefinerV2.RETURN_TYPES[FunPackVideoRefinerV2.RETURN_NAMES.index("encoded_prompts")] == "STRING"
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

    prompt, status, diagnostic, applied, _ = refiner._v2_prompt_advisor(
        clip,
        "Diagnostics",
        "person smoking",
        "person smoking",
        {"missing_axes": ["details"], "wrong_axes": []},
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
    assert "Prompt: person smoking" in advisor_prompt
    assert "Previous prompt: old encoded prompt" in advisor_prompt
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

    _, status, training_info, _, encoded_prompts, _, _ = refiner.refine_v2(
        "person smoking",
        clip,
        "Missing details",
        "advisor-repair-test",
        user_intent_prompt="person smoking",
        advisor_mode="Repair prompt",
        advisor_thinking=True,
    )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert "Advisor: applied repair" in status
    assert "Encoded as: advisor repaired prompt" in training_info
    assert "Positive prompt: person smoking, smoke trails drifting upward" in encoded_prompts
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

    _, status, training_info, _, _, _, _ = refiner.refine_v2(
        "person smoking",
        FakeClip(),
        "Missing details",
        "advisor-no-generation-test",
        advisor_mode="Repair prompt",
    )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert "does not expose text generation" in status
    # the message names the socket to wire a generator into, not just the fault
    assert "advisor_clip" in status
    assert "does not expose text generation" in training_info
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

    modified, status, training_info, _, encoded_prompts, _, _ = refiner.refine_v2(
        "person smoking",
        None,
        "Missing details",
        "learning-mode-test",
        positive_conditioning=positive_conditioning,
        mode="Learning",
    )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert "Mode: Learning" in training_info
    assert torch.equal(modified[0][0], positive_conditioning[0][0])
    assert state["last_run"]["encoded_prompt"] == "person smoking"
    assert "tiny smoke curls" not in state["last_run"]["encoded_prompt"]


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

    modified, status, training_info, _, _, _, _ = refiner.refine_v2(
        "woman walking through neon rain",
        None,
        "Perfect",
        "conditioning-input-test",
        positive_conditioning=positive_conditioning,
    )

    assert tokenizer_modes == ["ltx2"]
    assert modified[0][0].shape == positive_conditioning[0][0].shape


def test_refiner_v2_keeps_the_wired_conditioning_when_both_are_connected(tmp_path, monkeypatch):
    refiner = FunPackVideoRefinerV2()
    state_path = tmp_path / "state.json"
    refiner._v2_state_path = lambda refinement_key: str(state_path)
    monkeypatch.setattr(refiner, "_get_tokenizer", lambda mode="ltx2": (_ for _ in ()).throw(AssertionError("unexpected tokenizer load")))
    positive_conditioning = [(torch.full((1, 4, 3), 9.0), {"pooled_output": torch.ones(1, 3)})]

    modified, status, training_info, _, _, _, _ = refiner.refine_v2(
        "woman walking through neon rain",
        FakeClip(),
        "Perfect",
        "clip-priority-test",
        positive_conditioning=positive_conditioning,
    )

    # The wired tensor survives: re-encoding it from text is not the same conditioning when
    # the node that built it saw a reference image Studio cannot reach.
    assert torch.allclose(modified[0][0], torch.full((1, 4, 3), 9.0))


def test_refiner_v2_errors_without_clip_or_conditioning(tmp_path):
    refiner = FunPackVideoRefinerV2()
    state_path = tmp_path / "state.json"
    refiner._v2_state_path = lambda refinement_key: str(state_path)

    modified, status, training_info, _, encoded_prompts, _, _ = refiner.refine_v2(
        "woman walking through neon rain",
        None,
        "Perfect",
        "missing-conditioning-test",
    )

    assert modified == []
    assert "Positive prompt: woman walking through neon rain" in encoded_prompts
    assert "ERROR: V2 could not prepare conditioning" in status


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


def test_refiner_v2_caches_repeated_clip_category_encodes():
    refiner = FunPackVideoRefinerV2()
    clip = CountingClip()
    phrases = [
        {"text": "soft glow", "tokens": ["soft", "glow"]},
        {"text": "soft glow", "tokens": ["soft", "glow"]},
    ]

    refiner._v2_classify_phrases(clip, phrases, {"phrase_memory": {}}, encode_cache={})

    assert clip.calls <= len(refiner.CATEGORY_DESCRIPTIONS) + 1


def test_split_by_transitions_shows_scenes_in_encoded_prompts(tmp_path):
    refiner = FunPackVideoRefinerV2()
    state_path = tmp_path / "state.json"
    refiner._v2_state_path = lambda refinement_key: str(state_path)

    # `scene <N>` is the canonical split delimiter (split_scenes): a real transition trigger
    # from the user's DB, or this generic label. Word-numbers ("scene ten") and bare prose
    # ("cut to") deliberately do NOT cut — an incidental phrase must never split a prompt.
    cond, status, _, _, encoded_prompts, _, _ = refiner.refine_v2(
        "a woman in a red dress, scene 1 she runs, scene 2 she stops",
        FakeClip(),
        "Perfect",
        "transition-test",
        split_by_transitions=True,
    )

    assert len(cond) == 2
    assert cond[0][1]["funpack_scene_index"] == 0
    assert cond[0][1]["funpack_scene_count"] == 2
    # the leading run is the anchor and folds into every scene; each scene keeps its own text
    assert "a woman in a red dress" in cond[0][1]["funpack_scene_text"]
    assert "a woman in a red dress" in cond[1][1]["funpack_scene_text"]
    assert "she runs" in cond[0][1]["funpack_scene_text"]
    assert "she stops" in cond[1][1]["funpack_scene_text"]
    assert "she stops" not in cond[0][1]["funpack_scene_text"]
    # the delimiter is a boundary, never content — encoding it would pollute the prompt
    assert "scene 1" not in cond[0][1]["funpack_scene_text"]
    assert "scene 2" not in cond[1][1]["funpack_scene_text"]
    assert "Scene chain mode" in status
    assert "Transition split" in status
    assert "Detected scenes" in encoded_prompts
    assert "Scene 1" in encoded_prompts


def test_split_by_transitions_disabled_returns_single_entry(tmp_path):
    refiner = FunPackVideoRefinerV2()
    state_path = tmp_path / "state.json"
    refiner._v2_state_path = lambda refinement_key: str(state_path)

    cond, status, _, _, _, _, _ = refiner.refine_v2(
        "a woman in a red dress, then she runs, suddenly stops",
        FakeClip(),
        "Perfect",
        "transition-test-off",
        split_by_transitions=False,
    )

    assert len(cond) == 1
    assert "Transition split" not in status


def test_split_by_transitions_has_no_hard_scene_cap(tmp_path):
    refiner = FunPackVideoRefinerV2()
    state_path = tmp_path / "state.json"
    refiner._v2_state_path = lambda refinement_key: str(state_path)
    prompt = "anchor, " + ", ".join(f"scene {i} shows action {i}" for i in range(1, 12))

    cond, status, _, _, encoded_prompts, _, _ = refiner.refine_v2(
        prompt,
        FakeClip(),
        "Perfect",
        "transition-cap-test",
        split_by_transitions=True,
    )

    assert len(cond) == 11
    assert cond[-1][1]["funpack_scene_count"] == 11
    assert "capped at 8 scenes" not in status
    assert "Scene 11" in encoded_prompts


def test_studio_detects_seed_output_links():
    prompt = {
        "output": {
            "10": {"class_type": "FunPackStudio", "inputs": {}},
            "20": {"class_type": "Sampler", "inputs": {"seed": ["10", 3]}},
        }
    }

    assert FunPackStudio._is_output_connected(prompt, "10", 3) is True
    assert FunPackStudio._is_output_connected(prompt, "10", 4) is False


def test_successful_seed_memory_requires_connected_seed_output(tmp_path):
    refiner = FunPackVideoRefinerV2()
    state_path = tmp_path / "state.json"
    refiner._v2_state_path = lambda refinement_key: str(state_path)

    refiner.refine_v2("KaiSa walking through neon", FakeClip(), "Perfect", "seed-test", _seed=111)
    refiner.refine_v2(
        "KaiSa walking through neon",
        FakeClip(),
        "Perfect",
        "seed-test",
        _seed=222,
        seed_output_connected=False,
    )

    state = json.loads(state_path.read_text())
    assert state["global"]["successful_seed_memory"] == {}


def test_successful_seed_memory_stores_previous_seed_not_current(tmp_path):
    refiner = FunPackVideoRefinerV2()
    state_path = tmp_path / "state.json"
    refiner._v2_state_path = lambda refinement_key: str(state_path)

    refiner.refine_v2(
        "KaiSa walking through neon",
        FakeClip(),
        "Missing action",
        "seed-test",
        _seed=111,
        seed_output_connected=True,
    )
    refiner.refine_v2(
        "KaiSa walking through neon",
        FakeClip(),
        "Loved it",
        "seed-test",
        _seed=222,
        seed_output_connected=True,
    )

    state = json.loads(state_path.read_text())
    memory = state["global"]["successful_seed_memory"]
    stored = [item["seed"] for entry in memory.values() for item in entry["seeds"]]
    assert 111 in stored
    assert 222 not in stored


def test_successful_seed_reuse_matches_concepts(monkeypatch, tmp_path):
    refiner = FunPackVideoRefinerV2()
    state_path = tmp_path / "state.json"
    refiner._v2_state_path = lambda refinement_key: str(state_path)
    state_path.write_text(json.dumps({
        "version": 2,
        "global": {
            "successful_seed_memory": {
                "neon": {
                    "concept": "neon",
                    "seeds": [{
                        "seed": 777,
                        "hit_count": 2,
                        "last_iteration": 5,
                        "scene_seeds": [777, 778, 779],
                    }],
                }
            }
        },
        "prompt_histories": {},
        "last_run": None,
    }))
    monkeypatch.setattr("conditioning.random.random", lambda: 0.05)
    monkeypatch.setattr("conditioning.random.choice", lambda items: items[0])

    seed, source, scene_seeds = refiner._v2_choose_successful_seed(
        "seed-test",
        "KaiSa in neon rain",
        123,
        seed_output_connected=True,
    )

    assert seed == 777
    assert scene_seeds == [777, 778, 779]
    assert "successful seed memory" in source


def test_path_outcomes_marks_quality_dislikes_for_avoidance_but_not_wrong_ratings():
    refiner = FunPackVideoRefinerV2()
    gs = {"phrase_memory": {}, "path_outcomes": {}}
    prompt = "woman walking through neon rain"

    refiner._v2_update_path_outcomes(gs, _path_run(refiner, prompt, gs, seed=10), normalize_refiner_v2_rating("Awful"), 1)
    refiner._v2_update_path_outcomes(gs, _path_run(refiner, prompt, gs, seed=20), normalize_refiner_v2_rating("Wrong appearance"), 2)
    refiner._v2_update_path_outcomes(gs, _path_run(refiner, prompt, gs, seed=30), normalize_refiner_v2_rating("Perfect"), 3)

    samples = {s["seed"]: s for arm in gs["path_outcomes"].values() for s in arm["seeds"]}
    assert samples[10]["avoid"] is True   # Awful = genuine low quality
    assert samples[20]["avoid"] is False  # Wrong appearance = seed was fine, words were off
    assert samples[30]["avoid"] is False  # Perfect

    disliked = refiner._v2_path_disliked_seeds(gs["path_outcomes"], ["neon", "rain"])
    assert disliked == {10}


def test_choose_seed_avoids_disliked_and_drops_later_disliked_reuse(monkeypatch, tmp_path):
    refiner = FunPackVideoRefinerV2()
    state_path = tmp_path / "state.json"
    refiner._v2_state_path = lambda refinement_key: str(state_path)
    state_path.write_text(json.dumps({
        "version": 2,
        "global": {
            # 'neon' seed 777 was liked once, but the path memory later marks it disliked.
            "successful_seed_memory": {
                "neon": {"concept": "neon", "seeds": [{"seed": 777, "hit_count": 2, "last_iteration": 5}]}
            },
            "path_outcomes": {
                "arm1": {
                    "concepts": ["neon", "rain"],
                    "seeds": [{"seed": 777, "avoid": True, "reward": -0.9}],
                },
            },
        },
        "prompt_histories": {},
        "last_run": None,
    }))
    monkeypatch.setattr("conditioning.random.random", lambda: 0.05)  # would reuse if any candidate left

    seed, source, _ = refiner._v2_choose_successful_seed(
        "seed-test", "KaiSa in neon rain", 123, seed_output_connected=True,
    )

    assert seed != 777, "a seed marked disliked in path memory must never be reused"
    assert "avoiding 1 disliked seed" in source


def test_path_arm_beta_counts_liked_and_disliked_only():
    refiner = FunPackVideoRefinerV2()
    arm = {"outcomes": {"like": 3.0, "awful": 1.0, "missing_action": 5.0}}  # missing_action(0.05)=neutral
    alpha, beta = refiner._v2_path_arm_beta(arm)
    assert (alpha, beta) == (4.0, 2.0)  # 1 + 3 likes, 1 + 1 dislike, neutral ignored


def test_path_explore_decision_exploits_good_arm(monkeypatch):
    refiner = FunPackVideoRefinerV2()
    memory = {"a": {"concepts": ["neon"], "n_pulls": 5, "outcomes": {"like": 5.0}}}
    draws = iter([0.9, 0.1])  # arm theta high, fresh theta low
    monkeypatch.setattr("conditioning.random.betavariate", lambda a, b: next(draws))
    explore, reason = refiner._v2_path_explore_decision(memory, ["neon", "rain"])
    assert explore is False
    assert "exploit" in reason


def test_path_explore_decision_routes_away_from_disliked_arm(monkeypatch):
    refiner = FunPackVideoRefinerV2()
    memory = {"a": {"concepts": ["neon"], "n_pulls": 5, "outcomes": {"awful": 5.0}}}
    draws = iter([0.1, 0.6])  # arm theta low, fresh theta higher
    monkeypatch.setattr("conditioning.random.betavariate", lambda a, b: next(draws))
    explore, reason = refiner._v2_path_explore_decision(memory, ["neon"])
    assert explore is True
    assert "explore fresh" in reason


def test_path_explore_decision_defers_without_data():
    refiner = FunPackVideoRefinerV2()
    explore, reason = refiner._v2_path_explore_decision({}, ["neon"])
    assert explore is None and reason == ""


def test_path_planner_exploits_liked_arm_far_more_than_disliked_arm():
    refiner = FunPackVideoRefinerV2()
    liked = {"a": {"concepts": ["neon"], "outcomes": {"like": 30.0}}}
    disliked = {"a": {"concepts": ["neon"], "outcomes": {"awful": 30.0}}}
    random.seed(0)
    liked_explores = sum(refiner._v2_path_explore_decision(liked, ["neon"])[0] for _ in range(300))
    disliked_explores = sum(refiner._v2_path_explore_decision(disliked, ["neon"])[0] for _ in range(300))
    # A strongly-liked arm is exploited (rarely explores); a hated arm is fled (mostly explores).
    assert liked_explores < 60
    assert disliked_explores > 240


def test_path_conditioning_plan_locks_liked_explores_disliked():
    refiner = FunPackVideoRefinerV2()
    liked = {"a": {"concepts": ["neon"], "outcomes": {"like": 10.0}}}
    disliked = {"a": {"concepts": ["neon"], "outcomes": {"awful": 10.0}}}
    mixed = {"a": {"concepts": ["neon"], "outcomes": {"like": 5.0, "awful": 5.0}}}

    assert refiner._v2_path_conditioning_plan(liked, ["neon"])[0] == "lock"
    assert refiner._v2_path_conditioning_plan(disliked, ["neon"])[0] == "explore"
    assert refiner._v2_path_conditioning_plan(mixed, ["neon"])[0] == "normal"
    assert refiner._v2_path_conditioning_plan({}, ["neon"]) == (None, "")
    # No concept overlap -> no matching arms -> defer.
    assert refiner._v2_path_conditioning_plan(liked, ["desert"]) == (None, "")


def test_split_by_transitions_attaches_scene_seeds(tmp_path):
    refiner = FunPackVideoRefinerV2()
    state_path = tmp_path / "state.json"
    refiner._v2_state_path = lambda refinement_key: str(state_path)

    cond, _, _, _, _, _, _ = refiner.refine_v2(
        "anchor, scene 1 she walks, scene 2 she turns",
        FakeClip(),
        "Perfect",
        "split-seed-test",
        split_by_transitions=True,
        _seed=500,
        seed_output_connected=True,
    )

    assert [item[1]["funpack_scene_seed"] for item in cond] == [500, 501]
    assert all(item[1]["funpack_seed_source"] == "base seed + scene index" for item in cond)


def test_split_by_transitions_uses_provided_scene_seeds(tmp_path):
    refiner = FunPackVideoRefinerV2()
    state_path = tmp_path / "state.json"
    refiner._v2_state_path = lambda refinement_key: str(state_path)

    cond, _, _, _, _, _, _ = refiner.refine_v2(
        "anchor, scene 1 she walks, scene 2 she turns",
        FakeClip(),
        "Perfect",
        "split-seed-test",
        split_by_transitions=True,
        _seed=500,
        _scene_seeds=[900, 901],
        _seed_source="successful seed memory: reused 900 from 'anchor'",
    )

    assert [item[1]["funpack_scene_seed"] for item in cond] == [900, 901]
    assert all(item[1]["funpack_seed_source"] == "successful seed memory" for item in cond)


# --- Stage 2 Part B: "Wrong appearance" consistency anchor ---

def _mean_cosine(a, b):
    a = a.reshape(-1, a.shape[-1]).float()
    b = b.reshape(-1, b.shape[-1]).float()
    a = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    b = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return float((a * b).sum(dim=-1).mean().item())


def test_appearance_anchor_stores_blessed_on_good_rating():
    refiner = FunPackVideoRefinerV2()
    global_state = {}
    payload = tensor_to_serializable(torch.randn(1, 6, 8))
    previous_run = {"conditioning": payload, "prompt": "woman in red dress"}

    status = refiner._v2_update_appearance_anchor(
        global_state, previous_run, normalize_refiner_v2_rating("Perfect"), 3
    )

    assert "blessed" in status.lower()
    assert global_state["appearance_anchor"]["conditioning"] is payload
    assert global_state["appearance_anchor"]["count"] == 1
    # A good rating should NOT populate the drift slot.
    assert "appearance_drift" not in global_state


def test_appearance_anchor_stores_drift_on_wrong_appearance():
    refiner = FunPackVideoRefinerV2()
    global_state = {}
    payload = tensor_to_serializable(torch.randn(1, 6, 8))
    previous_run = {"conditioning": payload, "prompt": "woman in red dress"}

    refiner._v2_update_appearance_anchor(
        global_state, previous_run, normalize_refiner_v2_rating("Wrong appearance"), 4
    )

    assert global_state["appearance_drift"]["conditioning"] is payload
    # Wrong appearance must not overwrite/seed the blessed anchor.
    assert "appearance_anchor" not in global_state


def test_good_rating_overwrites_blessed_anchor():
    refiner = FunPackVideoRefinerV2()
    global_state = {}
    first = tensor_to_serializable(torch.randn(1, 6, 8))
    second = tensor_to_serializable(torch.randn(1, 6, 8))

    refiner._v2_update_appearance_anchor(
        global_state, {"conditioning": first, "prompt": "a"}, normalize_refiner_v2_rating("Perfect"), 1
    )
    refiner._v2_update_appearance_anchor(
        global_state, {"conditioning": second, "prompt": "b"}, normalize_refiner_v2_rating("Nailed it"), 2
    )

    assert global_state["appearance_anchor"]["conditioning"] is second
    assert global_state["appearance_anchor"]["count"] == 2


def test_appearance_anchor_skips_when_no_conditioning_or_skip_learning():
    refiner = FunPackVideoRefinerV2()
    global_state = {}
    # No conditioning payload on the run.
    refiner._v2_update_appearance_anchor(
        global_state, {"prompt": "x"}, normalize_refiner_v2_rating("Perfect"), 1
    )
    assert "appearance_anchor" not in global_state
    # skip_learning rating (e.g. -Just forget it-) never stores.
    payload = tensor_to_serializable(torch.randn(1, 6, 8))
    refiner._v2_update_appearance_anchor(
        global_state, {"conditioning": payload}, normalize_refiner_v2_rating("-Just forget it-"), 1
    )
    assert "appearance_anchor" not in global_state


def test_wrong_appearance_pulls_toward_blessed_and_repels_drift():
    refiner = FunPackVideoRefinerV2()
    torch.manual_seed(0)
    original = torch.randn(1, 6, 8)
    blessed = torch.randn(1, 6, 8)
    drift = torch.randn(1, 6, 8)
    global_state = {
        "appearance_anchor": {"conditioning": tensor_to_serializable(blessed)},
        "appearance_drift": {"conditioning": tensor_to_serializable(drift)},
    }

    refined, status = refiner._v2_apply_conditioning_memory(
        original.clone(), global_state, normalize_refiner_v2_rating("Wrong appearance")
    )

    assert "appearance-anchor: pull→blessed + repel←drift" in status
    # Output moved toward the blessed appearance and away from the drift (direction).
    assert _mean_cosine(refined, blessed) > _mean_cosine(original, blessed)
    assert _mean_cosine(refined, drift) < _mean_cosine(original, drift)
    # Gentle: per-token norm is preserved and the net change is small.
    assert torch.allclose(
        refined.norm(dim=-1), original.norm(dim=-1), atol=1e-3
    )
    assert (refined - original).norm() / original.norm() < 0.08


def test_appearance_anchor_idle_without_blessed():
    refiner = FunPackVideoRefinerV2()
    torch.manual_seed(1)
    original = torch.randn(1, 6, 8)

    refined, status = refiner._v2_apply_conditioning_memory(
        original.clone(), {}, normalize_refiner_v2_rating("Wrong appearance")
    )

    assert "appearance-anchor idle" in status
    assert torch.allclose(refined, original, atol=1e-4)


def test_appearance_anchor_not_applied_on_non_wrong_appearance_rating():
    refiner = FunPackVideoRefinerV2()
    torch.manual_seed(2)
    original = torch.randn(1, 6, 8)
    blessed = torch.randn(1, 6, 8)
    global_state = {"appearance_anchor": {"conditioning": tensor_to_serializable(blessed)}}

    # A "Missing details" rating must not trigger the appearance anchor pull.
    refined, status = refiner._v2_apply_conditioning_memory(
        original.clone(), global_state, normalize_refiner_v2_rating("Missing details")
    )

    assert "appearance-anchor idle" in status


def _lora_relation(name, lora_type, phrase_text, phrase_category):
    """The relevance the suggester assigns one LoRA for one prompt phrase."""
    refiner = FunPackVideoRefinerV2()
    history = {}
    profile = normalize_refiner_v2_rating("Like")
    refiner._v2_update_lora_suggestions(
        {"loras": [{"id": "x", "name": name, "type": lora_type, "base_model_weight": 1.0}]},
        history,
        {"lora_weight_memory": {}},
        [{"text": phrase_text, "primary": phrase_category}],
        profile,
        refiner._v2_axis_feedback(profile, []),
    )
    return history["lora_weight_suggestions"]["x"]["relation"]


def test_a_loras_filename_does_not_decide_how_relevant_it_is():
    """Splitting the file's NAME into words and matching them against the prompt guessed at
    what a LoRA does from what someone happened to call it. Two identical LoRAs must score
    the same whether or not the filename echoes the prompt."""
    echoes = _lora_relation("running_motion.safetensors", "general", "running", "action")
    opaque = _lora_relation("v3_final_ep12.safetensors", "general", "running", "action")
    assert echoes == opaque


def test_relevance_still_comes_from_the_type_and_the_prompt():
    """The replacement must not be inert: a LoRA typed 'action' is still more relevant to an
    action prompt than an untyped one."""
    typed = _lora_relation("v3_final_ep12.safetensors", "action", "running", "action")
    untyped = _lora_relation("v3_final_ep12.safetensors", "general", "running", "action")
    assert typed > untyped
