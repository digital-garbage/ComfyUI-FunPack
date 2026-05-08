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
    learning_labels = [label for label in V2_RATING_LABELS if label != "-Just forget it-"]

    for label in learning_labels:
        entry, _, _ = train_phrase(refiner, f"running test {label}", label)

        assert entry["category_evidence_count"] == 1
        assert set(entry["category_weights"]) == set(refiner.CATEGORY_DESCRIPTIONS)
        assert set(entry["clip_heuristic_scores"]) == set(refiner.CATEGORY_DESCRIPTIONS)
        assert set(entry["effective_category_scores"]) == set(refiner.CATEGORY_DESCRIPTIONS)


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
        {"prompt": "woman walking reaching hand tiny particles rain reflections", "phrases": liked_phrases},
        profile,
        1,
        feedback,
    )

    missing_profile = normalize_refiner_v2_rating("Missing details + action")
    missing_feedback = refiner._v2_axis_feedback(missing_profile, None)
    current_phrases = refiner._v2_classify_phrases(
        None,
        prompt_items(refiner, ["woman", "rain"]),
        global_state,
    )

    repaired, status, candidates = refiner._v2_repair_prompt_for_missing_axes(
        "woman, rain",
        current_phrases,
        global_state,
        None,
        missing_feedback,
    )

    assert "Preferred context stored" in refiner._v2_update_preferred_context_memory(
        global_state,
        {"phrases": liked_phrases},
        profile,
        2,
        feedback,
    )
    assert "walking" in repaired
    assert "tiny particles" in repaired
    assert "preferred_context" in status
    assert len(candidates) >= 2


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
