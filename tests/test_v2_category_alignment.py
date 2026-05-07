import sys
import types

sys.modules.setdefault("folder_paths", types.SimpleNamespace(models_dir=""))

from conditioning import FunPackVideoRefinerV2


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
