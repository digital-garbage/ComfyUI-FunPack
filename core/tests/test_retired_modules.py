"""A project saved while a module existed must still run after that module is gone, whatever its name."""
import json


def test_values_saved_for_a_module_that_no_longer_exists_are_ignored_not_refused(comfyui):
    from modules.sampling.modifiers.nodes import FunPackModifierSettings as N
    for gone in ("appearance", "appearence", "any_future_removed_module"):
        saved = json.dumps({gone: {"enabled": True}})
        assert N._problems(saved) == []
        assert N.execute(saved).args[1] == "settings for nothing set"


def test_a_name_with_no_module_is_said_in_the_log(comfyui):
    from core import log
    from modules.sampling.modifiers.nodes import FunPackModifierSettings as N
    log._reset()
    N.execute(json.dumps({"gone_module": {"x": 1}}))
    assert any("'gone_module'" in r["message"] and "ignored" in r["message"] for r in log.history())
