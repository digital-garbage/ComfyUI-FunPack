"""A project saved while a module existed must still run after that module is gone, whatever its name."""
import json


def test_values_saved_for_a_module_that_no_longer_exists_are_ignored_not_refused(comfyui):
    from modules.sampling.modifiers.nodes import FunPackModifierSettings as N
    for gone in ("appearance", "appearence", "any_future_removed_module"):
        saved = json.dumps({gone: {"enabled": True}})
        assert N._problems(saved) == []
        assert N.execute(saved).args[1] == "settings for nothing set"
