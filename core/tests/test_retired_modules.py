"""A project saved while a module existed must still run after that module is removed."""
import json


def test_values_saved_for_a_retired_module_are_ignored_but_a_typo_is_still_refused(comfyui):
    from modules.sampling.modifiers.nodes import FunPackModifierSettings as N
    old = json.dumps({"appearance": {"enabled": True}})
    assert N._problems(old) == []
    assert N.execute(old).args[1] == "settings for nothing set"
    assert any("no module named 'appearence'" in p for p in N._problems(json.dumps({"appearence": {}})))
