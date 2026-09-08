"""build(): anchor + scene text + postfix, shortcuts expanded across the
whole combined string, then $variables resolved last. Order is what makes a
shortcut trigger spanning the anchor/scene boundary still match, and a
variable's value able to contain another variable or a shortcut trigger.
"""

from core import prompt_build, shortcuts


def _sc(triggers, replacements):
    return shortcuts.Shortcut(name=triggers[0], triggers=triggers, replacements=replacements)


def test_anchor_and_postfix_wrap_the_scene_text():
    out = prompt_build.build("a fox runs", anchor="cinematic", postfix="4k")
    assert out == "cinematic a fox runs 4k"


def test_postfix_is_dropped_when_disabled():
    out = prompt_build.build("a fox runs", anchor="cinematic", postfix="4k", postfix_enabled=False)
    assert out == "cinematic a fox runs"


def test_empty_anchor_and_postfix_leave_only_the_scene_text():
    assert prompt_build.build("a fox runs") == "a fox runs"


def test_a_shortcut_expands_inside_the_combined_string():
    out = prompt_build.build("a fox runs", anchor="", postfix="",
                              shortcuts=[_sc(["fox"], ["red fox"])])
    assert out == "a red fox runs"


def test_a_shortcut_can_span_the_anchor_and_scene_boundary():
    out = prompt_build.build("hour scene", anchor="golden",
                              shortcuts=[_sc(["golden hour"], ["warm light"])])
    assert out == "warm light scene"


def test_variables_resolve_after_shortcuts_expand():
    out = prompt_build.build("a $animal runs", shortcuts=[_sc(["runs"], ["runs fast"])],
                              variables=[{"name": "animal", "value": "fox"}])
    assert out == "a fox runs fast"


def test_a_variables_own_text_is_not_rescanned_for_shortcut_triggers():
    # variables resolve AFTER shortcut expansion, so a $var's own text is not
    # re-scanned for shortcut triggers -- only the scene/anchor/postfix text is.
    out = prompt_build.build("$style", shortcuts=[_sc(["fox"], ["red fox"])],
                              variables=[{"name": "style", "value": "a fox runs"}])
    assert out == "a fox runs"


def test_whitespace_only_scene_text_with_no_anchor_or_postfix_is_empty():
    assert prompt_build.build("   ") == ""
