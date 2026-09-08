"""The shortcut library and its expansion algorithm -- a close port of v4's
`templates.py` (`apply_prompt_shortcuts`/`resolve_variables`), pinned against
the same behaviour: longest-trigger-first, seeded multi-choice, cycle-safe
variables, undefined variables left literal.
"""

import pytest

from core import config, shortcuts


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SHORTCUTS_FILE", tmp_path / "shortcuts.json")
    return tmp_path


# ── the library (CRUD) ───────────────────────────────────────────────────

def test_a_fresh_install_has_no_shortcuts(store):
    assert shortcuts.listing() == []


def test_a_saved_shortcut_round_trips(store):
    shortcuts.save({"name": "Golden hour", "triggers": ["golden hour"],
                     "replacements": ["warm golden-hour lighting"]})
    items = shortcuts.listing()
    assert len(items) == 1
    assert items[0].name == "Golden hour"
    assert items[0].triggers == ["golden hour"]


def test_a_shortcut_with_no_triggers_is_refused(store):
    with pytest.raises(ValueError):
        shortcuts.save({"name": "Nothing", "triggers": [], "replacements": ["x"]})
    assert shortcuts.listing() == []


def test_saving_under_original_name_updates_in_place(store):
    shortcuts.save({"name": "A", "triggers": ["a"], "replacements": ["1"]})
    shortcuts.save({"name": "B", "triggers": ["b"], "replacements": ["2"]})
    shortcuts.save({"name": "A renamed", "triggers": ["a"], "replacements": ["1!"]},
                    original_name="A")
    items = shortcuts.listing()
    assert [i.name for i in items] == ["A renamed", "B"]  # position kept, not moved to the end


def test_saving_under_a_new_name_appends(store):
    shortcuts.save({"name": "A", "triggers": ["a"], "replacements": ["1"]})
    shortcuts.save({"name": "B", "triggers": ["b"], "replacements": ["2"]})
    assert [i.name for i in shortcuts.listing()] == ["A", "B"]


def test_delete_removes_by_name(store):
    shortcuts.save({"name": "A", "triggers": ["a"], "replacements": ["1"]})
    shortcuts.save({"name": "B", "triggers": ["b"], "replacements": ["2"]})
    shortcuts.delete("A")
    assert [i.name for i in shortcuts.listing()] == ["B"]


def test_clear_empties_the_library(store):
    shortcuts.save({"name": "A", "triggers": ["a"], "replacements": ["1"]})
    shortcuts.clear()
    assert shortcuts.listing() == []


def test_replacement_and_trigger_lists_are_cleaned(store):
    shortcuts.save({"name": "A", "triggers": ["  a  ", "a", "b"],
                     "replacements": ["  x  ", "x", ""]})
    item = shortcuts.listing()[0]
    assert item.triggers == ["a", "b"]         # trimmed, de-duped
    assert item.replacements == ["x", ""]       # "" kept -- a deliberate remove entry


# ── expansion ─────────────────────────────────────────────────────────────

def _sc(name, triggers, replacements, enabled=True):
    return shortcuts.Shortcut(name=name, triggers=triggers, replacements=replacements,
                               enabled=enabled)


def test_a_trigger_expands_to_its_replacement():
    out = shortcuts.expand("a man at golden hour", [_sc("gh", ["golden hour"], ["warm light"])])
    assert out == "a man at warm light"


def test_matching_is_case_insensitive():
    out = shortcuts.expand("GOLDEN HOUR scene", [_sc("gh", ["golden hour"], ["warm light"])])
    assert out == "warm light scene"


def test_a_disabled_shortcut_never_matches():
    out = shortcuts.expand("golden hour", [_sc("gh", ["golden hour"], ["warm light"], enabled=False)])
    assert out == "golden hour"


def test_longest_trigger_wins_over_a_shorter_one_it_contains():
    scs = [_sc("short", ["golden"], ["SHORT"]), _sc("long", ["golden hour"], ["LONG"])]
    assert shortcuts.expand("golden hour", scs) == "LONG"


def test_a_trigger_inside_a_longer_word_does_not_match():
    out = shortcuts.expand("goldenrod field", [_sc("gh", ["golden"], ["warm"])])
    assert out == "goldenrod field"  # word-boundary guard, not a substring replace


def test_expansion_is_deterministic_for_the_same_text():
    sc = _sc("many", ["cat"], ["tabby", "siamese", "calico", "tuxedo"])
    a = shortcuts.expand("a cat sits", [sc])
    b = shortcuts.expand("a cat sits", [sc])
    assert a == b  # same text -> same seed -> same random.Random draw


def test_expansion_can_differ_for_different_seeds():
    sc = _sc("many", ["cat"], ["tabby", "siamese", "calico", "tuxedo", "ragdoll", "sphynx"])
    picks = {shortcuts.expand("a cat", [sc], seed=s) for s in range(1, 8)}
    assert len(picks) > 1  # different explicit seeds draw different replacements


def test_an_empty_replacement_removes_the_phrase_and_its_stray_comma():
    out = shortcuts.expand("tall, blurry, man", [_sc("rm", ["blurry"], [""])])
    assert out == "tall, man"


def test_no_shortcuts_leaves_text_untouched():
    assert shortcuts.expand("plain text", []) == "plain text"


def test_empty_text_returns_empty():
    assert shortcuts.expand("", [_sc("a", ["a"], ["b"])]) == ""


def test_a_shortcut_with_no_replacements_never_matches():
    assert shortcuts.expand("trigger word", [_sc("x", ["trigger"], [])]) == "trigger word"


# ── $variables ────────────────────────────────────────────────────────────

def test_a_declared_variable_is_substituted():
    out = shortcuts.resolve_variables("a $subject walks", [{"name": "subject", "value": "fox"}])
    assert out == "a fox walks"


def test_an_undefined_variable_is_left_literal():
    out = shortcuts.resolve_variables("a $ghost walks", [])
    assert out == "a $ghost walks"


def test_a_variable_may_reference_another():
    out = shortcuts.resolve_variables("$a", [{"name": "a", "value": "$b"}, {"name": "b", "value": "done"}])
    assert out == "done"


def test_a_self_referencing_variable_is_left_literal_not_infinitely_expanded():
    out = shortcuts.resolve_variables("$a", [{"name": "a", "value": "x $a y"}])
    assert out == "x $a y"


def test_a_two_variable_cycle_is_left_literal():
    out = shortcuts.resolve_variables(
        "$a", [{"name": "a", "value": "$b"}, {"name": "b", "value": "$a"}])
    assert out == "$a"  # the cycle is detected, not expanded forever


def test_a_leading_dollar_in_the_declared_name_is_ignored():
    out = shortcuts.resolve_variables("$subject", [{"name": "$subject", "value": "fox"}])
    assert out == "fox"


def test_no_variables_declared_returns_text_unchanged():
    assert shortcuts.resolve_variables("plain $x text", []) == "plain $x text"
