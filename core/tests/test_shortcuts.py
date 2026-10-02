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
    monkeypatch.setattr(config, "SHORTCUT_CATEGORIES_FILE", tmp_path / "cats.json")
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


def test_the_same_cycle_resolves_differently_depending_on_which_end_is_asked_for():
    # A="$B", B="$A": which one comes back literal depends on where resolution
    # STARTS -- the memoization added to bound fan-out below must key on the
    # stack a variable is reached through, not on its name alone, or one of
    # these two would silently answer with the other's result.
    variables = [{"name": "a", "value": "$b"}, {"name": "b", "value": "$a"}]
    assert shortcuts.resolve_variables("$a", variables) == "$a"
    assert shortcuts.resolve_variables("$b", variables) == "$b"


def test_a_variable_referencing_the_same_name_twice_is_not_a_cycle_and_stays_fast():
    # "$v1 $v1" is not a cycle (v1 never appears in its own expansion chain),
    # so the cycle guard never trips -- and without memoization, evaluating
    # the two "$v1" occurrences independently makes a chain of variables that
    # each reference the next twice cost 2**N evaluations of the deepest one.
    # 24 variables to double every one of the last variable's evaluations
    # would need real seconds and tens of megabytes without a cache or an
    # output-size cap; this finishes near-instantly and stays bounded.
    depth = 24
    variables = [{"name": f"v{i}", "value": f"$v{i + 1} $v{i + 1}"} for i in range(depth)]
    variables.append({"name": f"v{depth}", "value": "leaf"})
    out = shortcuts.resolve_variables("$v0", variables)
    assert "leaf" in out
    assert len(out) <= shortcuts._VARIABLE_MAX_OUTPUT + 32  # capped, not 2**24 copies of "leaf "


def test_a_leading_dollar_in_the_declared_name_is_ignored():
    out = shortcuts.resolve_variables("$subject", [{"name": "$subject", "value": "fox"}])
    assert out == "fox"


def test_no_variables_declared_returns_text_unchanged():
    assert shortcuts.resolve_variables("plain $x text", []) == "plain $x text"


# ── categories, export, import ───────────────────────────────────────────

def test_a_made_category_survives_empty_and_a_used_one_needs_no_entry(store):
    shortcuts.add_category("Camera", "Moves")
    shortcuts.save({"name": "x", "triggers": ["x"], "replacements": ["y"], "category": "Light"})
    assert shortcuts.categories() == [
        {"name": "Camera", "sub_categories": ["Moves"]}, {"name": "Light", "sub_categories": []}]
    with pytest.raises(ValueError):
        shortcuts.add_category("  ")
    shortcuts.clear()
    assert shortcuts.categories() == []


def test_export_then_replace_import_is_the_same_library(store):
    shortcuts.save({"name": "Fox", "triggers": ["fox", "vixen"], "replacements": ["red fox", ""],
                    "category": "Animals", "sub_category": "Wild"})
    shortcuts.add_category("Empty")
    payload = shortcuts.export_payload()
    shortcuts.clear()
    assert shortcuts.import_payload(payload, "replace") == 1
    assert [s.to_dict() for s in shortcuts.listing()] == payload["shortcuts"]
    assert shortcuts.categories() == payload["categories"]


def test_import_reads_a_v4_file_and_merge_keeps_what_is_here(store):
    shortcuts.save({"name": "Keep", "triggers": ["keep"], "replacements": ["k"]})
    v4 = {"version": 1, "shortcuts": {"fox": {"name": "Fox", "activation_words": ["fox"],
          "replacement": ["red fox"], "refinement_key": "k1", "category": "Animals"},
          "bad": {"name": "NoTrigger"}}, "categories": [{"name": "Animals", "sub_categories": ["Wild"]}]}
    assert shortcuts.import_payload(v4, "merge") == 1
    assert sorted(s.name for s in shortcuts.listing()) == ["Fox", "Keep"]
    assert shortcuts.categories() == [{"name": "Animals", "sub_categories": ["Wild"]}]
    shortcuts.import_payload(v4, "merge")
    assert len(shortcuts.listing()) == 2          # same name replaces, never duplicates


def test_import_refuses_what_holds_nothing_and_changes_nothing(store):
    shortcuts.save({"name": "Keep", "triggers": ["keep"], "replacements": ["k"]})
    for bad in ({"shortcuts": {"a": {"name": "no trigger"}}}, "nope", {"shortcuts": 5}):
        with pytest.raises(ValueError):
            shortcuts.import_payload(bad, "replace")
    with pytest.raises(ValueError):
        shortcuts.import_payload([{"triggers": ["a"]}], "bogus")
    assert [s.name for s in shortcuts.listing()] == ["Keep"]


def test_a_bad_categories_shape_is_refused_before_anything_is_written(store):
    shortcuts.save({"name": "Keep", "triggers": ["keep"], "replacements": ["k"]})
    ok = [{"name": "A", "triggers": ["a"], "replacements": ["1"]}]
    # a wrong-typed sub_categories is ignored, not exploded into letters or crashed on
    assert shortcuts.import_payload({"shortcuts": ok, "categories": [{"name": "C", "sub_categories": 5}, {"name": "D", "sub_categories": "Wild"}]}, "replace") == 1
    assert [s.name for s in shortcuts.listing()] == ["A"]
    assert shortcuts.categories() == [{"name": "C", "sub_categories": []}, {"name": "D", "sub_categories": []}]


def test_names_are_one_identity_everywhere(store):
    shortcuts.save({"name": "Fox", "triggers": ["fox"], "replacements": ["a"]})
    shortcuts.save({"name": "fox", "triggers": ["fox2"], "replacements": ["b"]})     # same shortcut, updated
    assert [s.triggers for s in shortcuts.listing()] == [["fox2"]]
    shortcuts.save({"name": "Cat", "triggers": ["cat"], "replacements": ["c"]})
    with pytest.raises(ValueError):                       # a rename onto an existing name
        shortcuts.save({"name": "CAT", "triggers": ["cat"], "replacements": ["c"]}, original_name="fox")
    assert sorted(s.name for s in shortcuts.listing()) == ["Cat", "fox"]


def test_import_counts_what_was_stored_and_the_key_names_a_nameless_row(store):
    data = {"shortcuts": {"Alpha": {"name": "", "triggers": ["x", "y"]},
                          "b": {"triggers": ["x", "z"]}, "b2": {"name": "B", "triggers": ["q"]}}}
    assert shortcuts.import_payload(data) == 2            # "b" and "B" are one name: the later row wins
    assert sorted(s.name for s in shortcuts.listing()) == ["Alpha", "B"]


def test_odd_types_never_become_their_repr(store):
    shortcuts.save({"name": "T", "triggers": "a, b;c", "replacements": [{"a": 1}, "ok"], "category": 5, "sub_category": ["x"]})
    s = shortcuts.listing()[0]
    assert s.triggers == ["a", "b", "c"] and s.replacements == ["ok"] and (s.category, s.sub_category) == ("", "")
    with pytest.raises(ValueError):
        shortcuts.add_category(5)


def test_replacements_are_prose_split_on_newlines_only(store):
    s = shortcuts.Shortcut.from_dict({"name": "P", "triggers": ["p"], "replacements": "a big, red fox; run\nsecond"})
    assert s.replacements == ["a big, red fox; run", "second"]


def test_a_library_with_case_duplicates_edits_and_deletes_the_named_one(store):
    shortcuts._save_all([shortcuts.Shortcut(name="Fox", triggers=["a"], replacements=["1"]),
                         shortcuts.Shortcut(name="fox", triggers=["b"], replacements=["2"])])
    shortcuts.save({"name": "fox", "triggers": ["b2"], "replacements": ["2"]}, original_name="fox")
    assert [(s.name, s.triggers) for s in shortcuts.listing()] == [("Fox", ["a"]), ("fox", ["b2"])]
    shortcuts.delete("fox")
    assert [s.name for s in shortcuts.listing()] == ["Fox"]
    shortcuts.delete("FOX")                      # no exact match: the case-insensitive one goes
    assert shortcuts.listing() == []
