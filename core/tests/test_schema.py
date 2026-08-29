"""Validation refuses rather than repairs.

Every case here is a module that would otherwise load and then behave in a way
nobody declared -- a panel rendering the wrong control, a headless run with an
invented default, a row that can never appear.
"""

import pytest

from core.contract import CONTRACT_VERSION, ModuleSpec
from core.schema import SchemaError, validate


def announce(**overrides):
    base = {
        "id": "audio_clock", "title": "Audio clock", "mount": "generation.timing",
        "settings": {"enabled": {"type": "bool", "default": True, "label": "On"}},
    }
    base.update(overrides)
    return base


def test_a_valid_declaration_becomes_a_spec():
    spec = validate(announce())
    assert isinstance(spec, ModuleSpec)
    assert spec.id == "audio_clock"
    assert spec.stage == "sampling"
    assert spec.status == "experimental"


def test_defaults_come_from_the_same_declaration_the_panel_renders():
    # This is the whole point of one declaration doing three jobs: a headless
    # run and a rendered panel cannot disagree about a starting value.
    spec = validate(announce(settings={
        "enabled": {"type": "bool", "default": True, "label": "On"},
        "strength": {"type": "float", "default": 0.65, "min": 0, "max": 1, "label": "Strength"},
    }))
    assert spec.defaults() == {"enabled": True, "strength": 0.65}


@pytest.mark.parametrize("missing", ["id", "title", "mount"])
def test_the_essentials_are_required(missing):
    raw = announce()
    del raw[missing]
    with pytest.raises(SchemaError, match=missing):
        validate(raw)


def test_a_blank_id_is_not_an_id():
    with pytest.raises(SchemaError):
        validate(announce(id="   "))


# --- types and renderers ---------------------------------------------------

def test_unknown_type_is_refused():
    with pytest.raises(SchemaError, match="unknown type"):
        validate(announce(settings={"x": {"type": "colour", "default": "#fff", "label": "C"}}))


def test_unknown_ui_hint_is_refused():
    # Accepting it would render something other than what was asked for.
    with pytest.raises(SchemaError, match="not available for a bool"):
        validate(announce(settings={"x": {"type": "bool", "default": True, "label": "X", "ui": "slider"}}))


def test_a_valid_ui_hint_passes():
    spec = validate(announce(settings={"x": {"type": "bool", "default": True, "label": "X", "ui": "toggle"}}))
    assert spec.settings["x"]["ui"] == "toggle"


def test_enum_may_ask_for_the_wheel():
    # How a module gets a radial picker without writing any JavaScript.
    spec = validate(announce(settings={"x": {
        "type": "enum", "default": "a", "label": "X", "ui": "wheel",
        "options": [{"value": "a", "label": "A"}, {"value": "b", "label": "B"}],
    }}))
    assert spec.settings["x"]["ui"] == "wheel"


# --- defaults --------------------------------------------------------------

def test_a_setting_without_a_default_is_refused():
    with pytest.raises(SchemaError, match="no default"):
        validate(announce(settings={"x": {"type": "float", "min": 0, "max": 1, "label": "X"}}))


@pytest.mark.parametrize("spec", [
    {"type": "bool", "default": "yes", "label": "X"},
    {"type": "int", "default": 1.5, "label": "X"},
    {"type": "float", "default": "0.5", "label": "X"},
    {"type": "text", "default": 3, "label": "X"},
    {"type": "bool", "default": 1, "label": "X"},
])
def test_a_default_of_the_wrong_type_is_refused(spec):
    with pytest.raises(SchemaError):
        validate(announce(settings={"x": spec}))


def test_a_default_outside_its_own_bounds_is_refused():
    with pytest.raises(SchemaError, match="below its min"):
        validate(announce(settings={"x": {"type": "float", "default": -1, "min": 0, "max": 1, "label": "X"}}))
    with pytest.raises(SchemaError, match="above its max"):
        validate(announce(settings={"x": {"type": "float", "default": 9, "min": 0, "max": 1, "label": "X"}}))


def test_inverted_bounds_are_refused():
    with pytest.raises(SchemaError, match="above max"):
        validate(announce(settings={"x": {"type": "int", "default": 0, "min": 5, "max": 1, "label": "X"}}))


def test_a_non_positive_step_is_refused():
    with pytest.raises(SchemaError, match="step"):
        validate(announce(settings={"x": {"type": "float", "default": 0.5, "step": 0, "label": "X"}}))


# --- content is text -------------------------------------------------------

@pytest.mark.parametrize("key", ["label", "hint", "unit"])
def test_content_must_be_text(key):
    # The UI renders these as text and will not interpret anything else.
    with pytest.raises(SchemaError, match=key):
        validate(announce(settings={"x": {"type": "bool", "default": True, key: {"markup": "<b>"}}}))


def test_option_labels_must_be_text_and_values_scalar():
    with pytest.raises(SchemaError, match="label"):
        validate(announce(settings={"x": {"type": "enum", "default": "a", "label": "X",
                                          "options": [{"value": "a", "label": ["a"]}]}}))
    with pytest.raises(SchemaError, match="scalar"):
        validate(announce(settings={"x": {"type": "enum", "default": "a", "label": "X",
                                          "options": [{"value": {"a": 1}, "label": "A"}]}}))


# --- enums -----------------------------------------------------------------

def test_an_enum_needs_options():
    with pytest.raises(SchemaError, match="no options"):
        validate(announce(settings={"x": {"type": "enum", "default": "a", "label": "X", "options": []}}))


def test_an_enum_default_must_be_one_of_its_options():
    with pytest.raises(SchemaError, match="not one of its options"):
        validate(announce(settings={"x": {"type": "enum", "default": "z", "label": "X",
                                          "options": [{"value": "a", "label": "A"}]}}))


def test_duplicate_option_values_are_refused():
    with pytest.raises(SchemaError, match="duplicate"):
        validate(announce(settings={"x": {"type": "enum", "default": "a", "label": "X",
                                          "options": [{"value": "a", "label": "A"}, {"value": "a", "label": "B"}]}}))


# --- conditional visibility ------------------------------------------------

def test_when_may_only_name_siblings():
    # A condition on a key that does not exist is always false, so the row would
    # simply never appear -- a typo that reads as a deliberate decision.
    with pytest.raises(SchemaError, match="does not declare"):
        validate(announce(settings={
            "a": {"type": "bool", "default": True, "label": "A"},
            "b": {"type": "bool", "default": True, "label": "B", "when": {"nope": True}},
        }))


def test_when_may_not_name_itself():
    with pytest.raises(SchemaError, match="itself"):
        validate(announce(settings={"a": {"type": "bool", "default": True, "label": "A", "when": {"a": True}}}))


def test_a_valid_when_survives_into_the_spec():
    spec = validate(announce(settings={
        "enabled": {"type": "bool", "default": True, "label": "On"},
        "strength": {"type": "float", "default": 0.5, "label": "S", "when": {"enabled": True}},
    }))
    assert spec.settings["strength"]["when"] == {"enabled": True}


# --- the rest of the announcement ------------------------------------------

def test_unknown_stage_is_refused():
    with pytest.raises(SchemaError, match="unknown stage"):
        validate(announce(stage="whenever"))


def test_unknown_status_is_refused():
    with pytest.raises(SchemaError, match="unknown status"):
        validate(announce(status="probably-fine"))


def test_relations_must_be_lists_of_ids():
    with pytest.raises(SchemaError, match="after"):
        validate(announce(after="audio_clock"))     # a bare string is a common slip
    with pytest.raises(SchemaError, match="requires"):
        validate(announce(requires=[1, 2]))


def test_the_manifest_carries_what_the_ui_needs():
    spec = validate(announce(requires=["audio_stream"], after=["x"], stage="latent"))
    manifest = spec.to_manifest()
    assert manifest["id"] == "audio_clock"
    assert manifest["requires"] == ["audio_stream"]
    assert manifest["stage"] == "latent"
    assert "source" not in manifest, "the import path is core's business, not the browser's"


def test_the_contract_has_a_version():
    assert isinstance(CONTRACT_VERSION, int)


# --- the door nobody was watching ------------------------------------------

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_modules_own_default_cannot_be_non_finite(bad):
    """The override path refused NaN while a module's own default sailed
    through -- and a default is the value MOST runs use, because a setting
    nobody names never reaches the override check at all. `sigma > nan` is
    False at every step, so it becomes a permanent silent no-op."""
    with pytest.raises(SchemaError, match="finite"):
        validate({"id": "m", "title": "M", "mount": "x", "settings": {
            "strength": {"type": "float", "default": bad, "min": 0.0, "max": 1.0,
                         "label": "S"}}})


@pytest.mark.parametrize("bound", ["min", "max"])
def test_a_non_finite_bound_is_refused_too(bound):
    """An infinite bound makes the range check meaningless in one direction."""
    settings = {"strength": {"type": "float", "default": 0.5, "label": "S",
                             bound: float("inf")}}
    with pytest.raises(SchemaError, match="finite"):
        validate({"id": "m", "title": "M", "mount": "x", "settings": settings})


def test_an_ordinary_default_still_passes():
    spec = validate({"id": "m", "title": "M", "mount": "x", "settings": {
        "strength": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "label": "S"}}})
    assert spec.defaults() == {"strength": 0.5}
