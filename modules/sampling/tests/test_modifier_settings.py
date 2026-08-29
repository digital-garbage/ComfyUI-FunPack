"""The settings payload, checked rather than trusted.

v4 carried these as JSON in a string widget that nothing validated. The cost was
visible in the code: a hand-mirrored table of every default, "regenerated from
engine_settings.js by a test", plus a list of keys that no longer meant anything
and had to be filtered out so they could not reach the render. Every test here
is a thing that silently did nothing in v4.
"""

import pytest


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy."""


@pytest.fixture
def registry(monkeypatch):
    from core import registry as registry_mod
    from core.contract import ModuleSpec

    fake = registry_mod.Registry()
    fake.specs = {
            "alg": ModuleSpec(id="alg", title="ALG", mount="", settings={
                "enabled": {"type": "bool", "default": False, "label": "On"},
                "strength": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "label": "S"},
                "steps": {"type": "int", "default": 2, "min": 0, "max": 8, "label": "N"},
                "mode": {"type": "enum", "default": "beat", "label": "M", "options": [
                    {"value": "beat", "label": "Beat"}, {"value": "flat", "label": "Flat"}]},
            }),
    }
    monkeypatch.setattr(registry_mod, "current", lambda rescan=False: fake)
    return fake


def _run(text):
    from modules.sampling.modifiers.nodes import FunPackModifierSettings
    return FunPackModifierSettings.execute(text).result


def test_a_valid_payload_comes_back_merged_over_the_defaults(registry):
    values, _status = _run('{"alg": {"enabled": true}}')
    assert values["alg"] == {"enabled": True, "strength": 0.5, "steps": 2, "mode": "beat"}


def test_an_empty_payload_is_fine(registry):
    values, _status = _run("")
    assert values == {}


def test_malformed_json_says_so_instead_of_meaning_nothing(registry):
    with pytest.raises(RuntimeError, match="not valid JSON"):
        _run("{not json")


def test_a_setting_no_module_declares_is_reported(registry):
    """In v4 this was simply ignored, which is how dead keys accumulated."""
    with pytest.raises(RuntimeError, match="no setting named 'stength'"):
        _run('{"alg": {"stength": 0.5}}')


def test_a_module_that_is_not_installed_is_reported(registry):
    with pytest.raises(RuntimeError, match="no module named 'ghost'"):
        _run('{"ghost": {"x": 1}}')


@pytest.mark.parametrize("payload,expected", [
    ('{"alg": {"strength": "loud"}}', "float"),
    ('{"alg": {"strength": 2.0}}', "above its maximum"),
    ('{"alg": {"strength": -1.0}}', "below its minimum"),
    ('{"alg": {"steps": 1.5}}', "int"),
    ('{"alg": {"enabled": "yes"}}', "bool"),
    ('{"alg": {"mode": "swing"}}', "must be one of"),
])
def test_a_wrong_value_is_refused_not_coerced(registry, payload, expected):
    with pytest.raises(RuntimeError, match=expected):
        _run(payload)


def test_a_bool_is_not_accepted_as_a_number(registry):
    """True == 1 in Python, so an unguarded numeric check lets a checkbox through
    as a strength."""
    with pytest.raises(RuntimeError, match="float"):
        _run('{"alg": {"strength": true}}')


def test_a_payload_that_is_not_an_object_is_refused(registry):
    with pytest.raises(RuntimeError, match="keyed by module id"):
        _run("[1, 2, 3]")


def test_every_problem_is_reported_at_once(registry):
    """One error per run means finding them one at a time."""
    with pytest.raises(RuntimeError) as raised:
        _run('{"alg": {"strength": 9.0, "steps": 99}}')
    message = str(raised.value)
    assert "above its maximum 1.0" in message and "above its maximum 8" in message


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_a_non_finite_number_is_refused(registry, literal):
    """json.loads accepts all three by default, and NaN compares False against
    BOTH bounds -- so a range check alone lets it through and it reaches sampling
    as a strength that poisons everything it touches."""
    with pytest.raises(RuntimeError, match="finite|maximum|minimum"):
        _run('{"alg": {"strength": %s}}' % literal)


def test_nan_specifically_is_not_silently_accepted(registry):
    import math
    try:
        values, _ = _run('{"alg": {"strength": NaN}}')
    except RuntimeError:
        return                      # refused, which is the point
    pytest.fail(f"NaN was accepted as {values['alg']['strength']!r} "
                f"(isnan={math.isnan(values['alg']['strength'])})")
