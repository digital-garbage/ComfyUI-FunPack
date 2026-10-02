"""Module control: the project's switch and the persistent quarantine."""

import types

from core import control, patching


def spec(mid, provides=("modifier",), source="nowhere"):
    return types.SimpleNamespace(id=mid, provides={p: (lambda *a, **k: None) for p in provides}, source=source)


def test_only_modules_that_modify_a_run_can_be_switched():
    assert control.controllable(spec("a", ("modifier",)))
    assert control.controllable(spec("a", ("sampler_modifier",)))
    assert not control.controllable(spec("loader", ("routes",)))


def test_a_module_the_project_turned_off_is_left_out_and_said():
    kept, notes = control.partition([spec("a"), spec("b")], {"_off": {"modules": ["a"]}})
    assert [s.id for s in kept] == ["b"] and "a: turned off for this project" in notes[0]


def test_structure_cannot_be_turned_off():
    kept, _ = control.partition([spec("loader", ("routes",))], {"_off": {"modules": ["loader"]}})
    assert [s.id for s in kept] == ["loader"]


def test_a_fault_is_remembered_across_runs_until_released():
    a = spec("a")
    control.quarantine(a, "RuntimeError: boom")
    kept, notes = control.partition([a], {})
    assert kept == [] and "a: OFF" in notes[0] and "boom" in notes[0] and "Settings ▸ Modules" in notes[0]
    assert control.fingerprint() == "a"
    assert control.release("a") and not control.release("a")
    assert [s.id for s in control.partition([a], {})[0]] == ["a"]


def test_a_module_whose_code_changed_since_it_failed_gets_another_try(monkeypatch):
    a = spec("a")
    monkeypatch.setattr(control, "signature", lambda s: "v1")
    control.quarantine(a, "boom")
    monkeypatch.setattr(control, "signature", lambda s: "v2")
    assert [s.id for s in control.partition([a], {})[0]] == ["a"]
    assert control.fingerprint() == ""


def test_an_interrupt_or_out_of_memory_is_not_the_modules_fault(monkeypatch):
    from core import registry
    a = spec("a")
    monkeypatch.setattr(registry, "current", lambda: types.SimpleNamespace(specs={"a": a}))

    class InterruptProcessingException(Exception): ...
    class OutOfMemoryError(Exception): ...

    control.fault("funpack.a", InterruptProcessingException())
    control.fault("funpack.a", OutOfMemoryError())
    assert control.fingerprint() == ""
    control.fault("funpack.a", ValueError("bad"))
    assert control.fingerprint() == "a"


def test_a_hook_that_raised_in_a_run_reaches_the_quarantine(monkeypatch):
    from core import registry
    a = spec("a")
    monkeypatch.setattr(registry, "current", lambda: types.SimpleNamespace(specs={"a": a}))
    dropped = patching.Dropped()
    assert dropped.record("funpack.a", RuntimeError("step 3"))
    assert "a" in control.quarantined([a]) and "step 3" in control.quarantined([a])["a"]["reason"]


def test_a_malformed_off_entry_is_named():
    assert control.bad_off({"modules": ["a"]}) is None and control.bad_off(None) is None
    assert "modules" in control.bad_off(["a"]) and "modules" in control.bad_off({"modules": [1]})
