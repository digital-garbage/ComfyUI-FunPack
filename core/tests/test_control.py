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


def test_an_off_or_quarantined_modules_values_are_not_checked(monkeypatch):
    from core import registry
    a, b = spec("a"), spec("b")
    monkeypatch.setattr(registry, "current", lambda: types.SimpleNamespace(specs={"a": a, "b": b}))
    control.quarantine(b, "boom")
    assert control.skipped({"_off": {"modules": ["a"]}}) == {"a", "b"}
    assert control.skipped({}) == {"b"}


def test_edited_code_that_this_process_has_not_loaded_stays_off_and_says_restart(monkeypatch):
    a = spec("a")
    sig = {"now": "v1"}
    monkeypatch.setattr(control, "signature", lambda s: sig["now"])
    control._loaded["a"] = "v1"
    control.quarantine(a, "boom")
    sig["now"] = "v2"                                   # edited, not restarted
    kept, notes = control.partition([a], {})
    assert kept == [] and "restart ComfyUI" in notes[0]
    control._loaded["a"] = "v2"                         # after a restart the new code is what ran
    assert [s.id for s in control.partition([a], {})[0]] == ["a"]


def test_a_quarantine_that_cannot_be_written_still_holds_for_the_session(monkeypatch, tmp_path):
    a = spec("a")
    monkeypatch.setattr(control.config, "QUARANTINE_FILE", tmp_path / "no" / "such" / "dir" / "q.json")
    control.quarantine(a, "boom")
    assert control.partition([a], {})[0] == []
    control.release("a")
    assert [s.id for s in control.partition([a], {})[0]] == ["a"]


def test_running_out_of_memory_by_message_is_not_the_modules_fault(monkeypatch):
    from core import registry
    a = spec("a")
    monkeypatch.setattr(registry, "current", lambda: types.SimpleNamespace(specs={"a": a}))
    control.fault("funpack.a", RuntimeError("MPS backend out of memory (MPS allocated: 9 GiB)"))
    assert control.fingerprint() == ""

