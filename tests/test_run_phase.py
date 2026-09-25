"""run_phase — the live "what is the sampler doing" label behind the progress readout.

The one invariant that actually breaks in production: the WRITER (samplers.py) reaches this
module as a package-relative import and the READER (movie_editor's bridge) imports FunPack
modules top-level by name, so the two can hold different module objects. State in module
globals would be two separate copies and the label would never arrive. It lives on `sys`
for exactly that reason, and this pins it down.
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_phase  # noqa: E402


def _second_import():
    """A second, independent module object for the same file — what the other import path
    produces at runtime."""
    spec = importlib.util.spec_from_file_location("other_pkg.run_phase", ROOT / "run_phase.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_both_import_paths_see_the_same_label():
    other = _second_import()
    run_phase.set_phase("scene 2/3 · pass 2 of 2")
    assert other.current()["label"] == "scene 2/3 · pass 2 of 2"
    other.clear()
    assert run_phase.current()["label"] == ""


def test_seq_only_ever_increases():
    """A poller uses it to tell "still the same phase" from "this phase came round again",
    so it must never reset — not even when the label is cleared."""
    before = run_phase.current()["seq"]
    run_phase.set_phase("a")
    run_phase.set_phase("a")
    run_phase.clear()
    assert run_phase.current()["seq"] == before + 3


def test_a_bad_label_never_raises():
    """This is a readout on the sampling path: it must not be able to fail a render."""
    run_phase.set_phase(None)
    assert run_phase.current()["label"] == ""
    run_phase.set_phase(12)
    assert run_phase.current()["label"] == "12"
    run_phase.clear()


def test_the_bridge_reads_the_same_sys_key_without_importing_this_module():
    """movie_editor's progress poll runs every 700ms on ComfyUI's event loop WHILE the
    worker thread samples, so it reads the label straight off `sys` rather than importing
    anything. That makes the key name a contract between two files — pin it."""
    src = (ROOT / "movie_editor" / "backend" / "bridge.py").read_text()
    assert run_phase._SYS_KEY == "_funpack_run_phase"
    assert f'getattr(sys, "{run_phase._SYS_KEY}"' in src
    # ...and the poll must not have grown an import back.
    poll = src.split("def current_progress(")[1].split("\ndef ")[0]
    assert "import_module" not in poll and "_funpack_attr" not in poll


# ── live prompt-enhancer readout ─────────────────────────────────────────────

def _on_run(monkeypatch, run_id):
    import server
    monkeypatch.setattr(server.PromptServer, "instance",
                        type("S", (), {"last_prompt_id": run_id})(), raising=False)


def test_rewrites_are_published_live_and_tagged_with_their_run(monkeypatch):
    other = _second_import()
    _on_run(monkeypatch, "run-1")
    run_phase.reset_enhanced()
    run_phase.publish_enhanced({"scene": 0, "after": "A"})
    other.publish_enhanced({"scene": 1, "after": "B"})      # the other import path adds to it
    live = sys._funpack_run_phase["enhanced"]
    assert live["prompt_id"] == "run-1" and [i["after"] for i in live["items"]] == ["A", "B"]


def test_a_new_run_never_carries_the_last_runs_rewrite(monkeypatch):
    _on_run(monkeypatch, "run-1")
    run_phase.publish_enhanced({"after": "old"})
    _on_run(monkeypatch, "run-2")
    run_phase.publish_enhanced({"after": "new"})
    live = sys._funpack_run_phase["enhanced"]
    assert live["prompt_id"] == "run-2" and [i["after"] for i in live["items"]] == ["new"]
