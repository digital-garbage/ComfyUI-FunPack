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
