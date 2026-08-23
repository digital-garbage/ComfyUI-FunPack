"""Turning a silent death into a traceback.

Two failures this project has actually hit leave nothing in the log: a native crash (no
Python traceback at all) and a hang (nothing to print by definition). faulthandler covers
both — the crash handlers dump the stack before dying, and SIGUSR1 dumps it on demand from
another terminal without restarting anything.
"""
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import diagnostics as dg  # noqa: E402


class _FH:
    def __init__(self, fail=False):
        self.enabled = None
        self.registered = []
        self._fail = fail

    def enable(self, **kw):
        if self._fail:
            raise RuntimeError("no can do")
        self.enabled = kw

    def register(self, signum, **kw):
        self.registered.append((signum, kw))


@pytest.mark.parametrize("text", ["1", "true", "yes", "on", "ENABLE", " Enabled "])
def test_the_on_words(text):
    assert dg.wanted(text)


@pytest.mark.parametrize("text", ["", None, "0", "off", "no", "maybe"])
def test_everything_else_is_off(text):
    assert not dg.wanted(text)


def test_it_does_nothing_unless_asked():
    """Installing signal handlers in someone else's process should be a choice."""
    fh = _FH()
    assert dg.enable("", fh=fh) is None
    assert fh.enabled is None and fh.registered == []


def test_it_installs_the_crash_handlers_for_every_thread():
    fh = _FH()
    note = dg.enable("1", fh=fh, sig=types.SimpleNamespace())
    assert fh.enabled["all_threads"] is True
    assert "native crash" in note


def test_sigusr1_dumps_a_hung_process_on_demand():
    """The half that matters for a hang: no py-spy, no gdb, no restart."""
    fh = _FH()
    note = dg.enable("1", fh=fh, sig=types.SimpleNamespace(SIGUSR1=10))
    assert fh.registered and fh.registered[0][0] == 10
    assert fh.registered[0][1]["all_threads"] is True
    assert "kill -USR1" in note


def test_no_sigusr1_is_not_an_error():
    """Windows has no SIGUSR1; the crash handler is still the part that matters there."""
    fh = _FH()
    note = dg.enable("1", fh=fh, sig=types.SimpleNamespace())
    assert fh.registered == [] and "kill -USR1" not in note and note


def test_a_broken_faulthandler_never_breaks_the_import():
    """This runs at import. A diagnostic that stops FunPack loading is worse than none."""
    note = dg.enable("1", fh=_FH(fail=True), sig=types.SimpleNamespace())
    assert "could not be enabled" in note
