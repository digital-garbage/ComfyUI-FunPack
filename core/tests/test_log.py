"""The log: a severity, a source, and a message.

The format is the contract -- the app will parse it and a person will scan it --
so it is pinned rather than left to drift line by line.
"""

import pytest

from core import log


@pytest.fixture(autouse=True)
def clean():
    log._reset()
    yield
    log._reset()


def test_the_shape_is_label_severity_source_message():
    assert log.line("Warning", "FunPack LoRA Loader", "style.safetensors is empty") == (
        "[Log] Warning: FunPack LoRA Loader | style.safetensors is empty")


def test_a_message_with_no_source_does_not_print_an_empty_one():
    assert log.line("Info", "", "ready") == "[Log] Info: ready"


@pytest.mark.parametrize("fn,level", [
    (log.info, "Info"), (log.alert, "Alert"),
    (log.warning, "Warning"), (log.critical, "Critical"),
])
def test_each_level_records_itself(fn, level):
    fn("somewhere", "something")
    assert log.history()[-1] == {
        "at": log.history()[-1]["at"], "level": level,
        "source": "somewhere", "message": "something"}


def test_it_prints_as_well_as_records(capsys):
    log.warning("loader", "a file went missing")
    assert "[Log] Warning: loader | a file went missing" in capsys.readouterr().err


def test_history_can_be_filtered_by_level():
    log.info("a", "one")
    log.warning("b", "two")
    log.info("c", "three")
    assert [r["source"] for r in log.history(log.INFO)] == ["a", "c"]


def test_the_history_is_bounded():
    """A long rental session must not turn the log into a memory leak."""
    for i in range(log.HISTORY + 50):
        log.info("loop", str(i))
    assert len(log.history()) == log.HISTORY
    assert log.history()[-1]["message"] == str(log.HISTORY + 49)


def test_failed_is_a_warning_because_it_means_a_feature_is_absent():
    log.failed("some.module", ValueError("bad"))
    entry = log.history()[-1]
    assert entry["level"] == log.WARNING
    assert "ValueError" in entry["message"] and "bad" in entry["message"]


# --- once, and what "once" means -------------------------------------------

def test_once_says_it_a_single_time():
    for _ in range(10):
        log.once("k", log.ALERT, "src", "inert")
    assert len(log.history()) == 1


def test_once_is_per_run_not_per_process():
    """v4 deduped for the life of the interpreter, so a session reported the
    first generation that went inert and stayed quiet for every one after."""
    log.once("k", log.ALERT, "src", "inert")
    log.new_run()
    log.once("k", log.ALERT, "src", "inert")
    assert len(log.history()) == 2


def test_different_keys_are_independent():
    log.once("a", log.ALERT, "src", "one")
    log.once("b", log.ALERT, "src", "two")
    assert len(log.history()) == 2
