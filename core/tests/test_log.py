"""The log: a severity, a source, and a message.

The format is the contract -- the app will parse it and a person will scan it --
so it is pinned rather than left to drift line by line.
"""

import pytest

from core import log, run


@pytest.fixture(autouse=True)
def clean():
    log._reset()
    run._reset()
    yield
    log._reset()
    run._reset()


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
    entry = log.history()[-1]
    assert entry["level"] == level
    assert entry["source"] == "somewhere"
    assert entry["message"] == "something"


def test_a_record_carries_the_run_it_happened_in():
    """So a report can say which generation a line belongs to, and two runs in
    one session can be told apart."""
    log.info("before", "no run yet")
    assert log.history()[-1]["run"] is None

    first = run.start()
    log.info("during", "inside a run")
    assert log.history()[-1]["run"] == first

    second = run.start()
    log.info("during", "inside the next one")
    assert log.history()[-1]["run"] == second
    assert first != second


def test_history_can_be_filtered_to_one_run():
    first = run.start()
    log.info("a", "one")
    run.start()
    log.info("b", "two")
    assert [r["source"] for r in log.history(run=first)] == ["a"]


def test_starting_a_run_clears_what_was_said_once():
    run.start()
    log.once("k", log.ALERT, "src", "inert")
    run.start()
    log.once("k", log.ALERT, "src", "inert")
    assert len(log.history()) == 2


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


def test_broke_says_it_worked_and_then_did_not_which_is_a_different_search():
    """The distinction, stated where it is defined.

    "did not load" points a reader at an import; "failed while ..." points at
    the work. Something that loaded, recognised its model and then raised was
    reported as absent in three separate files before anyone noticed, which is
    why this is one function and not three hand-written strings.
    """
    log.broke("some.module", ValueError("bad"), "building this model's latent")
    entry = log.history()[-1]
    assert entry["level"] == log.WARNING
    assert "did not load" not in entry["message"]
    assert "failed while building this model's latent" in entry["message"]
    assert "ValueError" in entry["message"] and "bad" in entry["message"]


def test_broke_still_says_something_useful_with_no_description():
    log.broke("some.module", ValueError("bad"))
    assert "failed while working" in log.history()[-1]["message"]


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


def test_the_log_survives_the_process_dying(tmp_path, monkeypatch):
    """Each record is on disk as soon as it is said; the next start shows the previous session's last lines."""
    from core import config
    monkeypatch.setattr(config, "LOG_FILE", tmp_path / "log.jsonl")
    monkeypatch.setattr(log, "_persist", True)
    monkeypatch.setattr(log, "_file", None)
    log._reset()
    log.info("training", "step 3: backward")
    log.warning("training", "step 4: forward")
    log._file.close()
    log._records.clear()
    monkeypatch.setattr(log, "_file", None)
    log._load()
    assert [r["message"] for r in log.history()] == ["step 3: backward", "step 4: forward"]


# --- colour only where attention is needed ----------------------

from core import backend_log  # noqa: E402

_SESSION = [
    "got prompt",
    "[Log] Info: FunPack Sampler | 4 steps, sage attention",
    "[Log] Alert: FunPack Sampler | no image is connected, starting from noise",
    "[Log] Warning: FunPack LoRA Loader | style.safetensors is empty",
    "[Log] Critical: torch | cannot allocate 14 GB",
    "Traceback (most recent call last):",
    '  File "x.py", line 3, in run',
    "    boom()",
    "ValueError: nope",
    "back to normal after the error text",   # prose that mentions "error" is not an alarm
    "model failed to be loved, said nobody",  # nor is "failed" in prose
    "[VideoHelperSuite] - WARNING - bad frame rate",
    "UserWarning: something deprecated",
    "Prompt executed in 12.3 seconds",
]


def test_only_lines_that_need_a_look_are_coloured():
    tones = backend_log.tones(_SESSION)
    assert tones == ["plain", "plain", "warn", "warn", "error",
                     "error", "error", "error", "error",
                     "plain", "plain", "warn", "warn", "plain"]


def test_a_healthy_run_has_no_colour_at_all():
    healthy = ["got prompt", "[Log] Info: FunPack Sampler | loaded", "Prompt executed in 3 seconds",
               "100%|##########| 4/4 [00:08<00:00]"]
    assert set(backend_log.tones(healthy)) == {"plain"}


def test_the_log_answer_carries_one_tone_per_line(monkeypatch):
    from collections import deque
    writes = deque([{"m": "\n".join(_SESSION) + "\n"}])
    monkeypatch.setattr(backend_log, "_memory", lambda: writes)
    answer = backend_log.recent()
    assert len(answer["tones"]) == len(answer["lines"]) == len(_SESSION)
    assert answer["tones"][2] == "warn" and answer["tones"][4] == "error"


@pytest.mark.parametrize("level,painted", [
    (log.INFO, False), (log.ALERT, True), (log.WARNING, True), (log.CRITICAL, True)])
def test_the_terminal_paints_only_alert_and_up(monkeypatch, level, painted):
    monkeypatch.setenv("FUNPACK_LOG_COLOR", "1")
    out = log._paint(level, "text")
    assert (out != "text") is painted
    if painted:
        assert out.startswith("\x1b[") and out.endswith("\x1b[0m")


def test_no_colour_when_the_terminal_is_not_one_or_the_user_says_no(monkeypatch):
    monkeypatch.delenv("FUNPACK_LOG_COLOR", raising=False)
    monkeypatch.setenv("NO_COLOR", "1")
    assert log._paint(log.CRITICAL, "t") == "t"
    monkeypatch.delenv("NO_COLOR")
    monkeypatch.setenv("FUNPACK_LOG_COLOR", "0")
    assert log._paint(log.CRITICAL, "t") == "t"


def test_colour_codes_never_reach_the_panel(monkeypatch):
    from collections import deque
    painted = "\x1b[38;5;208m[Log] Warning: x | y\x1b[0m"
    monkeypatch.setattr(backend_log, "_memory", lambda: deque([{"m": painted + "\n"}]))
    answer = backend_log.recent()
    assert answer["lines"] == ["[Log] Warning: x | y"] and answer["tones"] == ["warn"]
