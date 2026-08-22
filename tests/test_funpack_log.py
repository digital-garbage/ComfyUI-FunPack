"""The console voice: say what did not happen, without saying it twenty times."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import funpack_log as L  # noqa: E402


@pytest.fixture(autouse=True)
def _clean():
    L.reset()
    yield
    L.reset()


def test_a_standing_condition_is_stated_once_not_every_run(capsys):
    for _ in range(5):
        L.note_on_change("family", "FunPackEnhancements", "non-LTX model — enhancements skipped")
    assert capsys.readouterr().out.count("non-LTX") == 1


def test_a_standing_condition_speaks_again_when_the_answer_changes(capsys):
    L.note_on_change("family", "T", "non-LTX model — enhancements skipped")
    L.note_on_change("family", "T", "LTX model — enhancements active")
    L.note_on_change("family", "T", "non-LTX model — enhancements skipped")
    out = capsys.readouterr().out
    assert out.count("non-LTX") == 2 and out.count("enhancements active") == 1


def test_a_new_run_does_not_reset_standing_conditions(capsys):
    L.note_on_change("family", "T", "non-LTX")
    L.begin_run()
    L.note_on_change("family", "T", "non-LTX")
    assert capsys.readouterr().out.count("non-LTX") == 1


def test_a_failure_says_what_stopped_and_what_the_output_looks_like(capsys):
    L.failed("FunPackSceneChain", "anchor pin restore", ValueError("shape mismatch"),
             "the scene keeps its unpinned latent")
    out = capsys.readouterr().out
    assert "anchor pin restore failed" in out
    assert "shape mismatch" in out
    assert "the scene keeps its unpinned latent" in out


def test_a_per_step_failure_is_collapsed_to_one_line_per_run(capsys):
    for _ in range(20):
        L.failed("T", "taste steering", RuntimeError("boom"), "the step runs unsteered")
    assert capsys.readouterr().out.count("taste steering failed") == 1


def test_the_next_run_reports_the_same_failure_again(capsys):
    L.failed("T", "taste steering", RuntimeError("boom"), "unsteered")
    L.begin_run()
    L.failed("T", "taste steering", RuntimeError("boom"), "unsteered")
    assert capsys.readouterr().out.count("taste steering failed") == 2


def test_a_failure_without_an_error_object_still_reads_as_a_sentence(capsys):
    L.failed("T", "guide append", None, "the scene renders without the guide")
    assert "guide append failed — the scene renders without the guide" in capsys.readouterr().out


def test_a_deliberate_skip_reads_differently_from_a_failure(capsys):
    L.skipped("T", "SLA attention", "Triton is not installed")
    out = capsys.readouterr().out
    assert "SKIPPED: Triton is not installed" in out
    assert "failed" not in out


def test_note_once_keys_on_the_message_so_distinct_values_each_speak(capsys):
    L.note_once("T", "scene 1 of 3")
    L.note_once("T", "scene 2 of 3")
    L.note_once("T", "scene 2 of 3")
    assert capsys.readouterr().out.count("scene") == 2


def test_note_once_can_be_pinned_to_one_key_so_a_varying_line_says_it_once(capsys):
    L.note_once("T", "dropped 1 frame", key="dropped")
    L.note_once("T", "dropped 7 frames", key="dropped")
    assert capsys.readouterr().out.count("dropped") == 1


def test_note_always_speaks(capsys):
    for _ in range(3):
        L.note("T", "scene finished")
    assert capsys.readouterr().out.count("scene finished") == 3


def test_the_helpers_report_whether_they_actually_printed():
    assert L.note_on_change("k", "T", "a") is True
    assert L.note_on_change("k", "T", "a") is False
    assert L.failed("T", "x", None, "y") is True
    assert L.failed("T", "x", None, "y") is False


# --- wiring ---------------------------------------------------------------------
# The helper is only worth anything if the modules that swallow failures actually use it.

def test_the_sampler_and_studio_share_one_logger():
    import samplers
    import conditioning
    import ltx_enhancements
    assert samplers._log is L
    assert conditioning._log is L
    assert ltx_enhancements._log is L


def test_a_standing_condition_and_a_failure_do_not_share_a_reset(capsys):
    # begin_run() must clear failures (so the next run reports them again) without clearing
    # standing conditions (so a model family is not re-announced every generation).
    L.note_on_change("family", "T", "non-LTX")
    L.failed("T", "guide append", None, "no guide")
    capsys.readouterr()
    L.begin_run()
    L.note_on_change("family", "T", "non-LTX")
    L.failed("T", "guide append", None, "no guide")
    out = capsys.readouterr().out
    assert "non-LTX" not in out          # standing: still true, still silent
    assert "guide append failed" in out  # failure: news again this run
