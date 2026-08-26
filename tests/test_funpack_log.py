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
    assert "anchor pin restore: Failed" in out
    # The effect, not the exception: an exception message rarely says what the output looks
    # like, and that is the half rule 1 exists for. The error is one FUNPACK_LOG away.
    assert "The scene keeps its unpinned latent" in out
    L.reset()
    L.set_verbose(True)
    try:
        L.failed("FunPackSceneChain", "anchor pin restore", ValueError("shape mismatch"),
                 "the scene keeps its unpinned latent")
        assert "shape mismatch" in capsys.readouterr().out
    finally:
        L.set_verbose(False)


def test_a_per_step_failure_is_collapsed_to_one_line_per_run(capsys):
    for _ in range(20):
        L.failed("T", "taste steering", RuntimeError("boom"), "the step runs unsteered")
    assert capsys.readouterr().out.count("taste steering: Failed") == 1


def test_the_next_run_reports_the_same_failure_again(capsys):
    L.failed("T", "taste steering", RuntimeError("boom"), "unsteered")
    L.begin_run()
    L.failed("T", "taste steering", RuntimeError("boom"), "unsteered")
    assert capsys.readouterr().out.count("taste steering: Failed") == 2


def test_a_failure_without_an_error_object_still_reads_as_a_sentence(capsys):
    L.failed("T", "guide append", None, "the scene renders without the guide")
    assert "guide append: Failed" in capsys.readouterr().out


def test_a_deliberate_skip_reads_differently_from_a_failure(capsys):
    L.skipped("T", "SLA attention", "Triton is not installed")
    out = capsys.readouterr().out
    assert "SLA attention: Inactive | Triton is not installed" in out
    assert "Failed" not in out


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
    assert "guide append: Failed" in out  # failure: news again this run


# ── the shape: one line, one clause ─────────────────────────────────────────
#
# A paragraph per feature is how a log stops being read, and a log nobody reads reports
# nothing. The reasoning is still written at every call site — it is worth having the first
# time someone hits a problem — so FUNPACK_LOG=verbose prints it in full.

def test_a_feature_reads_as_state_then_reason(capsys):
    L.feature("FunPack", "K/V conditioning patch", False, "non-LTX model loaded")
    assert capsys.readouterr().out == \
        "[FunPack] K/V conditioning patch: Inactive | Non-LTX model loaded\n"


def test_an_active_feature_needs_no_reason(capsys):
    L.feature("FunPack", "Region lock", True)
    assert capsys.readouterr().out == "[FunPack] Region lock: Active\n"


def test_an_inactive_feature_without_a_reason_says_so_rather_than_trailing_off(capsys):
    """Inactive is only half an answer. A missing why is a gap worth seeing, not hiding."""
    L.feature("FunPack", "Region lock", False)
    assert "Inactive | No reason given" in capsys.readouterr().out


def test_the_explanation_is_cut_and_the_answer_is_kept(capsys):
    L.feature("FunPack", "Taste push", False,
              "needs 3 liked runs on this key. The Refiner learns the direction from your "
              "ratings and there is nothing to learn from yet.")
    assert capsys.readouterr().out == \
        "[FunPack] Taste push: Inactive | Needs 3 liked runs on this key\n"


def test_verbose_prints_the_whole_thing(capsys):
    L.set_verbose(True)
    try:
        L.feature("FunPack", "Taste push", False,
                  "needs 3 liked runs on this key. The Refiner learns the direction from "
                  "your ratings.")
        assert "The Refiner learns the direction" in capsys.readouterr().out
    finally:
        L.set_verbose(False)


def test_an_em_dash_is_not_a_cut(capsys):
    """It introduces the reason as often as the essay, and a renderer that guesses wrong eats
    the one part of the line worth reading."""
    L.feature("FunPack", "Region lock", False, "not a MiniMax H3 model — an H3-only lane")
    assert "an H3-only lane" in capsys.readouterr().out


def test_a_single_long_sentence_is_trimmed_rather_than_left_to_run(capsys):
    L.feature("FunPack", "Thing", False, "because " + "a" * 400)
    line = capsys.readouterr().out.strip()
    assert len(line) < 220 and line.endswith("…")


def test_a_feature_repeats_only_when_its_state_changes(capsys):
    """A standing condition is news once. Twenty identical lines are what makes a log stop
    being read at all."""
    for _ in range(5):
        L.feature("FunPack", "Region lock", False, "not a MiniMax H3 model")
    assert capsys.readouterr().out.count("Region lock") == 1
    L.feature("FunPack", "Region lock", True, "384 of 1024 patches")
    assert "Region lock: Active" in capsys.readouterr().out


def test_a_skip_and_an_inactive_feature_read_the_same(capsys):
    """They are the same fact — the user should not have to learn two spellings of it."""
    L.feature("FunPack", "SLA attention", False, "Triton is not installed")
    L.reset()
    L.skipped("FunPack", "SLA attention", "Triton is not installed")
    lines = [x for x in capsys.readouterr().out.strip().splitlines()]
    assert lines[0] == lines[1]


def test_a_diagnostic_can_ask_for_its_whole_payload(capsys):
    """The rare line whose PAYLOAD is the detail — two sets of key names to compare — where
    trimming removes the only reason to print it."""
    L.note("FunPack", "LoRA matched NOTHING. Its keys: a.b.c. This model wants: d.e.f.",
           full=True)
    out = capsys.readouterr().out
    assert "a.b.c" in out and "d.e.f" in out
