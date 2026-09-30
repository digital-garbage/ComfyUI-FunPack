"""Bridge import helpers and parse error formatting."""
import pytest

from movie_editor.backend import bridge


def test_format_funpack_error_with_message():
    assert bridge.format_funpack_error(ValueError("bad trigger")) == "ValueError: bad trigger"


def test_format_funpack_error_empty_message():
    assert bridge.format_funpack_error(KeyError()) == "KeyError (no message)"


def test_ensure_funpack_path_adds_repo_root():
    bridge._FUNPACK_PATH_ENSURED = False
    import sys
    root = str(bridge._FUNPACK_ROOT)
    sys.path[:] = [p for p in sys.path if p != root]
    bridge._ensure_funpack_path()
    assert root in sys.path


def _q_item(prompt_id, client_id, funpack=None):
    extra = {"client_id": client_id}
    if funpack is not None:
        extra["funpack"] = funpack
    return [1, prompt_id, {"graph": True}, extra, ["9"]]


def test_select_active_picks_editor_running_job_with_metadata():
    state = {
        "queue_running": [_q_item("p1", bridge.EDITOR_CLIENT_ID,
                                  {"pid": "proj-7", "scene_ids": ["a", "b"], "only_scene": None})],
        "queue_pending": [_q_item("p2", bridge.EDITOR_CLIENT_ID, {"pid": "proj-7"})],
    }
    out = bridge._select_active(state)
    assert out["running"] is True
    assert out["prompt_id"] == "p1"
    assert out["pid"] == "proj-7"
    assert out["scene_ids"] == ["a", "b"]
    assert out["pending"] == 1


def test_select_active_ignores_foreign_client_jobs():
    state = {
        "queue_running": [_q_item("other", "some-comfy-tab")],
        "queue_pending": [_q_item("other2", "some-comfy-tab")],
    }
    out = bridge._select_active(state)
    assert out["running"] is False
    assert out["prompt_id"] is None
    assert out["pending"] == 0


def test_select_active_empty_queue():
    out = bridge._select_active({})
    assert out == {"running": False, "prompt_id": None, "pid": None,
                   "scene_ids": [], "only_scene": None, "pending": 0}


def test_select_active_running_without_funpack_meta():
    state = {"queue_running": [_q_item("p1", bridge.EDITOR_CLIENT_ID)]}
    out = bridge._select_active(state)
    assert out["running"] is True
    assert out["prompt_id"] == "p1"
    assert out["pid"] is None
    assert out["scene_ids"] == []


def test_rating_labels_hide_internal_editor_values(monkeypatch):
    def _fake_attr(_mod, name):
        table = {
            "V2_RATING_LABELS": ["Perfect", "__funpack_continue__", "__funpack_fresh_prompt__"],
            "MOVIE_EDITOR_CONTINUE_RATING": "__funpack_continue__",
            "MOVIE_EDITOR_FRESH_PROMPT_RATING": "__funpack_fresh_prompt__",
        }
        return table[name]

    monkeypatch.setattr(bridge, "_funpack_attr", _fake_attr)
    labels = bridge.rating_labels().get("labels") or []
    assert "__funpack_continue__" not in labels
    assert "__funpack_fresh_prompt__" not in labels
    assert all(not str(l).startswith("__funpack_") for l in labels)
    assert "Perfect" in labels


# ── the log panel has to survive a restart ────────────────────────────────────
# The in-memory buffer only holds what THIS process printed, so a crash-and-restart left the
# panel empty — exactly when the lines before the crash are the ones worth reading.


@pytest.fixture
def logfile(tmp_path, monkeypatch):
    path = tmp_path / "comfyui.log"
    monkeypatch.setattr(bridge, "_comfy_log_file", lambda: path if path.is_file() else None)
    bridge._LOG_FILE_CACHE.update({"at": 0.0, "lines": []})
    with bridge._LOG_LOCK:
        bridge._LOG.clear()
    return path


def test_a_short_buffer_is_backfilled_from_comfyuis_own_log(logfile):
    logfile.write_text("older 1\nolder 2\nolder 3\n")
    assert bridge.recent_log(10) == ["older 1", "older 2", "older 3"]


def test_the_seam_is_not_shown_twice(logfile):
    """The file contains what the buffer holds. Without cutting the overlap the panel shows
    the same lines once from the file and once from the buffer."""
    logfile.write_text("old\nlive 1\nlive 2\n")
    with bridge._LOG_LOCK:
        bridge._LOG.extend(["live 1", "live 2"])
    assert bridge.recent_log(10) == ["old", "live 1", "live 2"]


def test_a_full_buffer_never_touches_the_file(logfile, monkeypatch):
    """Polled every 1.5s — once this process has enough of its own output, reading the file
    every time would be a file read per poll for nothing."""
    logfile.write_text("should not be read\n")
    monkeypatch.setattr(bridge, "_log_file_tail",
                        lambda n: pytest.fail("read the file with a full buffer"))
    with bridge._LOG_LOCK:
        bridge._LOG.extend(["a", "b", "c"])
    assert bridge.recent_log(3) == ["a", "b", "c"]


def test_no_log_file_is_not_an_error(logfile):
    with bridge._LOG_LOCK:
        bridge._LOG.extend(["only live"])
    assert bridge.recent_log(10) == ["only live"]


def test_an_unreadable_log_file_is_not_an_error(logfile, monkeypatch):
    logfile.write_text("x\n")
    monkeypatch.setattr(bridge, "_comfy_log_file",
                        lambda: (_ for _ in ()).throw(OSError("nope")))
    assert isinstance(bridge.recent_log(10), list)


def test_only_the_tail_of_a_huge_log_is_read(logfile):
    """A log can be hundreds of MB on a long-running box; the panel wants the last lines."""
    logfile.write_text("\n".join(f"line {i}" for i in range(200_000)) + "\n")
    out = bridge.recent_log(5)
    assert len(out) == 5 and out[-1] == "line 199999"


def test_progress_carries_the_live_enhanced_rewrite(monkeypatch):
    import sys as _sys
    from movie_editor.backend import bridge
    monkeypatch.setattr(bridge, "_install_progress_hook", lambda: None)
    monkeypatch.setattr(_sys, "_funpack_run_phase", {"label": "", "seq": 1, "enhanced": {
        "prompt_id": "p1", "items": [{"after": "A cat in the rain."}]}}, raising=False)
    got = bridge.current_progress()["enhanced"]
    assert got == {"prompt_id": "p1", "items": [{"after": "A cat in the rain."}]}
    monkeypatch.setattr(_sys, "_funpack_run_phase", {"label": "", "seq": 1}, raising=False)
    assert bridge.current_progress()["enhanced"] is None


def test_log_levels_read_what_a_line_says():
    lines = ["[FunPackSceneChain] dynashift failed (x), passing through",
             "Traceback (most recent call last):",
             '  File "samplers.py", line 3, in f',
             "[FunPack] Region locks: Inactive | This model is not MiniMax H3",
             "[FunPack] Region locks: Active",
             "Prompt executed in 12.3 seconds",
             "[FunPackSceneChain] scene 1/2 sampling",
             "ImportError: cannot import name 'x' from 'y'",
             "No OpenGL_accelerate module loaded: No module named 'OpenGL_accelerate'"]
    assert bridge.log_levels(lines) == ["error", "error", "error", "warn", "ok", "ok", "info",
                                        "error", "warn"]


def test_terminal_colours_only_funpack_lines_and_the_buffer_stays_plain():
    import io
    out = io.StringIO()
    tee = bridge._Tee(out, color=True)
    before = len(bridge._LOG)
    tee.write("[FunPack] thing failed")
    tee.write("\n")
    tee.write("some other pack: error\n")
    tee.write("[FunPack] scene 1/2 sampling\n")
    assert out.getvalue() == ("\x1b[91m[FunPack] thing failed\x1b[0m\n"
                              "some other pack: error\n[FunPack] scene 1/2 sampling\n")
    assert list(bridge._LOG)[before:] == ["[FunPack] thing failed", "some other pack: error",
                                          "[FunPack] scene 1/2 sampling"]
    plain = io.StringIO()
    bridge._Tee(plain).write("[FunPack] thing failed")
    assert plain.getvalue() == "[FunPack] thing failed"


def test_colour_codes_in_comfyuis_log_file_never_reach_the_panel(logfile):
    logfile.write_text("\x1b[91m[FunPack] boom failed\x1b[0m\n")
    assert bridge.recent_log(10)[-1] == "[FunPack] boom failed"


def test_other_packs_colour_codes_are_stripped_for_the_panel(logfile):
    with bridge._LOG_LOCK:
        bridge._LOG.append("[VideoHelperSuite] - \x1b[0;33mWARNING\x1b[0m - x")
    assert bridge.recent_log(5)[-1] == "[VideoHelperSuite] - WARNING - x"


def test_plural_and_ing_forms_keep_their_colour():
    assert [bridge.log_level(x) for x in (
        "errors occurred while loading", "3 exceptions were raised", "multiple crashes",
        "some warnings were shown", "the load fails silently", "3 attempts fail before success",
        "skipping scene 2", "10 rated gens, all good", "[CRITICAL] disk full")] == [
        "error", "error", "error", "warn", "error", "error", "warn", "info", "error"]


def test_a_funpack_line_after_a_newline_is_still_coloured():
    assert bridge._paint("\n[FunPack] x failed\n") == "\x1b[91m\n[FunPack] x failed\n\x1b[0m"
    assert bridge._paint("\n[FunPack] Restarting ComfyUI...\n") == "\n[FunPack] Restarting ComfyUI...\n"


def test_colour_switch(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FUNPACK_LOG_COLOR", "1")
    assert bridge._color_wanted()
    monkeypatch.setenv("FUNPACK_LOG_COLOR", "0")
    assert not bridge._color_wanted()
    monkeypatch.delenv("FUNPACK_LOG_COLOR")
    monkeypatch.setenv("NO_COLOR", "1")
    assert not bridge._color_wanted()


def test_set_mempolicy_noise_is_dropped_and_said_once(monkeypatch):
    import io
    monkeypatch.setattr(bridge, "_noise_said", False)
    out = io.StringIO()
    tee = bridge._Tee(out)
    before = len(bridge._LOG)
    tee.write("set_mempolicy: Operation not permitted\nframe=  24 fps=0.0\n"
              "set_mempolicy: Operation not permitted\n")
    tee.write("set_mempolicy: Operation not permitted\n")
    text = out.getvalue()
    assert "set_mempolicy: Operation not permitted\n" not in text.replace('"set_mempolicy: Operation not permitted" lines', "")
    assert "frame=  24 fps=0.0" in text and text.count("Hiding") == 1
    assert not any(l.startswith("set_mempolicy") for l in list(bridge._LOG)[before:])


def test_logging_output_reaches_the_panel_buffer(logfile):
    import logging
    handler = bridge._LogToBuffer()
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        record = logging.getLogger("t").makeRecord(
            "t", logging.ERROR, __file__, 1, "!!! Exception during processing !!!", None,
            __import__("sys").exc_info())
    handler.emit(record)
    lines = bridge.recent_log(50)
    assert lines[0] == "[ERROR] !!! Exception during processing !!!"
    assert lines[-1] == "RuntimeError: boom"
    assert set(bridge.log_levels(lines)) == {"error"}


def test_the_seam_ignores_a_logging_level_tag(logfile):
    logfile.write_text("old line\nPrompt executed in 3.1 seconds\n")
    with bridge._LOG_LOCK:
        bridge._LOG.append("[INFO] Prompt executed in 3.1 seconds")
    assert bridge.recent_log(10) == ["old line", "[INFO] Prompt executed in 3.1 seconds"]


def test_a_traceback_is_red_through_its_closing_line():
    lines = ["[FunPack][movie] x failed:", "Traceback (most recent call last):",
             '  File "a.py", line 1, in f', "KeyboardInterrupt", "next unrelated line"]
    assert bridge.log_levels(lines) == ["error", "error", "error", "error", "info"]


def test_the_new_features_lines_carry_their_state_colour():
    """Green = it is acting, orange = it cannot or barely does, red = it failed."""
    P = "[FunPackSceneChain] "
    green = (
        "STAS: Active | alpha 2.00 at block 15: steered 12 call(s), channels 3 and 41",
        "STAS: Active | alpha learned 2.00 from 4 rating(s), trying 2.10",
        "late-branch guidance: Active | strength 0.50, a copy with block 43 left out, sharing "
        "blocks 0-42 (~14% extra per step, none on the last), edits the picture (the sound can react).",
        "late-branch guidance: Active | guided 3 step call(s)",
        "decisiveness: Active | learned k 1.020 from 7 rating(s), trying 1.050, push up to 9% of a step's noise",
        "shot memory: Active | reusing a liked shot of 5 (reuse 0.71 > fresh 0.40), amount 0.70",
        "shot memory: Active | fresh (reuse 0.30 < fresh 0.55)",
        "steering window: Active | 1 of 4 steps (peak gate 0.50)",
        "[FunPack] upscaled a.mp4 with 4x.safetensors: 96 frames -> funpack_upscaled/a_4x.mp4",
    )
    orange = (
        "STAS: Inactive | no channel at block 15 is over 50x the mean, nothing steered -- try another block",
        "STAS: Inactive | H3 only",
        "late-branch guidance: Inactive | no step call reached it, nothing guided",
        "late-branch guidance: Inactive | did not apply this run, so this run's rating will not teach its learned strength",
        "decisiveness: barely acts on this schedule (its push is at most 1.9% of a step's noise, the "
        "last step is never edited), so this run's rating will not teach k | learned k 0.900",
        "decisiveness: Inactive | a 1-step schedule has no step before the output to carry an edit into",
        "shot memory: Inactive | left 1 sampling pass(es) alone, their latent already held a picture",
        "steering window: Inactive | nothing will steer on this schedule, every rating-driven mechanism is gated off",
    )
    for line in green:
        text = line if line.startswith("[") else P + line
        assert bridge.log_level(text) == "ok", text
    for line in orange:
        text = line if line.startswith("[") else P + line
        assert bridge.log_level(text) == "warn", text
    assert bridge.log_level(P + "decisiveness failed (boom), passing through") == "error"
