"""The last lines ComfyUI printed.

Users do not open devtools, and "it did not work" with nothing to read is where
a bug report dies. This is the log they can actually reach.

Read from ComfyUI's own in-memory copy of its output (`app.logger.logs`, what
its terminal printed). Current ComfyUI writes no log file unless ComfyUI-Manager
adds one, so a file-only reader found nothing on a plain install. The copy is
lengthened here from 300 writes, and a file is read only when there is no copy.

Never raises. A log panel that fails is worse than one that says it found no
file, because the reason someone opened it is that something else already broke.
"""

from __future__ import annotations

import re
from collections import deque
from pathlib import Path

#: Read from the end. A ComfyUI log runs to megabytes over a long session and
#: nobody scrolls back through a boot from three days ago.
MAX_BYTES = 512 * 1024
MAX_LINES = 2000


def _memory():
    """ComfyUI's deque of `{t, m}` writes, lengthened once; None outside ComfyUI's main."""
    try:
        import app.logger as comfy_log
    except Exception:  # noqa: BLE001
        return None
    logs = comfy_log.logs
    if logs is not None and (logs.maxlen or 0) < 8000:
        # its interceptor looks the global up on every write, so the longer deque takes over
        comfy_log.logs = logs = deque(logs, maxlen=8000)
    return logs


_memory()


def log_file() -> Path | None:
    """ComfyUI's log, or None when this install does not write one."""
    try:
        import folder_paths
        base = Path(folder_paths.base_path)
    except Exception:  # noqa: BLE001
        return None
    # ComfyUI-Manager names it per port (comfyui_8188.log); the newest is this run's
    try:
        found = [(p.stat().st_mtime, p) for p in (base / "user").glob("comfyui*.log") if ".prev" not in p.name]
    except OSError:
        return None
    return max(found, default=(0, None))[1]


# Colour means "look here": a line is plain unless somebody should look at it. Two tones only
# (v4 also had a green "ok", which made a healthy run look busy). The patterns
# are strict on purpose -- a line that merely contains the word "failed" in
# prose is not an alarm, and an alarm that cries wolf is switched off by eye.
_CRITICAL = re.compile(r"^\[Log\] Critical:|^\s*Traceback \(most recent call last\)|"
                       r"^\s*!!! Exception|\bout of memory\b|^\s*[\w.]*(Error|Exception)\b(:|$)|"
                       r"^\s*(ERROR|CRITICAL|FATAL)\b|\[(ERROR|CRITICAL)\]|\b(ERROR|CRITICAL) -", 0)
_ATTENTION = re.compile(r"^\[Log\] (Alert|Warning):|^\s*(WARNING|WARN)\b|\[WARNING\]|\bWARNING -|"
                        r"\b\w*Warning:")


def tones(lines: list) -> list:
    """"error" | "warn" | "plain" for each line.

    A traceback's indented body takes its head's tone, so a stack trace is one
    red block rather than one red line and thirty white ones.
    """
    out, in_trace = [], False
    for line in lines:
        indented = line[:1] in (" ", "\t")
        if "Traceback (most recent call last)" in line:
            in_trace, tone = True, "error"
        elif in_trace and indented:
            tone = "error"
        elif in_trace:
            in_trace, tone = False, "error"      # the exception line closes the trace
        elif _CRITICAL.search(line):
            tone = "error"
        elif _ATTENTION.search(line):
            tone = "warn"
        else:
            tone = "plain"
        out.append(tone)
    return out


def _answer(lines: list, path, detail: str = "") -> dict:
    return {"lines": lines, "tones": tones(lines), "path": path, "detail": detail}


def recent(limit: int = 600) -> dict:
    """`{lines, path, detail}`. `detail` is set only when there is nothing to show.

    The absence of a log is a real state with a real cause -- ComfyUI started
    without one -- so it is reported as an answer rather than as an empty list
    that looks like a quiet log.
    """
    limit = max(1, min(int(limit or 600), MAX_LINES))
    logs = _memory()
    if logs:
        text = re.sub(r"\x1b\[[0-9;]*m", "", "".join(str(e.get("m", "")) for e in list(logs)))   # terminal colours
        lines = [ln.rsplit("\r", 1)[-1] for ln in text.split("\n")]   # a progress bar keeps its last state
        if lines and not lines[-1]:
            lines.pop()
        return _answer(lines[-limit:], None)
    path = log_file()
    if path is None:
        return _answer([], None, "ComfyUI is not writing a log file here, so there is nothing "
                                 "to show. Its output is in the terminal it was started from.")
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            if size > MAX_BYTES:
                fh.seek(size - MAX_BYTES)
                fh.readline()               # drop the partial first line
            text = fh.read().decode("utf-8", errors="replace")
    except OSError as exc:
        return _answer([], str(path), f"Could not read {path.name}: {exc.strerror or exc}.")
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    return _answer(text.splitlines()[-limit:], str(path))
