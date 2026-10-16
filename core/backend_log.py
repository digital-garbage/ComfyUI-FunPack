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
        return {"lines": lines[-limit:], "path": None, "detail": ""}
    path = log_file()
    if path is None:
        return {"lines": [], "path": None,
                "detail": "ComfyUI is not writing a log file here, so there is nothing "
                          "to show. Its output is in the terminal it was started from."}
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            if size > MAX_BYTES:
                fh.seek(size - MAX_BYTES)
                fh.readline()               # drop the partial first line
            text = fh.read().decode("utf-8", errors="replace")
    except OSError as exc:
        return {"lines": [], "path": str(path),
                "detail": f"Could not read {path.name}: {exc.strerror or exc}."}
    return {"lines": text.splitlines()[-limit:], "path": str(path), "detail": ""}
