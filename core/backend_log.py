"""The last lines ComfyUI printed.

Users do not open devtools, and "it did not work" with nothing to read is where
a bug report dies. This is the log they can actually reach.

Read from ComfyUI's own file rather than from a buffer in this process. v4 kept
both and its own docstring says why the file is the half that matters: a
CRASH AND RESTART empties an in-memory buffer, and the lines before the crash
are exactly the ones worth reading. A file survives it. So this is the smaller
half and the more useful one, and the buffer can be added if a case turns up
that the file misses.

Never raises. A log panel that fails is worse than one that says it found no
file, because the reason someone opened it is that something else already broke.
"""

from __future__ import annotations

from pathlib import Path

#: Read from the end. A ComfyUI log runs to megabytes over a long session and
#: nobody scrolls back through a boot from three days ago.
MAX_BYTES = 512 * 1024
MAX_LINES = 2000


def log_file() -> Path | None:
    """ComfyUI's log, or None when this install does not write one."""
    try:
        import folder_paths
        base = Path(folder_paths.base_path)
    except Exception:  # noqa: BLE001
        return None
    for name in ("comfyui.log", "comfyui.prev.log"):
        candidate = base / "user" / name
        if candidate.is_file():
            return candidate
    return None


def recent(limit: int = 600) -> dict:
    """`{lines, path, detail}`. `detail` is set only when there is nothing to show.

    The absence of a log is a real state with a real cause -- ComfyUI started
    without one -- so it is reported as an answer rather than as an empty list
    that looks like a quiet log.
    """
    limit = max(1, min(int(limit or 600), MAX_LINES))
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
