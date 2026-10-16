"""What ComfyUI has in its temp directory.

Previews and other transient outputs land there, and they are wiped when the
server restarts -- so "where did that go?" has an answer for as long as the
server has been up, and this is where it is.

Only media, and only what can be shown: a temp directory also fills with
whatever any node felt like writing, and a browser listing 400 .pt files is not
a media bin.
"""

from __future__ import annotations

import os

#: What ComfyUI serves through /view and a browser can draw.
KINDS = {
    "image": {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"},
    "video": {".mp4", ".webm", ".mov", ".mkv"},
    "audio": {".wav", ".mp3", ".flac", ".ogg"},
}

#: A long session can leave thousands. The window shows the newest; the rest are
#: on disk and nobody scrolls to them.
MAX_FILES = 500


def kind_of(name: str) -> str | None:
    ext = os.path.splitext(name)[1].lower()
    for kind, exts in KINDS.items():
        if ext in exts:
            return kind
    return None


def temp_dir() -> str | None:
    """ComfyUI's temp directory, or None when ComfyUI is not here to ask."""
    try:
        import folder_paths
        return folder_paths.get_temp_directory()
    except Exception:  # noqa: BLE001
        return None


def listing(limit: int = MAX_FILES) -> dict:
    """`{files, path, detail}`, newest first.

    `detail` carries the reason there is nothing rather than leaving an empty
    list to be read as "nothing was made" -- which is a different thing from
    "this is not a ComfyUI install" and from "the directory was wiped".
    """
    limit = max(1, min(int(limit or MAX_FILES), MAX_FILES))
    base = temp_dir()
    if not base:
        return {"files": [], "path": None,
                "detail": "ComfyUI is not here to ask where its temp files go."}
    if not os.path.isdir(base):
        return {"files": [], "path": base,
                "detail": "The temp directory does not exist yet. It is made when "
                          "something is first written to it."}

    found: list[dict] = []
    for root, _dirs, names in os.walk(base):
        for name in names:
            kind = kind_of(name)
            if not kind:
                continue
            full = os.path.join(root, name)
            try:
                stat = os.stat(full)
            except OSError:
                continue                    # vanished between walk and stat
            sub = os.path.relpath(root, base)
            found.append({
                "filename": name,
                "subfolder": "" if sub == "." else sub.replace(os.sep, "/"),
                "kind": kind,
                "size": stat.st_size,
                "mtime": stat.st_mtime,
            })

    found.sort(key=lambda f: f.get("mtime", 0), reverse=True)
    detail = "" if found else "Nothing here. Temp files are wiped when ComfyUI restarts."
    return {"files": found[:limit], "path": base, "detail": detail}
