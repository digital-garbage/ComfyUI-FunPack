"""Static serving for the app and for module JS.

The resolution is a pure function returning a `Served` value, so the allowlist and
the traversal guard are testable without aiohttp or a running ComfyUI.
"""

import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import FrozenSet

# The editor iterates fast; a cached stale module is worse than a re-read.
NO_STORE = {"Cache-Control": "no-store, max-age=0"}

# Some platforms guess text/plain for .js. With ES modules that is fatal — the
# browser refuses the module outright — so every type we care about is explicit.
EXPLICIT_TYPES = {
    ".js": "text/javascript",
    ".css": "text/css",
    ".html": "text/html",
    ".svg": "image/svg+xml",
    ".woff2": "font/woff2",
    ".json": "application/json",
}


@dataclass(frozen=True)
class Served:
    status: int
    body: bytes = b""
    content_type: str = "text/plain"
    headers: dict = field(default_factory=lambda: dict(NO_STORE))


def content_type_for(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in EXPLICIT_TYPES:
        return EXPLICIT_TYPES[suffix]
    return mimetypes.guess_type(str(path))[0] or "application/octet-stream"


def resolve(root: Path, rel: str, allowed: FrozenSet[str]) -> Path:
    """Path for `rel` under `root`, or raise.

    ValueError  -> the request tried to leave the root (403)
    LookupError -> nothing to serve here (404)
    """
    root = root.resolve()
    cleaned = (rel or "").lstrip("/")
    if "\x00" in cleaned:
        raise ValueError("null byte in path")
    try:
        target = (root / cleaned).resolve()
    except (OSError, ValueError) as exc:  # malformed path, symlink loop
        raise ValueError("unresolvable path") from exc

    # The guard is on the RESOLVED path, so `..` and symlinks are both covered.
    if target != root and root not in target.parents:
        raise ValueError("outside root")

    if target.is_dir():
        if ".html" not in allowed:
            raise LookupError("directory")
        target = target / "index.html"

    if target.suffix.lower() not in allowed:
        raise LookupError("extension not allowed")
    if not target.is_file():
        raise LookupError("not a file")
    return target


def serve(root: Path, rel: str, allowed: FrozenSet[str]) -> Served:
    try:
        target = resolve(root, rel, allowed)
    except ValueError:
        return Served(403)
    except LookupError:
        return Served(404)
    return Served(200, target.read_bytes(), content_type_for(target))
