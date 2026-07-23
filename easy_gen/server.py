"""FunPack Easy Gen — simplified single-scene generation UI.

UI:   GET  /funpack/easy/   (static assets under /funpack/easy/<file>)

No new backend API on purpose: this frontend talks to the exact same
/funpack/movie/api/* endpoints the Movie Editor uses (same Project files,
same on-disk storage), so a project created here opens unchanged in the
full Editor and vice versa. The only job of this module is serving Easy
Gen's own frontend files, falling back to the Editor's frontend directory
for shared scripts/styles (api.js excluded — see easy_gen/frontend/api.js)
so those stay a single source of truth instead of being copy-pasted.
"""
from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Optional

try:
    from aiohttp import web
    from server import PromptServer
except Exception:  # pragma: no cover - only available inside ComfyUI
    web = None
    PromptServer = None

from ..movie_editor.backend import config as movie_config

UI_PREFIX = "/funpack/easy"
FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"


def _resolve(tail: str) -> Optional[Path]:
    rel = (tail or "index.html").lstrip("/")
    for root in (FRONTEND_DIR, movie_config.FRONTEND_DIR):
        root = root.resolve()
        target = (root / rel).resolve()
        if root not in target.parents and target != root:
            continue  # path traversal guard
        if target.is_dir():
            target = target / "index.html"
        if target.is_file():
            return target
    return None


def _serve_static(tail: str) -> "web.Response":
    target = _resolve(tail)
    if target is None:
        raise web.HTTPNotFound()
    ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
    if target.suffix == ".js":
        ctype = "text/javascript"  # be explicit; some platforms guess text/plain
    return web.Response(
        body=target.read_bytes(), content_type=ctype,
        headers={"Cache-Control": "no-store, max-age=0"},
    )


if web is not None and PromptServer is not None:
    routes = PromptServer.instance.routes

    @routes.get(UI_PREFIX)
    async def _root_redirect(_req):
        raise web.HTTPFound(UI_PREFIX + "/")

    @routes.get(UI_PREFIX + "/")
    async def _index(_req):
        return _serve_static("index.html")

    @routes.get(UI_PREFIX + "/{tail:.*}")
    async def _static(req):
        return _serve_static(req.match_info["tail"])

    print(f"[FunPack] Easy Gen available at {movie_config.comfy_display_url()}{UI_PREFIX}/")
