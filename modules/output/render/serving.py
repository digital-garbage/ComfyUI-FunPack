"""Playing a clip in the browser: serving the file, and the trimmed preview segments.

Two rules this layer exists to keep (each cost a release in v4):

* ffmpeg never runs on the event loop. ComfyUI has ONE loop for generation, every playing
  video stream and the editor API; a seconds-long encode inline stalls all of them.
* A media URL serves the same bytes for as long as it exists. A file that cannot be served
  correctly YET (still being written) answers 503 so the player retries; one that cannot be
  made playable (remux failed) answers 502, rate-limited, and is NEVER served raw -- raw
  moov-at-end video fails every deep seek and the player's retries fail the same way forever.
"""

from __future__ import annotations

import asyncio
import hashlib
import mimetypes
import os
import time
from collections import OrderedDict

from ..._core import projects
from . import effects, files

try:
    from aiohttp import web
except Exception:                                  # noqa: BLE001 -- tests without aiohttp
    web = None

RETRY_AFTER = {"Retry-After": "2"}
REMUX_RETRY_SEC = 30.0
CACHE_MAX = 400                      # files kept for playback; the oldest are deleted past this


class _Cache(OrderedDict):
    """key -> (signature, path); evicts (and deletes the file of) the oldest past CACHE_MAX."""

    def put(self, key, value):
        self[key] = value
        self.move_to_end(key)
        while len(self) > CACHE_MAX:
            _, (_sig, path) = self.popitem(last=False)
            try:
                os.remove(path)
            except OSError:
                pass


_remuxed = _Cache()                  # source path -> (signature, faststart copy)
_remux_failed: dict = {}             # source path -> (signature, monotonic time)
_segments = _Cache()                 # segment key -> (None, path)
_locks: dict = {}


def _lock(key):
    return _locks.setdefault(key, asyncio.Lock())


def _unavailable(reason):
    return web.HTTPServiceUnavailable(reason=reason, headers=RETRY_AFTER)


async def playable(path: str) -> str:
    """`path`, or a seek-safe copy of it. See the module docstring for the 503/502 rules."""
    if not files.is_isobmff(path):
        return path
    try:
        sig = files.signature(path)
    except OSError:
        return path                                  # the caller re-checks existence: a definitive 404
    hit = _remuxed.get(path)
    if hit and hit[0] == sig and os.path.isfile(hit[1]):
        return hit[1]
    async with _lock(path):
        try:
            sig = files.signature(path)
        except OSError:
            return path
        hit = _remuxed.get(path)
        if hit and hit[0] == sig and os.path.isfile(hit[1]):
            return hit[1]
        where = files.moov_position(path)
        if where == "front":
            return path
        if where == "none":
            raise _unavailable("Video file is still being written.")
        failed = _remux_failed.get(path)
        if failed and failed[0] == sig and time.monotonic() - failed[1] < REMUX_RETRY_SEC:
            raise web.HTTPBadGateway(reason="Video could not be prepared for playback (remux failed).")
        digest = hashlib.md5(f"{path}|{sig[0]}|{sig[1]}".encode()).hexdigest()[:16]
        try:
            out = files.temp_file(f"funpack_faststart_{digest}{os.path.splitext(path)[1].lower()}")
        except files.ClipError:
            raise web.HTTPServiceUnavailable(reason="Temp directory unavailable.", headers={"Retry-After": "5"})
        error = None
        try:
            await asyncio.to_thread(files.remux_faststart, path, out)
            ok = os.path.getsize(out) > 0 and files.moov_position(out) == "front"
        except Exception as exc:                     # noqa: BLE001
            ok, error = False, exc
        try:
            unchanged = files.signature(path) == sig
        except OSError:
            unchanged = False                        # deleted under us: the retry 404s definitively
        if ok and unchanged:
            _remuxed.put(path, (sig, out))
            _remux_failed.pop(path, None)
            return out
        try:
            os.remove(out)
        except OSError:
            pass
        if not unchanged:
            raise _unavailable("Video file is still being written.")      # rewritten under the remux
        _remux_failed[path] = (sig, time.monotonic())
        from ..._core import log
        log.warning("FunPack Render", f"could not prepare {os.path.basename(path)} for playback: {error}")
        raise web.HTTPBadGateway(reason="Video could not be prepared for playback (remux failed).")


async def result(request) -> "web.StreamResponse":
    """GET/HEAD a ComfyUI render by (filename, subfolder, type), seekable and Range-served."""
    q = request.query
    path = files.comfy_path(q.get("filename", ""), q.get("subfolder", ""), q.get("type", "output"))
    if not path or not os.path.isfile(path):
        raise web.HTTPNotFound()
    if request.method == "HEAD":
        # Mirrors GET's mid-write gate: the player HEAD-probes after a media error to tell
        # "still being written" (show "processing") from a dead file, so it must not lie 200.
        if files.is_isobmff(path) and files.moov_position(path) == "none":
            raise _unavailable("Video file is still being written.")
        return web.Response()
    ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
    name = q.get("filename", "") or os.path.basename(path)
    if ctype.startswith("video/"):
        path = await playable(path)
    return web.FileResponse(path, headers={"Content-Type": ctype,
                                           "Content-Disposition": f'inline; filename="{name}"'})


# --- preview segments ---------------------------------------------------------


def scene_clip(project, scene_id: str, render: dict | None = None, window: dict | None = None) -> dict:
    """The trim a scene's preview segment must show: the same window the export uses.

    `window` ({dur, src_in} from the URL) wins over the saved project: the editor autosaves
    seconds after a trim, and a segment is cached for an hour under its URL, so the bytes
    must follow what the URL says, never a project that has not caught up."""
    scene = next((s for s in project.scenes if s.id == scene_id), None)
    if scene is None or scene.excluded:
        raise KeyError(f"scene {scene_id}")
    render = render if render is not None else (project.scene_renders.get(scene_id) or {})
    media = render.get("media") if isinstance(render.get("media"), dict) else {}
    if not media.get("filename"):
        raise KeyError(f"no render for {scene_id}")
    fps = float(scene.eff_fps(project) or 25)
    dur = float(scene.source_dur) if scene.source_dur is not None else scene.eff_frames(project) / fps
    src_in = float(scene.source_in or 0)
    window = window or {}
    if window.get("dur") is not None and window["dur"] > 0:
        dur = window["dur"]
    if window.get("src_in") is not None and window["src_in"] >= 0:
        src_in = window["src_in"]
    return {"filename": media["filename"], "subfolder": media.get("subfolder") or "",
            "type": media.get("type") or "output",
            "in": src_in + _f(render.get("inSec")), "dur": dur, "fps": fps}


def _f(value, default=0.0):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return default if v != v else v


def query_window(q) -> dict:
    """The trim the player asked for; a missing or junk value reads as "use the project's"."""
    def num(key):
        try:
            v = float(q.get(key))
        except (TypeError, ValueError):
            return None
        return v if v == v and v not in (float("inf"), float("-inf")) else None
    return {"dur": num("dur"), "src_in": num("src_in")}


def query_render(q) -> dict | None:
    """A render named in the query string overrides the scene's saved one (a clip just made)."""
    if not q.get("filename"):
        return None
    return {"inSec": _f(q.get("render_in")),
            "media": {"filename": q["filename"], "subfolder": q.get("subfolder") or "",
                      "type": q.get("type") or "output"}}


def ghost_clip(q) -> dict | None:
    """A removed scene whose clip still previews has no scene: the whole window is in the query."""
    if not q.get("filename") or q.get("dur") is None:
        return None
    try:
        start, dur = float(q.get("render_in") or 0) + float(q.get("src_in") or 0), float(q["dur"])
    except (TypeError, ValueError):
        return None
    if start != start or dur != dur:
        return None
    return {"filename": q["filename"], "subfolder": q.get("subfolder") or "",
            "type": q.get("type") or "output", "in": start, "dur": dur}


DARK_BELOW = 20.0                    # average brightness (0..255) under which a last frame is a fade-out, not a picture to continue from
_frames: dict = {}                   # what a window's last frame was saved as: key -> (media id, brightness)


def last_frame(project, scene_id: str, render: dict | None = None, window: dict | None = None) -> dict:
    """The last picture of a scene's render, saved in the media bin: {media_id, brightness, dark}.

    Dark means the clip ends in a fade to black -- starting the next shot from it gives a black start
    (v4's i2i lesson), so the caller may refuse it. The same window answers from the bin it saved to."""
    from ..._core import media
    clip = scene_clip(project, scene_id, render, window)       # a render just made, and the trim the page shows, win over the saved project
    src = files.clip_path(clip)
    sig = files.signature(src)
    key = (clip["filename"], clip["subfolder"], clip["type"], clip["in"], clip["dur"], sig)
    hit = _frames.get(key)
    if hit and media.path_for(hit[0]):
        return {"media_id": hit[0], "brightness": hit[1], "dark": hit[1] < DARK_BELOW}
    out = files.temp_file(f"funpack_last_{projects.safe_part(project.id)[:8]}_{projects.safe_part(scene_id)}_{int(time.time() * 1000)}.png")
    try:
        brightness = files.last_frame(src, out, clip["in"], clip["dur"], clip["fps"])
        with open(out, "rb") as fh:
            entry = media.save_upload(f"last_frame_{projects.safe_part(scene_id)[:12]}.png", fh.read())
    finally:
        try:
            os.remove(out)
        except OSError:
            pass
    _frames[key] = (entry["id"], brightness)
    return {"media_id": entry["id"], "brightness": brightness, "dark": brightness < DARK_BELOW}


async def segment(request, project) -> "web.StreamResponse":
    q = request.query
    scene_id = request.match_info["scene_id"]
    try:
        clip = scene_clip(project, scene_id, query_render(q), query_window(q))
    except KeyError as exc:
        clip = ghost_clip(q)
        if clip is None:
            return web.json_response({"detail": f"No render for this scene ({exc})."}, status=404)
    try:
        src = files.clip_path(clip)
        sig = files.signature(src)
    except files.ClipError as exc:
        return web.json_response({"detail": str(exc)}, status=400)
    except OSError as exc:
        return web.json_response({"detail": str(exc)}, status=503)
    if files.is_isobmff(src) and files.moov_position(src) == "none":
        return web.json_response({"detail": "Source video is still being written."}, status=503, headers=RETRY_AFTER)
    reverse = str(q.get("rev") or "") in ("1", "true", "yes")
    if reverse:
        refusal = effects.reverse_refusal(clip.get("dur"), clip.get("fps") or 24)
        if refusal:
            return web.json_response({"detail": refusal}, status=400)
    # Everything that changes the bytes is in the key (the source's signature too: a clip
    # re-rendered at the same path must not keep serving the old encode).
    key = (f"{project.id}:{scene_id}:{clip['filename']}:{clip['subfolder']}:{clip['type']}:"
           f"{clip['in']}:{clip['dur']}:{'rev' if reverse else ''}:{sig[0]}:{sig[1]}")
    async with _lock(key):
        hit = _segments.get(key)
        if hit and os.path.isfile(hit[1]):
            return web.FileResponse(hit[1], headers={"Cache-Control": "private, max-age=3600"})
        try:
            out = files.temp_file(f"funpack_preview_{projects.safe_part(project.id)[:8]}_"
                                  f"{projects.safe_part(scene_id)}_{int(time.time() * 1000)}.mp4")
            await asyncio.to_thread(files.trim, src, out, clip.get("in"), clip.get("dur"),
                                    fast=True, reverse=reverse)
            got = files.duration(out)
            if got is not None and got <= 0:           # a window past the end trims to a file with no picture
                os.remove(out)
                return web.json_response({"detail": "This clip's window starts past the end of its render: "
                                                    "generate it again."}, status=400)
        except files.ClipError as exc:         # ffmpeg itself failed: not "still being written", so not a retry
            return web.json_response({"detail": str(exc)}, status=502)
        _segments.put(key, (None, out))
    return web.FileResponse(out, headers={"Cache-Control": "private, max-age=3600"})
