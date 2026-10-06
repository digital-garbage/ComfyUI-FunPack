"""Render: playing, previewing, exporting and finishing the clips on the timeline.

Everything between "a clip was generated" and "a file you can keep": serving a render
so a browser can seek in it, the trimmed preview segment each timeline clip plays,
exporting a selection, and the final stitch with its seams, effects, audio and
overlays, plus the one node that upscales a finished file. This runs beside ComfyUI,
on the files it wrote.

Unvalidated on a GPU rental (there is no GPU in it); the ffmpeg parts are tested
against real ffmpeg.
"""

import asyncio
import os
import time

from ..._core import media, projects
from . import files, jobs, library, serving, stitch, upscale

try:
    from aiohttp import web
except Exception:                                  # noqa: BLE001 -- tests without aiohttp
    web = None

ID = "render"
TITLE = "Render"
STAGE = "post"
CATEGORY = "post"
STATUS = "experimental"
NODES = [upscale.FunPackUpscaleVideo] if upscale.io is not None else []


def routes(table, base, web):
    def project_of(req):
        found = projects.get(req.match_info["pid"])
        if found is None:
            raise web.HTTPNotFound(reason="No such project.")
        return found

    async def body_of(req):
        try:
            body = await req.json()
        except Exception:                          # noqa: BLE001
            return {}
        return body if isinstance(body, dict) else {}

    def bad(detail, status=400):
        return web.json_response({"detail": detail}, status=status)

    @table.get(base + "/result")
    async def _result(req):
        return await serving.result(req)

    @table.get(base + "/poster")
    async def _poster(req):
        return await serving.poster(req)

    @table.get(base + "/projects/{pid}/preview-segment/{scene_id}")
    async def _segment(req):
        return await serving.segment(req, project_of(req))

    @table.post(base + "/projects/{pid}/last-frame")
    async def _last_frame(req):
        proj = project_of(req)
        body = await body_of(req)
        scene_id = body.get("scene_id")
        if not isinstance(scene_id, str) or not scene_id:
            return bad("Send {scene_id}.")
        render = body.get("render") if isinstance(body.get("render"), dict) and isinstance((body["render"].get("media") or None), dict) else None
        window = serving.query_window({"dur": body.get("dur"), "src_in": body.get("src_in")})
        try:
            return web.json_response(await asyncio.to_thread(serving.last_frame, proj, scene_id, render, window, bool(body.get("reverse"))))
        except KeyError as exc:
            return bad(f"No render for this scene ({exc}).", 404)
        except (files.ClipError, ValueError, OSError) as exc:
            return bad(str(exc), 502)

    @table.get(base + "/upscale_models")
    async def _upscale_models(_req):
        return web.json_response({"models": upscale.models()})

    @table.get(base + "/library")
    async def _library(_req):
        return web.json_response(library.payload())

    @table.post(base + "/projects/{pid}/render")
    async def _render(req):
        proj = project_of(req)
        clips = (await body_of(req)).get("clips")
        clips = clips if isinstance(clips, list) else []
        if not clips and not stitch.has_graphics(proj):
            return bad("Nothing to render: add overlays, audio, or generated clips.")
        job = jobs.new()
        asyncio.create_task(jobs.run(job, stitch.render, proj, clips, error=stitch.RenderError))
        return web.json_response({"job_id": job})

    @table.get(base + "/projects/{pid}/render/{job}")
    async def _render_status(req):
        project_of(req)
        job = jobs.get(req.match_info["job"])
        return web.json_response(job) if job else bad("Render job not found.", 404)

    @table.post(base + "/projects/{pid}/export-clip")
    async def _export_clip(req):
        project_of(req)
        clip = (await body_of(req)).get("clip")
        if not isinstance(clip, dict) or not (clip.get("filename") or clip.get("bin_media_ref")):
            return bad("Missing clip filename.")
        try:
            out = await asyncio.to_thread(stitch.concat, [clip])
        except stitch.RenderError as exc:
            return bad(str(exc), 503)
        return web.json_response({"media": out["media"]})

    @table.post(base + "/projects/{pid}/export-clips")
    async def _export_clips(req):
        project_of(req)
        clips = (await body_of(req)).get("clips")
        if not isinstance(clips, list) or not clips:
            return bad("Nothing to export.")
        job = jobs.new()
        asyncio.create_task(jobs.run(job, stitch.concat, clips, error=stitch.RenderError))
        return web.json_response({"job_id": job})

    @table.get(base + "/projects/{pid}/export-clips/{job}")
    async def _export_status(req):
        project_of(req)
        job = jobs.get(req.match_info["job"])
        return web.json_response(job) if job else bad("Export job not found.", 404)

    @table.post(base + "/import-clip")
    async def _import_clip(req):
        body = await body_of(req)
        clip = body.get("clip")
        if not isinstance(clip, dict) or not (clip.get("bin_media_ref") or clip.get("filename")):
            return bad("Missing clip source.")
        name = str(body.get("name") or "clip.mp4").strip() or "clip.mp4"
        if not name.lower().endswith((".mp4", ".webm", ".mov", ".mkv", ".avi")):
            name += ".mp4"
        try:
            data = await asyncio.to_thread(_clip_bytes, clip)
            entry = media.save_upload(name, data, limit=None)      # our own render: a long one is over the upload limit
        except (files.ClipError, stitch.RenderError) as exc:
            return bad(str(exc), 400)
        except ValueError as exc:
            return bad(str(exc), 400)
        return web.json_response({"media": entry})


def _clip_bytes(clip: dict) -> bytes:
    """The clip's bytes, trimmed when the spec asks for a window. Blocking."""
    src = files.clip_path(clip)
    inn, dur = clip.get("in"), clip.get("dur")
    try:
        needs_trim = (inn is not None and float(inn) > 0.001) or dur is not None
    except (TypeError, ValueError):
        needs_trim = True
    if not needs_trim:
        with open(src, "rb") as fh:
            return fh.read()
    out = files.temp_file(f"funpack_import_{int(time.time() * 1000)}.mp4")
    try:
        files.trim(src, out, inn, dur)
        with open(out, "rb") as fh:
            return fh.read()
    finally:
        try:
            os.remove(out)
        except OSError:
            pass


PROVIDES = {"routes": routes}
