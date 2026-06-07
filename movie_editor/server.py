"""Movie Editor routes, served by ComfyUI's own aiohttp server.

UI:   GET  /funpack/movie/            (and static assets under /funpack/movie/<file>)
API:  /funpack/movie/api/*

Registered on PromptServer like the other /funpack/* routes (see templates.py,
batch_training.py). Pure logic lives in backend/ (timeline, projects, workflow);
bridge.py talks to ComfyUI (parse/library in-process, queue/history/view over loopback).
"""
from __future__ import annotations

import mimetypes
from typing import Optional

try:
    from aiohttp import web
    from server import PromptServer
except Exception:  # pragma: no cover - only available inside ComfyUI
    web = None
    PromptServer = None

from .backend import bridge, builder, config, media, nodes, projects
from .backend.timeline import Project, build_combined_prompt

UI_PREFIX = "/funpack/movie"


def _restart_comfy() -> None:
    """Relaunch the ComfyUI process in place. Mirrors ComfyUI-Manager's reboot so it
    works the same whether launched directly, as a module, or via comfy-cli."""
    import os
    import sys
    try:
        sys.stdout.close_log()  # type: ignore[attr-defined]  # Manager's tee logger, if present
    except Exception:
        pass
    # comfy-cli watches for a .reboot file and relaunches us itself.
    if "__COMFY_CLI_SESSION__" in os.environ:
        try:
            open(os.environ["__COMFY_CLI_SESSION__"] + ".reboot", "w").close()
        except Exception:
            pass
        print("\n[FunPack] Restarting ComfyUI...\n", flush=True)
        os._exit(0)
    sys_argv = sys.argv.copy()
    if "--windows-standalone-build" in sys_argv:
        sys_argv.remove("--windows-standalone-build")
    if sys_argv and sys_argv[0].endswith("__main__.py"):  # python -m comfy
        module_name = os.path.basename(os.path.dirname(sys_argv[0]))
        cmds = [sys.executable, "-m", module_name] + sys_argv[1:]
    elif sys.platform.startswith("win32"):
        cmds = ['"' + sys.executable + '"', '"' + sys_argv[0] + '"'] + sys_argv[1:]
    else:
        cmds = [sys.executable] + sys_argv
    print(f"\n[FunPack] Restarting ComfyUI... {cmds}\n", flush=True)
    os.execv(sys.executable, cmds)


# ── helpers ──────────────────────────────────────────────────────────────────

def _media_from_history(hist_entry: dict) -> list[dict]:
    out: list[dict] = []
    for node_id, node_out in (hist_entry.get("outputs") or {}).items():
        for key in ("gifs", "videos", "images"):
            for item in node_out.get(key, []) or []:
                fn = item.get("filename")
                if not fn:
                    continue
                out.append({
                    "node_id": node_id, "kind": key, "filename": fn,
                    "subfolder": item.get("subfolder", ""), "type": item.get("type", "output"),
                    "format": item.get("format"),
                })
    return out


def _prepare_media(proj: Project) -> Optional[dict]:
    """If a scene has an image asset + a chosen target, copy it into ComfyUI's input
    folder (ephemeral) and return {filename, target} for the builder to LoadImage."""
    import os
    import shutil
    try:
        import folder_paths
        indir = folder_paths.get_input_directory()
    except Exception:
        return None
    for sc in proj.scenes:
        src = sc.source
        ref = getattr(src, "media_ref", None)
        tgt = getattr(src, "target", None)
        # A target is OPTIONAL now (routing is automatic — the image becomes a LoadImage
        # IMAGE producer that auto-wires). Only an asset reference is required.
        if not (src and getattr(src, "type", "") in ("image", "generated_frame") and ref):
            continue
        path = media.path_for(ref)
        if not path:
            continue
        fn = f"funpack_movie_{ref}{path.suffix}"
        try:
            shutil.copy(str(path), os.path.join(indir, fn))
        except OSError:
            return None
        return {"filename": fn, "target": tgt}
    return None


def _project_or_404(pid: str) -> Project:
    p = projects.get(pid)
    if p is None:
        raise web.HTTPNotFound(reason=f"Project {pid} not found")
    return p


def _project_models(p: Optional[Project]) -> dict:
    """The project's own pipeline config, or the global default when it has none yet."""
    m = getattr(p, "models", None) or {}
    if m.get("slots") or m.get("links") or m.get("core_overrides"):
        return m
    return nodes.load_models()


def _solo(p: Project, only_scene: Optional[str]) -> Project:
    if not only_scene:
        return p
    scene = next((s for s in p.scenes if s.id == only_scene), None)
    if scene is None:
        raise web.HTTPNotFound(reason=f"Scene {only_scene} not found")
    clone = Project.from_dict(p.to_dict())
    clone.scenes = [scene]
    return clone


def _run_studio_inputs(target: Project, active_scenes: list) -> dict:
    """Studio widget overrides for this run, incl. the RLHF rating of the run's last
    render (Studio refines from it). Uses the run's first user-rated scene; when no scene
    was rated by the user, send "-Just forget it-" so Studio does NOT learn from a rating
    it assigned itself."""
    if target.conditioning_slot != "funpack":
        return {}
    si = dict(target.studio_inputs or {})
    rating = next((s.rating for s in active_scenes if (getattr(s, "rating", "") or "").strip()), None)
    si["rating"] = rating or "-Just forget it-"
    return si


def _run_sampler_inputs(target: Project, scene_count: int) -> dict:
    """Sampler widget overrides for one chain run. Carries inside a multi-scene run
    must overlap, so carry_i2v_guides is forced on; a 1-scene run leaves it alone."""
    if target.sampler_slot != "funpack":
        return {}
    samp = dict(target.sampler_inputs or {})
    if scene_count > 1:
        samp["carry_i2v_guides"] = True
    return samp


_XFADE_MAP = {
    "crossfade": "fade", "fadeblack": "fadeblack",
    "wipeleft": "wipeleft", "wiperight": "wiperight", "dissolve": "dissolve",
}


def _build_render_filter(clips: list, tracks: Optional[list] = None,
                         keep_original: bool = True, base_input: int = 0) -> tuple[str, bool]:
    """ffmpeg filter_complex for the final stitch.

    Video — per clip: normalize to one canvas, then blur / fade in-out / virtual Ken-Burns
    zoom (zoompan, fixed output size). Fold left-to-right with xfade (overlap) or concat.

    Audio — per-clip original audio gets its volume; when keep_original the per-clip streams
    are folded alongside the video (acrossfade at crossfades, concat otherwise) into [aorig].
    Extra `tracks` (inserted audio, input index base_input+j) are delayed to their start and
    volume-scaled, then everything is amixed into [aout]. keep_original=False drops the
    original LTXAV audio entirely; with no tracks either, the output is silent (no [aout]).

    Returns (filter_complex, has_audio).
    """
    tracks = tracks or []
    n = len(clips)
    parts: list[str] = []

    # One canvas for the whole render (clips may have differing native sizes/fps — xfade and
    # concat require identical size/fps/sar, so every clip is normalized to this).
    cw = int(clips[0].get("w") or 0) or 768
    ch = int(clips[0].get("h") or 0) or 768
    cfps = float(clips[0].get("fps") or 0) or 25.0

    for i, c in enumerate(clips):
        fx = c.get("fx") or {}
        dur = float(c.get("dur") or 0) or 0.0
        # Normalize first: fit into the canvas (letterbox), fixed fps + square pixels.
        vf: list[str] = [
            f"scale={cw}:{ch}:force_original_aspect_ratio=decrease",
            f"pad={cw}:{ch}:-1:-1:color=black",
            "setsar=1", f"fps={cfps:g}",
        ]
        zoom = fx.get("zoom")
        if zoom in ("in", "out") and dur > 0:
            # VIRTUAL zoom: zoompan keeps the output fixed at the canvas size and scales the
            # content within it (never changes the actual frame size). in: 1.0->1.2 (push in);
            # out: 1.2->1.0 (pull back). Centered.
            nframes = max(1, round(dur * cfps))
            z = f"1+0.20*on/{nframes}" if zoom == "in" else f"1.20-0.20*on/{nframes}"
            vf.append(
                f"zoompan=z='{z}':d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
                f":s={cw}x{ch}:fps={cfps:g}"
            )
            vf.append("setsar=1")
        blur = float(fx.get("blur") or 0)
        if blur > 0:
            vf.append(f"gblur=sigma={blur * 20:.2f}")
        fi = float(fx.get("fade_in") or 0)
        if fi > 0:
            vf.append(f"fade=t=in:st=0:d={fi:.3f}")
        fo = float(fx.get("fade_out") or 0)
        if fo > 0 and dur > 0:
            vf.append(f"fade=t=out:st={max(0.0, dur - fo):.3f}:d={fo:.3f}")
        vf.append("format=yuv420p")
        vf.append("setsar=1")
        parts.append(f"[{i}:v:0]{','.join(vf)}[v{i}]")
        if keep_original:
            vol = float(c.get("volume", 1.0))
            af = "aformat=sample_fmts=fltp:channel_layouts=stereo"
            if abs(vol - 1.0) > 1e-3:
                af += f",volume={max(0.0, vol):.3f}"
            parts.append(f"[{i}:a:0]{af}[a{i}]")

    # Fold video (and the original audio when kept) left-to-right.
    acc_v, acc_a = "[v0]", "[a0]"
    acc_dur = float(clips[0].get("dur") or 0) or 0.0
    for i in range(1, n):
        prev = clips[i - 1]
        trans = (prev.get("transition") or "").strip()
        td = float(prev.get("tdur") or 0)
        dur_i = float(clips[i].get("dur") or 0) or 0.0
        if trans in _XFADE_MAP and td > 0 and acc_dur > td:
            off = max(0.0, acc_dur - td)
            ov = f"[vx{i}]"
            parts.append(f"{acc_v}[v{i}]xfade=transition={_XFADE_MAP[trans]}:duration={td:.3f}:offset={off:.3f}{ov}")
            if keep_original:
                oa = f"[ax{i}]"; parts.append(f"{acc_a}[a{i}]acrossfade=d={td:.3f}{oa}"); acc_a = oa
            acc_v, acc_dur = ov, acc_dur + dur_i - td
        else:
            ov = f"[vc{i}]"
            parts.append(f"{acc_v}[v{i}]concat=n=2:v=1:a=0{ov}")
            if keep_original:
                oa = f"[ac{i}]"; parts.append(f"{acc_a}[a{i}]concat=n=2:v=0:a=1{oa}"); acc_a = oa
            acc_v, acc_dur = ov, acc_dur + dur_i

    parts.append(f"{acc_v}null[vout]")
    total = max(0.01, acc_dur)

    # Assemble the audio mix: original (if kept) + each inserted track (delayed/volume'd).
    mix: list[str] = []
    if keep_original and n > 0:
        mix.append(acc_a)
    for j, t in enumerate(tracks):
        idx = base_input + j
        start = float(t.get("start_sec") or 0)
        tvol = float(t.get("volume", 1.0))
        ms = int(max(0.0, start) * 1000)
        lbl = f"[at{j}]"
        chain = f"[{idx}:a:0]aformat=sample_fmts=fltp:channel_layouts=stereo,volume={max(0.0, tvol):.3f}"
        if ms > 0:
            chain += f",adelay={ms}|{ms}"
        parts.append(f"{chain}{lbl}")
        mix.append(lbl)

    if not mix:
        return ";".join(parts), False  # silent output (video only)
    if len(mix) == 1:
        parts.append(f"{mix[0]}atrim=0:{total:.3f},asetpts=PTS-STARTPTS[aout]")
    else:
        parts.append("".join(mix) + f"amix=inputs={len(mix)}:normalize=0:duration=longest[amx]")
        parts.append(f"[amx]atrim=0:{total:.3f},asetpts=PTS-STARTPTS[aout]")
    return ";".join(parts), True


def _segment(p: Project, scene_ids: list) -> Project:
    """Project clone holding only `scene_ids` (in their project order). Used to
    generate one chain run: its first scene supplies the i2v anchor, the rest are
    carries that overlap inside the run (one ComfyUI chain-sampler request)."""
    ids = set(scene_ids)
    clone = Project.from_dict(p.to_dict())
    clone.scenes = [s for s in clone.scenes if s.id in ids]
    if not clone.scenes:
        raise web.HTTPNotFound(reason="No scenes matched the requested segment")
    return clone


def _serve_static(tail: str) -> "web.Response":
    root = config.FRONTEND_DIR.resolve()
    rel = (tail or "index.html").lstrip("/")
    target = (root / rel).resolve()
    if root not in target.parents and target != root:
        raise web.HTTPForbidden()
    if target.is_dir():
        target = target / "index.html"
    if not target.is_file():
        raise web.HTTPNotFound()
    ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
    if target.suffix == ".js":
        ctype = "text/javascript"  # be explicit; some platforms guess text/plain
    return web.Response(
        body=target.read_bytes(), content_type=ctype,
        headers={"Cache-Control": "no-store, max-age=0"},  # editor iterates fast; never cache
    )


# ── registration ─────────────────────────────────────────────────────────────

if web is not None and PromptServer is not None:
    routes = PromptServer.instance.routes
    config.ensure_dirs()
    bridge.install_log_capture()  # tee ComfyUI stdout/stderr for the Log viewer

    # --- API: projects ---
    @routes.get(UI_PREFIX + "/api/health")
    async def _health(_req):
        return web.json_response({
            "ok": True,
            "comfy_url": config.comfy_base_url(),
            "reference_loaded": bool(builder.load_reference().get("nodes")),
            "configured_slots": len(nodes.load_models().get("slots", [])),
        })

    @routes.get(UI_PREFIX + "/api/projects")
    async def _list(_req):
        return web.json_response({"projects": projects.list_projects()})

    @routes.post(UI_PREFIX + "/api/projects")
    async def _create(req):
        body = await req.json()
        p = projects.create(str(body.get("name") or "Untitled"))
        # seed the new project's pipeline config from the global default
        glob = nodes.load_models()
        if glob.get("slots") or glob.get("links"):
            p.models = glob
            projects.save(p)
        return web.json_response(p.to_dict())

    @routes.post(UI_PREFIX + "/api/projects/import")
    async def _import(req):
        try:
            body = await req.json()
        except Exception:
            return web.json_response({"detail": "Request body must be a project JSON."}, status=400)
        if not isinstance(body, dict) or "scenes" not in body:
            return web.json_response({"detail": "Payload does not look like a project file."}, status=400)
        import time as _time
        body.pop("id", None)
        body["created_at"] = _time.time()
        body["updated_at"] = _time.time()
        p = projects.save(Project.from_dict(body))
        return web.json_response(p.to_dict())

    @routes.get(UI_PREFIX + "/api/projects/{pid}")
    async def _get(req):
        return web.json_response(_project_or_404(req.match_info["pid"]).to_dict())

    @routes.put(UI_PREFIX + "/api/projects/{pid}")
    async def _update(req):
        pid = req.match_info["pid"]
        existing = _project_or_404(pid)
        body = await req.json()
        body["id"] = pid
        body.setdefault("created_at", existing.created_at)
        return web.json_response(projects.save(Project.from_dict(body)).to_dict())

    @routes.delete(UI_PREFIX + "/api/projects/{pid}")
    async def _delete(req):
        pid = req.match_info["pid"]
        if not projects.delete(pid):
            raise web.HTTPNotFound(reason=f"Project {pid} not found")
        return web.json_response({"deleted": pid})

    @routes.get(UI_PREFIX + "/api/projects/{pid}/download")
    async def _download(req):
        p = _project_or_404(req.match_info["pid"])
        safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in p.name).strip()[:64]
        filename = f"{safe_name or p.id}.funpack_project.json"
        import json as _json
        return web.Response(
            body=_json.dumps(p.to_dict(), indent=2).encode(),
            content_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"',
                     "Cache-Control": "no-store"},
        )

    # --- API: timeline preview (round-trip integrity check) ---
    @routes.get(UI_PREFIX + "/api/projects/{pid}/preview")
    async def _preview(req):
        p = _project_or_404(req.match_info["pid"])
        include_excluded = req.query.get("include_excluded") == "true"
        prompt = build_combined_prompt(p, include_excluded=include_excluded)
        result = {"combined_prompt": prompt}
        try:
            parsed = bridge.parse_timeline(prompt, seed=p.seed)
            result["parsed"] = parsed
            result["parsed_raw"] = bridge.parse_timeline_raw(prompt)
            result["parsed_verbatim"] = bridge.parse_timeline_verbatim(prompt)
            expected = len([s for s in p.scenes if include_excluded or not s.excluded])
            got = len(parsed.get("scenes", []))
            result["expected_scenes"] = expected
            result["parsed_scenes"] = got
            if expected and got != expected:
                result["warning"] = (
                    f"Studio split produced {got} scene(s) but the timeline has {expected}. "
                    f"Check transition markers (each scene needs a recognized trigger before it)."
                )
        except Exception as e:  # noqa: BLE001
            result["parse_error"] = str(e)
        return web.json_response(result)

    # --- API: parse an arbitrary prompt (global-prompt → anchor/scenes/transitions) ---
    @routes.post(UI_PREFIX + "/api/projects/{pid}/parse")
    async def _parse(req):
        p = _project_or_404(req.match_info["pid"])
        body = await req.json() if req.can_read_body else {}
        prompt = str(body.get("prompt", ""))
        try:
            parsed = bridge.parse_timeline(prompt, seed=p.seed)
            return web.json_response({
                "parsed": parsed,
                "parsed_raw": bridge.parse_timeline_raw(prompt),
                "parsed_verbatim": bridge.parse_timeline_verbatim(prompt),
                "combined_prompt": prompt,
            })
        except Exception as e:  # noqa: BLE001
            return web.json_response({"detail": f"Parse failed: {e}"}, status=502)

    # --- API: library (shortcuts + transitions, FunPack in-process) ---
    @routes.get(UI_PREFIX + "/api/library/transitions")
    async def _transitions(_req):
        try:
            return web.json_response(bridge.transitions())
        except Exception as e:  # noqa: BLE001
            return web.json_response({"transitions": [], "error": str(e)})

    @routes.post(UI_PREFIX + "/api/library/transitions")
    async def _transition_save(req):
        try:
            return web.json_response(bridge.save_transition(await req.json()))
        except Exception as e:  # noqa: BLE001
            raise web.HTTPBadRequest(reason=str(e))

    @routes.delete(UI_PREFIX + "/api/library/transitions/{name}")
    async def _transition_delete(req):
        try:
            return web.json_response(bridge.delete_transition(req.match_info["name"]))
        except Exception as e:  # noqa: BLE001
            raise web.HTTPBadRequest(reason=str(e))

    @routes.get(UI_PREFIX + "/api/library/shortcuts")
    async def _shortcuts(_req):
        try:
            return web.json_response(bridge.shortcuts())
        except Exception as e:  # noqa: BLE001
            return web.json_response({"shortcuts": [], "error": str(e)})

    @routes.post(UI_PREFIX + "/api/library/shortcuts")
    async def _shortcut_save(req):
        try:
            return web.json_response(bridge.save_shortcut(await req.json()))
        except Exception as e:  # noqa: BLE001
            raise web.HTTPBadRequest(reason=str(e))

    @routes.delete(UI_PREFIX + "/api/library/shortcuts/{name}")
    async def _shortcut_delete(req):
        try:
            return web.json_response(bridge.delete_shortcut(req.match_info["name"]))
        except Exception as e:  # noqa: BLE001
            raise web.HTTPBadRequest(reason=str(e))

    @routes.get(UI_PREFIX + "/api/library/shortcuts/export")
    async def _shortcuts_export(_req):
        try:
            data = bridge.export_shortcuts()
            return web.json_response(data, headers={
                "Content-Disposition": "attachment; filename=funpack_shortcuts.json",
                "Cache-Control": "no-store",
            })
        except Exception as e:  # noqa: BLE001
            raise web.HTTPInternalServerError(reason=str(e))

    @routes.post(UI_PREFIX + "/api/library/shortcuts/import")
    async def _shortcuts_import(req):
        try:
            incoming = await req.json()
            if not isinstance(incoming, dict) or "shortcuts" not in incoming:
                raise web.HTTPBadRequest(reason="Payload must be a shortcuts database JSON.")
            return web.json_response(bridge.import_shortcuts(incoming))
        except web.HTTPException:
            raise
        except Exception as e:  # noqa: BLE001
            raise web.HTTPBadRequest(reason=str(e))

    @routes.get(UI_PREFIX + "/api/library/transitions/export")
    async def _transitions_export(_req):
        try:
            data = bridge.export_transitions()
            return web.json_response(data, headers={
                "Content-Disposition": "attachment; filename=funpack_transitions.json",
                "Cache-Control": "no-store",
            })
        except Exception as e:  # noqa: BLE001
            raise web.HTTPInternalServerError(reason=str(e))

    @routes.post(UI_PREFIX + "/api/library/transitions/import")
    async def _transitions_import(req):
        try:
            incoming = await req.json()
            if not isinstance(incoming, dict) or "transitions" not in incoming:
                raise web.HTTPBadRequest(reason="Payload must be a transitions database JSON.")
            return web.json_response(bridge.import_transitions(incoming))
        except web.HTTPException:
            raise
        except Exception as e:  # noqa: BLE001
            raise web.HTTPBadRequest(reason=str(e))

    # --- API: media bin ---
    @routes.get(UI_PREFIX + "/api/media")
    async def _media_list(_req):
        return web.json_response({"media": media.list_media()})

    @routes.post(UI_PREFIX + "/api/media")
    async def _media_upload(req):
        reader = await req.multipart()
        saved = []
        async for part in reader:
            if part.name not in ("file", "files"):
                continue
            data = await part.read(decode=False)
            if data:
                saved.append(media.save_upload(part.filename or "upload.bin", data))
        if not saved:
            raise web.HTTPBadRequest(reason="No file in upload.")
        return web.json_response({"media": saved})

    @routes.get(UI_PREFIX + "/api/media/{mid}")
    async def _media_get(req):
        p = media.path_for(req.match_info["mid"])
        if p is None:
            raise web.HTTPNotFound()
        return web.Response(body=p.read_bytes(), content_type=media.content_type(req.match_info["mid"]))

    @routes.delete(UI_PREFIX + "/api/media/{mid}")
    async def _media_delete(req):
        if not media.delete(req.match_info["mid"]):
            raise web.HTTPNotFound()
        return web.json_response({"deleted": req.match_info["mid"]})

    # --- API: generate / status / result ---
    @routes.post(UI_PREFIX + "/api/projects/{pid}/generate")
    async def _generate(req):
        p = _project_or_404(req.match_info["pid"])
        body = await req.json() if req.can_read_body else {}
        scene_ids = body.get("scene_ids")
        if scene_ids:
            target = _segment(p, scene_ids)
        else:
            target = _solo(p, body.get("only_scene"))
        prompt = build_combined_prompt(target, for_generation=True)
        if not prompt.strip():
            return web.json_response({"detail": "Nothing to generate — no active scene text."}, status=400)
        try:
            oi = await bridge.object_info()
        except Exception as e:  # noqa: BLE001
            return web.json_response({"detail": f"Node registry unavailable: {e}"}, status=502)
        bridge.current_progress()  # ensure the sampler step-progress hook is installed
        active_scenes = [s for s in target.scenes if not s.excluded]
        active_scene_count = len(active_scenes)
        # V1 uniform chain: if trimmed scenes all agree on a frame count, use it.
        # This makes the timeline trim handle actually affect generation length,
        # provided the user has linked EmptyLTXVLatent.num_frames → Project Frames
        # in Models → Linked inputs.
        trimmed = [
            s.eff_frames(target)
            for s in active_scenes
            if s.frames_mode != "project" and s.frames is not None
        ]
        effective_frames = (
            trimmed[0]
            if trimmed and all(f == trimmed[0] for f in trimmed)
            else target.num_frames_per_scene
        )
        graph, report = builder.build(oi, _project_models(target), {
            "prompt": prompt, "seed": target.seed,
            "num_frames_per_scene": effective_frames,
            "frame_rate": target.frame_rate,
            "width": target.width, "height": target.height,
            "negative_prompt": target.negative_prompt or None,
            "max_scenes": active_scene_count,
            "studio_inputs": _run_studio_inputs(target, active_scenes),
            "sampler_inputs": _run_sampler_inputs(target, active_scene_count),
            "reset_session": bool(body.get("reset_session")),
        }, media=_prepare_media(target))
        if report["blocking"]:
            detail = "Generation blocked — " + "; ".join(report["blocking"])
            return web.json_response({"detail": detail, "report": report}, status=400)
        try:
            result = await bridge.queue_prompt(graph)
        except Exception as e:  # noqa: BLE001
            return web.json_response({"detail": f"Failed to queue with ComfyUI: {e}"}, status=502)
        return web.json_response({"prompt_id": result.get("prompt_id"), "report": report})

    @routes.get(UI_PREFIX + "/api/projects/{pid}/status/{prompt_id}")
    async def _status(req):
        prompt_id = req.match_info["prompt_id"]
        try:
            hist = await bridge.history(prompt_id)
        except Exception as e:  # noqa: BLE001
            raise web.HTTPBadGateway(reason=f"history unavailable: {e}")
        if prompt_id in hist:
            entry = hist[prompt_id]
            raw_status = entry.get("status") or {}
            media = _media_from_history(entry)
            # Extract ComfyUI execution errors from history so the frontend can show them.
            exec_error: Optional[str] = None
            for _kind, payload in raw_status.get("messages") or []:
                if _kind == "execution_error" and isinstance(payload, dict):
                    node_type = payload.get("node_type", "unknown node")
                    exc = payload.get("exception_message") or payload.get("traceback") or "unknown error"
                    exec_error = f"{node_type}: {exc}"
                    break
            is_error = raw_status.get("status_str") == "error"
            return web.json_response({
                "state": "error" if (is_error and not media) else "completed",
                "media": media,
                "error": exec_error,
                "status": raw_status,
            })
        try:
            running = await bridge.is_running(prompt_id)
        except Exception:  # noqa: BLE001
            running = True
        return web.json_response({"state": "running" if running else "pending", "media": []})

    @routes.get(UI_PREFIX + "/api/rating-labels")
    async def _rating_labels(_req):
        try:
            return web.json_response(bridge.rating_labels())
        except Exception as e:  # noqa: BLE001
            return web.json_response({"labels": [], "error": str(e)})

    @routes.get(UI_PREFIX + "/api/progress")
    async def _progress(_req):
        return web.json_response(bridge.current_progress())

    @routes.get(UI_PREFIX + "/api/log")
    async def _log(req):
        try:
            limit = int(req.query.get("limit", "600"))
        except (TypeError, ValueError):
            limit = 600
        return web.json_response({"lines": bridge.recent_log(limit)})

    @routes.post(UI_PREFIX + "/api/interrupt")
    async def _interrupt(_req):
        try:
            return web.json_response(await bridge.interrupt())
        except Exception as e:  # noqa: BLE001
            return web.json_response({"detail": str(e)}, status=502)

    @routes.get(UI_PREFIX + "/api/projects/{pid}/result")
    async def _result(req):
        try:
            data, ctype = await bridge.fetch_view(
                req.query.get("filename", ""),
                req.query.get("subfolder", ""),
                req.query.get("type", "output"),
            )
        except Exception as e:  # noqa: BLE001
            raise web.HTTPBadGateway(reason=f"could not fetch result: {e}")
        return web.Response(body=data, content_type=ctype.split(";")[0])

    # --- API: final render (hard-cut concat of kept clips, with per-clip in/out) ---
    @routes.post(UI_PREFIX + "/api/projects/{pid}/render")
    async def _render_final(req):
        proj = _project_or_404(req.match_info["pid"])
        body = await req.json() if req.can_read_body else {}
        clips = body.get("clips") or []  # ordered [{filename, subfolder, type, in?, dur?, volume?}]
        if not clips:
            return web.json_response({"detail": "Nothing to render — no generated clips."}, status=400)
        import os
        import shutil
        import subprocess
        import time as _time
        ff = shutil.which("ffmpeg")
        if not ff:
            return web.json_response({"detail": "ffmpeg not found on PATH — install it to render the final video."}, status=503)
        try:
            import folder_paths
            outdir = folder_paths.get_output_directory()
            tempdir = folder_paths.get_temp_directory()
        except Exception as e:  # noqa: BLE001
            return web.json_response({"detail": f"Output directory unavailable: {e}"}, status=500)

        def _resolve(c):
            base = outdir if c.get("type", "output") == "output" else tempdir
            return os.path.join(base, c.get("subfolder", ""), c.get("filename", ""))

        paths = [_resolve(c) for c in clips]
        missing = [p for p in paths if not os.path.isfile(p)]
        if missing:
            return web.json_response({"detail": f"{len(missing)} clip file(s) not found on disk — regenerate then render."}, status=400)

        # Inserted audio tracks (project-level): resolve each media asset to a file. Skip any
        # that no longer exist on disk rather than failing the whole render.
        keep_original = bool(getattr(proj, "keep_original_audio", True))
        tracks = []
        for t in (getattr(proj, "audio_tracks", None) or []):
            mp = media.path_for(t.get("media_ref") or "")
            if mp is None:
                continue
            tracks.append({"path": str(mp), "start_sec": t.get("start_sec") or 0, "volume": t.get("volume", 1.0)})

        # The final render is ephemeral (temp dir) — persist it via the Export Save dialog.
        out_name = f"funpack_final_{int(_time.time())}.mp4"
        out_path = os.path.join(tempdir, out_name)
        # Per-clip in/out via -ss/-t before each input, then a filter graph that applies
        # per-clip video effects + seam transitions and the audio mix (per-clip volume,
        # original-audio toggle, inserted tracks). Re-encode keeps/mixes the audio.
        cmd = [ff, "-y"]
        for c, pth in zip(clips, paths):
            inn, dur = c.get("in"), c.get("dur")
            if inn is not None:
                cmd += ["-ss", f"{float(inn):.3f}"]
            if dur is not None:
                cmd += ["-t", f"{float(dur):.3f}"]
            cmd += ["-i", pth]
        n = len(clips)
        for t in tracks:  # inserted-audio inputs follow the clip inputs (indices n..n+k-1)
            cmd += ["-i", t["path"]]
        filt, has_audio = _build_render_filter(clips, tracks=tracks, keep_original=keep_original, base_input=n)
        cmd += ["-filter_complex", filt, "-map", "[vout]"]
        if has_audio:
            cmd += ["-map", "[aout]", "-c:a", "aac", "-b:a", "192k"]
        cmd += [
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", out_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            tail = (proc.stderr or "")[-1000:]
            return web.json_response({"detail": f"ffmpeg render failed: {tail}"}, status=500)
        return web.json_response({
            "media": {"filename": out_name, "subfolder": "", "type": "temp", "kind": "videos"},
            "clips": n,
        })

    # --- API: models / pluggable node slots ---
    @routes.get(UI_PREFIX + "/api/node-roles")
    async def _node_roles(_req):
        return web.json_response({"roles": nodes.roles_payload()})

    @routes.get(UI_PREFIX + "/api/node-candidates/{role}")
    async def _node_candidates(req):
        role = req.match_info["role"]
        refresh = req.query.get("refresh") == "true"
        try:
            oi = await bridge.object_info(refresh=refresh)
        except Exception as e:  # noqa: BLE001
            raise web.HTTPBadGateway(reason=f"object_info unavailable: {e}")
        try:
            return web.json_response({"role": role, "candidates": nodes.candidates(oi, role)})
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"[FunPack][movie] node-candidates({role}) failed:\n{traceback.format_exc()}")
            raise web.HTTPInternalServerError(reason=f"candidates({role}) failed: {e}")

    @routes.get(UI_PREFIX + "/api/pipeline-ports")
    async def _pipeline_ports(_req):
        try:
            oi = await bridge.object_info()
        except Exception:
            oi = None
        return web.json_response({
            "ports": nodes.pipeline_ports(oi),
            "core_producers": nodes.core_producers(),
            "requirements": nodes.pipeline_requirements(),
        })

    @routes.get(UI_PREFIX + "/api/image-targets")
    async def _image_targets(req):
        try:
            oi = await bridge.object_info()
        except Exception:
            oi = {}
        out = []
        for p in nodes.pipeline_ports(oi):
            if p.get("type") == "IMAGE":
                out.append({"value": "port:" + p["id"], "label": p["label"]})
        pid = req.query.get("pid")
        models_cfg = _project_models(projects.get(pid)) if pid else nodes.load_models()
        for slot in models_cfg.get("slots", []):
            nd = oi.get(slot.get("node_class")) or {}
            for ci in nodes.connection_inputs(nd):
                if ci["type"] == "IMAGE":
                    label = (slot.get("label") or slot.get("node_class")) + " · " + ci["name"]
                    out.append({"value": f"node:{slot['id']}:{ci['name']}", "label": label})
        return web.json_response({"targets": out})

    @routes.get(UI_PREFIX + "/api/core-graph")
    async def _core_graph(req):
        try:
            oi = await bridge.object_info()
        except Exception:
            oi = {}
        pid = req.query.get("pid")
        models_cfg = _project_models(projects.get(pid)) if pid else nodes.load_models()
        return web.json_response({"nodes": builder.core_graph(oi, models_cfg)})

    @routes.get(UI_PREFIX + "/api/all-nodes")
    async def _all_nodes(_req):
        try:
            oi = await bridge.object_info()
        except Exception as e:  # noqa: BLE001
            raise web.HTTPBadGateway(reason=f"object_info unavailable: {e}")
        try:
            return web.json_response({"nodes": nodes.all_nodes(oi)})
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"[FunPack][movie] all-nodes failed:\n{traceback.format_exc()}")
            raise web.HTTPInternalServerError(reason=f"all-nodes failed: {e}")

    @routes.get(UI_PREFIX + "/api/node/{cls}")
    async def _node(req):
        try:
            oi = await bridge.object_info()
        except Exception as e:  # noqa: BLE001
            raise web.HTTPBadGateway(reason=f"object_info unavailable: {e}")
        spec = nodes.describe_node(oi, req.match_info["cls"])
        if spec is None:
            raise web.HTTPNotFound(reason="Unknown node class")
        return web.json_response(spec)

    @routes.post(UI_PREFIX + "/api/restart")
    async def _restart(_req):
        import asyncio
        # defer so this 200 flushes to the browser before the process is replaced
        asyncio.get_event_loop().call_later(0.7, _restart_comfy)
        return web.json_response({"restarting": True})

    @routes.post(UI_PREFIX + "/api/models/refresh")
    async def _models_refresh(_req):
        try:
            await bridge.object_info(refresh=True)
        except Exception as e:  # noqa: BLE001
            raise web.HTTPBadGateway(reason=f"refresh failed: {e}")
        return web.json_response({"ok": True})

    @routes.get(UI_PREFIX + "/api/models")
    async def _models_get(_req):
        return web.json_response(nodes.load_models())

    @routes.put(UI_PREFIX + "/api/models")
    async def _models_put(req):
        body = await req.json()
        return web.json_response(nodes.save_models(body))

    # Per-project pipeline config (slots + links). Falls back to the global default
    # when the project has none yet; saving also updates the global default so the
    # next NEW project inherits your latest loaders.
    @routes.get(UI_PREFIX + "/api/projects/{pid}/models")
    async def _project_models_get(req):
        return web.json_response(_project_models(_project_or_404(req.match_info["pid"])))

    @routes.put(UI_PREFIX + "/api/projects/{pid}/models")
    async def _project_models_put(req):
        p = _project_or_404(req.match_info["pid"])
        body = await req.json()
        if not isinstance(body, dict):
            body = {"slots": []}
        body.setdefault("slots", [])
        p.models = body
        projects.save(p)
        nodes.save_models(body)  # keep the global default in sync (seeds new projects)
        return web.json_response(body)

    # --- UI: static frontend (must be registered AFTER api routes) ---
    @routes.get(UI_PREFIX)
    async def _root_redirect(_req):
        raise web.HTTPFound(UI_PREFIX + "/")

    @routes.get(UI_PREFIX + "/")
    async def _index(_req):
        return _serve_static("index.html")

    @routes.get(UI_PREFIX + "/{tail:.*}")
    async def _static(req):
        return _serve_static(req.match_info["tail"])

    print(f"[FunPack] Movie Editor available at {UI_PREFIX}/")
