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

from .backend import bridge, builder, config, git_update, media, nodes, pipeline_caps, pipeline_deps, pipeline_wiring, projects, workflow_import
from .backend.nle_effects import zoompan_z_expr
from .backend.nle_overlays import build_overlay_video_filter, prepare_overlay_export
from .backend.timeline import (
    Project,
    build_combined_prompt,
    collapse_generative_units,
    continuity_media_refs,
    effective_negative_prompt,
    gen_unit_id,
    group_generative_units,
    is_video_clip,
)

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


def _scene_layout_from_history(hist_entry: dict, fps: float) -> Optional[list]:
    from .backend.chain_layout import layout_from_history_entry
    return layout_from_history_entry(hist_entry, fps)


def _resolve_comfy_media_path(filename: str, subfolder: str = "", type_: str = "output") -> Optional[str]:
    """Absolute path to a ComfyUI output/temp media file (None if dirs unavailable)."""
    import os
    if not filename:
        return None
    try:
        import folder_paths
        base = folder_paths.get_output_directory() if type_ == "output" else folder_paths.get_temp_directory()
    except Exception:
        return None
    return os.path.join(base, subfolder or "", filename)


def _resolve_clip_src_path(clip: dict) -> str:
    """Absolute path to a render or media-bin file referenced by an export/import clip spec."""
    import os
    ref = clip.get("bin_media_ref")
    if ref:
        mp = media.path_for(ref)
        if mp is None:
            raise FileNotFoundError("Media bin file not found — it may have been deleted.")
        return str(mp)
    fp = _resolve_comfy_media_path(
        clip.get("filename", ""),
        clip.get("subfolder", ""),
        clip.get("type", "output"),
    )
    if not fp or not os.path.isfile(fp):
        raise FileNotFoundError("Source clip file not found on disk — regenerate then try again.")
    return fp


def _clip_needs_trim(clip: dict) -> bool:
    inn = float(clip.get("in") or 0)
    dur = clip.get("dur")
    return inn > 0.001 or dur is not None


def _ffmpeg_trim_clip(src_path: str, out_path: str, inn, dur) -> None:
    import shutil
    import subprocess
    ff = shutil.which("ffmpeg")
    if not ff:
        raise RuntimeError("ffmpeg not found on PATH — install it to export or import clips.")
    cmd = [ff, "-y"]
    if inn is not None and float(inn) > 0:
        cmd += ["-ss", f"{float(inn):.3f}"]
    if dur is not None:
        cmd += ["-t", f"{float(dur):.3f}"]
    cmd += [
        "-i", src_path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", out_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or "ffmpeg failed")[-1000:])


def _playback_render_from_query(query) -> Optional[dict]:
    """Optional live render override from preview-segment query params."""
    filename = query.get("filename") if query else None
    if not filename:
        return None
    return {
        "inSec": float(query.get("render_in") or 0),
        "media": {
            "filename": filename,
            "subfolder": query.get("subfolder") or "",
            "type": query.get("type") or "output",
        },
    }


def _scene_playback_clip_spec(
    project: Project,
    scene_id: str,
    *,
    render_override: Optional[dict] = None,
) -> dict:
    """Export-matching trim spec for one scene's preview segment."""
    scene = next((s for s in project.scenes if s.id == scene_id), None)
    if not scene or scene.excluded:
        raise KeyError(f"scene {scene_id}")
    if render_override is not None:
        render = render_override
    else:
        render = (project.scene_renders or {}).get(scene_id) or {}
    media = render.get("media") or {}
    if not media.get("filename"):
        raise KeyError(f"no render for {scene_id}")
    in_sec = float(scene.source_in or 0) + float(render.get("inSec") or 0)
    if scene.source_dur is not None:
        dur = float(scene.source_dur)
    else:
        fps = scene.eff_fps(project) or 25
        dur = scene.eff_frames(project) / float(fps)
    return {
        "filename": media.get("filename", ""),
        "subfolder": media.get("subfolder") or "",
        "type": media.get("type") or "output",
        "in": in_sec,
        "dur": dur,
    }


_preview_segment_cache: dict[str, str] = {}


def _preview_segment_cache_key(pid: str, scene_id: str, clip: dict) -> str:
    return (
        f"{pid}:{scene_id}:{clip.get('filename')}:{clip.get('subfolder') or ''}:"
        f"{clip.get('type') or 'output'}:{clip.get('in')}:{clip.get('dur')}"
    )


def _clip_bytes_for_media(clip: dict) -> bytes:
    """Read (and optionally trim) a clip spec into bytes for the media bin."""
    import os
    import time
    from pathlib import Path
    src = _resolve_clip_src_path(clip)
    if not _clip_needs_trim(clip):
        return Path(src).read_bytes()
    try:
        import folder_paths
        tempdir = folder_paths.get_temp_directory()
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Temp directory unavailable: {e}") from e
    out_path = os.path.join(tempdir, f"funpack_import_{int(time.time())}.mp4")
    try:
        _ffmpeg_trim_clip(src, out_path, clip.get("in"), clip.get("dur"))
        return Path(out_path).read_bytes()
    finally:
        Path(out_path).unlink(missing_ok=True)


def _copy_scene_media(ref: str, indir: str) -> Optional[str]:
    import os
    import shutil
    path = media.path_for(ref)
    if not path:
        return None
    fn = f"funpack_movie_{ref}{path.suffix}"
    try:
        shutil.copy(str(path), os.path.join(indir, fn))
    except OSError:
        return None
    return fn


def _prepare_media(proj: Project, extra_refs: Optional[list] = None, *, chain_available: bool = True) -> Optional[dict]:
    """Copy scene image assets into ComfyUI's input folder.

    Returns {"primary": {filename, target}, "by_scene": {scene_id: {filename, media_ref, target}}}
    for the builder (run anchor) and sampler guide lookups. `extra_refs` copies additional
    media-bin ids (e.g. prior-scene guide images for solo mixed runs).
    """
    import os
    try:
        import folder_paths
        indir = folder_paths.get_input_directory()
    except Exception:
        return None
    primary = None
    by_scene: dict = {}
    seen_refs: set = set()

    def _add_ref(ref: str, tgt=None, sid: Optional[str] = None, *, set_primary: bool = True) -> None:
        nonlocal primary
        if not ref or ref in seen_refs:
            return
        fn = _copy_scene_media(ref, indir)
        if not fn:
            return
        seen_refs.add(ref)
        entry = {"filename": fn, "media_ref": ref, "target": tgt}
        if sid:
            by_scene[sid] = entry
        else:
            by_scene[f"_ref_{ref}"] = entry
        if set_primary and primary is None:
            primary = {"filename": fn, "target": tgt}

    for sc in proj.scenes:
        if sc.excluded:
            continue
        src = sc.source
        ref = getattr(src, "media_ref", None)
        tgt = getattr(src, "target", None)
        stype = pipeline_caps.effective_source_type(sc, chain_available)
        if stype not in ("image", "generated_frame", "mixed") or not ref:
            continue
        _add_ref(ref, tgt, sc.id, set_primary=True)
    for ref in (extra_refs or []):
        # Guide-only assets (identity pin, prior-scene borrow) must not become the run anchor.
        _add_ref(str(ref), set_primary=False)
    if not by_scene:
        return None
    return {"primary": primary, "by_scene": by_scene}


def _project_or_404(pid: str) -> Project:
    p = projects.get(pid)
    if p is None:
        raise web.HTTPNotFound(reason=f"Project {pid} not found")
    return p


def _parse_has_scenes(parsed: Optional[dict]) -> bool:
    return isinstance(parsed, dict) and bool(parsed.get("scenes"))


def _parse_prompt_variants(prompt: str, seed: int) -> tuple[dict, dict]:
    """Run Studio/raw/verbatim splits independently; never fail all because one path threw."""
    parsed = parsed_raw = parsed_verbatim = None
    errors: dict[str, str] = {}
    for key, fn in (
        ("parsed", lambda: bridge.parse_timeline(prompt, seed=seed)),
        ("parsed_raw", lambda: bridge.parse_timeline_raw(prompt)),
        ("parsed_verbatim", lambda: bridge.parse_timeline_verbatim(prompt)),
    ):
        try:
            value = fn()
        except Exception as exc:  # noqa: BLE001
            errors[key] = bridge.format_funpack_error(exc)
            continue
        if key == "parsed":
            parsed = value
        elif key == "parsed_raw":
            parsed_raw = value
        else:
            parsed_verbatim = value
    return (
        {
            "parsed": parsed,
            "parsed_raw": parsed_raw,
            "parsed_verbatim": parsed_verbatim,
            "combined_prompt": prompt,
            **({"parse_errors": errors} if errors else {}),
        },
        errors,
    )


def _project_models(p: Optional[Project]) -> dict:
    """The project's own pipeline config, or the global default when it has none yet."""
    m = getattr(p, "models", None)
    if not isinstance(m, dict):
        return nodes.load_models()
    # Treat slots: [] as intentional (user removed all loaders) — do not fall back to global.
    if "slots" in m or m.get("links") or m.get("core_overrides") or m.get("disable_core") or m.get("full_control"):
        return m
    return nodes.load_models()


def _solo(p: Project, only_scene: Optional[str]) -> Project:
    if not only_scene:
        return collapse_generative_units(p)
    scene = next((s for s in p.scenes if s.id == only_scene), None)
    if scene is None:
        raise web.HTTPNotFound(reason=f"Scene {only_scene} not found")
    uid = gen_unit_id(scene)
    ids = [s.id for s in p.scenes if gen_unit_id(s) == uid and not s.excluded]
    return collapse_generative_units(_segment(p, ids))


def _run_studio_inputs(
    target: Project,
    active_scenes: list,
    *,
    prompt_changed: bool = False,
    models: Optional[dict] = None,
) -> dict:
    """Studio widget overrides for this run, incl. the RLHF rating of the run's last
    render (Studio refines from it).

    Single-scene runs: one scene rating maps to the Studio rating widget (same as before).
    Multi-scene chain runs: per-scene ratings are passed in studio_settings.refiner
    (movie_editor_scene_ratings); the top-level rating is continue/fresh so global learning
    does not collapse multiple scene ratings into one.

    When no scene was rated: send MOVIE_EDITOR_CONTINUE_RATING so Studio still applies
    session memory / active repairs but does not treat the run as '-Just forget it-'.
    When the generation prompt changed since the last queue, send
    MOVIE_EDITOR_FRESH_PROMPT_RATING instead — clears stale repairs only, keeps training.
    refine_v2 uses Initial discovery when there is no previous run to refine from."""
    if not pipeline_caps.uses_funpack_studio(target, models):
        return {}
    si = dict(target.studio_inputs or {})
    try:
        from conditioning import (
            MOVIE_EDITOR_CONTINUE_RATING,
            MOVIE_EDITOR_FRESH_PROMPT_RATING,
        )
    except ImportError:
        try:
            from ..conditioning import (  # type: ignore
                MOVIE_EDITOR_CONTINUE_RATING,
                MOVIE_EDITOR_FRESH_PROMPT_RATING,
            )
        except ImportError:
            MOVIE_EDITOR_CONTINUE_RATING = "__funpack_continue__"
            MOVIE_EDITOR_FRESH_PROMPT_RATING = "__funpack_fresh_prompt__"
    default_rating = (
        MOVIE_EDITOR_FRESH_PROMPT_RATING if prompt_changed else MOVIE_EDITOR_CONTINUE_RATING
    )
    scene_ratings = []
    for i, sc in enumerate(active_scenes):
        label = (getattr(sc, "rating", "") or "").strip()
        if label:
            scene_ratings.append({"index": i, "rating": label})
    if len(active_scenes) > 1 and scene_ratings:
        si["rating"] = default_rating
        si["_movie_editor_scene_ratings"] = scene_ratings
    elif scene_ratings:
        si["rating"] = scene_ratings[0]["rating"]
    else:
        si["rating"] = default_rating
    return si


def _shortcut_seed(project: Project) -> int:
    """Seed for prompt shortcut / wildcard expansion in split preview."""
    si = project.sampler_inputs or {}
    if si.get("seed") is not None:
        return int(si["seed"])
    return 0


def _resolve_run_seed(target: Project) -> int:
    """Sampler seed for one Generate click — fixed when set in Chain Sampler, else random."""
    import random
    si = target.sampler_inputs or {}
    if si.get("seed") is not None:
        return int(si["seed"])
    return random.randint(1, 0xFFFFFFFFFFFFFFFF)


def _run_sampler_inputs(
    target: Project,
    scene_count: int,
    full: Optional[Project] = None,
    *,
    models: Optional[dict] = None,
) -> dict:
    """Sampler widget overrides for one generation run.

    Auto continuity (project.continuity_settings) builds guides and sampler knobs per run
    for carry, mixed, image, and empty sources. Manual guide_settings.stack_enabled wins.
    """
    if not pipeline_caps.uses_chain_sampler(target, models):
        return {}
    import json
    from .backend.timeline import (
        build_auto_continuity_guides,
        build_scene_guides_payload,
        continuity_settings_for_run,
        is_mixed_source,
        normalize_guide_settings,
    )

    proj = full or target
    samp = dict(target.sampler_inputs or {})
    active = [s for s in target.scenes if not s.excluded]
    cs = continuity_settings_for_run(proj)
    manual_stack = normalize_guide_settings(proj.guide_settings)["stack_enabled"]

    auto_guides = build_auto_continuity_guides(proj, target) if cs["auto_enabled"] else None
    manual_guides = build_scene_guides_payload(target) if manual_stack else None
    guides = manual_guides or auto_guides

    if guides:
        samp["funpack_scene_guides"] = json.dumps(guides)
        samp["carry_i2v_guides"] = False

    solo_run = scene_count == 1 and len(active) == 1
    if solo_run:
        samp["frame_overlap"] = 0
        if not guides:
            from .backend.timeline import source_type
            if source_type(active[0]) in ("image", "generated_frame", "mixed", "empty"):
                samp["carry_i2v_guides"] = False

    if scene_count > 1 and not guides:
        samp["carry_i2v_guides"] = True

    if cs["auto_enabled"] and scene_count > 1 and cs["mid_scene_guide"]:
        samp["mid_scene_guide"] = True
        samp["mid_scene_guide_strength"] = cs["mid_scene_guide_strength"]

    return samp


def _scene_media_ref_map(media_pack: Optional[dict]) -> dict[str, str]:
    if not media_pack:
        return {}
    by_scene = media_pack.get("by_scene") or {}
    return {
        entry["media_ref"]: entry["filename"]
        for entry in by_scene.values()
        if isinstance(entry, dict) and entry.get("media_ref") and entry.get("filename")
    }


def _missing_continuity_media(full: Project, target: Project) -> list[str]:
    """Media-bin ids referenced by active continuity or manual guide stacks."""
    from .backend.timeline import continuity_media_refs

    missing: list[str] = []
    for ref in continuity_media_refs(full, target):
        if not media.path_for(ref):
            missing.append(ref)
    return missing


def _missing_scene_anchor_media(target: Project, *, chain_available: bool = True) -> list[str]:
    """i2v anchor images referenced on scenes but missing from the media bin."""
    if not chain_available:
        return []
    from .backend.timeline import scene_anchor_media_refs

    missing: list[str] = []
    for ref in scene_anchor_media_refs(target):
        if not media.path_for(ref):
            missing.append(ref)
    return missing


def _attach_scene_anchors(samp: dict, media_pack: Optional[dict], target: Project) -> dict:
    """Mixed-source i2v anchors (Img2Video starting latents), separate from guides."""
    import json
    from .backend.timeline import build_scene_anchors_payload

    by_ref = _scene_media_ref_map(media_pack)
    if by_ref:
        samp["funpack_scene_media_refs"] = json.dumps(by_ref)

    anchors = build_scene_anchors_payload(target)
    if not anchors or not media_pack:
        return samp
    by_scene = media_pack.get("by_scene") or {}
    enriched: dict = {}
    for idx, meta in anchors.items():
        sid = meta.get("scene_id")
        file_entry = by_scene.get(sid) if sid else None
        fn = (file_entry or {}).get("filename")
        if fn:
            enriched[idx] = {**meta, "filename": fn}
    if enriched:
        samp["funpack_scene_anchors"] = json.dumps(enriched)
    return samp


_XFADE_MAP = {
    "crossfade": "fade", "fadeblack": "fadeblack",
    "wipeleft": "wipeleft", "wiperight": "wiperight", "dissolve": "dissolve",
}


def _build_render_filter(clips: list, tracks: Optional[list] = None,
                         keep_original: bool = True, base_input: int = 0,
                         *, blank_canvas: Optional[dict] = None) -> tuple[str, bool]:
    """ffmpeg filter_complex for the final stitch.

    Video — per clip: normalize to one canvas, then blur / fade in-out / virtual Ken-Burns
    zoom (zoompan, fixed output size). Fold left-to-right with xfade (overlap) or concat.
    When `blank_canvas` is set, input 0 is a lavfi color source (no scene clips).

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

    if blank_canvas:
        total = max(0.01, float(blank_canvas.get("dur") or 0.01))
        parts.append("[0:v]format=yuv420p,setsar=1[vbase]")
        acc_a = None
    else:
        if not clips:
            raise ValueError("clips required when blank_canvas is not set")
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
                nframes = max(1, round(dur * cfps))
                z = zoompan_z_expr(zoom, fx, nframes)
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
            gap = float(prev.get("gap_after") or 0)
            if gap > 0.001:
                gv = f"[vgap{i}]"
                ga = f"[agap{i}]"
                parts.append(
                    f"color=c=black:s={cw}x{ch}:r={cfps:g}:d={gap:.3f},format=yuv420p,setsar=1{gv}"
                )
                parts.append(
                    f"anullsrc=r=48000:cl=stereo:d={gap:.3f},"
                    f"aformat=sample_fmts=fltp:channel_layouts=stereo{ga}"
                )
                ov = f"[vcgap{i}]"
                parts.append(f"{acc_v}{gv}concat=n=2:v=1:a=0{ov}")
                if keep_original:
                    oa = f"[acgap{i}]"
                    parts.append(f"{acc_a}{ga}concat=n=2:v=0:a=1{oa}")
                    acc_a = oa
                acc_v = ov
                acc_dur += gap
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

        parts.append(f"{acc_v}null[vbase]")
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


def _append_overlay_filters(
    parts: list[str],
    export_overlays: list[dict],
    *,
    canvas_w: int,
    canvas_h: int,
    image_paths: list[str],
    image_input_base: int,
) -> None:
    """Append overlay compositing after [vbase]; sets final [vout]."""
    if not export_overlays:
        parts.append("[vbase]null[vout]")
        return
    labels = [f"[{image_input_base + i}:v:0]" for i in range(len(image_paths))]
    ov_parts, final = build_overlay_video_filter(
        "[vbase]", export_overlays, canvas_w=canvas_w, canvas_h=canvas_h, image_input_labels=labels,
    )
    parts.extend(ov_parts)
    if final == "[vbase]":
        parts.append("[vbase]null[vout]")
    else:
        parts.append(f"{final}null[vout]")


class RenderJobError(Exception):
    pass


def _audio_track_end_sec(t: dict) -> float:
    start = float(t.get("start_sec") or 0)
    for key in ("source_dur", "pinned_dur"):
        if t.get(key) is not None:
            return start + float(t[key])
    return start + 1.0


def _timeline_duration_sec(proj) -> float:
    total = 0.0
    for ov in getattr(proj, "overlay_tracks", None) or []:
        end = float(ov.get("start_sec") or 0) + float(ov.get("duration_sec") or 0)
        total = max(total, end)
    for t in getattr(proj, "audio_tracks", None) or []:
        total = max(total, _audio_track_end_sec(t))
    return max(total, 0.01)


def _has_graphics_export_content(proj) -> bool:
    if list(getattr(proj, "audio_tracks", None) or []):
        return True
    return any(float(ov.get("duration_sec") or 0) > 0 for ov in (getattr(proj, "overlay_tracks", None) or []))


_render_jobs: dict[str, dict] = {}
_export_jobs: dict[str, dict] = {}


def _ffmpeg_concat_clips_plain(clips: list) -> dict:
    """Join trimmed clips with hard cuts only — no effects, transitions, or extra audio mix."""
    import os
    import shutil
    import subprocess
    import time as _time

    if not clips:
        raise RenderJobError("Nothing to export.")
    ff = shutil.which("ffmpeg")
    if not ff:
        raise RenderJobError("ffmpeg not found on PATH — install it to export clips.")
    try:
        import folder_paths
        tempdir = folder_paths.get_temp_directory()
    except Exception as e:  # noqa: BLE001
        raise RenderJobError(f"Output directory unavailable: {e}") from e

    stamp = int(_time.time())
    if len(clips) == 1:
        out_name = f"funpack_export_{stamp}.mp4"
        out_path = os.path.join(tempdir, out_name)
        try:
            src_path = _resolve_clip_src_path(clips[0])
            _ffmpeg_trim_clip(src_path, out_path, clips[0].get("in"), clips[0].get("dur"))
        except FileNotFoundError as e:
            raise RenderJobError(str(e)) from e
        except RuntimeError as e:
            raise RenderJobError(str(e)) from e
        return {
            "media": {"filename": out_name, "subfolder": "", "type": "temp", "kind": "videos"},
            "clips": 1,
        }

    seg_paths: list[str] = []
    for i, c in enumerate(clips):
        try:
            src_path = _resolve_clip_src_path(c)
            seg = os.path.join(tempdir, f"funpack_seg_{stamp}_{i}.mp4")
            _ffmpeg_trim_clip(src_path, seg, c.get("in"), c.get("dur"))
            seg_paths.append(seg)
        except FileNotFoundError as e:
            raise RenderJobError(str(e)) from e
        except RuntimeError as e:
            raise RenderJobError(str(e)) from e

    out_name = f"funpack_export_{stamp}.mp4"
    out_path = os.path.join(tempdir, out_name)
    list_path = os.path.join(tempdir, f"funpack_concat_{stamp}.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for p in seg_paths:
            esc = p.replace("'", "'\\''")
            f.write(f"file '{esc}'\n")
    cmd = [
        ff, "-y", "-f", "concat", "-safe", "0", "-i", list_path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", out_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or "")[-1000:]
        raise RenderJobError(f"ffmpeg concat failed: {tail}")
    return {
        "media": {"filename": out_name, "subfolder": "", "type": "temp", "kind": "videos"},
        "clips": len(clips),
    }


def _ffmpeg_stitch_final(proj, clips: list) -> dict:
    """Blocking ffmpeg stitch — run via asyncio.to_thread from the render job handler."""
    import os
    import shutil
    import subprocess
    import time as _time

    blank_mode = not clips
    if blank_mode and not _has_graphics_export_content(proj):
        raise RenderJobError("Nothing to render — add overlays, audio, or generated clips.")
    ff = shutil.which("ffmpeg")
    if not ff:
        raise RenderJobError("ffmpeg not found on PATH — install it to render the final video.")
    try:
        import folder_paths
        outdir = folder_paths.get_output_directory()
        tempdir = folder_paths.get_temp_directory()
    except Exception as e:  # noqa: BLE001
        raise RenderJobError(f"Output directory unavailable: {e}") from e

    cw = int(getattr(proj, "width", 768) or 768)
    ch = int(getattr(proj, "height", 512) or 512)
    cfps = float(getattr(proj, "frame_rate", 25) or 25)
    blank_canvas = None
    n_video = 0

    def _resolve(c):
        if c.get("bin_media_ref"):
            mp = media.path_for(c.get("bin_media_ref") or "")
            if mp is None:
                return ""
            return str(mp)
        base = outdir if c.get("type", "output") == "output" else tempdir
        return os.path.join(base, c.get("subfolder", ""), c.get("filename", ""))

    paths: list = []
    if blank_mode:
        duration = _timeline_duration_sec(proj)
        blank_canvas = {"w": cw, "h": ch, "fps": cfps, "dur": duration}
        n_video = 1
    else:
        paths = [_resolve(c) for c in clips]
        missing = [p for p in paths if not os.path.isfile(p)]
        if missing:
            raise RenderJobError(f"{len(missing)} clip file(s) not found on disk — regenerate then render.")
        cw = int(clips[0].get("w") or 0) or cw
        ch = int(clips[0].get("h") or 0) or ch
        n_video = len(clips)

    keep_original = bool(getattr(proj, "keep_original_audio", True)) and not blank_mode
    clip_by_scene = {c["scene_id"]: c for c in clips if c.get("scene_id")}
    media_tracks: list = []
    separated_tracks: list = []
    for t in (getattr(proj, "audio_tracks", None) or []):
        is_sep = t.get("kind") == "separated" or (
            t.get("scene_id") and not t.get("media_ref") and t.get("kind") != "overlay"
        )
        if is_sep:
            sid = t.get("scene_id")
            pinned = t.get("pinned_media") if isinstance(t.get("pinned_media"), dict) else None
            pth = None
            if pinned and pinned.get("filename"):
                pth = _resolve_comfy_media_path(
                    pinned.get("filename", ""),
                    pinned.get("subfolder", ""),
                    pinned.get("type", "output"),
                )
            bin_ref = t.get("pinned_bin_ref")
            if (not pth or not os.path.isfile(pth)) and bin_ref:
                mp = media.path_for(str(bin_ref))
                if mp is not None:
                    pth = str(mp)
            c = clip_by_scene.get(sid) if sid else None
            if not pth or not os.path.isfile(pth):
                if not c:
                    continue
                pth = _resolve(c)
                if not os.path.isfile(pth):
                    continue
            src_in = t.get("pinned_in_sec", t.get("source_in_sec"))
            if src_in is None:
                src_in = (c.get("in") if c else None) or 0
            src_dur = t.get("pinned_dur", t.get("source_dur"))
            if src_dur is None:
                src_dur = (c.get("dur") if c else None) or 0
            separated_tracks.append({
                "path": pth,
                "source_in": float(src_in),
                "source_dur": float(src_dur),
                "start_sec": t.get("start_sec") or 0,
                "volume": t.get("volume", 1.0),
            })
            continue
        mp = media.path_for(t.get("media_ref") or "")
        if mp is None:
            continue
        entry = {
            "path": str(mp),
            "start_sec": t.get("start_sec") or 0,
            "volume": t.get("volume", 1.0),
        }
        src_in = t.get("source_in_sec")
        src_dur = t.get("source_dur")
        if src_in is not None:
            entry["source_in"] = float(src_in)
        if src_dur is not None:
            entry["source_dur"] = float(src_dur)
        media_tracks.append(entry)

    out_name = f"funpack_final_{int(_time.time())}.mp4"
    out_path = os.path.join(tempdir, out_name)
    cmd = [ff, "-y"]
    if blank_mode:
        cmd += ["-f", "lavfi", "-i", f"color=c=black:s={cw}x{ch}:r={cfps:g}:d={blank_canvas['dur']:.3f}"]
    else:
        for c, pth in zip(clips, paths):
            inn, dur = c.get("in"), c.get("dur")
            if inn is not None:
                cmd += ["-ss", f"{float(inn):.3f}"]
            if dur is not None:
                cmd += ["-t", f"{float(dur):.3f}"]
            cmd += ["-i", pth]
    n = n_video
    for t in media_tracks:
        if t.get("source_in") is not None and t.get("source_dur") is not None:
            cmd += ["-ss", f"{t['source_in']:.3f}", "-t", f"{t['source_dur']:.3f}", "-i", t["path"]]
        else:
            cmd += ["-i", t["path"]]
    for t in separated_tracks:
        cmd += ["-ss", f"{t['source_in']:.3f}", "-t", f"{t['source_dur']:.3f}", "-i", t["path"]]
    tracks = media_tracks + separated_tracks

    def _overlay_image_path(media_ref: str | None) -> str | None:
        mp = media.path_for(media_ref or "")
        return str(mp) if mp is not None else None

    export_overlays, overlay_image_paths = prepare_overlay_export(
        list(getattr(proj, "overlay_tracks", None) or []),
        list(getattr(proj, "overlay_lanes", None) or []),
        canvas_w=cw,
        canvas_h=ch,
        tempdir=tempdir,
        resolve_image_path=_overlay_image_path,
    )
    for pth in overlay_image_paths:
        cmd += ["-i", pth]

    filt, has_audio = _build_render_filter(
        clips, tracks=tracks, keep_original=keep_original, base_input=n,
        blank_canvas=blank_canvas,
    )
    filt_parts = filt.split(";") if filt else []
    ov_input_base = n + len(tracks)
    _append_overlay_filters(
        filt_parts, export_overlays, canvas_w=cw, canvas_h=ch,
        image_paths=overlay_image_paths, image_input_base=ov_input_base,
    )
    filt = ";".join(filt_parts)
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
        raise RenderJobError(f"ffmpeg render failed: {tail}")
    return {
        "media": {"filename": out_name, "subfolder": "", "type": "temp", "kind": "videos"},
        "clips": n,
    }


def _segment(p: Project, scene_ids: list) -> Project:
    """Project clone holding only `scene_ids` (in their project order). Used to
    generate one chain run: its first scene supplies the i2v anchor, the rest are
    carries that overlap inside the run (one ComfyUI chain-sampler request)."""
    ids = set(scene_ids)
    clone = Project.from_dict(p.to_dict())
    clone.scenes = [s for s in clone.scenes if s.id in ids]
    if not clone.scenes:
        raise web.HTTPNotFound(reason="No scenes matched the requested segment")
    return collapse_generative_units(clone)


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
            "template_exists": config.TEMPLATE_PATH.is_file(),
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
        try:
            p = projects.save(Project.from_dict(body))
        except Exception as e:  # noqa: BLE001
            return web.json_response({"detail": f"Invalid project file: {e}"}, status=400)
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
        for_generation = req.query.get("for_generation", "true") == "true"
        preview_target = p
        if not include_excluded:
            preview_target = Project.from_dict(p.to_dict())
            preview_target.scenes = [s for s in preview_target.scenes if not s.excluded]
        if for_generation:
            try:
                validation = bridge.validate_generation_prompt(p, preview_target)
            except Exception as e:  # noqa: BLE001
                return web.json_response({
                    "combined_prompt": "",
                    "display_prompt": "",
                    "for_generation": True,
                    "parse_error": str(e),
                    "validation_error": str(e),
                })
            prompt = validation["generation_prompt"]
            result = {
                "combined_prompt": prompt,
                "display_prompt": validation["display_prompt"],
                "for_generation": True,
                "validation": validation,
            }
            try:
                parsed = bridge.parse_timeline(prompt, seed=_shortcut_seed(p))
                result["parsed"] = parsed
                result["parsed_raw"] = bridge.parse_timeline_raw(prompt)
                result["parsed_verbatim"] = bridge.parse_timeline_verbatim(prompt)
                result["expected_scenes"] = validation["expected_scenes"]
                result["parsed_scenes"] = len(parsed.get("scenes", []))
                result["scene_refinement_keys"] = bridge.scene_refinement_keys(
                    prompt, len(parsed.get("scenes", [])), p.refinement_key)
                result["refinement_key_pool"] = bridge.refinement_key_pool(prompt)
            except Exception as e:  # noqa: BLE001
                result["parse_error"] = str(e)
            return web.json_response(result)

        prompt = build_combined_prompt(p, include_excluded=include_excluded)
        result = {"combined_prompt": prompt, "for_generation": False}
        try:
            parsed = bridge.parse_timeline(prompt, seed=_shortcut_seed(p))
            result["parsed"] = parsed
            result["parsed_raw"] = bridge.parse_timeline_raw(prompt)
            result["parsed_verbatim"] = bridge.parse_timeline_verbatim(prompt)
            expected = len([s for s in p.scenes if include_excluded or not s.excluded])
            got = len(parsed.get("scenes", []))
            result["expected_scenes"] = expected
            result["parsed_scenes"] = got
            result["scene_refinement_keys"] = bridge.scene_refinement_keys(
                prompt, got, p.refinement_key)
            result["refinement_key_pool"] = bridge.refinement_key_pool(prompt)
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
        payload, errors = _parse_prompt_variants(prompt, _shortcut_seed(p))
        usable = any(_parse_has_scenes(payload.get(k)) for k in ("parsed", "parsed_raw", "parsed_verbatim"))
        if errors and not usable:
            detail = "; ".join(errors.values())
            return web.json_response({"detail": detail, "parse_errors": errors}, status=502)
        return web.json_response(payload)

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

    @routes.get(UI_PREFIX + "/api/library/characters")
    async def _characters(_req):
        try:
            from .backend import characters as char_lib
            return web.json_response(char_lib.list_characters())
        except Exception as e:  # noqa: BLE001
            return web.json_response({"characters": [], "error": str(e)})

    @routes.post(UI_PREFIX + "/api/library/characters")
    async def _character_save(req):
        try:
            from .backend import characters as char_lib
            return web.json_response(char_lib.save_character(await req.json()))
        except Exception as e:  # noqa: BLE001
            raise web.HTTPBadRequest(reason=str(e))

    @routes.delete(UI_PREFIX + "/api/library/characters/{cid}")
    async def _character_delete(req):
        try:
            from .backend import characters as char_lib
            return web.json_response(char_lib.delete_character(req.match_info["cid"]))
        except Exception as e:  # noqa: BLE001
            raise web.HTTPBadRequest(reason=str(e))

    @routes.get(UI_PREFIX + "/api/library/nle")
    async def _nle_library(_req):
        try:
            from .backend import nle_library
            return web.json_response(nle_library.load())
        except Exception as e:  # noqa: BLE001
            return web.json_response({"effects": [], "video_transitions": [], "error": str(e)})

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
                "Content-Disposition": "attachment; filename=funpack_promptsplit.json",
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

    @routes.patch(UI_PREFIX + "/api/media/{mid}")
    async def _media_rename(req):
        body = await req.json() if req.can_read_body else {}
        name = (body.get("name") or "").strip()
        if not name:
            raise web.HTTPBadRequest(reason="Name is required.")
        entry = media.rename(req.match_info["mid"], name)
        if entry is None:
            raise web.HTTPNotFound()
        return web.json_response({"media": entry})

    @routes.post(UI_PREFIX + "/api/media/import-clip")
    async def _media_import_clip(req):
        body = await req.json() if req.can_read_body else {}
        clip = body.get("clip") or {}
        name = (body.get("name") or "clip.mp4").strip() or "clip.mp4"
        if not clip.get("bin_media_ref") and not clip.get("filename"):
            return web.json_response({"detail": "Missing clip source."}, status=400)
        if not name.lower().endswith((".mp4", ".webm", ".mov", ".mkv", ".avi")):
            name = f"{name}.mp4"
        try:
            data = _clip_bytes_for_media(clip)
        except FileNotFoundError as e:
            return web.json_response({"detail": str(e)}, status=400)
        except RuntimeError as e:
            return web.json_response({"detail": str(e)}, status=503)
        entry = media.save_upload(name, data)
        return web.json_response({"media": entry})

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
        try:
            validation = bridge.validate_generation_prompt(p, target)
        except Exception as e:  # noqa: BLE001
            return web.json_response({"detail": f"Prompt validation failed: {e}"}, status=400)
        prompt = validation["generation_prompt"]
        if not prompt.strip():
            return web.json_response({"detail": "Nothing to generate — no active scene text."}, status=400)
        prompt_changed = bool(validation.get("prompt_changed_since_last_queue"))
        reset_session = bool(body.get("reset_session"))
        try:
            oi = await bridge.object_info()
        except Exception as e:  # noqa: BLE001
            return web.json_response({"detail": f"Node registry unavailable: {e}"}, status=502)
        models_cfg = _project_models(p)
        caps = pipeline_caps.capabilities(p, models_cfg)
        if not caps["studio"]:
            reset_session = False
        bridge.reset_progress()
        bridge.current_progress()  # ensure the sampler step-progress hook is installed
        active_scenes = [s for s in target.scenes if not s.excluded and not is_video_clip(s)]
        active_scene_count = len(active_scenes)
        if not active_scene_count:
            return web.json_response(
                {"detail": "Nothing to generate — all clips are video-only or excluded."},
                status=400,
            )
        missing_media = _missing_continuity_media(p, target) if caps["chain_sampler"] else []
        if missing_media:
            ids = ", ".join(missing_media)
            return web.json_response(
                {"detail": f"Missing media for continuity or guide stack ({ids}). Re-upload or clear the reference in Engine settings / Scene inspector."},
                status=400,
            )
        missing_anchors = _missing_scene_anchor_media(target, chain_available=caps["chain_sampler"])
        if missing_anchors:
            ids = ", ".join(missing_anchors)
            return web.json_response(
                {"detail": f"Missing i2v anchor image in Media bin ({ids}). Re-save the frame or pick another anchor."},
                status=400,
            )
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
        try:
            media_pack = _prepare_media(
                target,
                continuity_media_refs(p, target) if caps["chain_sampler"] else [],
                chain_available=caps["chain_sampler"],
            )
            sampler_inputs = _run_sampler_inputs(target, active_scene_count, full=p, models=models_cfg)
            if caps["chain_sampler"]:
                _attach_scene_anchors(sampler_inputs, media_pack, target)
            graph, report = builder.build(oi, models_cfg, {
                "prompt": prompt, "seed": _resolve_run_seed(target),
                "num_frames_per_scene": effective_frames,
                "frame_rate": target.frame_rate,
                "width": target.width, "height": target.height,
                "negative_prompt": effective_negative_prompt(target) or None,
                "max_scenes": active_scene_count,
                "studio_inputs": _run_studio_inputs(
                    target, active_scenes, prompt_changed=prompt_changed, models=models_cfg,
                ),
                "sampler_inputs": sampler_inputs,
                "reset_session": reset_session,
                "refinement_key": (target.refinement_key or "default"),
            }, media=(media_pack or {}).get("primary") if media_pack else None)
        except Exception as e:  # noqa: BLE001
            return web.json_response({"detail": f"Workflow build failed: {e}"}, status=502)
        if report["blocking"]:
            detail = "Generation blocked — " + "; ".join(report["blocking"])
            return web.json_response({"detail": detail, "report": report}, status=400)
        try:
            result = await bridge.queue_prompt(graph)
        except Exception as e:  # noqa: BLE001
            return web.json_response({"detail": f"Failed to queue with ComfyUI: {e}"}, status=502)
        prompt_id = result.get("prompt_id")
        p.generation_meta = {
            **(p.generation_meta or {}),
            "prompt_hash": validation["prompt_hash"],
            "run_hash": validation["run_hash"],
        }
        projects.save(p)
        return web.json_response({
            "prompt_id": result.get("prompt_id"),
            "report": report,
            "validation": validation,
            "prompt_repairs_cleared": prompt_changed,
            "reset_session": reset_session,
        })

    @routes.get(UI_PREFIX + "/api/projects/{pid}/status/{prompt_id}")
    async def _status(req):
        prompt_id = req.match_info["prompt_id"]
        proj = _project_or_404(req.match_info["pid"])
        run_fps = float(getattr(proj, "frame_rate", None) or 25.0)
        try:
            hist = await bridge.history(prompt_id)
        except Exception as e:  # noqa: BLE001
            raise web.HTTPBadGateway(reason=f"history unavailable: {e}")
        if prompt_id in hist:
            entry = hist[prompt_id]
            raw_status = entry.get("status") or {}
            media = _media_from_history(entry)
            scene_layout = _scene_layout_from_history(entry, run_fps)
            # Extract ComfyUI execution errors from history so the frontend can show them.
            exec_error: Optional[str] = None
            for _kind, payload in raw_status.get("messages") or []:
                if _kind == "execution_error" and isinstance(payload, dict):
                    node_type = payload.get("node_type", "unknown node")
                    exc = payload.get("exception_message") or payload.get("traceback") or "unknown error"
                    exec_error = f"{node_type}: {exc}"
                    break
            is_error = raw_status.get("status_str") == "error"
            payload = {
                "state": "error" if (is_error and not media) else "completed",
                "media": media,
                "error": exec_error,
                "status": raw_status,
            }
            if scene_layout:
                payload["scene_layout"] = scene_layout
            return web.json_response(payload)
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
        import os
        if req.method == "HEAD":
            _project_or_404(req.match_info["pid"])
            path = _resolve_comfy_media_path(
                req.query.get("filename", ""),
                req.query.get("subfolder", ""),
                req.query.get("type", "output"),
            )
            if not path or not os.path.isfile(path):
                raise web.HTTPNotFound()
            return web.Response()
        try:
            data, ctype = await bridge.fetch_view(
                req.query.get("filename", ""),
                req.query.get("subfolder", ""),
                req.query.get("type", "output"),
            )
        except Exception as e:  # noqa: BLE001
            raise web.HTTPBadGateway(reason=f"could not fetch result: {e}")
        return web.Response(body=data, content_type=ctype.split(";")[0])

    # --- API: final render (async job — ffmpeg can exceed Cloudflare tunnel timeouts) ---
    import asyncio
    import uuid as _uuid

    async def _run_render_job(job_id: str, proj, clips: list) -> None:
        _render_jobs[job_id] = {"state": "running"}
        try:
            result = await asyncio.to_thread(_ffmpeg_stitch_final, proj, clips)
            _render_jobs[job_id] = {"state": "done", **result}
        except RenderJobError as e:
            _render_jobs[job_id] = {"state": "error", "detail": str(e)}
        except Exception as e:  # noqa: BLE001
            _render_jobs[job_id] = {"state": "error", "detail": f"Render failed: {e}"}

    @routes.post(UI_PREFIX + "/api/projects/{pid}/render")
    async def _render_final(req):
        proj = _project_or_404(req.match_info["pid"])
        body = await req.json() if req.can_read_body else {}
        clips = body.get("clips") or []
        if not clips and not _has_graphics_export_content(proj):
            return web.json_response({"detail": "Nothing to render — add overlays, audio, or generated clips."}, status=400)
        job_id = _uuid.uuid4().hex
        _render_jobs[job_id] = {"state": "queued"}
        asyncio.create_task(_run_render_job(job_id, proj, clips))
        return web.json_response({"job_id": job_id})

    @routes.get(UI_PREFIX + "/api/projects/{pid}/render/{job_id}")
    async def _render_job_status(req):
        _project_or_404(req.match_info["pid"])
        job = _render_jobs.get(req.match_info["job_id"])
        if not job:
            return web.json_response({"detail": "Render job not found."}, status=404)
        return web.json_response(job)

    # --- API: preview segment (ffmpeg trim + faststart, same as export per scene) ---
    @routes.get(UI_PREFIX + "/api/projects/{pid}/preview-segment/{scene_id}")
    async def _preview_segment(req):
        import os
        pid = req.match_info["pid"]
        scene_id = req.match_info["scene_id"]
        proj = _project_or_404(pid)
        render_override = _playback_render_from_query(req.query)
        try:
            clip = _scene_playback_clip_spec(proj, scene_id, render_override=render_override)
        except KeyError as e:
            return web.json_response({"detail": str(e)}, status=404)
        key = _preview_segment_cache_key(pid, scene_id, clip)
        cached = _preview_segment_cache.get(key)
        if cached and os.path.isfile(cached):
            return web.FileResponse(cached, headers={"Cache-Control": "private, max-age=3600"})
        try:
            import folder_paths
            tempdir = folder_paths.get_temp_directory()
        except Exception as e:  # noqa: BLE001
            return web.json_response({"detail": f"Temp directory unavailable: {e}"}, status=500)
        import time as _time
        out_path = os.path.join(
            tempdir,
            f"funpack_preview_{pid[:8]}_{scene_id}_{int(_time.time() * 1000)}.mp4",
        )
        try:
            src_path = _resolve_clip_src_path(clip)
            _ffmpeg_trim_clip(src_path, out_path, clip.get("in"), clip.get("dur"))
        except FileNotFoundError as e:
            return web.json_response({"detail": str(e)}, status=400)
        except RuntimeError as e:
            return web.json_response({"detail": str(e)}, status=503)
        _preview_segment_cache[key] = out_path
        return web.FileResponse(out_path, headers={"Cache-Control": "private, max-age=3600"})

    # --- API: export one clip (ffmpeg trim from a chain output using in/dur) ---
    @routes.post(UI_PREFIX + "/api/projects/{pid}/export-clip")
    async def _export_clip(req):
        _project_or_404(req.match_info["pid"])
        body = await req.json() if req.can_read_body else {}
        clip = body.get("clip") or {}
        if not clip.get("filename") and not clip.get("bin_media_ref"):
            return web.json_response({"detail": "Missing clip filename."}, status=400)
        import os
        import time as _time
        try:
            import folder_paths
            tempdir = folder_paths.get_temp_directory()
        except Exception as e:  # noqa: BLE001
            return web.json_response({"detail": f"Output directory unavailable: {e}"}, status=500)

        out_name = f"funpack_export_{int(_time.time())}.mp4"
        out_path = os.path.join(tempdir, out_name)
        try:
            src_path = _resolve_clip_src_path(clip)
            _ffmpeg_trim_clip(src_path, out_path, clip.get("in"), clip.get("dur"))
        except FileNotFoundError as e:
            return web.json_response({"detail": str(e)}, status=400)
        except RuntimeError as e:
            return web.json_response({"detail": str(e)}, status=503)
        return web.json_response({
            "media": {"filename": out_name, "subfolder": "", "type": "temp", "kind": "videos"},
        })

    import asyncio
    import uuid as _uuid_export

    async def _run_export_concat_job(job_id: str, clips: list) -> None:
        _export_jobs[job_id] = {"state": "running"}
        try:
            result = await asyncio.to_thread(_ffmpeg_concat_clips_plain, clips)
            _export_jobs[job_id] = {"state": "done", **result}
        except RenderJobError as e:
            _export_jobs[job_id] = {"state": "error", "detail": str(e)}
        except Exception as e:  # noqa: BLE001
            _export_jobs[job_id] = {"state": "error", "detail": f"Export failed: {e}"}

    @routes.post(UI_PREFIX + "/api/projects/{pid}/export-clips")
    async def _export_clips(req):
        _project_or_404(req.match_info["pid"])
        body = await req.json() if req.can_read_body else {}
        clips = body.get("clips") or []
        if not clips:
            return web.json_response({"detail": "Nothing to export."}, status=400)
        job_id = _uuid_export.uuid4().hex
        _export_jobs[job_id] = {"state": "queued"}
        asyncio.create_task(_run_export_concat_job(job_id, clips))
        return web.json_response({"job_id": job_id})

    @routes.get(UI_PREFIX + "/api/projects/{pid}/export-clips/{job_id}")
    async def _export_clips_status(req):
        _project_or_404(req.match_info["pid"])
        job = _export_jobs.get(req.match_info["job_id"])
        if not job:
            return web.json_response({"detail": "Export job not found."}, status=404)
        return web.json_response(job)

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

    @routes.get(UI_PREFIX + "/api/pipeline-deps")
    async def _pipeline_deps(_req):
        try:
            oi = await bridge.object_info()
        except Exception:
            oi = {}
        mgr = await pipeline_deps.manager_available()
        return web.json_response(pipeline_deps.status_payload(
            oi,
            manager_available=mgr,
            manager_on_disk=pipeline_deps.manager_dir_on_disk(),
        ))

    @routes.post(UI_PREFIX + "/api/pipeline-deps/install")
    async def _pipeline_deps_install(req):
        import asyncio
        body = await req.json() if req.can_read_body else {}
        if body.get("install_manager"):
            if pipeline_deps.manager_dir_on_disk():
                return web.json_response(
                    {"detail": "ComfyUI-Manager is already present on disk. Restart ComfyUI to load it."},
                    status=400,
                )
            job = pipeline_deps.create_manager_install_job()
            asyncio.create_task(pipeline_deps.run_manager_install_job(job["job_id"], _restart_comfy))
            return web.json_response({"job_id": job["job_id"], "total": job["total"], "kind": "manager"})
        if not await pipeline_deps.manager_available():
            return web.json_response(
                {"detail": "ComfyUI-Manager is not installed or not reachable. Install it first."},
                status=503,
            )
        pack_ids = body.get("pack_ids") or body.get("packs") or []
        if not isinstance(pack_ids, list) or not pack_ids:
            return web.json_response({"detail": "pack_ids required."}, status=400)
        job = pipeline_deps.create_install_job([str(x) for x in pack_ids])
        asyncio.create_task(pipeline_deps.run_install_job(job["job_id"], _restart_comfy))
        return web.json_response({"job_id": job["job_id"], "total": job["total"], "kind": "packs"})

    @routes.get(UI_PREFIX + "/api/pipeline-deps/install/{job_id}")
    async def _pipeline_deps_install_status(req):
        job = pipeline_deps.get_install_job(req.match_info["job_id"])
        if not job:
            raise web.HTTPNotFound(reason="Install job not found")
        return web.json_response(job)

    @routes.post(UI_PREFIX + "/api/pipeline-deps/install/{job_id}/cancel")
    async def _pipeline_deps_install_cancel(req):
        import asyncio
        job_id = req.match_info["job_id"]
        if not pipeline_deps.cancel_install_job(job_id):
            job = pipeline_deps.get_install_job(job_id)
            if not job:
                raise web.HTTPNotFound(reason="Install job not found")
            return web.json_response({"detail": "Install cannot be cancelled."}, status=409)
        asyncio.create_task(pipeline_deps._reset_manager_queue())
        return web.json_response({"cancelled": True, "job_id": job_id})

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
            "wiring": pipeline_wiring.wiring_rules_payload(),
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

    @routes.get(UI_PREFIX + "/api/git/status")
    async def _git_status(_req):
        return web.json_response(git_update.status())

    @routes.post(UI_PREFIX + "/api/git/update")
    async def _git_update(req):
        import asyncio
        body = await req.json() if req.can_read_body else {}
        branch = body.get("branch")
        try:
            result = git_update.pull(str(branch).strip() if branch else None)
        except git_update.GitUpdateError as e:
            return web.json_response({"detail": str(e)}, status=400)
        asyncio.get_event_loop().call_later(0.7, _restart_comfy)
        return web.json_response({"restarting": True, **result})

    @routes.post(UI_PREFIX + "/api/git/checkout")
    async def _git_checkout(req):
        import asyncio
        body = await req.json() if req.can_read_body else {}
        branch = str(body.get("branch") or "").strip()
        if not branch:
            return web.json_response({"detail": "Branch name is required."}, status=400)
        try:
            result = git_update.checkout(branch, pull_after=True)
        except git_update.GitUpdateError as e:
            return web.json_response({"detail": str(e)}, status=400)
        asyncio.get_event_loop().call_later(0.7, _restart_comfy)
        return web.json_response({"restarting": True, **result})

    @routes.post(UI_PREFIX + "/api/restart")
    async def _restart(_req):
        import asyncio
        # defer so this 200 flushes to the browser before the process is replaced
        asyncio.get_event_loop().call_later(0.7, _restart_comfy)
        return web.json_response({"restarting": True})

    @routes.post(UI_PREFIX + "/api/workflow/parse")
    async def _workflow_parse(req):
        body = await req.json()
        workflow = body.get("workflow")
        try:
            oi = await bridge.object_info()
        except Exception as e:  # noqa: BLE001
            raise web.HTTPBadGateway(reason=f"object_info unavailable: {e}")
        result = workflow_import.parse_workflow(workflow, oi)
        if result.get("error"):
            return web.json_response(result, status=400)
        return web.json_response(result)

    @routes.post(UI_PREFIX + "/api/projects/{pid}/workflow/apply")
    async def _workflow_apply(req):
        p = _project_or_404(req.match_info["pid"])
        body = await req.json()
        workflow = body.get("workflow")
        bindings = body.get("bindings") or {}
        try:
            oi = await bridge.object_info()
        except Exception as e:  # noqa: BLE001
            raise web.HTTPBadGateway(reason=f"object_info unavailable: {e}")
        try:
            config_data = workflow_import.apply_workflow(workflow, bindings, oi)
        except ValueError as e:
            return web.json_response({"detail": str(e)}, status=400)
        p.models = config_data
        projects.save(p)
        nodes.save_models(config_data)
        return web.json_response(config_data)

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
    @routes.get(UI_PREFIX + "/api/projects/{pid}/pipeline-caps")
    async def _pipeline_caps(req):
        p = _project_or_404(req.match_info["pid"])
        return web.json_response(pipeline_caps.capabilities(p, _project_models(p)))

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
