"""The final cut: trimmed clips joined with their seams, mixed with extra audio and
overlays, into one file.

`build_filter` is pure: clips in, an ffmpeg `filter_complex` out, so what a render does
can be read and tested without running ffmpeg. `render` and `concat` run it.

Video: every clip is normalised to one canvas (xfade and concat need identical size, fps
and aspect), given its effects, then folded left to right with `xfade` (overlap) or
`concat` (hard cut), with black gaps where the editor left one. Audio: each clip's own
sound (a clip with none contributes silence, so one silent clip cannot sink the render)
follows the same fold, then extra tracks are delayed to their start and mixed in.
"""

from __future__ import annotations

import os
import time

from ..._core import media
from . import effects, files, overlays

#: Seam name (the editor's `video_transition`) -> ffmpeg xfade transition.
XFADE = {"crossfade": "fade", "fadeblack": "fadeblack", "wipeleft": "wipeleft",
         "wiperight": "wiperight", "dissolve": "dissolve"}

STEREO = "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo"
RATE = 48000


class RenderError(Exception):
    """A render that cannot be made, in words for the person who pressed Render."""


def _f(value, default=0.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return default if v != v or v in (float("inf"), float("-inf")) else v


def build_filter(clips: list[dict], tracks: list[dict] | None = None, *, keep_original: bool = True,
                 base_input: int = 0, blank: dict | None = None) -> tuple[str, bool]:
    """-> (filter_complex, has_audio). Input i is clip i; extra audio track j is input
    `base_input + j`. With `blank`, input 0 is a plain colour source and there are no clips.

    Raises RenderError when a clip cannot be made as asked (reverse too long)."""
    tracks = tracks or []
    parts: list[str] = []
    audio: str | None = None
    total = 0.01

    if blank:
        total = max(0.01, _f(blank.get("dur"), 0.01))
        parts.append("[0:v]format=yuv420p,setsar=1[vbase]")
    else:
        if not clips:
            raise RenderError("There is nothing to render: no clips, no overlays and no audio.")
        cw = int(_f(clips[0].get("w")) or 768)
        ch = int(_f(clips[0].get("h")) or 768)
        fps = _f(clips[0].get("fps")) or 25.0
        for i, c in enumerate(clips):
            dur = _f(c.get("dur"))
            try:
                vf = effects.clip_filters(c.get("fx") or {}, cw, ch, fps, dur)
            except ValueError as exc:
                raise RenderError(f"Clip {i + 1}: {exc}") from exc
            parts.append(f"[{i}:v:0]{','.join(vf)}[v{i}]")
            if keep_original:
                parts.append(_clip_audio(i, c, dur))
        acc_v, acc_a = "[v0]", "[a0]"
        acc_dur = _f(clips[0].get("dur"))
        for i in range(1, len(clips)):
            prev, dur_i = clips[i - 1], _f(clips[i].get("dur"))
            gap = _f(prev.get("gap_after"))
            if gap > 0.001:
                parts.append(f"color=c=black:s={cw}x{ch}:r={fps:g}:d={gap:.3f},format=yuv420p,setsar=1[vgap{i}]")
                parts.append(f"{acc_v}[vgap{i}]concat=n=2:v=1:a=0[vcgap{i}]")
                acc_v, acc_dur = f"[vcgap{i}]", acc_dur + gap
                if keep_original:
                    parts.append(f"anullsrc=r={RATE}:cl=stereo:d={gap:.3f},{STEREO}[agap{i}]")
                    parts.append(f"{acc_a}[agap{i}]concat=n=2:v=0:a=1[acgap{i}]")
                    acc_a = f"[acgap{i}]"
            name = str(prev.get("transition") or "").strip()
            td = _f(prev.get("tdur"))
            # A seam longer than either side cannot overlap: it is a hard cut, said nowhere
            # because xfade itself refuses it.
            if name in XFADE and td > 0 and acc_dur > td and dur_i > td:
                parts.append(f"{acc_v}[v{i}]xfade=transition={XFADE[name]}:duration={td:.3f}:"
                             f"offset={max(0.0, acc_dur - td):.3f}[vx{i}]")
                acc_v, acc_dur = f"[vx{i}]", acc_dur + dur_i - td
                if keep_original:
                    parts.append(f"{acc_a}[a{i}]acrossfade=d={td:.3f}[ax{i}]")
                    acc_a = f"[ax{i}]"
            else:
                parts.append(f"{acc_v}[v{i}]concat=n=2:v=1:a=0[vc{i}]")
                acc_v, acc_dur = f"[vc{i}]", acc_dur + dur_i
                if keep_original:
                    parts.append(f"{acc_a}[a{i}]concat=n=2:v=0:a=1[ac{i}]")
                    acc_a = f"[ac{i}]"
        parts.append(f"{acc_v}null[vbase]")
        total = max(0.01, acc_dur)
        audio = acc_a if keep_original else None

    mix = [audio] if audio else []
    for j, t in enumerate(tracks):
        ms = int(max(0.0, _f(t.get("start_sec"))) * 1000)
        chain = (f"[{base_input + j}:a:0]{STEREO},volume={max(0.0, _f(t.get('volume'), 1.0)):.3f}"
                 + (f",adelay={ms}|{ms}" if ms > 0 else ""))
        parts.append(f"{chain}[at{j}]")
        mix.append(f"[at{j}]")
    if not mix:
        return ";".join(parts), False
    if len(mix) == 1:
        parts.append(f"{mix[0]}atrim=0:{total:.3f},asetpts=PTS-STARTPTS[aout]")
    else:
        parts.append("".join(mix) + f"amix=inputs={len(mix)}:normalize=0:duration=longest[amx]")
        parts.append(f"[amx]atrim=0:{total:.3f},asetpts=PTS-STARTPTS[aout]")
    return ";".join(parts), True


def _clip_audio(i: int, clip: dict, dur: float) -> str:
    """The filter that makes clip i's sound: its own (volume, reversed with the picture), or silence."""
    if not clip.get("has_audio", True):
        return f"anullsrc=r={RATE}:cl=stereo:d={max(dur, 0.04):.3f},{STEREO}[a{i}]"
    fx = clip.get("fx") or {}
    af = STEREO
    if fx.get("reverse"):
        af += ",areverse"                  # a reversed picture must not play its soundtrack forwards
    vol = _f(clip.get("volume"), 1.0)
    if abs(vol - 1.0) > 1e-3:
        af += f",volume={max(0.0, vol):.3f}"
    return f"[{i}:a:0]{af}[a{i}]"


def _audio_tracks(project, clips_by_scene: dict, resolve) -> list[dict]:
    """The project's extra audio lanes as {path, start_sec, volume[, source_in, source_dur]}.

    A "separated" lane is a clip's own sound pulled onto a lane: it plays the audio pinned at
    separation time (so a later regeneration of the picture keeps the old sound), else the
    clip's file. A lane whose file cannot be found is left out."""
    out = []
    for t in project.audio_tracks:
        separated = t.get("kind") == "separated" or (t.get("scene_id") and not t.get("media_ref")
                                                    and t.get("kind") != "overlay")
        start, vol = _f(t.get("start_sec")), _f(t.get("volume"), 1.0)
        if separated:
            clip = clips_by_scene.get(t.get("scene_id")) or {}
            path = None
            pinned = t.get("pinned_media") if isinstance(t.get("pinned_media"), dict) else None
            if pinned and pinned.get("filename"):
                path = files.comfy_path(pinned["filename"], pinned.get("subfolder") or "", pinned.get("type") or "output")
            ref = t.get("pinned_bin_ref")
            if (not path or not os.path.isfile(path)) and isinstance(ref, str) and media.is_id(ref):
                found = media.path_for(ref)
                path = str(found) if found else None
            if (not path or not os.path.isfile(path)) and clip:
                path = resolve(clip)
            if not path or not os.path.isfile(path):
                continue
            src_in = t.get("pinned_in_sec", t.get("source_in_sec"))
            src_dur = t.get("pinned_dur", t.get("source_dur"))
            out.append({"path": path, "start_sec": start, "volume": vol,
                        "source_in": _f(src_in if src_in is not None else clip.get("in")),
                        "source_dur": _f(src_dur if src_dur is not None else clip.get("dur"))})
            continue
        ref = t.get("media_ref")
        found = media.path_for(ref) if isinstance(ref, str) and media.is_id(ref) else None
        if found is None:
            continue
        entry = {"path": str(found), "start_sec": start, "volume": vol}
        if t.get("source_in_sec") is not None and t.get("source_dur") is not None:
            entry.update(source_in=_f(t["source_in_sec"]), source_dur=_f(t["source_dur"]))
        out.append(entry)
    return out


def _track_end(t: dict) -> float:
    start = _f(t.get("start_sec"))
    for key in ("source_dur", "pinned_dur"):
        if t.get(key) is not None:
            return start + _f(t[key])
    return start + 1.0


def graphics_duration(project) -> float:
    """How long the audio lanes and overlays run: the length of a render with no clips."""
    end = 0.0
    for ov in project.overlay_tracks:
        end = max(end, _f(ov.get("start_sec")) + _f(ov.get("duration_sec")))
    for t in project.audio_tracks:
        end = max(end, _track_end(t))
    return max(end, 0.01)


def has_graphics(project) -> bool:
    return bool(project.audio_tracks) or any(_f(ov.get("duration_sec")) > 0 for ov in project.overlay_tracks)


def _clip_spec_ok(c) -> bool:
    return isinstance(c, dict) and (c.get("bin_media_ref") or c.get("filename"))


def render(project, clips: list[dict]) -> dict:
    """Run the final render: -> {"media": {...temp file...}, "clips": n}. Blocking: call it off the event loop."""
    clips = [c for c in clips if _clip_spec_ok(c)] if isinstance(clips, list) else []
    blank = not clips
    if blank and not has_graphics(project):
        raise RenderError("Nothing to render: add overlays, audio, or generated clips.")
    cw, ch, fps = int(project.width or 768), int(project.height or 512), float(project.frame_rate or 25)
    paths: list[str] = []
    try:
        paths = [files.clip_path(c) for c in clips]
    except files.ClipError as exc:
        raise RenderError(str(exc)) from exc
    if clips:
        cw, ch = int(_f(clips[0].get("w")) or cw), int(_f(clips[0].get("h")) or ch)
        fps = _f(clips[0].get("fps")) or fps
        for c, p in zip(clips, paths):
            c["has_audio"] = files.has_audio(p)
        blank_canvas = None
    else:
        blank_canvas = {"w": cw, "h": ch, "fps": fps, "dur": graphics_duration(project)}
    try:
        ffmpeg = files.ffmpeg()
        tempdir = os.path.dirname(files.temp_file("x"))
    except files.ClipError as exc:
        raise RenderError(str(exc)) from exc

    by_scene = {c["scene_id"]: c for c in clips if c.get("scene_id")}
    tracks = _audio_tracks(project, by_scene, files.clip_path) if clips or project.audio_tracks else []
    keep = bool(project.keep_original_audio) and not blank

    cmd = [ffmpeg, "-y"]
    if blank:
        cmd += ["-f", "lavfi", "-i", f"color=c=black:s={cw}x{ch}:r={fps:g}:d={blank_canvas['dur']:.3f}"]
    else:
        for c, path in zip(clips, paths):
            if c.get("in") is not None:
                cmd += ["-ss", f"{_f(c['in']):.3f}"]
            if c.get("dur") is not None:
                cmd += ["-t", f"{_f(c['dur']):.3f}"]
            cmd += ["-i", path]
    base = 1 if blank else len(clips)
    for t in tracks:
        if t.get("source_in") is not None:
            cmd += ["-ss", f"{t['source_in']:.3f}", "-t", f"{t['source_dur']:.3f}"]
        cmd += ["-i", t["path"]]

    def image_path(ref):
        found = media.path_for(ref) if isinstance(ref, str) and media.is_id(ref) else None
        return str(found) if found else None

    drawn, pictures = overlays.prepare(project.overlay_tracks, project.overlay_lanes, tempdir=tempdir,
                                       image_path=image_path)
    for path in pictures:
        cmd += ["-i", path]
    graph, has_audio = build_filter(clips, tracks, keep_original=keep, base_input=base, blank=blank_canvas)
    lines = graph.split(";") if graph else []
    labels = [f"[{base + len(tracks) + i}:v:0]" for i in range(len(pictures))]
    ov_lines, final = overlays.composite("[vbase]", drawn, canvas_w=cw, image_labels=labels)
    lines += ov_lines
    lines.append(f"{final}null[vout]")
    cmd += ["-filter_complex", ";".join(lines), "-map", "[vout]"]
    if has_audio:
        cmd += ["-map", "[aout]", "-c:a", "aac", "-b:a", "192k"]
    name = f"funpack_final_{int(time.time())}.mp4"
    out = os.path.join(tempdir, name)
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
    try:
        files.run(cmd)
    except files.ClipError as exc:
        raise RenderError(f"ffmpeg could not make the render: {exc}") from exc
    return {"media": {"filename": name, "subfolder": "", "type": "temp", "kind": "videos"},
            "clips": len(clips) or 1}


def concat(clips: list[dict]) -> dict:
    """Hard-cut the trimmed clips together with no effects, seams or extra audio (the selection
    export): -> {"media": {...}, "clips": n}. Blocking."""
    clips = [c for c in clips if _clip_spec_ok(c)] if isinstance(clips, list) else []
    if not clips:
        raise RenderError("There is nothing to export.")
    stamp = int(time.time())
    try:
        parts = []
        for i, c in enumerate(clips):
            out = files.temp_file(f"funpack_seg_{stamp}_{i}.mp4")
            files.trim(files.clip_path(c), out, c.get("in"), c.get("dur"))
            parts.append(out)
        name = f"funpack_export_{stamp}.mp4"
        out = files.temp_file(name)
        if len(parts) == 1:
            os.replace(parts[0], out)
        else:
            listing = files.temp_file(f"funpack_concat_{stamp}.txt")
            with open(listing, "w", encoding="utf-8") as fh:
                for p in parts:
                    fh.write("file '" + p.replace("'", "'\\''") + "'\n")
            try:
                files.run([files.ffmpeg(), "-y", "-f", "concat", "-safe", "0", "-i", listing,
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                            "-movflags", "+faststart", out])
            finally:
                for p in parts + [listing]:
                    try:
                        os.remove(p)
                    except OSError:
                        pass
    except files.ClipError as exc:
        raise RenderError(str(exc)) from exc
    return {"media": {"filename": name, "subfolder": "", "type": "temp", "kind": "videos"}, "clips": len(clips)}
