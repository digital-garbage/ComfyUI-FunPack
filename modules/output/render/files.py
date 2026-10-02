"""Where a clip's bytes are, and whether a browser can play them.

A clip is named the way ComfyUI names a file (filename, subfolder, type) or by a
media-library id. Names come out of a saved project, so they are never trusted as
paths: every one is resolved under the output or temp folder and refused if it
leaves it.

MP4 detail that cost real debugging in v4: ComfyUI's video saver writes the `moov`
index AFTER the picture data. A browser can play such a file from the start but a
deep seek into it is undecodable. So a file is served only when its `moov` is in
front (or has been remuxed to put it there), and a file with no `moov` at all is
still being written -- never served, because the same URL would later return
different bytes under a browser that is mid-stream.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from ..._core import media

ISOBMFF = (".mp4", ".m4v", ".mov")          # containers where moov/mdat order applies

AUDIO_RATE = 48000                          # every exported part is made at this rate, so parts can be joined

FFMPEG_MISSING = "ffmpeg was not found on PATH: install it to preview, export or render clips."


class ClipError(Exception):
    """A clip cannot be used, with the sentence the person should read."""


def ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise ClipError(FFMPEG_MISSING)
    return path


def comfy_dir(kind: str) -> str | None:
    """ComfyUI's output or temp folder, or None when ComfyUI is not here to ask."""
    try:
        import folder_paths
        return folder_paths.get_output_directory() if kind == "output" else folder_paths.get_temp_directory()
    except Exception:                         # noqa: BLE001 -- tests, headless
        return None


def comfy_path(filename, subfolder="", kind="output") -> str | None:
    """The file ComfyUI calls (filename, subfolder, kind), or None if the name does not
    resolve to a place under that folder."""
    if not isinstance(filename, str) or not filename:
        return None
    base = comfy_dir("output" if kind != "temp" else "temp")
    if not base:
        return None
    base = os.path.realpath(base)
    full = os.path.realpath(os.path.join(base, subfolder if isinstance(subfolder, str) else "",
                                         os.path.basename(filename)))
    return full if os.path.commonpath((full, base)) == base else None


def clip_path(clip: dict) -> str:
    """The file a clip spec points at: a media-library id, or a ComfyUI file."""
    ref = clip.get("bin_media_ref")
    if ref:
        found = media.path_for(ref) if media.is_id(ref) else None
        if found is None:
            raise ClipError("A media file in this clip is gone from the library: it may have been deleted.")
        return str(found)
    path = comfy_path(clip.get("filename"), clip.get("subfolder") or "", clip.get("type") or "output")
    if not path or not os.path.isfile(path):
        raise ClipError("A clip's file is not on disk: generate it again, then try this.")
    return path


def is_isobmff(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in ISOBMFF


def moov_position(path: str) -> str:
    """"front" (playable and seekable), "end" (complete, but undecodable on a deep seek: remux
    it), or "none" (no index anywhere: still being written, or a save was aborted)."""
    try:
        size = os.stat(path).st_size
        seen_mdat = False
        offset = 0
        with open(path, "rb") as f:
            while offset + 8 <= size:
                f.seek(offset)
                header = f.read(8)
                if len(header) < 8:
                    break
                box = int.from_bytes(header[:4], "big")
                kind = header[4:8]
                if kind == b"moov":
                    return "end" if seen_mdat else "front"
                if kind == b"mdat":
                    seen_mdat = True
                if box == 1:                  # 64-bit size follows
                    ext = f.read(8)
                    if len(ext) < 8:
                        break
                    box = int.from_bytes(ext, "big")
                elif box == 0:                # runs to EOF: an mdat still being streamed out
                    return "none"
                if box < 8:
                    break
                offset += box
    except OSError:
        pass
    return "none"


def signature(path: str) -> tuple:
    """(mtime_ns, size): the same pair means the same bytes, for cache keys."""
    st = os.stat(path)
    return (st.st_mtime_ns, st.st_size)


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise ClipError((proc.stderr or "ffmpeg failed")[-1000:])


def remux_faststart(src: str, out: str) -> None:
    """Copy the streams (no re-encode) with the index moved to the front."""
    run([ffmpeg(), "-y", "-i", src, "-c", "copy", "-movflags", "+faststart", out])


def trim(src: str, out: str, start=None, dur=None, *, fast=False, reverse=False) -> None:
    """Cut [start, start+dur) out of `src` into a seekable h264/aac mp4.

    `fast` is for previews, where the latency of a scrub across a clip boundary matters
    more than a few percent of bitrate."""
    cmd = [ffmpeg(), "-y"]
    if start is not None and float(start) > 0:
        cmd += ["-ss", f"{float(start):.3f}"]
    if dur is not None:
        cmd += ["-t", f"{float(dur):.3f}"]
    cmd += ["-i", src]
    if reverse:
        cmd += ["-vf", "reverse", "-af", "areverse"]     # after the trim: only the clip is buffered
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if fast:
        cmd += ["-preset", "veryfast"]
    cmd += ["-c:a", "aac", "-b:a", "192k", "-ar", str(AUDIO_RATE), "-ac", "2", "-movflags", "+faststart", out]
    run(cmd)


def has_audio(path: str) -> bool:
    """True when the file has a sound stream. ffprobe if there is one, else a probe run of ffmpeg."""
    probe = shutil.which("ffprobe")
    if probe:
        proc = subprocess.run([probe, "-v", "error", "-select_streams", "a", "-show_entries",
                               "stream=index", "-of", "csv=p=0", path], capture_output=True, text=True)
        return proc.returncode == 0 and bool(proc.stdout.strip())
    proc = subprocess.run([ffmpeg(), "-i", path], capture_output=True, text=True)
    return "Audio:" in (proc.stderr or "")


def duration(path: str) -> float | None:
    """Seconds of picture in `path`, or None when it has none (a window past the end of the source
    trims to a file with no streams at all) or ffprobe is not here to ask."""
    probe = shutil.which("ffprobe")
    if not probe:
        return None
    proc = subprocess.run([probe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                           "stream=duration:format=duration", "-of", "default=nw=1:nk=1", path],
                          capture_output=True, text=True)
    for line in proc.stdout.split():
        try:
            return float(line)
        except ValueError:
            continue
    return 0.0 if proc.returncode == 0 else None


def add_silence(src: str, out: str) -> None:
    """Copy `src` with a silent stereo track added, so it can be joined to clips that have sound."""
    run([ffmpeg(), "-y", "-i", src, "-f", "lavfi", "-i", f"anullsrc=r={AUDIO_RATE}:cl=stereo",
         "-shortest", "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
         "-movflags", "+faststart", out])


def temp_file(name: str) -> str:
    base = comfy_dir("temp")
    if not base:
        raise ClipError("ComfyUI's temp folder is not available here, so nothing can be written for playback.")
    Path(base).mkdir(parents=True, exist_ok=True)
    return os.path.join(base, name)
