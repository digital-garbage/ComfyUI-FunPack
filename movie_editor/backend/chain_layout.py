"""Map stitched chain output pixels to per-scene playback offsets (inSec).

The chain sampler blends overlap in latent space; decoded pixel scene starts do not
advance by (num_frames_per_scene - frame_overlap). Use boundary pixel positions from
the sampler (or the same latent math) instead of subtracting overlap on the client.
"""
from __future__ import annotations

import json
from typing import Any, Optional


def latent_to_pixel_frame(latent_frame: int, time_scale: int) -> int:
    latent_frame = max(0, int(latent_frame))
    time_scale = max(1, int(time_scale))
    if time_scale > 1:
        return int((latent_frame - 1) * time_scale + 1) if latent_frame > 0 else 0
    return latent_frame


def expected_latent_frames(pixel_frames: int, time_scale: int) -> int:
    return ((max(1, int(pixel_frames)) - 1) // max(1, int(time_scale))) + 1


def latent_overlap_frames(frame_overlap: int, time_scale: int) -> int:
    if int(frame_overlap) <= 0:
        return 0
    return expected_latent_frames(int(frame_overlap) + 1, time_scale) - 1


def _boundary_start_pixels(boundaries: list[dict], time_scale: int) -> dict[int, int]:
    out: dict[int, int] = {}
    for entry in boundaries or []:
        between = entry.get("between") or entry.get("between_scenes") or []
        if not between:
            continue
        try:
            scene_idx = int(between[0])
        except (TypeError, ValueError):
            continue
        if "pixel_frame" in entry:
            px = int(entry["pixel_frame"])
        elif "boundary_latent" in entry:
            px = latent_to_pixel_frame(int(entry["boundary_latent"]), time_scale)
        else:
            continue
        out[scene_idx] = max(0, px)
    return out


def scene_start_pixels(
    scene_count: int,
    *,
    num_frames_per_scene: int,
    frame_overlap: int,
    time_scale: int,
    boundaries: Optional[list[dict]] = None,
) -> list[int]:
    """Pixel-frame start index for each scene (0-based), length scene_count."""
    if scene_count <= 0:
        return []
    starts = [0]
    by_boundary = _boundary_start_pixels(boundaries or [], time_scale)
    for i in range(1, scene_count):
        if i in by_boundary:
            starts.append(by_boundary[i])
            continue
        latent_frames = expected_latent_frames(num_frames_per_scene, time_scale)
        latent_overlap = latent_overlap_frames(frame_overlap, time_scale)
        cum = latent_frames + (i - 1) * max(1, latent_frames - latent_overlap)
        starts.append(latent_to_pixel_frame(cum, time_scale))
    return starts


def scene_playback_layout(
    scene_count: int,
    *,
    fps: float,
    num_frames_per_scene: int,
    frame_overlap: int,
    time_scale: int,
    boundaries: Optional[list[dict]] = None,
) -> list[dict[str, Any]]:
    """Per chain scene: {scene_index, start_frame, in_sec}."""
    fps = max(1e-6, float(fps or 25.0))
    starts = scene_start_pixels(
        scene_count,
        num_frames_per_scene=num_frames_per_scene,
        frame_overlap=frame_overlap,
        time_scale=time_scale,
        boundaries=boundaries,
    )
    return [
        {"scene_index": i, "start_frame": starts[i], "in_sec": round(starts[i] / fps, 6)}
        for i in range(len(starts))
    ]


def layout_from_boundaries_json(data: dict, fps: float) -> Optional[list[dict[str, Any]]]:
    """Parse scene_boundaries JSON from the chain sampler STRING output."""
    if not isinstance(data, dict):
        return None
    fps = max(1e-6, float(fps or 25.0))
    playback = data.get("scene_playback")
    if isinstance(playback, list) and playback:
        out = []
        for i, entry in enumerate(playback):
            if not isinstance(entry, dict):
                continue
            if "start_frame" not in entry:
                break
            out.append({
                "scene_index": int(entry.get("scene_index", i)),
                "start_frame": int(entry["start_frame"]),
                "in_sec": round(int(entry["start_frame"]) / fps, 6),
            })
        if out:
            return out
    scene_count = int(data.get("scene_count") or 0)
    if scene_count <= 0:
        return None
    time_scale = int(data.get("time_scale") or 1)
    pixel_frames = int(data.get("frames_per_scene_pixel") or data.get("num_frames_per_scene") or 0)
    pixel_overlap = int(data.get("pixel_overlap") or 0)
    boundaries = []
    for entry in data.get("boundaries") or []:
        if not isinstance(entry, dict):
            continue
        boundaries.append({
            "between": entry.get("between_scenes") or entry.get("between"),
            "pixel_frame": entry.get("pixel_frame"),
            "boundary_latent": entry.get("latent_frame") or entry.get("boundary_latent"),
        })
    if not pixel_frames:
        latent_frames = int(data.get("frames_per_scene") or 0)
        if latent_frames > 0:
            pixel_frames = latent_to_pixel_frame(latent_frames, time_scale)
    if pixel_frames <= 0:
        ranges = data.get("scene_pixel_ranges") or []
        if ranges and isinstance(ranges[0], dict):
            pixel_frames = int(ranges[0].get("end", 0)) - int(ranges[0].get("start", 0)) + 1
    if pixel_frames <= 0:
        return None
    return scene_playback_layout(
        scene_count,
        fps=fps,
        num_frames_per_scene=pixel_frames,
        frame_overlap=pixel_overlap,
        time_scale=time_scale,
        boundaries=boundaries,
    )


def layout_from_history_entry(entry: dict, fps: float) -> Optional[list[dict[str, Any]]]:
    for node_out in (entry.get("outputs") or {}).values():
        for text in node_out.get("text") or []:
            if not isinstance(text, str):
                continue
            stripped = text.strip()
            if not stripped.startswith("{"):
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            layout = layout_from_boundaries_json(data, fps)
            if layout:
                return layout
    return None
