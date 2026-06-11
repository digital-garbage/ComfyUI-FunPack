"""Shared Ken Burns / clip-effect helpers for preview and ffmpeg render."""
from __future__ import annotations

from typing import Any


def zoom_effect_params(fx: dict[str, Any] | None, nframes: int) -> tuple[float, int, int]:
    """Clamp zoom ratio, start frame, and ramp length for a clip of ``nframes``."""
    fx = fx or {}
    nframes = max(1, int(nframes))
    ratio = float(fx.get("zoom_ratio") if fx.get("zoom_ratio") is not None else 0.15)
    ratio = max(0.01, min(0.5, ratio))
    start = int(fx.get("zoom_start_frame") or 0)
    start = max(0, min(start, max(0, nframes - 1)))
    default_len = min(max(1, nframes // 4), 25)
    length = int(fx.get("zoom_frames") if fx.get("zoom_frames") is not None else default_len)
    length = max(1, min(length, max(1, nframes - start)))
    return ratio, start, length


def zoom_scale_at_frame(zoom: str, fx: dict[str, Any] | None, frame: int, nframes: int) -> float:
    """Return virtual zoom scale at output frame index ``frame`` (0-based)."""
    if zoom not in ("in", "out"):
        return 1.0
    ratio, start, length = zoom_effect_params(fx, nframes)
    on = max(0, min(int(frame), nframes - 1))
    end = 1.0 + ratio
    if on < start:
        return 1.0 if zoom == "in" else end
    if on >= start + length:
        return end if zoom == "in" else 1.0
    t = (on - start) / float(length)
    return 1.0 + ratio * t if zoom == "in" else end - ratio * t


def zoompan_z_expr(zoom: str, fx: dict[str, Any] | None, nframes: int) -> str:
    """ffmpeg zoompan ``z`` expression for a timed Ken Burns ramp."""
    ratio, start, length = zoom_effect_params(fx, nframes)
    end = 1.0 + ratio
    if zoom == "in":
        return (
            f"if(lt(on,{start}),1,"
            f"if(lt(on,{start + length}),1+{ratio:.6f}*(on-{start})/{length},"
            f"{end:.6f}))"
        )
    return (
        f"if(lt(on,{start}),{end:.6f},"
        f"if(lt(on,{start + length}),{end:.6f}*(1-(on-{start})/{length}),"
        f"1))"
    )
