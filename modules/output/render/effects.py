"""Per-clip pixel effects, as ffmpeg filters.

The player draws these live with CSS (player.js `_applyFx`); the render must show
the same thing, so the geometry is applied in the same order the preview applies it:
flips, then crop, then fit into the canvas. Pure functions of one clip's `fx` dict.
"""

from __future__ import annotations

from typing import Any

# ffmpeg's `reverse` holds every frame of its input in memory (there is no streaming
# form). Inputs are trimmed to the clip first, so the bound is the clip's own length,
# but a long imported video would try to buffer gigabytes and take ComfyUI down with it.
# At 768x768 yuv420p a frame is ~0.9 MB: this cap is roughly 1 GB.
REVERSE_MAX_FRAMES = 1200


def _f(value, default=0.0) -> float:
    """`value` as a finite float; an effect field is whatever the editor (or a hand-edited
    file) stored, and a bad one must read as 'not set', not stop a render."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return default if v != v or v in (float("inf"), float("-inf")) else v


def zoom_params(fx: dict | None, nframes: int) -> tuple[float, int, int]:
    """(ratio, start frame, ramp length in frames), clamped to what fits in `nframes`."""
    fx = fx or {}
    nframes = max(1, int(nframes))
    ratio = max(0.01, min(0.5, _f(fx.get("zoom_ratio"), 0.15)))
    start = max(0, min(int(_f(fx.get("zoom_start_frame"))), nframes - 1))
    length = int(_f(fx.get("zoom_frames"), float(min(max(1, nframes // 4), 25))))
    return ratio, start, max(1, min(length, max(1, nframes - start)))


def zoom_scale_at(zoom: str, fx: dict | None, frame: int, nframes: int) -> float:
    """The virtual zoom scale at output frame `frame` (0-based)."""
    if zoom not in ("in", "out"):
        return 1.0
    ratio, start, length = zoom_params(fx, nframes)
    on = max(0, min(int(frame), nframes - 1))
    end = 1.0 + ratio
    if on < start:
        return 1.0 if zoom == "in" else end
    if on >= start + length:
        return end if zoom == "in" else 1.0
    t = (on - start) / float(length)
    return 1.0 + ratio * t if zoom == "in" else end - ratio * t


def zoompan_z(zoom: str, fx: dict | None, nframes: int) -> str:
    """The `z` expression of ffmpeg's zoompan for a timed Ken Burns ramp."""
    ratio, start, length = zoom_params(fx, nframes)
    end = 1.0 + ratio
    if zoom == "in":
        return (f"if(lt(on,{start}),1,"
                f"if(lt(on,{start + length}),1+{ratio:.6f}*(on-{start})/{length},{end:.6f}))")
    return (f"if(lt(on,{start}),{end:.6f},"
            f"if(lt(on,{start + length}),{end:.6f}*(1-(on-{start})/{length}),1))")


def crop_inset(fx: dict | None) -> float:
    """The fraction trimmed off EACH edge, 0..0.4. Junk reads as no crop."""
    v = _f((fx or {}).get("crop_inset"))
    return 0.0 if v <= 0 else min(0.4, v)


def geometry(fx: dict | None, cw: int, ch: int) -> list[str]:
    """Filters that place one clip into the `cw`x`ch` canvas: flips, crop, then fit.

    The crop is of the SOURCE, so it runs before the fit; flips commute with both.
    """
    fx = fx or {}
    out: list[str] = []
    if fx.get("flip_h"):
        out.append("hflip")
    if fx.get("flip_v"):
        out.append("vflip")
    inset = crop_inset(fx)
    if inset > 0:
        keep = 1.0 - 2.0 * inset
        # Even sizes: yuv420p subsamples chroma 2x2 and later filters reject an odd frame.
        out.append(f"crop=trunc(iw*{keep:.6f}/2)*2:trunc(ih*{keep:.6f}/2)*2")
    if fx.get("fit") == "fill":
        out += [f"scale={cw}:{ch}:force_original_aspect_ratio=increase", f"crop={cw}:{ch}"]
    else:
        out += [f"scale={cw}:{ch}:force_original_aspect_ratio=decrease", f"pad={cw}:{ch}:-1:-1:color=black"]
    return out


def reverse_frames(dur_sec: Any, fps: Any) -> int:
    try:
        return max(0, int(round(float(dur_sec or 0) * float(fps or 0))))
    except (TypeError, ValueError):
        return 0


def reverse_refusal(dur_sec: Any, fps: Any) -> str | None:
    """Why reverse cannot run on this clip, or None when it can. Said, never skipped: a
    clip rendered forwards when reverse was asked for is a wrong result that looks right."""
    frames = reverse_frames(dur_sec, fps)
    if frames <= REVERSE_MAX_FRAMES:
        return None
    secs = REVERSE_MAX_FRAMES / float(fps or 24)
    return (f"Reverse needs every frame in memory at once, and this clip is {frames} frames "
            f"(limit {REVERSE_MAX_FRAMES}, about {secs:.0f}s at {float(fps or 24):g} fps). "
            f"Split the clip and reverse the parts.")


def clip_filters(fx: dict | None, cw: int, ch: int, fps: float, dur: float) -> list[str]:
    """Everything one clip goes through before it is joined to the next, in order:
    reverse, geometry, fixed fps, zoom, blur, fades, then a common pixel format.
    Raises ValueError when reverse is asked for and cannot be done."""
    fx = fx or {}
    vf: list[str] = []
    if fx.get("reverse"):
        refusal = reverse_refusal(dur, fps)
        if refusal:
            raise ValueError(refusal)
        vf.append("reverse")                 # before the geometry: the SOURCE is what turns around
    vf += geometry(fx, cw, ch) + ["setsar=1", f"fps={fps:g}"]
    zoom = fx.get("zoom")
    if zoom in ("in", "out") and dur > 0:
        z = zoompan_z(zoom, fx, max(1, round(dur * fps)))
        vf.append(f"zoompan=z='{z}':d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={cw}x{ch}:fps={fps:g}")
        vf.append("setsar=1")
    blur = _f(fx.get("blur"))
    if blur > 0:
        vf.append(f"gblur=sigma={blur * 20:.2f}")
    fade_in = _f(fx.get("fade_in"))
    if fade_in > 0:
        vf.append(f"fade=t=in:st=0:d={fade_in:.3f}")
    fade_out = _f(fx.get("fade_out"))
    if fade_out > 0 and dur > 0:
        vf.append(f"fade=t=out:st={max(0.0, dur - fade_out):.3f}:d={fade_out:.3f}")
    vf += ["format=yuv420p", "setsar=1"]
    return vf
