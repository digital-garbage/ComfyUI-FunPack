"""Timeline graphics overlays (images + text) for preview/export ffmpeg compositing."""
from __future__ import annotations

import os
import re
from typing import Any


# Common system font paths (first match wins at export time).
_FONT_PATHS: dict[str, list[str]] = {
    "arial": [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ],
    "helvetica": [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ],
    "georgia": [
        "/System/Library/Fonts/Supplemental/Georgia.ttf",
        "/Library/Fonts/Georgia.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Georgia.ttf",
    ],
    "times": [
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/Library/Fonts/Times New Roman.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    ],
    "courier": [
        "/System/Library/Fonts/Supplemental/Courier New.ttf",
        "/Library/Fonts/Courier New.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Courier_New.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ],
    "verdana": [
        "/System/Library/Fonts/Supplemental/Verdana.ttf",
        "/Library/Fonts/Verdana.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Verdana.ttf",
    ],
    "impact": [
        "/System/Library/Fonts/Supplemental/Impact.ttf",
        "/Library/Fonts/Impact.ttf",
    ],
}


def _escape_drawtext(text: str) -> str:
    s = str(text or "")
    s = s.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'").replace("%", "\\%")
    s = re.sub(r"[\r\n]+", " ", s)
    return s


def _resolve_fontfile(font_family: str | None) -> str | None:
    key = (font_family or "system-ui").strip().lower()
    if key in ("", "system-ui", "default"):
        return None
    for path in _FONT_PATHS.get(key, []):
        if os.path.isfile(path):
            return path
    return None


def _image_target_size(ov: dict, cw: int, ch: int) -> tuple[int, int | None]:
    """Return (width_px, height_px or None for keep-aspect)."""
    wpx = ov.get("width_px")
    if wpx is not None:
        tw = max(8, int(wpx))
        if ov.get("keep_aspect", True) is not False:
            return tw, None
        th = max(8, int(ov.get("height_px") or tw))
        return tw, th
    scale = max(0.05, min(1.5, float(ov.get("scale") if ov.get("scale") is not None else 0.35)))
    return max(8, int(scale * cw)), None


def _apply_flip(label: str, flip_h: bool, flip_v: bool, seq: int) -> tuple[list[str], str]:
    parts: list[str] = []
    cur = label
    if flip_h:
        nxt = f"[ovfh{seq}]"
        parts.append(f"{cur}hflip{nxt}")
        cur = nxt
    if flip_v:
        nxt = f"[ovfv{seq}]"
        parts.append(f"{cur}vflip{nxt}")
        cur = nxt
    return parts, cur


def sort_overlays_for_composite(
    overlays: list[dict],
    lanes: list[dict] | None = None,
) -> list[dict]:
    """Bottom lanes first, then start time within a lane."""
    lane_list = list(lanes or [])
    if not lane_list:
        return list(overlays or [])
    order = {str(l.get("id") or ""): i for i, l in enumerate(lane_list)}
    fallback = str(lane_list[0].get("id") or "")

    def key(ov: dict) -> tuple[int, float]:
        lid = str(ov.get("lane_id") or fallback)
        return (order.get(lid, 0), float(ov.get("start_sec") or 0))

    return sorted(list(overlays or []), key=key)


def build_overlay_video_filter(
    base_label: str,
    overlays: list[dict],
    *,
    canvas_w: int,
    canvas_h: int,
    image_input_labels: list[str],
) -> tuple[list[str], str]:
    """Compose ``overlays`` onto ``base_label``. Returns (filter lines, final label).

    ``image_input_labels`` lists ffmpeg stream labels for image overlay inputs,
    in the same order as image entries in ``overlays``.
    """
    parts: list[str] = []
    cur = base_label
    img_i = 0
    cw = max(1, int(canvas_w))
    ch = max(1, int(canvas_h))
    seq = 0

    for ov in overlays or []:
        kind = ov.get("kind") or "image"
        start = float(ov.get("start_sec") or 0)
        dur = float(ov.get("duration_sec") or 0)
        if dur <= 0:
            continue
        end = start + dur
        enable = f"enable='between(t,{start:.3f},{end:.3f})'"
        nx = max(0.0, min(1.0, float(ov.get("x") if ov.get("x") is not None else 0.5)))
        ny = max(0.0, min(1.0, float(ov.get("y") if ov.get("y") is not None else 0.5)))
        opacity = max(0.0, min(1.0, float(ov.get("opacity") if ov.get("opacity") is not None else 1.0)))
        flip_h = bool(ov.get("flip_h"))
        flip_v = bool(ov.get("flip_v"))
        out = f"[vov{seq}]"
        seq += 1

        if kind == "text":
            text = _escape_drawtext(ov.get("text") or "Text")
            size = max(8, int(ov.get("font_size") or 42))
            color = str(ov.get("color") or "#ffffff").lstrip("#")
            fontcolor = f"0x{color}" if len(color) == 6 else "white"
            alpha = f":alpha={opacity:.3f}" if opacity < 0.999 else ""
            x_expr = f"{nx:.6f}*W-text_w/2"
            y_expr = f"{ny:.6f}*H-text_h/2"
            fontfile = _resolve_fontfile(ov.get("font_family"))
            font_opt = f":fontfile='{fontfile}'" if fontfile else ""
            draw = (
                f"drawtext=text='{text}':fontsize={size}:fontcolor={fontcolor}{alpha}{font_opt}:"
                f"x={x_expr}:y={y_expr}:{enable}"
            )
            if flip_h or flip_v:
                txt_layer = f"[txtl{seq}]"
                parts.append(f"color=c=black@0.0:s={cw}x{ch}:d=1,format=rgba,{draw}{txt_layer}")
                flip_parts, flipped = _apply_flip(txt_layer, flip_h, flip_v, seq)
                parts.extend(flip_parts)
                parts.append(f"{cur}{flipped}overlay=0:0:{enable}{out}")
            else:
                parts.append(f"{cur}{draw}{out}")
            cur = out
            continue

        if kind != "image" or img_i >= len(image_input_labels):
            continue
        in_lbl = image_input_labels[img_i]
        img_i += 1
        tw, th = _image_target_size(ov, cw, ch)
        scaled = f"[ovs{seq}]"
        if th is None:
            scale_expr = f"scale={tw}:-1"
        else:
            scale_expr = f"scale={tw}:{th}"
        if opacity < 0.999:
            parts.append(
                f"{in_lbl}{scale_expr},format=rgba,colorchannelmixer=aa={opacity:.3f}{scaled}"
            )
        else:
            parts.append(f"{in_lbl}{scale_expr}{scaled}")
        flip_parts, flipped = _apply_flip(scaled, flip_h, flip_v, seq)
        parts.extend(flip_parts)
        x_expr = f"{nx:.6f}*W-w/2"
        y_expr = f"{ny:.6f}*H-h/2"
        parts.append(f"{cur}{flipped}overlay=x={x_expr}:y={y_expr}:{enable}{out}")
        cur = out

    return parts, cur
