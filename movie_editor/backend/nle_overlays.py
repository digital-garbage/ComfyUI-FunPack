"""Timeline graphics overlays (images + text) for preview/export ffmpeg compositing."""
from __future__ import annotations

import re
from typing import Any


def _escape_drawtext(text: str) -> str:
    s = str(text or "")
    s = s.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'").replace("%", "\\%")
    s = re.sub(r"[\r\n]+", " ", s)
    return s


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
            parts.append(
                f"{cur}drawtext=text='{text}':fontsize={size}:fontcolor={fontcolor}{alpha}:"
                f"x={x_expr}:y={y_expr}:{enable}{out}"
            )
            cur = out
            continue

        if kind != "image" or img_i >= len(image_input_labels):
            continue
        in_lbl = image_input_labels[img_i]
        img_i += 1
        scale = max(0.05, min(1.5, float(ov.get("scale") if ov.get("scale") is not None else 0.35)))
        target_w = max(8, int(scale * cw))
        scaled = f"[ovs{seq}]"
        if opacity < 0.999:
            parts.append(
                f"{in_lbl}scale={target_w}:-1,format=rgba,colorchannelmixer=aa={opacity:.3f}{scaled}"
            )
        else:
            parts.append(f"{in_lbl}scale={target_w}:-1{scaled}")
        x_expr = f"{nx:.6f}*W-w/2"
        y_expr = f"{ny:.6f}*H-h/2"
        parts.append(f"{cur}{scaled}overlay=x={x_expr}:y={y_expr}:{enable}{out}")
        cur = out

    return parts, cur
