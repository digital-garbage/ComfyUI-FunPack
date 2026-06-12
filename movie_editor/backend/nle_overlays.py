"""Timeline graphics overlays (images + text) for preview/export ffmpeg compositing."""
from __future__ import annotations

import os
import re
from typing import Callable


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

_DEFAULT_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]


def _resolve_fontfile(font_family: str | None) -> str | None:
    key = (font_family or "system-ui").strip().lower()
    if key in ("", "system-ui", "default"):
        key = "arial"
    for path in _FONT_PATHS.get(key, []):
        if os.path.isfile(path):
            return path
    for path in _DEFAULT_FONT_CANDIDATES:
        if os.path.isfile(path):
            return path
    return None


def _hex_rgba(color: str, opacity: float) -> tuple[int, int, int, int]:
    c = str(color or "#ffffff").strip().lstrip("#")
    if len(c) == 6:
        r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    else:
        r, g, b = 255, 255, 255
    a = int(max(0.0, min(1.0, float(opacity))) * 255)
    return r, g, b, a


def _load_pil_font(font_family: str | None, size: int):
    from PIL import ImageFont

    size = max(8, int(size))
    path = _resolve_fontfile(font_family)
    if path:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    for path in _DEFAULT_FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_text_overlay_png(ov: dict, canvas_w: int, canvas_h: int, out_path: str) -> tuple[int, int]:
    """Rasterize a text overlay to a transparent PNG (canvas pixel coordinates).

    Returns ``(width, height)`` of the PNG. Flip is baked into the PNG so ffmpeg
    only needs the standard overlay filter (no drawtext dependency).
    """
    from PIL import Image, ImageDraw

    text = re.sub(r"\r\n?", "\n", str(ov.get("text") or "Text")).strip() or "Text"
    size = max(8, int(ov.get("font_size") or 42))
    rgba = _hex_rgba(str(ov.get("color") or "#ffffff"), ov.get("opacity", 1))
    font = _load_pil_font(ov.get("font_family"), size)

    probe = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    draw = ImageDraw.Draw(probe)
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=max(2, size // 10))
    tw = max(1, bbox[2] - bbox[0])
    th = max(1, bbox[3] - bbox[1])
    pad = max(4, size // 8)
    img = Image.new("RGBA", (tw + pad * 2, th + pad * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.multiline_text(
        (pad - bbox[0], pad - bbox[1]),
        text,
        font=font,
        fill=rgba,
        spacing=max(2, size // 10),
    )

    if ov.get("flip_h"):
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if ov.get("flip_v"):
        img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    img.save(out_path, "PNG")
    return img.size


def prepare_overlay_export(
    overlays: list[dict],
    lanes: list[dict] | None,
    *,
    canvas_w: int,
    canvas_h: int,
    tempdir: str,
    resolve_image_path: Callable[[str | None], str | None],
) -> tuple[list[dict], list[str]]:
    """Build ordered overlay list + ffmpeg input paths for export.

    Text overlays are rasterized to PNG via Pillow so export works even when
    ffmpeg lacks the drawtext filter (common on minimal Homebrew builds).
    """
    sorted_ovs = sort_overlays_for_composite(overlays, lanes)
    export_overlays: list[dict] = []
    paths: list[str] = []

    for i, ov in enumerate(sorted_ovs):
        kind = ov.get("kind") or "image"
        if kind == "text":
            png_path = os.path.join(tempdir, f"ov_text_{ov.get('id') or i}.png")
            w, h = render_text_overlay_png(ov, canvas_w, canvas_h, png_path)
            paths.append(png_path)
            export_overlays.append({
                **ov,
                "kind": "image",
                "width_px": w,
                "height_px": h,
                "keep_aspect": False,
                "flip_h": False,
                "flip_v": False,
            })
            continue
        if kind != "image":
            continue
        src = resolve_image_path(ov.get("media_ref"))
        if not src or not os.path.isfile(src):
            continue
        paths.append(src)
        export_overlays.append(dict(ov))

    return export_overlays, paths


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
    """Compose image overlays onto ``base_label``. Returns (filter lines, final label).

    Text overlays must be rasterized before calling this (see ``prepare_overlay_export``).
    """
    parts: list[str] = []
    cur = base_label
    img_i = 0
    cw = max(1, int(canvas_w))
    seq = 0

    for ov in overlays or []:
        kind = ov.get("kind") or "image"
        if kind != "image":
            continue
        start = float(ov.get("start_sec") or 0)
        dur = float(ov.get("duration_sec") or 0)
        if dur <= 0:
            continue
        if img_i >= len(image_input_labels):
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

        in_lbl = image_input_labels[img_i]
        img_i += 1
        tw, th = _image_target_size(ov, cw, max(1, int(canvas_h)))
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
