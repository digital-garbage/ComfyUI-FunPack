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


def _variant_path(regular: str, bold: bool, italic: bool) -> str | None:
    """Find a bold/italic sibling of a regular font file by common naming patterns."""
    if not regular or (not bold and not italic):
        return None
    d = os.path.dirname(regular)
    base, ext = os.path.splitext(os.path.basename(regular))
    if ext.lower() == ".ttc":
        return None
    # Candidate suffixes per family naming style, most specific first.
    space, ital_word = [], "Italic"
    if bold and italic:
        space = [" Bold Italic", "-BoldItalic", "-BoldOblique", " BdIt"]
    elif bold:
        space = [" Bold", "-Bold", " Bd"]
    elif italic:
        space = [" Italic", "-Italic", "-Oblique", " It"]
    # Strip a trailing "-Regular"/" Regular" stem before appending a variant suffix.
    stem = re.sub(r"[ -]?Regular$", "", base)
    for suf in space:
        cand = os.path.join(d, f"{stem}{suf}{ext}")
        if os.path.isfile(cand):
            return cand
    return None


def _load_pil_font(font_family: str | None, size: int, bold: bool = False, italic: bool = False):
    """Return ``(font, has_bold, has_italic)`` — the booleans say whether a real
    bold/italic face was found (so the caller can fake the missing ones)."""
    from PIL import ImageFont

    size = max(8, int(size))
    path = _resolve_fontfile(font_family)
    has_bold = has_italic = False
    if path:
        variant = _variant_path(path, bold, italic)
        if variant:
            try:
                f = ImageFont.truetype(variant, size)
                return f, bold, italic
            except OSError:
                pass
        try:
            return ImageFont.truetype(path, size), has_bold, has_italic
        except OSError:
            pass
    for cand in _DEFAULT_FONT_CANDIDATES:
        try:
            return ImageFont.truetype(cand, size), has_bold, has_italic
        except OSError:
            continue
    return ImageFont.load_default(), has_bold, has_italic


def render_text_overlay_png(ov: dict, canvas_w: int, canvas_h: int, out_path: str) -> tuple[int, int]:
    """Rasterize a text overlay to a transparent PNG (canvas pixel coordinates).

    Returns ``(width, height)`` of the PNG. Flip is baked into the PNG so ffmpeg
    only needs the standard overlay filter (no drawtext dependency).
    """
    from PIL import Image, ImageDraw

    text = re.sub(r"\r\n?", "\n", str(ov.get("text") or "Text")).strip() or "Text"
    size = max(8, int(ov.get("font_size") or 42))
    opacity = max(0.0, min(1.0, float(ov.get("opacity", 1) or 0)))
    fill = _hex_rgba(str(ov.get("color") or "#ffffff"), opacity)

    bold = bool(ov.get("bold"))
    italic = bool(ov.get("italic"))
    font, has_bold, has_italic = _load_pil_font(ov.get("font_family"), size, bold, italic)

    align = str(ov.get("text_align") or "center").lower()
    if align not in ("left", "center", "right"):
        align = "center"
    ls_mult = float(ov.get("line_spacing") if ov.get("line_spacing") is not None else 1.2)
    spacing = max(0, int(round(size * (ls_mult - 1))))

    stroke_w = max(0, int(round(float(ov.get("stroke_width") or 0))))
    # Fake bold by thickening glyphs with a same-color stroke when no bold face exists.
    faux_bold_extra = max(1, int(round(size * 0.045))) if (bold and not has_bold) else 0
    eff_stroke = stroke_w + faux_bold_extra
    if stroke_w > 0:
        stroke_fill = _hex_rgba(str(ov.get("stroke_color") or "#000000"), opacity)
    elif faux_bold_extra:
        stroke_fill = fill
    else:
        stroke_fill = None

    shadow = bool(ov.get("shadow"))
    shadow_off = max(1, int(round(size * 0.06)))
    shadow_fill = _hex_rgba(str(ov.get("shadow_color") or "#000000"), opacity * 0.85) if shadow else None

    probe = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    draw = ImageDraw.Draw(probe)
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing, align=align, stroke_width=eff_stroke)
    # Pillow >= 10 may return float bbox coords; Image.new() and crop boxes need ints.
    bx0, by0, bx1, by1 = (int(round(v)) for v in bbox)
    tw = max(1, bx1 - bx0)
    th = max(1, by1 - by0)
    margin = eff_stroke + (shadow_off if shadow else 0) + max(4, size // 8)
    layer = Image.new("RGBA", (tw + margin * 2 + shadow_off, th + margin * 2 + shadow_off), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    ox, oy = margin - bx0, margin - by0
    if shadow:
        draw.multiline_text((ox + shadow_off, oy + shadow_off), text, font=font, fill=shadow_fill,
                            spacing=spacing, align=align, stroke_width=eff_stroke, stroke_fill=shadow_fill)
    draw.multiline_text((ox, oy), text, font=font, fill=fill,
                        spacing=spacing, align=align, stroke_width=eff_stroke, stroke_fill=stroke_fill)

    # Fake italic by shearing the rendered text layer when no italic face exists.
    if italic and not has_italic:
        shear = 0.21
        new_w = layer.width + int(abs(shear) * layer.height)
        layer = layer.transform((new_w, layer.height), Image.Transform.AFFINE,
                                (1, shear, -shear * layer.height, 0, 1, 0),
                                resample=Image.Resampling.BICUBIC)

    crop = layer.getbbox()
    if crop:
        layer = layer.crop(crop)

    if ov.get("bg_enabled"):
        pad_x, pad_y = max(2, int(round(size * 0.4))), max(2, int(round(size * 0.22)))
        bw, bh = layer.width + pad_x * 2, layer.height + pad_y * 2
        bg = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
        bg_opacity = float(ov.get("bg_opacity") if ov.get("bg_opacity") is not None else 0.5)
        bg_rgba = _hex_rgba(str(ov.get("bg_color") or "#000000"), bg_opacity * opacity)
        ImageDraw.Draw(bg).rounded_rectangle([0, 0, bw - 1, bh - 1], radius=max(0, int(round(size * 0.1))), fill=bg_rgba)
        bg.alpha_composite(layer, (pad_x, pad_y))
        layer = bg

    if ov.get("flip_h"):
        layer = layer.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if ov.get("flip_v"):
        layer = layer.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    layer.save(out_path, "PNG")
    return layer.size


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
