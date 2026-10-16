"""Timeline graphics (images and text) composited over the montage.

Text is rasterised to a transparent PNG with Pillow and then composited like any
other image, so export does not depend on ffmpeg having the `drawtext` filter
(minimal builds do not). Flips are baked into that PNG for the same reason.
"""

from __future__ import annotations

import os
import re
from typing import Callable

_FONTS = {
    "arial": ["/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf",
              "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"],
    "helvetica": ["/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf",
                  "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
    "georgia": ["/System/Library/Fonts/Supplemental/Georgia.ttf", "/Library/Fonts/Georgia.ttf",
                "/usr/share/fonts/truetype/msttcorefonts/Georgia.ttf"],
    "times": ["/System/Library/Fonts/Supplemental/Times New Roman.ttf", "/Library/Fonts/Times New Roman.ttf",
              "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"],
    "courier": ["/System/Library/Fonts/Supplemental/Courier New.ttf", "/Library/Fonts/Courier New.ttf",
                "/usr/share/fonts/truetype/msttcorefonts/Courier_New.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"],
    "verdana": ["/System/Library/Fonts/Supplemental/Verdana.ttf", "/Library/Fonts/Verdana.ttf",
                "/usr/share/fonts/truetype/msttcorefonts/Verdana.ttf"],
    "impact": ["/System/Library/Fonts/Supplemental/Impact.ttf", "/Library/Fonts/Impact.ttf"],
}
_FALLBACK = ["/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
             "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"]


def _num(value, default, lo=None, hi=None) -> float:
    """A finite float from whatever the editor stored; junk reads as `default`."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if v != v or v in (float("inf"), float("-inf")):
        return default
    if lo is not None:
        v = max(lo, v)
    return min(hi, v) if hi is not None else v


def _fontfile(family) -> str | None:
    key = str(family or "system-ui").strip().lower()
    if key in ("", "system-ui", "default"):
        key = "arial"
    for path in _FONTS.get(key, []) + _FALLBACK:
        if os.path.isfile(path):
            return path
    return None


def _rgba(color, opacity) -> tuple[int, int, int, int]:
    c = str(color or "#ffffff").strip().lstrip("#")
    if len(c) in (3, 4):                                       # #rgb / #rgba
        c = "".join(ch * 2 for ch in c)
    alpha = 1.0
    if len(c) == 8:                                            # #rrggbbaa: its alpha scales the overlay's opacity
        try:
            alpha, c = int(c[6:], 16) / 255, c[:6]
        except ValueError:
            c = "ffffff"
    opacity = _num(opacity, 1.0, 0.0, 1.0) * alpha
    try:
        r, g, b = (int(c[i:i + 2], 16) for i in (0, 2, 4)) if len(c) == 6 else (255, 255, 255)
    except ValueError:
        r, g, b = 255, 255, 255
    return r, g, b, int(opacity * 255)


def _variant(regular: str, bold: bool, italic: bool) -> str | None:
    """A bold/italic sibling of a regular font file, by the usual naming patterns."""
    if not regular or not (bold or italic):
        return None
    folder, (base, ext) = os.path.dirname(regular), os.path.splitext(os.path.basename(regular))
    if ext.lower() == ".ttc":
        return None
    suffixes = ([" Bold Italic", "-BoldItalic", "-BoldOblique", " BdIt"] if bold and italic else
                [" Bold", "-Bold", " Bd"] if bold else [" Italic", "-Italic", "-Oblique", " It"])
    stem = re.sub(r"[ -]?Regular$", "", base)
    for suffix in suffixes:
        cand = os.path.join(folder, f"{stem}{suffix}{ext}")
        if os.path.isfile(cand):
            return cand
    return None


def _font(family, size: int, bold: bool, italic: bool):
    """(font, has real bold, has real italic): the caller fakes the faces that were not found."""
    from PIL import ImageFont
    size = max(8, int(size))
    path = _fontfile(family)
    if path:
        variant = _variant(path, bold, italic)
        for cand, got in ((variant, (bold, italic)), (path, (False, False))):
            if cand:
                try:
                    return ImageFont.truetype(cand, size), got[0], got[1]
                except OSError:
                    pass
    for cand in _FALLBACK:
        try:
            return ImageFont.truetype(cand, size), False, False
        except OSError:
            continue
    return ImageFont.load_default(), False, False


def text_png(ov: dict, out_path: str) -> tuple[int, int]:
    """Rasterise a text overlay to a transparent PNG; returns its (width, height)."""
    from PIL import Image, ImageDraw

    text = re.sub(r"\r\n?", "\n", str(ov.get("text") or "Text")).strip() or "Text"
    size = max(8, int(_num(ov.get("font_size"), 42)))
    opacity = _num(ov.get("opacity"), 1.0, 0.0, 1.0)
    fill = _rgba(ov.get("color"), opacity)
    bold, italic = bool(ov.get("bold")), bool(ov.get("italic"))
    font, has_bold, has_italic = _font(ov.get("font_family"), size, bold, italic)
    align = str(ov.get("text_align") or "center").lower()
    align = align if align in ("left", "center", "right") else "center"
    spacing = max(0, int(round(size * (_num(ov.get("line_spacing"), 1.2) - 1))))
    stroke = max(0, int(round(_num(ov.get("stroke_width"), 0))))
    faux_bold = max(1, int(round(size * 0.045))) if bold and not has_bold else 0
    eff_stroke = stroke + faux_bold
    stroke_fill = _rgba(ov.get("stroke_color") or "#000000", opacity) if stroke > 0 else (fill if faux_bold else None)
    shadow = bool(ov.get("shadow"))
    shadow_off = max(1, int(round(size * 0.06)))
    shadow_fill = _rgba(ov.get("shadow_color") or "#000000", opacity * 0.85) if shadow else None

    draw = ImageDraw.Draw(Image.new("RGBA", (4, 4)))
    bx0, by0, bx1, by1 = (int(round(v)) for v in draw.multiline_textbbox(
        (0, 0), text, font=font, spacing=spacing, align=align, stroke_width=eff_stroke))
    margin = eff_stroke + (shadow_off if shadow else 0) + max(4, size // 8)
    layer = Image.new("RGBA", (max(1, bx1 - bx0) + margin * 2 + shadow_off,
                               max(1, by1 - by0) + margin * 2 + shadow_off), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    ox, oy = margin - bx0, margin - by0
    if shadow:
        draw.multiline_text((ox + shadow_off, oy + shadow_off), text, font=font, fill=shadow_fill, spacing=spacing,
                            align=align, stroke_width=eff_stroke, stroke_fill=shadow_fill)
    draw.multiline_text((ox, oy), text, font=font, fill=fill, spacing=spacing, align=align,
                        stroke_width=eff_stroke, stroke_fill=stroke_fill)
    if italic and not has_italic:                              # no italic face: shear the layer
        shear = 0.21
        layer = layer.transform((layer.width + int(shear * layer.height), layer.height), Image.Transform.AFFINE,
                                (1, shear, -shear * layer.height, 0, 1, 0), resample=Image.Resampling.BICUBIC)
    crop = layer.getbbox()
    if crop:
        layer = layer.crop(crop)
    if ov.get("bg_enabled"):
        px, py = max(2, int(round(size * 0.4))), max(2, int(round(size * 0.22)))
        bg = Image.new("RGBA", (layer.width + px * 2, layer.height + py * 2), (0, 0, 0, 0))
        bg_op = _num(ov.get("bg_opacity"), 0.5)
        ImageDraw.Draw(bg).rounded_rectangle(
            [0, 0, bg.width - 1, bg.height - 1], radius=max(0, int(round(size * 0.1))),
            fill=_rgba(ov.get("bg_color") or "#000000", bg_op * opacity))
        bg.alpha_composite(layer, (px, py))
        layer = bg
    if ov.get("flip_h"):
        layer = layer.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if ov.get("flip_v"):
        layer = layer.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    layer.save(out_path, "PNG")
    return layer.size


def in_stacking_order(overlays: list[dict], lanes: list[dict] | None) -> list[dict]:
    """Bottom lanes first, then by start time within a lane."""
    lanes = [lane for lane in (lanes or []) if isinstance(lane, dict)]
    if not lanes:
        return list(overlays or [])
    order = {str(lane.get("id") or ""): i for i, lane in enumerate(lanes)}
    first = str(lanes[0].get("id") or "")
    return sorted(overlays or [], key=lambda ov: (order.get(str(ov.get("lane_id") or first), 0),
                                                  _num(ov.get("start_sec"), 0.0)))


def prepare(overlays: list[dict], lanes: list[dict] | None, *, tempdir: str,
            image_path: Callable[[str | None], str | None]) -> tuple[list[dict], list[str]]:
    """(overlays ready to composite, the image files they read -- one per overlay, in order).

    A text overlay becomes an image overlay at its rendered size; an image overlay whose file is
    gone is left out (the caller says so by comparing counts)."""
    ready, paths = [], []
    for i, ov in enumerate(in_stacking_order(overlays, lanes)):
        kind = ov.get("kind") or "image"
        if kind == "text":
            png = os.path.join(tempdir, f"ov_text_{re.sub(r'[^A-Za-z0-9_-]', '', str(ov.get('id') or i))}.png")
            w, h = text_png(ov, png)
            paths.append(png)
            ready.append({**ov, "kind": "image", "width_px": w, "height_px": h, "keep_aspect": False,
                          "flip_h": False, "flip_v": False})
        elif kind == "image":
            src = image_path(ov.get("media_ref"))
            if src and os.path.isfile(src):
                paths.append(src)
                ready.append(dict(ov))
    return ready, paths


def _size(ov: dict, canvas_w: int) -> tuple[int, int | None]:
    """(width px, height px or None to keep the aspect ratio)."""
    wpx = ov.get("width_px")
    if wpx is not None:
        tw = max(8, int(_num(wpx, 8)))
        if ov.get("keep_aspect", True) is not False:
            return tw, None
        return tw, max(8, int(_num(ov.get("height_px"), tw)))
    return max(8, int(_num(ov.get("scale"), 0.35, 0.05, 1.5) * canvas_w)), None


def composite(base: str, overlays: list[dict], *, canvas_w: int, image_labels: list[str]) -> tuple[list[str], str]:
    """Filter lines laying each image overlay over `base`; returns (lines, final label). Each overlay
    reads its own input, so an overlay that is not drawn leaves its input unused (ffmpeg allows that)."""
    parts: list[str] = []
    cur = base
    for seq, ov in enumerate(overlays):
        # image_labels[i] is overlay i's picture (prepare() keeps them 1:1); a skipped overlay
        # must not shift the pictures of the ones after it.
        if (ov.get("kind") or "image") != "image" or seq >= len(image_labels):
            continue
        start, dur = _num(ov.get("start_sec"), 0.0), _num(ov.get("duration_sec"), 0.0)
        if dur <= 0:
            continue
        nx, ny = _num(ov.get("x"), 0.5, 0.0, 1.0), _num(ov.get("y"), 0.5, 0.0, 1.0)
        opacity = _num(ov.get("opacity"), 1.0, 0.0, 1.0)
        tw, th = _size(ov, max(1, int(canvas_w)))
        scale = f"scale={tw}:-1" if th is None else f"scale={tw}:{th}"
        scaled = f"[ovs{seq}]"
        src = image_labels[seq]
        if opacity < 0.999:
            parts.append(f"{src}{scale},format=rgba,colorchannelmixer=aa={opacity:.3f}{scaled}")
        else:
            parts.append(f"{src}{scale}{scaled}")
        flipped = scaled
        for flag, name in (("flip_h", "hflip"), ("flip_v", "vflip")):
            if ov.get(flag):
                nxt = f"[ov{name}{seq}]"
                parts.append(f"{flipped}{name}{nxt}")
                flipped = nxt
        out = f"[vov{seq}]"
        parts.append(f"{cur}{flipped}overlay=x={nx:.6f}*W-w/2:y={ny:.6f}*H-h/2:"
                     f"enable='between(t,{start:.3f},{start + dur:.3f})'{out}")
        cur = out
    return parts, cur
