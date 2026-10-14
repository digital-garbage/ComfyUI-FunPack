"""NVIDIA RTX Video Super Resolution as an upscale engine, through the nvidia-vfx add-on.

Same contract as a model upscale: `make(options)` returns a function [B,H,W,3] float 0-1 -> [B,h,w,3],
which upscale_file streams a video through. The option names and ranges follow NVIDIA's RTX node
for ComfyUI (quality, scale or target size, strength, 8/10-bit), so a setup carries over.
"""

import json

NAME = "NVIDIA RTX Super Resolution"        # sent as the model name when this engine is picked (the UI holds the same string)
_LEVELS = ("LOW", "MEDIUM", "HIGH", "ULTRA")
QUALITIES = ("BICUBIC", *_LEVELS,
             *(f"DENOISE_{l}" for l in _LEVELS), *(f"DEBLUR_{l}" for l in _LEVELS),
             *(f"HIGHBITRATE_{l}" for l in _LEVELS), "STREAMING_MEDIUM", "STREAMING_ULTRA")
SAME_SIZE = ("DENOISE_", "DEBLUR_")          # these clean a video at its own size
DEFAULTS = {"quality": "ULTRA", "resize": "scale", "scale": 2.0, "width": 1920, "height": 1080, "strength": 1.0, "tenbit": False}


def options(raw) -> dict:
    """The user's choices, checked. A value out of range is refused by name, never quietly swapped."""
    try:
        given = json.loads(raw) if isinstance(raw, str) and raw.strip() else (raw or {})
    except ValueError as exc:
        raise ValueError(f"RTX options are not valid JSON: {exc}") from exc
    if not isinstance(given, dict):
        raise ValueError("RTX options must be an object.")
    o = {**DEFAULTS, **given}
    if o["quality"] not in QUALITIES:
        raise ValueError(f"RTX quality {o['quality']!r} is not one of: {', '.join(QUALITIES)}")
    if o["resize"] not in ("scale", "size"):
        raise ValueError("RTX resize must be 'scale' or 'size'.")
    for key, lo, hi in (("scale", 1.0, 4.0), ("strength", 0.0, 1.0), ("width", 64, 8192), ("height", 64, 8192)):
        if not isinstance(o[key], (int, float)) or isinstance(o[key], bool) or not lo <= o[key] <= hi:
            raise ValueError(f"RTX {key} must be a number from {lo} to {hi}.")
    o["tenbit"] = bool(o["tenbit"])
    return o


def out_size(w, h, o):
    """Output size: the input's own for denoise/deblur, else the scale or the target, each snapped to a multiple of 8."""
    if o["quality"].startswith(SAME_SIZE):
        return w, h
    ow, oh = (int(w * o["scale"]), int(h * o["scale"])) if o["resize"] == "scale" else (int(o["width"]), int(o["height"]))
    snap = lambda v: max(8, round(v / 8) * 8)
    return snap(ow), snap(oh)


def pack10(x):
    """[3,H,W] float 0-1 -> (H,W) uint32 in the SDK's RGB10A2 layout: R bits 0-9, G 10-19, B 20-29, A 30-31."""
    import torch
    v = (x.clamp(0, 1) * 1023).round().to(torch.int64)
    return (v[0] | (v[1] << 10) | (v[2] << 20) | (3 << 30)).to(torch.uint32)


def unpack10(p):
    """(H,W) uint32 -> [3,H,W] float 0-1."""
    import torch
    v = p.to(torch.int64)
    return torch.stack([v & 1023, (v >> 10) & 1023, (v >> 20) & 1023]).float() / 1023.0


def make(o):
    """-> the upscale function for these (already checked) options. Raises a sentence when the add-on is missing."""
    try:
        from nvvfx import VideoSuperRes
    except ImportError as exc:
        raise RuntimeError(f"{NAME} is not installed: switch it on in Settings ▸ Modules and restart ComfyUI.") from exc
    import torch
    quality = getattr(VideoSuperRes.QualityLevel, o["quality"], None)
    if quality is None:
        raise RuntimeError(f"This nvidia-vfx has no quality level {o['quality']}.")
    effects = {}

    def effect(h, w):
        if (h, w) not in effects:
            kw = {"quality": quality}
            if o["strength"] < 1.0:
                kw["strength"] = o["strength"]
            if o["tenbit"]:
                kw["image_encoding"] = VideoSuperRes.ImageEncoding.RGB10A2
            e = VideoSuperRes(**kw)
            e.output_width, e.output_height = out_size(w, h, o)
            e.load()
            effects[(h, w)] = e
        return effects[(h, w)]

    def run(x):
        out = []
        for frame in x:
            h, w = frame.shape[0], frame.shape[1]
            f = frame.to("cuda", torch.float32).permute(2, 0, 1).contiguous()
            res = effect(h, w).run(pack10(f) if o["tenbit"] else f)
            y = torch.from_dlpack(res.image).clone()          # the effect reuses its buffer: copy before the next frame
            out.append((unpack10(y) if o["tenbit"] else y).permute(1, 2, 0))
        return torch.stack(out)
    return run
