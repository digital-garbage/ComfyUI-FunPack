"""Pictures and clips as tensors for the trainer: sized to a bucket, cropped to cover it. Blocking."""

from __future__ import annotations

import math
from pathlib import Path

import torch


def bucket_for(aspect: float, resolution: int, step: int) -> tuple[int, int]:
    """(width, height) with about resolution x resolution pixels, the aspect of `aspect` (width / height) and sides a multiple of `step`."""
    area = float(resolution) ** 2
    w = math.sqrt(area * aspect)
    h = w / aspect
    snap = lambda v: max(step, int(round(v / step)) * step)
    return snap(w), snap(h)


def buckets_for(aspects, resolution: int, step: int, single: bool) -> list[tuple[int, int]]:
    """One (width, height) per aspect. With `single` every item gets the bucket of the median aspect, so all share one shape."""
    if not aspects:
        return []
    if single:
        median = sorted(aspects)[len(aspects) // 2]
        return [bucket_for(median, resolution, step)] * len(aspects)
    # Few distinct shapes keep training fast: aspects snap to the nearest of a coarse ladder.
    ladder = sorted({bucket_for(r, resolution, step) for r in (0.5, 0.625, 0.75, 0.875, 1.0, 1.15, 1.35, 1.6, 2.0)})
    return [min(ladder, key=lambda b: abs(math.log(b[0] / b[1]) - math.log(a))) for a in aspects]


def cover(img, width: int, height: int):
    """A PIL image scaled to cover width x height, then centre-cropped to exactly that."""
    from PIL import Image
    scale = max(width / img.width, height / img.height)
    w, h = max(width, round(img.width * scale)), max(height, round(img.height * scale))
    img = img.resize((w, h), Image.LANCZOS)
    left, top = (w - width) // 2, (h - height) // 2
    return img.crop((left, top, left + width, top + height))


def to_tensor(imgs) -> torch.Tensor:
    """PIL RGB images of one size -> float [N, H, W, 3] in 0..1."""
    import numpy as np
    return torch.from_numpy(np.stack([np.asarray(i.convert("RGB"), dtype=np.float32) / 255.0 for i in imgs]))


def size_of(path, kind: str) -> tuple[int, int]:
    """(width, height) without decoding the whole file."""
    if kind == "image":
        from PIL import Image, ImageOps
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)
            return img.width, img.height
    import av
    with av.open(str(path)) as box:
        s = box.streams.video[0]
        return int(s.codec_context.width), int(s.codec_context.height)


def clip_info(path) -> dict | None:
    """{"frames", "fps"} of a video file at its own rate, or None when it cannot be read."""
    try:
        import av
        with av.open(str(path)) as box:
            s = box.streams.video[0]
            src = float(s.average_rate or s.guessed_rate or 24.0)
            total = int(s.frames or 0)
            if not total and s.duration and s.time_base:
                total = int(float(s.duration * s.time_base) * src)
    except Exception:                                     # noqa: BLE001 -- an unreadable file is just left without a number
        return None
    return {"frames": total, "fps": round(src, 3)} if total else None


def frames_at(path, fps: float) -> int:
    """How many frames a clip gives when read at `fps` (0 = its own rate)."""
    import av
    with av.open(str(path)) as box:
        s = box.streams.video[0]
        src = float(s.average_rate or s.guessed_rate or 24.0)
        step = src / (float(fps) if fps else src)
        total = int(s.frames or 0)
        if not total and s.duration and s.time_base:
            total = int(float(s.duration * s.time_base) * src)
    return int((total - 1) / step) + 1 if total else 0


def clip_frames(path, frames: int, fps: float, start: str, width: int, height: int):
    """[frames, H, W, 3] float 0..1 taken in order from the clip at `fps` (0 = the clip's own rate), plus the rate used.

    Raises ValueError, naming the numbers, when the clip is shorter than the frames asked for.
    """
    import av
    with av.open(str(path)) as box:
        s = box.streams.video[0]
        src = float(s.average_rate or s.guessed_rate or 24.0)
        rate = float(fps) if fps else src
        step = src / rate
        total = int(s.frames or 0)
        if not total and s.duration and s.time_base:
            total = int(float(s.duration * s.time_base) * src)
        span = math.ceil((frames - 1) * step) + 1
        first = max(0, (total - span) // 2) if start == "middle" and total else 0
        want = [first + int(round(i * step)) for i in range(frames)]
        need = set(want)
        got: dict[int, object] = {}
        for index, frame in enumerate(box.decode(s)):
            if index in need:
                got[index] = cover(frame.to_image().convert("RGB"), width, height)
            if index >= want[-1]:
                break
    missing = [i for i in want if i not in got]
    if missing:
        raise ValueError(f"{Path(path).name} is too short: it needs {want[-1] + 1} frames at {rate:g} fps and has {len(got)} there (its own rate is {src:g} fps)")
    return to_tensor([got[i] for i in want]), rate


def read_audio(path, seconds: float, start_seconds: float = 0.0):
    """(waveform [1, channels, samples], sample rate) for `seconds` of the clip's sound, or None when it has none."""
    import av
    import numpy as np
    with av.open(str(path)) as box:
        if not box.streams.audio:
            return None
        s = box.streams.audio[0]
        rate = int(s.rate or s.codec_context.sample_rate)
        chunks = []
        for frame in box.decode(s):
            arr = frame.to_ndarray()
            arr = arr.astype(np.float32) / (32768.0 if arr.dtype.kind == "i" else 1.0)
            if frame.format.is_planar:
                chunks.append(arr)
            else:
                chunks.append(arr.reshape(-1, len(frame.layout.channels)).T)
    if not chunks:
        return None
    wave = np.concatenate(chunks, axis=1)
    a, b = int(start_seconds * rate), int((start_seconds + seconds) * rate)
    wave = wave[:, a:b]
    if wave.shape[1] < int(seconds * rate):
        wave = np.pad(wave, ((0, 0), (0, int(seconds * rate) - wave.shape[1])))
    return torch.from_numpy(wave.copy())[None], rate
