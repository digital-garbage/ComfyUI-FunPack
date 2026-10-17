"""One picture out of an image file or a clip, for the tagger. Blocking; call it from a worker thread."""

from __future__ import annotations

import io
import shutil
import subprocess

from .store import kind_of


def picture(path, at: float = 0.5):
    """A PIL RGB image: the file itself, or the frame `at` (0 to 1) of the way through a clip."""
    from PIL import Image, ImageOps
    if kind_of(str(path)) == "image":
        with Image.open(path) as img:
            return ImageOps.exif_transpose(img).convert("RGB")
    try:
        import av
    except ImportError:
        av = None
    if av is not None:
        with av.open(str(path)) as box:
            stream = box.streams.video[0]
            if stream.duration and stream.time_base:
                box.seek(int(stream.duration * min(max(at, 0.0), 1.0)), stream=stream)
            for frame in box.decode(stream):
                return frame.to_image().convert("RGB")
        raise ValueError("the clip has no picture in it")
    ff = shutil.which("ffmpeg")
    if not ff:
        raise ValueError("reading a clip needs PyAV or ffmpeg, and neither is here")
    out = subprocess.run([ff, "-v", "error", "-i", str(path), "-vf", "thumbnail", "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
                         capture_output=True, check=True).stdout
    return Image.open(io.BytesIO(out)).convert("RGB")
