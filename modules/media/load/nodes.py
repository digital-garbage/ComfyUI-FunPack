"""Load one file from FunPack's media library (core/media.py): a picture, a video or sound.

The bridge between "picked in the Media bin" and "wired into the graph": the
media store has no idea what a reference image or an aspect ratio source is, and
the graph has no idea what a media id is. This is the one place both are true.

`media_id` is a plain string widget rather than a picker combo on purpose --
the id comes from the app (the Media bin hands it to this node's input before
a run queues, the same way a scene's prompt text lands on CLIPTextEncode's
`text`), not from someone typing a file name into this node by hand.
"""

import torch
from comfy_api.latest import InputImpl, io

from ..._core import log, media as media_mod


#: Frame rate the H3 reference-video input takes.
REF_VIDEO_FPS = 24
#: What the audio output holds when the file has no sound (or nothing was asked for): one silent sample.
_SILENCE = {"waveform": torch.zeros((1, 1, 1)), "sample_rate": 44100}


def _read_audio(path):
    """A file's sound as ComfyUI AUDIO, or None when it has none."""
    import av
    import numpy as np
    chunks, rate = [], 0
    with av.open(str(path)) as container:
        if not container.streams.audio:
            return None
        for frame in container.decode(audio=0):
            rate = frame.sample_rate
            data = frame.to_ndarray()
            if data.dtype.kind in "iu":
                data = data.astype(np.float32) / float(np.iinfo(data.dtype).max)
            if frame.format.is_packed:
                data = data.reshape(-1, len(frame.layout.channels)).T
            chunks.append(data.astype(np.float32))
    if not chunks:
        return None
    return {"waveform": torch.from_numpy(np.concatenate(chunks, axis=1)).unsqueeze(0), "sample_rate": rate}


class FunPackLoadMedia(io.ComfyNode):
    """Takes a picture, a video or an audio file and fills what it has: the app wires the output that fits."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="FunPackLoadMedia",
            display_name="FunPack Load Media",
            category="FunPack/Media",
            description="Load a picture, a video or an audio file from FunPack's media library by id. "
                        "image: the picture (a video's first frame). frames: a video's frames at 24 fps. "
                        "audio: a video's soundtrack or an audio file.",
            inputs=[
                io.String.Input(
                    "media_id", default="",
                    tooltip="A media library id, picked in the Media bin."),
                io.Int.Input(
                    "blank_width", default=0, min=0, max=16384, optional=True,
                    tooltip="With nothing picked, make a plain grey picture this wide instead of refusing. 0 = refuse."),
                io.Int.Input(
                    "blank_height", default=0, min=0, max=16384, optional=True,
                    tooltip="With nothing picked, make a plain grey picture this tall instead of refusing. 0 = refuse."),
                io.Boolean.Input(
                    "full", default=False, optional=True,
                    tooltip="Read the whole video and its sound. Off: only the first frame, which is all a picture input needs."),
            ],
            outputs=[io.Image.Output(display_name="image"), io.Image.Output(display_name="frames"),
                     io.Audio.Output(display_name="audio")],
        )

    @classmethod
    def execute(cls, media_id: str, blank_width: int = 0, blank_height: int = 0, full: bool = False) -> io.NodeOutput:
        mid = (media_id or "").strip()
        if not mid and blank_width > 0 and blank_height > 0:
            # A text-only start: the picture exists only to give the pipeline a SHAPE (see the H3 preset).
            log.info("FunPack Load Media", f"nothing picked: a plain {blank_width}x{blank_height} picture stands in")
            grey = torch.full((1, blank_height, blank_width, 3), 0.5)
            return io.NodeOutput(grey, grey, _SILENCE)
        if not mid:
            raise RuntimeError(
                "FunPack Load Media: nothing is picked for this input. "
                "Choose a file in the Media bin.")
        entry = media_mod.get(mid)
        path = media_mod.path_for(mid)
        if entry is None or path is None:
            raise RuntimeError(
                f"FunPack Load Media: media {mid!r} is not in the library any "
                f"more -- it may have been deleted. Pick another.")
        name, kind = entry.get("name", mid), entry.get("kind")
        if kind not in ("image", "video", "audio"):
            raise RuntimeError(f"FunPack Load Media: {name!r} is {kind or 'a file'}, which is not a picture, a video or sound.")

        if kind == "audio":
            try:
                sound = _read_audio(path)
            except Exception:                                # noqa: BLE001 -- a damaged file raises whatever PyAV raises
                sound = None
            if sound is None:
                raise RuntimeError(f"FunPack Load Media: could not read {name!r}.")
            log.info("FunPack Load Media", f"loaded {name} (audio)")
            grey = torch.full((1, 64, 64, 3), 0.5)       # no picture in it: the outputs for pictures are never wired
            return io.NodeOutput(grey, grey, sound)

        try:
            components = InputImpl.VideoFromFile(str(path)).get_components()
        except Exception as exc:                             # noqa: BLE001 -- a damaged file raises whatever PyAV raises
            raise RuntimeError(f"FunPack Load Media: could not read {name!r} ({exc}).") from exc
        images = components.images
        if images is None or images.shape[0] == 0:
            raise RuntimeError(f"FunPack Load Media: could not read {name!r}.")
        first = images[:1]
        if kind == "image" or not full:
            log.info("FunPack Load Media", f"loaded {name}")
            return io.NodeOutput(first, first, _SILENCE)

        fps = float(components.frame_rate or REF_VIDEO_FPS)
        # Whatever the source rate, the model wants 24 fps: pick the frame showing at each 1/24 s.
        count = max(1, int(images.shape[0] / fps * REF_VIDEO_FPS))
        index = torch.clamp((torch.arange(count) * fps / REF_VIDEO_FPS).long(), max=images.shape[0] - 1)
        sound = components.audio or _SILENCE
        log.info("FunPack Load Media", f"loaded {name}: {count} frames at {REF_VIDEO_FPS} fps")
        return io.NodeOutput(first, images[index], sound)
