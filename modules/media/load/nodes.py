"""Load one file from FunPack's media library (core/media.py) as an IMAGE.

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


class FunPackLoadMedia(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="FunPackLoadMedia",
            display_name="FunPack Load Media",
            category="FunPack/Media",
            description="Load an image (or the first frame of a video) from FunPack's media library by id.",
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
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    def execute(cls, media_id: str, blank_width: int = 0, blank_height: int = 0) -> io.NodeOutput:
        mid = (media_id or "").strip()
        if not mid and blank_width > 0 and blank_height > 0:
            # A text-only start: the picture exists only to give the pipeline a SHAPE (see the H3 preset).
            log.info("FunPack Load Media", f"nothing picked: a plain {blank_width}x{blank_height} picture stands in")
            return io.NodeOutput(torch.full((1, blank_height, blank_width, 3), 0.5))
        if not mid:
            raise RuntimeError(
                "FunPack Load Media: nothing is picked for this input. "
                "Choose an image in the Media bin.")
        entry = media_mod.get(mid)
        path = media_mod.path_for(mid)
        if entry is None or path is None:
            raise RuntimeError(
                f"FunPack Load Media: media {mid!r} is not in the library any "
                f"more -- it may have been deleted. Pick another.")
        kind = entry.get("kind")
        if kind not in ("image", "video"):
            raise RuntimeError(
                f"FunPack Load Media: {entry.get('name', mid)!r} is {kind or 'a file'}, "
                f"which has no picture to use. Pick an image or a video.")

        components = InputImpl.VideoFromFile(str(path)).get_components()
        images = components.images
        if images is None or images.shape[0] == 0:
            raise RuntimeError(
                f"FunPack Load Media: could not read {entry.get('name', mid)!r}.")

        log.info("FunPack Load Media", f"loaded {entry.get('name', mid)}")
        # First frame only: this node hands off a still image, never a clip --
        # a video reference stands in as its first frame, since everything
        # downstream expects one image.
        if kind == "video":
            log.info("FunPack Load Media", f"{entry.get('name', mid)} is a video: its first frame is used")
        return io.NodeOutput(images[:1])


#: Frame rate and longest stretch the H3 reference-video input takes (its tooltip: 24 fps, 2-15 s).
REF_VIDEO_FPS = 24
REF_VIDEO_MAX_SECONDS = 15


def _entry(mid: str, kinds: tuple[str, ...], wanted: str):
    mid = (mid or "").strip()
    if not mid:
        raise RuntimeError(f"FunPack Load Media: nothing is picked for this input. Choose {wanted} in the Media bin.")
    entry, path = media_mod.get(mid), media_mod.path_for(mid)
    if entry is None or path is None:
        raise RuntimeError(f"FunPack Load Media: media {mid!r} is not in the library any more -- it may have been deleted. Pick another.")
    if entry.get("kind") not in kinds:
        raise RuntimeError(f"FunPack Load Media: {entry.get('name', mid)!r} is {entry.get('kind', 'a file')}, not {wanted}.")
    return entry, path


class FunPackLoadMediaVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="FunPackLoadMediaVideo",
            display_name="FunPack Load Media (video)",
            category="FunPack/Media",
            description="Load a video from FunPack's media library by id as frames at 24 fps, at most 15 seconds.",
            inputs=[io.String.Input("media_id", default="", tooltip="A media library id, picked in the Media bin.")],
            outputs=[io.Image.Output(display_name="frames")],
        )

    @classmethod
    def execute(cls, media_id: str) -> io.NodeOutput:
        entry, path = _entry(media_id, ("video",), "a video")
        components = InputImpl.VideoFromFile(str(path)).get_components()
        frames, fps = components.images, float(components.frame_rate or REF_VIDEO_FPS)
        if frames is None or frames.shape[0] == 0:
            raise RuntimeError(f"FunPack Load Media: could not read {entry.get('name', media_id)!r}.")
        seconds = frames.shape[0] / fps
        # Whatever the source rate, the model wants 24 fps: pick the frame showing at each 1/24 s.
        count = int(min(seconds, REF_VIDEO_MAX_SECONDS) * REF_VIDEO_FPS)
        index = torch.clamp((torch.arange(count) * fps / REF_VIDEO_FPS).long(), max=frames.shape[0] - 1)
        log.info("FunPack Load Media", f"loaded {entry.get('name', media_id)}: {count} frames at {REF_VIDEO_FPS} fps")
        return io.NodeOutput(frames[index])


class FunPackLoadMediaAudio(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="FunPackLoadMediaAudio",
            display_name="FunPack Load Media (audio)",
            category="FunPack/Media",
            description="Load an audio file from FunPack's media library by id.",
            inputs=[io.String.Input("media_id", default="", tooltip="A media library id, picked in the Media bin.")],
            outputs=[io.Audio.Output(display_name="audio")],
        )

    @classmethod
    def execute(cls, media_id: str) -> io.NodeOutput:
        import av
        import numpy as np
        entry, path = _entry(media_id, ("audio",), "an audio file")
        chunks, rate = [], 0
        with av.open(str(path)) as container:
            if not container.streams.audio:
                raise RuntimeError(f"FunPack Load Media: {entry.get('name', media_id)!r} has no sound.")
            for frame in container.decode(audio=0):
                rate = frame.sample_rate
                data = frame.to_ndarray()
                if data.dtype.kind in "iu":
                    data = data.astype(np.float32) / float(np.iinfo(data.dtype).max)
                if frame.format.is_packed:
                    data = data.reshape(-1, len(frame.layout.channels)).T
                chunks.append(data.astype(np.float32))
        if not chunks:
            raise RuntimeError(f"FunPack Load Media: could not read {entry.get('name', media_id)!r}.")
        log.info("FunPack Load Media", f"loaded {entry.get('name', media_id)}")
        return io.NodeOutput({"waveform": torch.from_numpy(np.concatenate(chunks, axis=1)).unsqueeze(0), "sample_rate": rate})
