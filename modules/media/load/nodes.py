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
