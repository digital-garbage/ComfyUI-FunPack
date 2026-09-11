"""Load one file from FunPack's media library (core/media.py) as an IMAGE.

The bridge between "picked in the Media bin" and "wired into the graph": the
media store has no idea what a reference image or a resolution source is, and
the graph has no idea what a media id is. This is the one place both are true.

`media_id` is a plain string widget rather than a picker combo on purpose --
the id comes from the app (the Media bin hands it to this node's input before
a run queues, the same way a scene's prompt text lands on CLIPTextEncode's
`text`), not from someone typing a file name into this node by hand.
"""

from comfy_api.latest import InputImpl, io

from ..._core import log, media as media_mod


class FunPackLoadMedia(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="FunPackLoadMedia",
            display_name="FunPack Load Media",
            category="FunPack/Media",
            description="Load an image from FunPack's media library by id.",
            inputs=[
                io.String.Input(
                    "media_id", default="",
                    tooltip="A media library id, picked in the Media bin."),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    def execute(cls, media_id: str) -> io.NodeOutput:
        mid = (media_id or "").strip()
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
        if entry.get("kind") != "image":
            raise RuntimeError(
                f"FunPack Load Media: {entry.get('name', mid)!r} is a "
                f"{entry.get('kind', 'file')}, not an image.")

        components = InputImpl.VideoFromFile(str(path)).get_components()
        images = components.images
        if images is None or images.shape[0] == 0:
            raise RuntimeError(
                f"FunPack Load Media: could not read {entry.get('name', mid)!r}.")

        log.info("FunPack Load Media", f"loaded {entry.get('name', mid)}")
        # First frame only: this node hands off a still image, never a clip --
        # a multi-frame source (an image sequence, or a video picked by mistake
        # before the kind check above existed) would otherwise silently become
        # a batch, and everything downstream expects one image.
        return io.NodeOutput(images[:1])
