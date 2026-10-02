"""Latent to pictures, and to sound where there is any.

This is the only part of getting a result out that FunPack needs to own. Core
already turns images into a video (`CreateVideo`) and writes the file
(`SaveVideo`, `SaveWEBM`, `SaveImage`), and those handle containers, codecs and
audio muxing properly -- writing our own would be duplication with a worse
version of somebody else's tested code. VHS is unnecessary either way.

What core cannot generalise is the decode itself for a model whose latent is not
one tensor. An AV latent carries a video branch and an audio branch that need
different VAEs, and how they are arranged is the model's business. So the same
rule as the empty latent applies: a model's own module may claim the decode, and
otherwise it is the ordinary single-tensor path.

The node takes the MODEL for one reason: so a provider can answer "is this mine"
by IDENTITY rather than by shape. Without it, the H3 module claimed any latent
that happened to have two parts -- correct today, because H3 is the only thing
producing one, and silently wrong the moment a second AV model exists. Deciding
from shape is how a model mismatch turns into an unrelated-looking fault instead
of an error.
"""

from comfy_api.latest import io

from ..._core import log, registry as registry_mod

CAPABILITY = "decode"


def _plain_decode(vae, latent, tile_size):
    """The ordinary decode, tiled the way ComfyUI's own VAE Decode (Tiled) does it when
    asked: a quarter-tile overlap, 64-frame temporal tiles with an 8-frame overlap."""
    if not tile_size or tile_size <= 0:
        return vae.decode(latent)
    overlap = tile_size // 4
    t_size, t_overlap = 64, 8
    t_comp = vae.temporal_compression_decode()
    if t_comp is not None:
        t_size, t_overlap = max(2, t_size // t_comp), max(1, min(t_size // t_comp // 2, t_overlap // t_comp))
    else:
        t_size = t_overlap = None
    c = vae.spacial_compression_decode()
    return vae.decode_tiled(latent, tile_x=tile_size // c, tile_y=tile_size // c, overlap=overlap // c,
                            tile_t=t_size, overlap_t=t_overlap)


class FunPackDecode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="FunPackDecode",
            display_name="FunPack Decode",
            category="FunPack/Output",
            description="Decode a latent to images, and to audio when the model has any.",
            inputs=[
                io.Latent.Input("samples"),
                io.Vae.Input("vae", tooltip="The picture VAE."),
                io.Model.Input("model", optional=True,
                               tooltip="The model this latent came from. Needed only when a "
                                       "model has its own way of being decoded."),
                io.Vae.Input("audio_vae", optional=True,
                             tooltip="Only for models that generate sound alongside the video."),
                io.Int.Input("tile_size", default=0, min=0, max=4096, step=64, optional=True,
                             tooltip="Decode in tiles of this many pixels, to fit less VRAM. 0 = in one "
                                     "piece. Bigger = faster and more VRAM. An X2 Detail VAE is always "
                                     "tiled (256 at least)."),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
                io.Audio.Output(display_name="audio"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, samples, vae, model=None, audio_vae=None, tile_size=0) -> io.NodeOutput:
        latent = samples["samples"]

        for spec, decode in registry_mod.current().providers(CAPABILITY):
            try:
                claimed = decode(latent, model=model, vae=vae, audio_vae=audio_vae, tile_size=tile_size)
            except Exception as exc:             # noqa: BLE001
                # Same protocol as the empty latent: returning None means "not my
                # model", so getting this far means it WAS and it broke. Falling
                # back would decode a nested latent as if it were one tensor.
                log.broke(f"{spec.id}.{CAPABILITY}", exc, "decoding this model's latent")
                raise RuntimeError(
                    f"{spec.id} handles this model's decode and failed: "
                    f"{type(exc).__name__}: {exc}. Refusing to decode it as a plain "
                    f"latent, which would not be the same picture."
                ) from exc
            if claimed is not None:
                images, audio = claimed
                return io.NodeOutput(images, audio, f"{spec.id} decoded this")

        if getattr(latent, "is_nested", False):
            # Nothing claimed it and it is not one tensor: vae.decode would either
            # raise somewhere confusing or quietly decode the wrong branch.
            raise RuntimeError(
                "This latent has more than one part and no installed module knows "
                "how to decode it."
                + ("" if model is not None else
                   " The model input is not wired, so no module could recognise it.")
                + " The module for this model is missing.")

        images = _plain_decode(vae, latent, tile_size)
        if len(images.shape) == 5:               # a batch of clips -> a strip of frames
            images = images.reshape(-1, *images.shape[-3:])
        return io.NodeOutput(images, None, "decoded as a single latent")
