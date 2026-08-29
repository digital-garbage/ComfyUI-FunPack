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
"""

from comfy_api.latest import io

from ..._core import log, registry as registry_mod

CAPABILITY = "decode"


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
                io.Vae.Input("audio_vae", optional=True,
                             tooltip="Only for models that generate sound alongside the video."),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
                io.Audio.Output(display_name="audio"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, samples, vae, audio_vae=None) -> io.NodeOutput:
        latent = samples["samples"]

        for spec, decode in registry_mod.current().providers(CAPABILITY):
            try:
                claimed = decode(latent, vae=vae, audio_vae=audio_vae)
            except Exception as exc:             # noqa: BLE001
                # Same protocol as the empty latent: returning None means "not my
                # model", so getting this far means it WAS and it broke. Falling
                # back would decode a nested latent as if it were one tensor.
                log.failed(f"{spec.id}.{CAPABILITY}", exc)
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
                "how to decode it. The module for this model is missing.")

        images = vae.decode(latent)
        if len(images.shape) == 5:               # a batch of clips -> a strip of frames
            images = images.reshape(-1, *images.shape[-3:])
        return io.NodeOutput(images, None, "decoded as a single latent")
