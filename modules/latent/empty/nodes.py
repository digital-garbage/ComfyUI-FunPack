"""An empty latent, shaped by whatever model is wired in.

Two paths, and the order matters:

1. **A model's own module may claim it.** Some latents cannot be derived from
   anything ComfyUI publishes -- MiniMax H3's is a NestedTensor of a video and an
   audio branch, its video channel count is not the `latent_channels` its format
   reports, and its length snaps to a model-specific frame grid. So a model module
   may provide `empty_latent` and build it itself.
2. **Otherwise it is derived** from `latent_format`, which is correct for every
   single-tensor model with a spatial latent: SDXL (4ch, /8), Wan21 (16ch, /8,
   /4) and LTXV (128ch, /32, /8) all come out right, including models that ship
   after this is written.

A one-axis latent is neither: its length is not published anywhere, and the same
rank means seconds of audio for StableAudio and tokens of geometry for
Hunyuan3D. That is refused here rather than guessed, on the same reasoning as
point 1 -- deriving it produced a latent a thousandth of the length asked for and
reported success.

The provider protocol, and it matters: **return None for "not my model", and let
anything else raise.** A provider decides whether it recognises the model FIRST,
before touching anything that can fail, so an exception means it had already
claimed the model. Such a failure therefore stops the node instead of falling
through to the derivation -- deriving a claimed model produces a latent that is
quietly wrong rather than absent. For H3 at length=124 the derivation gives 31
latent frames where the real grid is 37: a 16% shorter video, reported as success.

The consequence worth stating: nothing here knows the name of a single model, and
nothing here assumes video. A 2-D latent format yields an image latent because
that is what the model says it wants.
"""

import comfy.model_management
import torch
from comfy_api.latest import io

from ..._core import log, registry as registry_mod

CAPABILITY = "empty_latent"


def latent_format_of(model):
    inner = getattr(model, "model", None)
    config = getattr(inner, "model_config", None)
    return getattr(config, "latent_format", None)


def derive(model, width, height, length, batch_size):
    """The generic shape, from what the model publishes about its own latent."""
    fmt = latent_format_of(model)
    if fmt is None:
        raise RuntimeError(
            "This model does not publish a latent format, so its latent shape "
            "cannot be worked out. A model module can supply one.")

    channels = int(getattr(fmt, "latent_channels", 4))
    spatial = int(getattr(fmt, "spacial_downscale_ratio", 8) or 8)
    temporal = int(getattr(fmt, "temporal_downscale_ratio", 1) or 1)
    rank = getattr(fmt, "latent_dimensions", 2)

    # Never below one: a small size divided by a large ratio floors to zero, and
    # a zero-sized latent fails somewhere much further along.
    w = max(1, width // spatial)
    h = max(1, height // spatial)
    device = comfy.model_management.intermediate_device()

    if rank == 3:
        frames = ((max(1, length) - 1) // temporal) + 1
        samples = torch.zeros([batch_size, channels, frames, h, w], device=device)
    elif rank == 1:
        # Refused, not guessed. A one-axis latent's length is not derivable from
        # anything the format publishes: core's own EmptyLatentAudio turns
        # SECONDS into samples with a rate and a hop that live in the node
        # (44100 and 2048), not in latent_format, and EmptyLatentHunyuan3Dv2
        # counts tokens for a resolution -- the same rank meaning two unrelated
        # things. Building `length` samples produced a latent three orders of
        # magnitude short of a soundtrack and called it a success.
        raise RuntimeError(
            "This model's latent has a single axis whose length is not derivable "
            "from what the model publishes -- seconds, sample rate and hop are "
            "not in its latent format. A module for this model can supply it.")
    else:
        samples = torch.zeros([batch_size, channels, h, w], device=device)

    # These say what ratios the latent was BUILT at, so the sampler can rescale
    # if the model wants different ones (comfy.sample.fix_empty_latent_channels).
    # Reporting what we actually used is right by construction.
    return {"samples": samples,
            "downscale_ratio_spacial": spatial,
            "downscale_ratio_temporal": temporal}


class FunPackEmptyLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="FunPackEmptyLatent",
            display_name="FunPack Empty Latent",
            category="FunPack/Latent",
            description="An empty latent shaped for the wired model, image or video.",
            inputs=[
                io.Model.Input("model", tooltip="The latent's shape comes from this model."),
                io.Int.Input("width", default=768, min=16, max=16384, step=16),
                io.Int.Input("height", default=512, min=16, max=16384, step=16),
                io.Int.Input("length", default=1, min=1, max=16384,
                             tooltip="Frames. Ignored by models whose latent has no "
                                     "time axis, and it is frames -- not seconds and "
                                     "not samples."),
                io.Int.Input("batch_size", default=1, min=1, max=4096, optional=True),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, model, width: int, height: int, length: int,
                batch_size: int = 1) -> io.NodeOutput:
        for spec, build in registry_mod.current().providers(CAPABILITY):
            try:
                claimed = build(model, width=width, height=height,
                                length=length, batch_size=batch_size)
            except Exception as exc:             # noqa: BLE001
                # It got past recognising the model, so this IS its model and it
                # broke. Carrying on to the derivation would answer a question
                # this module had already claimed, with a shape that is wrong in
                # a way nothing downstream reports.
                # Not log.failed: that word means "did not load", and this one
                # loaded, claimed the model, and broke while working. Reporting
                # it as absent sends whoever reads the log looking for an import
                # error that never happened.
                log.warning(f"{spec.id}.{CAPABILITY}",
                            f"handles this model's latent and failed while building it "
                            f"-- {type(exc).__name__}: {exc}")
                raise RuntimeError(
                    f"{spec.id} handles this model's latent and failed to build it: "
                    f"{type(exc).__name__}: {exc}. Refusing to substitute a generic "
                    f"shape, which would be silently wrong for this model."
                ) from exc
            if claimed is not None:
                return io.NodeOutput(claimed, f"{spec.id} built this latent")

        return io.NodeOutput(derive(model, width, height, length, batch_size),
                             "shaped from the model's own latent format")
