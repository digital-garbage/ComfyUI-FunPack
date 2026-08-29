"""An empty latent, shaped by whatever model is wired in.

Two paths, and the order matters:

1. **A model's own module may claim it.** Some latents cannot be derived from
   anything ComfyUI publishes -- MiniMax H3's is a NestedTensor of a video and an
   audio branch, its video channel count is not the `latent_channels` its format
   reports, and its length snaps to a model-specific frame grid. So a model module
   may provide `empty_latent` and build it itself.
2. **Otherwise it is derived** from `latent_format`, which is correct for every
   single-tensor model: SDXL (4ch, /8), Wan21 (16ch, /8, /4) and LTXV (128ch,
   /32, /8) all come out right, including models that ship after this is written.

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
        samples = torch.zeros([batch_size, channels, max(1, length)], device=device)
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
                             tooltip="Frames. Ignored by models whose latent has no time axis."),
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
                # One module's mistake must not decide the shape for every model.
                log.failed(f"{spec.id}.{CAPABILITY}", exc)
                continue
            if claimed is not None:
                return io.NodeOutput(claimed, f"{spec.id} built this latent")

        return io.NodeOutput(derive(model, width, height, length, batch_size),
                             "shaped from the model's own latent format")
