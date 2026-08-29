"""Detail: sharpen or soften what the model is producing, while it produces it.

Declares NO traits, so it applies to every model -- image, video, anything that
ships later. That is the other half of the compatibility rule: ALG narrows itself
to models with a time axis, and this narrows itself to nothing.

It is also the modifier that can be SEEN on a laptop. Ratings, anchors and audio
clocks all need hardware or history before they mean anything; this changes the
picture on a 512x512 image in ten seconds, so "the setting did what it said" is
something to look at rather than infer.

The maths is ComfyUI's own `LatentOperationSharpen`, not a copy of it. The
technique here is FunPack's -- v4 measured this and settled on it -- but the
kernel is core's, tested by core, and a second implementation of a gaussian
unsharp mask would only be a second thing to keep correct.
"""

from ..._core import log, patching

ID = "sharpen"
TITLE = "Detail"
MOUNT = "generation.sampling"
STAGE = "sampling"
STATUS = "proven"

SETTINGS = {
    "enabled": {
        "type": "bool", "default": False,
        "label": "Adjust detail",
        "hint": "Sharpens or softens as the picture forms, not afterwards.",
    },
    # Range set from measurement, not from what looked like a round number:
    # 0.4 gives a visibly richer picture on SD1.5 that still reads correctly,
    # and 1.2 already blows out the highlights and smears the subject. A slider
    # whose top half destroys the image is a slider that lies about its range.
    "amount": {
        "type": "float", "default": 0.4, "min": -1.0, "max": 1.0, "step": 0.05,
        "label": "Amount", "ui": "slider",
        "hint": "Above zero sharpens, below zero softens.",
        "when": {"enabled": True},
    },
    "radius": {
        "type": "int", "default": 5, "min": 1, "max": 31, "step": 2,
        "label": "Spread", "ui": "slider",
        "hint": "How far the effect reaches from each detail.",
        "when": {"enabled": True},
    },
}


def _operation(radius, amount):
    """Core's sharpen kernel, asked for by name rather than reimplemented.

    `nodes_post_processing` is imported FIRST on purpose. ComfyUI has a genuine
    cycle between it and `nodes_latent` -- each imports from the other -- which
    resolves only because init_extra_nodes happens to reach post_processing
    first. Importing nodes_latent on its own raises ImportError, so a custom
    node that reaches for it has to break the cycle the same way.
    """
    import comfy_extras.nodes_post_processing  # noqa: F401  (breaks the cycle)
    from comfy_extras.nodes_latent import LatentOperationSharpen
    return LatentOperationSharpen.execute(
        sharpen_radius=int(radius), sigma=1.0, alpha=float(amount)).result[0]


def install(patcher, values, key):
    if not values.get("enabled"):
        return None

    amount = float(values.get("amount", 0.4))
    if amount == 0.0:
        # Zero is a request for nothing, and saying so beats reporting that a
        # modifier ran when it could not have changed a single pixel.
        log.alert("FunPack Detail", "enabled at amount 0, which changes nothing")
        return None

    radius = int(values.get("radius", 5))
    operation = _operation(radius, abs(amount))
    sharpening = amount > 0

    def pre_cfg(args):
        conds_out = args["conds_out"]
        # Applied to the guidance DIFFERENCE when there is one, exactly as
        # core's LatentApplyOperationCFG does -- operating on the raw
        # conditioned output instead would fight the guidance rather than
        # shape it.
        if len(conds_out) == 2:
            adjusted = operation(latent=(conds_out[0] - conds_out[1]))
            changed = adjusted + conds_out[1]
        else:
            changed = operation(latent=conds_out[0])
        conds_out[0] = changed if sharpening else (2 * conds_out[0] - changed)
        return conds_out

    # Tagged, because this hook is an anonymous entry in a list: without a
    # mark there is nothing to identify it by when it has to come off again.
    patcher.set_model_sampler_pre_cfg_function(patching.tag(pre_cfg, key))
    return f"{'sharpening' if sharpening else 'softening'} by {abs(amount)} at spread {radius}"


PROVIDES = {"modifier": install}
