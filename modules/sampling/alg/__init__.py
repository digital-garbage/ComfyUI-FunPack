"""ALG: stop an i2v model copying its anchor frame.

An image-to-video model is shown a sharp reference at step 0 and can satisfy the
objective by barely moving -- the result matches the reference and hardly
animates. ALG (arXiv 2506.08456) low-passes the anchor for the early, high-sigma
steps, removing the detail that shortcut depends on, and swaps the sharp anchor
back once sigma drops so the later steps still resolve real detail.

The first module that is only behaviour: settings and a modifier, no ComfyUI node
of its own. It attaches to the MODEL, so it applies to whatever sampler is wired.

`REQUIRES` is `temporal_latent` and nothing more. It is not "for LTX" or "for
H3" -- it needs a latent with a time axis and a pinned anchor, and any model with
both gets it, including ones nobody here has heard of.

KNOWN ASSUMPTION: the anchor is latent frame 0. That is true of every i2v setup
this can currently be wired into, and it is what v4 assumed. It is NOT true in
general -- H3 can pin a keyframe at any frame -- so if an interior-pin mechanism
is ever ported, ALG must take the frame indices from the pin rather than assume
the first. The information to do it properly is already present in the
denoise_mask, which is what marks the locked frames; deriving from it is left
undone deliberately, because there is nothing yet that pins anywhere else and a
derivation with no consumer cannot be checked against reality.

What is NOT ported from v4: the packed-layout path, which reached into a
model-specific arrangement of the latent to find the video stream. Where the
anchor is not a plain [B, C, T, H, W] tensor this stands down and says so once,
rather than reshaping something it is guessing about. A model module can teach it
that layout later.
"""

from comfy.patcher_extension import WrappersMP

from ..._core import log
from .blur import AnchorSwap, blur_frames

ID = "alg"
TITLE = "Anchor de-staticking"
MOUNT = "generation.sampling"
STAGE = "sampling"
STATUS = "proven"
REQUIRES = ["temporal_latent"]

SETTINGS = {
    "enabled": {
        "type": "bool", "default": False,
        "label": "Loosen the starting image",
        "hint": "Stops a still first frame from holding the whole clip still.",
    },
    "strength": {
        "type": "float", "default": 4.0, "min": 1.0, "max": 16.0, "step": 0.5,
        "label": "How much", "ui": "slider",
        "hint": "Higher frees up more motion and keeps less of the original detail.",
        "when": {"enabled": True},
    },
    "until_sigma": {
        "type": "float", "default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05,
        "label": "How long for", "ui": "slider",
        "hint": "The starting image comes back sharp once the picture settles.",
        "when": {"enabled": True},
    },
}

_said = set()


def _once(message: str) -> None:
    """Say an inert reason once per process, not once per step."""
    if message not in _said:
        _said.add(message)
        log.note(message)


def install(patcher, values, key):
    """Put the anchor swap on the model, for whatever sampler runs it."""
    if not values.get("enabled"):
        return None

    strength = float(values.get("strength", 4.0))
    threshold = float(values.get("until_sigma", 0.6))

    def wrapper(executor, model_wrap, sigmas, extra_args, callback, noise,
                latent_image=None, denoise_mask=None, disable_pbar=False):
        sampler = getattr(executor, "class_obj", None)
        inner_fn = getattr(sampler, "sampler_function", None)

        # No anchor pinned means nothing to de-static; a sampler that is not
        # KSAMPLER-shaped has no loop to stand between.
        if inner_fn is None or latent_image is None or denoise_mask is None:
            _once("[FunPack] ALG is off this run: no pinned anchor to loosen.")
            return executor(model_wrap, sigmas, extra_args, callback, noise,
                            latent_image, denoise_mask, disable_pbar)

        blurred = blur_frames(latent_image, strength, frame_indices=(0,))
        if blurred is None:
            _once("[FunPack] ALG is off this run: this model's anchor is not a "
                  "plain video latent, and reshaping it would be guesswork.")
            return executor(model_wrap, sigmas, extra_args, callback, noise,
                            latent_image, denoise_mask, disable_pbar)

        def alg_sampler_function(model_k, x, step_sigmas, extra_args=None,
                                 callback=None, disable=None, **options):
            proxy = AnchorSwap(model_k, latent_image, blurred, threshold)
            try:
                return inner_fn(proxy, x, step_sigmas, extra_args=extra_args,
                                callback=callback, disable=disable, **options)
            finally:
                # Whatever happened, the real denoiser goes back to the sharp
                # anchor: it outlives this call and the next reader must not
                # inherit a blurred one.
                model_k.latent_image = latent_image

        import comfy.samplers
        replacement = comfy.samplers.KSAMPLER(
            alg_sampler_function,
            extra_options=getattr(sampler, "extra_options", {}),
            inpaint_options=getattr(sampler, "inpaint_options", {}),
        )
        return replacement.sample(model_wrap, sigmas, extra_args, callback, noise,
                                  latent_image, denoise_mask, disable_pbar)

    patcher.add_wrapper_with_key(WrappersMP.SAMPLER_SAMPLE, key, wrapper)
    return f"loosening the starting image (amount {strength}, until sigma {threshold})"


PROVIDES = {"modifier": install}
