"""Video -> audio coupling: how strongly the sound follows the picture.

LTX's blocks carry a trained cross-attention from the video stream into the audio stream. Scaling
its output scales how much the picture steers the sound: above 1 the sound follows the picture
harder (JoyAI's `v2a_grad_scale`), 0 mutes the link. Nothing is installed at 1.0, so it costs
nothing when off.
"""

from ..._core import patching

ID = "audio_coupling"
TITLE = "Video→audio coupling"
MOUNT = "generation.sampling"
STAGE = "guidance"
CATEGORY = "guidance"
STATUS = "experimental"
REQUIRES = ["ltx_av"]

SETTINGS = {
    "scale": {
        "type": "float", "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.25,
        "label": "Video→audio coupling", "ui": "slider",
        "hint": "How strongly the sound follows the picture. 1 = the model as trained, above = tighter, "
                "0 = the sound ignores the picture.",
    },
}

_SUB = "video_to_audio_attn"


def install(patcher, values, key):
    scale = float(values.get("scale", 1.0))
    if abs(scale - 1.0) < 1e-6:
        return None
    blocks = patcher.get_model_object("diffusion_model.transformer_blocks")
    wired = 0
    for index, block in enumerate(blocks):
        sub = getattr(block, _SUB, None)
        if sub is None:
            continue
        original = type(sub).forward.__get__(sub)

        def forward(*args, __original=original, **kwargs):
            out = __original(*args, **kwargs)
            if isinstance(out, tuple):
                return (out[0] * scale,) + tuple(out[1:])
            return out * scale

        guarded = patcher.guarded(forward, lambda *a, __original=original, **k: __original(*a, **k)) \
            if hasattr(patcher, "guarded") else forward
        wrapped = (lambda g: lambda *a, **k: g(*a, **k))(guarded)
        patching.tag(wrapped, key)
        patcher.add_object_patch(f"diffusion_model.transformer_blocks.{index}.{_SUB}.forward", wrapped)
        wired += 1
    if not wired:
        raise RuntimeError("this model's blocks have no video-to-audio attention to scale")
    return f"{scale:g}x across {wired} blocks"


PROVIDES = {"modifier": install}
