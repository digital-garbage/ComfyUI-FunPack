"""Video detail: crisper or softer picture, with the sound untouched.

Every block shares one attention pass, so anything that changes the picture
earlier also reaches the soundtrack. The final layer is past the last attention:
video and audio are modulated on separate rows and leave through separate heads,
so an edit here has no path to the audio at all.

WHAT is scaled matters. The layer computes `norm(x) * (1 + s) + shift`, and H3's
`s` is negative there, so scaling `s` ran the dial backwards (measured on a
rental in v4: 0.8 gave more contrast, 1.2 less). Scaling the norm's output --
the whole `(1 + s)` term's input -- is monotone whichever sign the checkpoint
carries. Upstream's own forward does everything else, so its PDD head bank and
signature changes are inherited, not copied.

v4 by eye: 1.25 read as more detail; reach for 1.4-1.8 before calling it inert.
"""

from ..._core import log

ID = "video_detail"
TITLE = "Video detail"
MOUNT = "generation.sampling"
STAGE = "post"
CATEGORY = "post"
STATUS = "experimental"
REQUIRES = ["adaln_modalities", "audio_stream"]

SETTINGS = {
    "enabled": {
        "type": "bool", "default": False,
        "label": "Picture detail",
        "hint": "Crisper or softer picture. The sound is not affected.",
    },
    "amount": {
        "type": "float", "default": 1.25, "min": 0.0, "max": 2.0, "step": 0.05,
        "label": "Amount", "ui": "slider",
        "hint": "Above 1 = more detail and contrast, below 1 = softer.",
        "when": {"enabled": True},
    },
}

_PARTS = ("norm", "adaln_proj", "video_out", "audio_out")
_KEY = "diffusion_model.final_layer.forward"


def install(patcher, values, key):
    if not values.get("enabled"):
        return None
    gain = float(values.get("amount", 1.0))
    if gain == 1.0:
        return None
    layer = patcher.get_model_object("diffusion_model.final_layer")
    if not all(hasattr(layer, part) for part in _PARTS):
        raise RuntimeError("this final layer is not the shape the edit was written against")
    original = type(layer).forward.__get__(layer)
    norm = layer.norm

    def forward(x, t_emb, video_seg, *args, **kwargs):
        va, vb = int(video_seg[0]), int(video_seg[1])
        if vb <= va:
            return original(x, t_emb, video_seg, *args, **kwargs)
        # The layer calls norm(x[a:b]) once per stream; the video call is the
        # one whose input IS the video slice. Matched by storage, not by order,
        # so an upstream reordering cannot scale the audio instead.
        video_ptr, rows = x[va:vb].data_ptr(), vb - va

        def scaled(inp, *a, **k):
            out = type(norm).forward(norm, inp, *a, **k)
            return out * gain if inp.data_ptr() == video_ptr and inp.shape[0] == rows else out

        norm.forward = scaled
        try:
            return original(x, t_emb, video_seg, *args, **kwargs)
        finally:
            del norm.forward

    guarded = patcher.guarded(forward, original) if hasattr(patcher, "guarded") else forward
    wrapped = lambda *a, **k: guarded(*a, **k)            # noqa: E731 -- taggable
    from ..._core import patching
    patching.tag(wrapped, key)
    patcher.add_object_patch(_KEY, wrapped)
    return f"{gain:g}x on the picture, past the last attention"


PROVIDES = {"modifier": install}
