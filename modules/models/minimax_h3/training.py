"""What LoRA training needs to know about H3, handed to the training module when it asks.

H3's latent is a video part and a sound part packed as one nested tensor, on a 17k+5 frame
grid at 24 fps, and its picture-and-sound forward uses a fused op that has no gradient in
comfy-kitchen 0.2.38. ComfyUI's own trainer knows none of that, so the two small repairs it
needs live here, applied only for the length of a training run and only when probed to be needed.
Proven on a tiny H3 on CPU (the loss moves, a LoRA comes out). NOT yet run on the real weights.
"""

import contextlib

import torch
from comfy.nested_tensor import NestedTensor
from comfy_extras.nodes_minimax_h3 import FPS, align_frame_count, temporal_shape

AUDIO_CHANNELS = 32
SPATIAL_STEP = 32


def _differentiable() -> bool:
    """Whether the fused rotary op the forward picks while training can carry a gradient."""
    from comfy import quant_ops
    x = torch.randn(1, 2, 2, 4, requires_grad=True)
    try:
        out = quant_ops.ck.rms_rope_split_half(x, torch.ones(4), torch.ones(1, 2, 1, 2), torch.zeros(1, 2, 1, 2), 1e-6)
        out = out[0] if isinstance(out, tuple) else out
        return bool(out.requires_grad)
    except Exception:                                   # noqa: BLE001 -- any refusal means "swap it"
        return False


@contextlib.contextmanager
def shims():
    """The two repairs, put back afterwards: the trainer accepts a nested latent, and the rotary op is differentiable."""
    import comfy_extras.nodes_train as nt
    from comfy import quant_ops
    original_count = nt._prepare_latents_and_count
    original_rope = quant_ops.ck.rms_rope_split_half

    def count(latents, dtype, bucket_mode):
        if isinstance(latents, NestedTensor):
            return latents.to(dtype), latents.tensors[0].shape[0], False
        return original_count(latents, dtype, bucket_mode)

    nt._prepare_latents_and_count = count
    try:
        if not _differentiable():
            from comfy_kitchen.backends.eager import rope
            quant_ops.ck.rms_rope_split_half = rope.rms_rope_split_half
        yield
    finally:
        nt._prepare_latents_and_count = original_count
        quant_ops.ck.rms_rope_split_half = original_rope


def pack(videos, sounds):
    """One latent dict from N video latents [1,24,t,h,w] and N sound latents [1,32,2,Ta], batched as H3 expects."""
    return {"samples": NestedTensor((torch.cat(videos, 0), torch.cat(sounds, 0)))}


def silence_latent(audio_vae, seconds: float):
    """The sound latent of `seconds` of silence, for a picture or a clip with no sound track."""
    rate = int(getattr(audio_vae, "audio_sample_rate", 44100))
    wave = torch.zeros(1, 2, max(1, int(seconds * rate)))
    return audio_vae.encode(wave.movedim(1, -1))


def adapt(model):
    """None for any model but H3. For H3, the facts the training module plans around."""
    from . import is_h3
    if not is_h3(model):
        return None
    return {"name": "H3", "step": SPATIAL_STEP, "video": True, "batch_limit": 1, "fps": FPS,
            "checkpoint_depth": 2,                              # one checkpoint per block and one for the text refiner: depth 1 would be the whole model as a single piece
            "frames": lambda n: align_frame_count(max(5, int(n))),
            "temporal": temporal_shape, "shims": shims, "pack": pack, "silence": silence_latent}
