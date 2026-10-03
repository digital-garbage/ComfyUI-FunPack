"""LTX 2.x (audio+video) support.

The second model-support module, built the way the first was: it teaches the system to recognise
this model, build its latent and decode it, and offers pipelines for it. Nothing outside this
folder mentions LTX.

What it contributes, and what it deliberately does NOT:

* **Traits** -- a joint audio+video model with a compressed time axis. It does NOT announce
  `dit_block_hooks`: the block hooks other modules install (attention temperature, block repeat...)
  are written for H3's packed block layout, and LTX's blocks take `(video, audio)` as a pair. Until
  a module is written for that layout it is absent here, said, rather than half-working.
* **An empty latent** -- video 128 channels at /32 and /8, plus an audio latent whose length comes
  from the AUDIO VAE and the frame rate. That is why this provider asks for `audio_vae` and
  `frame_rate`: without them the audio half cannot be sized, and a video-only latent would make a
  silent clip while reporting success, so it refuses instead.
* **Decode** -- the picture through the video VAE, the sound through the audio VAE.

Not validated on a real LTX model yet: the shapes are tested against fakes and ComfyUI's own
stock nodes, and the pipelines against the graph builder.
"""

import comfy.model_management
import torch
from comfy.nested_tensor import NestedTensor

from ..._core import log, traits as _traits
from .pipeline import presets as _pipeline_presets

has_block = _traits.has_block

ID = "model_ltx"
TITLE = "LTX 2.x"
STAGE = "load"
CATEGORY = "system"
STATUS = "experimental"

# Qualified: a bare class name matches anything built with a class of that name.
MODEL_CLASS = "comfy.ldm.lightricks.av_model.LTXAVModel"

VIDEO_CHANNELS = 128
VIDEO_SPATIAL_RATIO = 32
VIDEO_TEMPORAL_RATIO = 8

# ComfyUI tells the audio+video architecture from the video-only one by this key
# (comfy/model_detection.py: `audio_adaln_single.linear.weight`).
_PREFIXES = ("", "model.diffusion_model.")
_SIGNATURE_KEYS = ("audio_adaln_single.linear.weight",)


def is_ltx(model) -> bool:
    return has_block(model, MODEL_CLASS)


def detect(keys) -> bool:
    """Whether a checkpoint's tensor names are an LTX audio+video model's, from its header alone."""
    keyset = set(keys)
    return any(all(f"{prefix}{key}" in keyset for key in _SIGNATURE_KEYS) for prefix in _PREFIXES)


def probe_traits(keys) -> list:
    """What `traits()` would say, read before anything loads."""
    if not detect(keys):
        return []
    return ["audio_stream", "temporal_latent", "temporal_compression"]


def traits(model):
    if not is_ltx(model):
        return ()
    return ["audio_stream"]


def frames_for(length: int) -> int:
    """LTX makes 8k+1 frames; a length off that grid is rounded UP to it."""
    length = max(1, int(length))
    return ((length - 1 + VIDEO_TEMPORAL_RATIO - 1) // VIDEO_TEMPORAL_RATIO) * VIDEO_TEMPORAL_RATIO + 1


def empty_latent(model, width, height, length, batch_size=1, audio_vae=None, frame_rate=25.0):
    """The joint latent, or None when this is not an LTX audio+video model."""
    if not is_ltx(model):
        return None
    if audio_vae is None:
        raise RuntimeError(
            "this model makes sound with the picture and sizes it from the audio VAE: wire the audio "
            "VAE into the empty latent (a video-only latent would give a silent clip)")

    frames = frames_for(length)
    device = comfy.model_management.intermediate_device()
    video = torch.zeros(
        [batch_size, VIDEO_CHANNELS, ((frames - 1) // VIDEO_TEMPORAL_RATIO) + 1,
         max(1, height // VIDEO_SPATIAL_RATIO), max(1, width // VIDEO_SPATIAL_RATIO)],
        device=device)
    z_channels = audio_vae.latent_channels
    bins = audio_vae.first_stage_model.latent_frequency_bins
    steps = audio_vae.first_stage_model.num_of_latents_from_frames(frames, frame_rate)
    audio = torch.zeros((batch_size, z_channels, steps, bins), device=device)
    return {"samples": NestedTensor((video, audio)),
            "downscale_ratio_spacial": VIDEO_SPATIAL_RATIO,
            "downscale_ratio_temporal": VIDEO_TEMPORAL_RATIO}


def decode(latent, model=None, vae=None, audio_vae=None, tile_size=0):
    """The picture and the sound, each through the VAE that understands it."""
    if model is None or not is_ltx(model):
        return None
    if not getattr(latent, "is_nested", False):
        return None
    parts = latent.unbind()
    if len(parts) != 2:
        raise RuntimeError(f"expected a video and an audio part, got {len(parts)}")
    video_latent, audio_latent = parts
    if audio_vae is None:
        raise RuntimeError("this model generates sound and no audio VAE is wired into the decode")

    if tile_size and tile_size > 0:
        try:
            images = vae.decode_tiled(video_latent, tile_x=max(1, tile_size // VIDEO_SPATIAL_RATIO),
                                      tile_y=max(1, tile_size // VIDEO_SPATIAL_RATIO))
        except Exception as exc:                             # noqa: BLE001
            log.once(f"ltx_decode_tiles:{type(exc).__name__}", log.ALERT, "FunPack LTX decode",
                     f"the tiled decode failed ({type(exc).__name__}: {exc}); decoded in one piece instead")
            images = vae.decode(video_latent)
    else:
        images = vae.decode(video_latent)
    if len(images.shape) == 5:
        images = images.reshape(-1, *images.shape[-3:])

    waveform = audio_vae.decode(audio_latent).movedim(-1, 1).to(audio_latent.device)
    audio = {"waveform": waveform, "sample_rate": int(audio_vae.first_stage_model.output_sample_rate)}
    return images, audio


TRAITS = traits
PROVIDES = {"empty_latent": empty_latent, "decode": decode, "detect": detect,
            "probe_traits": probe_traits, "pipeline_presets": _pipeline_presets}
