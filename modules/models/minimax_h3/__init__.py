"""MiniMax H3 support.

The first model-support module, and the shape every later one should copy: it
teaches the system to recognise H3 and to build H3's latent, and NOTHING outside
this folder mentions H3 at all. Supporting the next model is another folder like
this one.

Two things are contributed:

* **Traits** -- what is true of this model that core cannot read for itself. Core
  reads shape and prediction type off any model; that H3 carries an audio stream,
  and that it exposes per-modality AdaLN projections, are facts about this
  architecture.
* **An empty latent** -- because H3's genuinely cannot be derived. Its latent is a
  NestedTensor of a video branch and an audio branch; the video channel count (24)
  is not the `latent_channels` its format reports (32); and its length snaps to a
  frame grid at a fixed frame rate. None of that is in `latent_format`.

The frame grid is imported from ComfyUI rather than copied. Copying it would be a
second definition of the same rule, and the two would drift the first time
upstream touched theirs.
"""

import comfy.model_management
import torch
from comfy.nested_tensor import NestedTensor
from comfy_extras.nodes_minimax_h3 import temporal_shape

from ..._core import traits as _traits
from .pipeline import presets as _pipeline_presets

has_block = _traits.has_block

ID = "model_minimax_h3"
TITLE = "MiniMax H3"
STAGE = "load"
CATEGORY = "system"
STATUS = "experimental"

# Qualified on purpose: a bare class name matches anything built with a class of
# that name, and this decides which model gets H3's latent.
MODEL_CLASS = "comfy.ldm.minimax.model.MiniMaxH3Model"
ADALN_CLASS = "comfy.ldm.minimax.model.AdalnProj"

VIDEO_CHANNELS = 24
AUDIO_CHANNELS = 32
VIDEO_SPATIAL_RATIO = 16

# Mirrors comfy/model_detection.py's own signature for this architecture: key
# NAMES only, no shapes or values, so a checkpoint file can be identified by
# its safetensors header alone -- without loading a single weight.
#
# Only the two prefixes ComfyUI's own loading path can actually end up
# reading H3's tensors under: bare (a standalone unet-only file, where
# `unet_prefix_from_state_dict`'s guess strips nothing and the original
# unprefixed keys survive), and "model.diffusion_model." (the first, and
# only H3-relevant, candidate that function tries for a full-checkpoint
# layout). A bare "diffusion_model." (no "model." in front) was here too,
# copied verbatim from the old v4 probe this was ported from -- traced
# through comfy/model_detection.py and comfy/sd.py and confirmed it is not
# reachable: ComfyUI's own prefix guess for that layout falls through to
# "model.", strips nothing (no key starts with "model."), and the ORIGINAL
# "diffusion_model."-prefixed keys are what detect_unet_config then searches
# under key_prefix="" -- which never finds them, so ComfyUI's real loader
# would refuse a file this probe would have called H3. A false "yes" is
# exactly the failure this feature exists to prevent, so it is gone rather
# than left in on the chance some file happens to use it.
_PREFIXES = ("", "model.diffusion_model.")
_SIGNATURE_KEYS = ("video_patch_proj.weight", "audio_patch_proj.weight")


def is_h3(model) -> bool:
    return has_block(model, MODEL_CLASS)


def detect(keys) -> bool:
    """Whether a safetensors file's tensor names are MiniMax H3's own.

    Read before anything is loaded: this is what lets core refuse a
    mismatched checkpoint (or say what a file actually is) before queueing
    a run that would otherwise fail deep inside sampling, far from the
    mistake.
    """
    keyset = set(keys)
    return any(all(f"{prefix}{key}" in keyset for key in _SIGNATURE_KEYS)
               for prefix in _PREFIXES)


def traits(model):
    """What core cannot read off the model for itself."""
    if not is_h3(model):
        return ()
    found = ["audio_stream"]
    if has_block(model, ADALN_CLASS):
        found.append("adaln_modalities")
    return found


def decode(latent, model=None, vae=None, audio_vae=None):
    """H3's two branches, each through the VAE that understands it.

    Claims by IDENTITY, not by shape. "Two parts" describes plenty of models that
    are not this one, and the branch order is H3's own arrangement -- reading
    someone else's wrongly would decode noise as a picture rather than fail. When
    the model is not wired we decline rather than guess, and the node then says
    what is missing.
    """
    if model is None or not is_h3(model):
        return None
    if not getattr(latent, "is_nested", False):
        return None

    parts = latent.unbind()
    if len(parts) != 2:
        raise RuntimeError(
            f"expected a video and an audio branch, got {len(parts)} parts")
    video_latent, audio_latent = parts

    images = vae.decode(video_latent)
    if len(images.shape) == 5:
        images = images.reshape(-1, *images.shape[-3:])

    if audio_vae is None:
        # Silence would look like a decoded soundtrack. Say what is missing.
        raise RuntimeError(
            "this model generates sound and no audio VAE is wired into the "
            "decode, so there is nothing to turn the audio branch into")

    from comfy_extras.nodes_audio import vae_decode_audio
    audio = vae_decode_audio(audio_vae, {"samples": audio_latent})
    return images, audio


def empty_latent(model, width, height, length, batch_size=1):
    """H3's joint video+audio latent, or None if this is not an H3 model."""
    if not is_h3(model):
        return None

    frame_count, latent_t, audio_t = temporal_shape(length)
    device = comfy.model_management.intermediate_device()
    video = torch.zeros(
        [batch_size, VIDEO_CHANNELS, latent_t,
         max(1, height // VIDEO_SPATIAL_RATIO), max(1, width // VIDEO_SPATIAL_RATIO)],
        device=device)
    audio = torch.zeros([batch_size, AUDIO_CHANNELS, 2, audio_t], device=device)
    # Nested, not concatenated: the two branches have different ranks, and the
    # sampler reads them as one latent with two parts.
    return {"samples": NestedTensor((video, audio))}


TRAITS = traits
PROVIDES = {"empty_latent": empty_latent, "decode": decode, "detect": detect,
            "pipeline_presets": _pipeline_presets}
