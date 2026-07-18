"""Segmented detailing: ADetailer-style region refine for the Scene Chain sampler.

After a scene finishes denoising, CLIPSeg (text-prompted segmentation, the same
model family face-swap stacks use for occlusion masking) locates user-named
regions ("hands", "feet", ...) on a few decoded keyframes. The matched region is
cut out of the video latent as a spatiotemporal tube, pushed through Lightricks'
own trained LatentUpsampler (2x spatial, latent space - the official two-stage
pipeline's stage-2 model), re-noised to the official stage-2 re-entry sigma and
re-denoised for the tail schedule, then downscaled back to its original latent
size and pasted through the feathered CLIPSeg soft mask. Final resolution never
changes: the upsample exists only so the model can resolve structure (fingers,
edges) at a higher working resolution before the crop returns to the grid.

Boundary law: this is a visual-behavior op, so it lives on the Chain Sampler.
Audio is never modified - the refine pass carries the scene's audio stream so
joint attention sees a legal AV latent, but its audio output is discarded.

No-op guarantee: empty targets, no upsampler, or no detection above threshold
returns the input latent object untouched (bit-identical run).
"""

import math

import torch
import torch.nn.functional as F

# Official LTX-2.3 two-stage distilled workflow, stage-2 ManualSigmas: re-noise the
# upsampled latent to 0.85 and run the tail of the 8-step distilled schedule. Using
# Lightricks' published re-entry point instead of inventing one: too much noise and
# the region redraws (drifts from the surrounding frame), too little and no detail
# is gained.
STAGE2_SIGMAS = (0.85, 0.725, 0.421875, 0.0)

# CLIPSeg heat above this (post-sigmoid) counts as "the named thing is here".
DEFAULT_THRESHOLD = 0.35

# Tubes larger than this fraction of the frame are refused: at that size the pass
# stops being region detailing and becomes a second full render (cost ~= 4x area
# fraction x tail steps), which belongs to a different feature.
MAX_TUBE_AREA = 0.35

# Latent cells of context padding around the thresholded region, and the minimum
# tube edge. Padding gives the refine pass surrounding context to blend against;
# the minimum keeps the crop a legal latent for conv stacks and attention.
TUBE_PAD = 2
MIN_TUBE_EDGE = 4

# Keyframes decoded per scene for detection. Single latent frames decode cheaply
# through the causal VAE (treated as stills) - enough for a temporal-union tube.
DETECTION_FRAMES = 3

# Cond keys that tie a scene's conditioning to the full-frame latent layout
# (i2v anchors / guide keyframes). The crop is its own small latent with no
# appended guide frames, so these must not travel into the refine pass.
_LAYOUT_COND_KEYS = ("keyframe_idxs", "guiding_latent", "concat_latent_image", "concat_mask")

_CLIPSEG = None  # (processor, model), lazy - loaded on first enabled run only


def _log(debug, msg):
    if debug:
        print(f"[FunPackDetail] {msg}")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def load_latent_upsampler(model_name):
    """Load Lightricks' LatentUpsampler from models/latent_upscale_models.

    Mirrors the LTX branch of comfy's LatentUpscaleModelLoader (safe_load +
    config from safetensors metadata); the architecture itself ships in comfy
    core (comfy.ldm.lightricks.latent_upsampler), so there is nothing to port.
    """
    import json

    import comfy.model_management
    import comfy.utils
    import folder_paths
    from comfy.ldm.lightricks.latent_upsampler import LatentUpsampler

    model_path = folder_paths.get_full_path_or_raise("latent_upscale_models", model_name)
    sd, metadata = comfy.utils.load_torch_file(model_path, safe_load=True, return_metadata=True)
    if "post_upsample_res_blocks.0.conv2.bias" not in sd:
        raise ValueError(
            f"{model_name} is not an LTX latent upsampler (expected post_upsample_res_blocks keys).")
    config = json.loads(metadata["config"])
    model = LatentUpsampler.from_config(config).to(
        dtype=comfy.model_management.vae_dtype(allowed_dtypes=[torch.bfloat16, torch.float32]))
    model.load_state_dict(sd)
    model.eval()
    return model


def _get_clipseg():
    """CLIPSeg processor+model, loaded once and kept on CPU.

    CPU on purpose: ~150M params, a handful of 352px frames per scene - the cost
    is milliseconds against a video denoise, and it never competes with the
    diffusion model for VRAM. Loaded straight from transformers (which comfy
    already ships); no GPL face-swap code involved.
    """
    global _CLIPSEG
    if _CLIPSEG is None:
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

        name = "CIDAS/clipseg-rd64-refined"
        processor = CLIPSegProcessor.from_pretrained(name)
        model = CLIPSegForImageSegmentation.from_pretrained(name)
        model.eval()
        _CLIPSEG = (processor, model)
    return _CLIPSEG


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def parse_targets(detail_targets):
    """Comma-separated user text -> clean prompt list ('' -> [])."""
    if not detail_targets:
        return []
    return [t.strip() for t in str(detail_targets).split(",") if t.strip()]


def _decode_detection_frames(vae, video, count, debug=False):
    """Decode `count` evenly spaced single latent frames to [H, W, C] float images.

    Each latent frame is decoded alone as [B, C, 1, h, w]: the causal VAE treats
    a length-1 clip as a still, which is exactly good enough for segmentation and
    avoids decoding the whole scene just to find a region.
    """
    frames = []
    f = int(video.shape[2])
    idxs = sorted({min(f - 1, round(i * (f - 1) / max(1, count - 1))) for i in range(max(1, count))})
    for i in idxs:
        try:
            decoded = vae.decode(video[:, :, i:i + 1])
        except Exception as exc:
            _log(debug, f"keyframe {i} decode failed ({exc}); skipping")
            continue
        img = decoded
        if img.dim() == 5:  # [B, T, H, W, C]
            img = img[0, 0]
        elif img.dim() == 4:  # [B, H, W, C]
            img = img[0]
        frames.append((i, img.detach().float().clamp(0.0, 1.0).cpu()))
    return frames


def _clipseg_heat(frames, prompts, debug=False):
    """Max-over-frames, max-over-prompts CLIPSeg heat map.

    Returns [H, W] in [0, 1] at the first frame's pixel size, or None when
    nothing was decodable. Max (not mean) across frames on purpose: the tube
    must cover the region wherever it appears during the scene.
    """
    import numpy as np
    from PIL import Image

    if not frames or not prompts:
        return None
    processor, model = _get_clipseg()
    h, w = frames[0][1].shape[0], frames[0][1].shape[1]
    heat = None
    with torch.no_grad():
        for _, img in frames:
            pil = Image.fromarray((img.numpy() * 255.0).astype(np.uint8))
            inputs = processor(
                text=list(prompts), images=[pil] * len(prompts), return_tensors="pt")
            logits = model(**inputs).logits  # [P, 352, 352] (or [352, 352] for one prompt)
            if logits.dim() == 2:
                logits = logits[None]
            frame_heat = torch.sigmoid(logits).amax(dim=0)[None, None]  # union over prompts
            frame_heat = F.interpolate(frame_heat, size=(h, w), mode="bilinear", align_corners=False)[0, 0]
            heat = frame_heat if heat is None else torch.maximum(heat, frame_heat)
    return heat


def find_tube(heat, latent_h, latent_w, threshold=DEFAULT_THRESHOLD,
              area_cap=MAX_TUBE_AREA, debug=False):
    """Threshold pixel-space heat into a latent-space tube box + soft paste mask.

    Returns (y0, y1, x0, x1, mask) in latent coords - mask is [1, 1, 1, th, tw]
    matching the tube, feathered from the CLIPSeg heat itself so the paste
    follows the actual silhouette instead of the rectangle. None when nothing
    crossed the threshold or the tube breached the area cap.
    """
    if heat is None:
        return None
    # Heat -> latent grid first; box math happens where the crop happens.
    lat_heat = F.interpolate(heat[None, None], size=(latent_h, latent_w),
                             mode="bilinear", align_corners=False)[0, 0]
    hot = lat_heat >= threshold
    if not bool(hot.any()):
        _log(debug, f"no region above threshold {threshold}")
        return None
    ys, xs = torch.where(hot)
    y0 = max(0, int(ys.min()) - TUBE_PAD)
    y1 = min(latent_h, int(ys.max()) + 1 + TUBE_PAD)
    x0 = max(0, int(xs.min()) - TUBE_PAD)
    x1 = min(latent_w, int(xs.max()) + 1 + TUBE_PAD)
    # Enforce a minimum edge (grow symmetrically, clamped to the frame).
    while (y1 - y0) < min(MIN_TUBE_EDGE, latent_h):
        y0, y1 = max(0, y0 - 1), min(latent_h, y1 + 1)
    while (x1 - x0) < min(MIN_TUBE_EDGE, latent_w):
        x0, x1 = max(0, x0 - 1), min(latent_w, x1 + 1)
    area = (y1 - y0) * (x1 - x0) / float(latent_h * latent_w)
    if area > area_cap:
        _log(debug, f"tube area {area:.0%} exceeds cap {area_cap:.0%}; refusing "
                    "(a region that large is a re-render, not a detail pass)")
        return None
    # Soft mask: the heat inside the tube, floored at 0 outside the threshold and
    # lightly blurred so the seam never lands on a hard edge.
    m = lat_heat[y0:y1, x0:x1].clone()
    m = ((m - threshold) / max(1e-6, 1.0 - threshold)).clamp(0.0, 1.0)
    m = F.avg_pool2d(m[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
    return y0, y1, x0, x1, m[None, None, None]


# ---------------------------------------------------------------------------
# Refine
# ---------------------------------------------------------------------------

def _strip_layout_conds(conds):
    """Copy conditioning minus full-frame layout keys (anchors/guides).

    The crop refine is its own small denoise: text still applies, but keyframe
    indices and guide latents describe positions in the full-frame latent and
    would fight (or crash) the crop-sized pass.
    """
    out = []
    for entry in conds or []:
        if isinstance(entry, (list, tuple)) and len(entry) == 2 and isinstance(entry[1], dict):
            meta = {k: v for k, v in entry[1].items() if k not in _LAYOUT_COND_KEYS}
            out.append([entry[0], meta])
        else:
            out.append(entry)
    return out


def _per_channel_stats(vae):
    return getattr(getattr(vae, "first_stage_model", None), "per_channel_statistics", None)


def _run_upsampler(upsampler, video_crop, vae, debug=False):
    """un_normalize -> LatentUpsampler(2x spatial) -> normalize, on the crop.

    Same recipe as comfy's LTXVLatentUpsampler node: the model was trained on
    raw (denormalized) VAE latents, so the trip through per_channel_statistics
    is mandatory, not cosmetic.
    """
    import comfy.model_management

    device = comfy.model_management.get_torch_device()
    model_dtype = next(upsampler.parameters()).dtype
    in_dtype, in_device = video_crop.dtype, video_crop.device
    stats = _per_channel_stats(vae)
    try:
        upsampler.to(device)
        x = video_crop.to(device=device, dtype=model_dtype)
        if stats is not None:
            x = stats.un_normalize(x)
        with torch.no_grad():
            x = upsampler(x)
        if stats is not None:
            x = stats.normalize(x)
    finally:
        upsampler.cpu()
    return x.to(device=in_device, dtype=in_dtype)


def _downscale_to(video, h, w):
    """Area-downscale [B, C, F, H, W] spatially to (h, w); frames independent."""
    b, c, f, hh, ww = video.shape
    x = video.permute(0, 2, 1, 3, 4).reshape(b * f, c, hh, ww)
    x = F.interpolate(x.float(), size=(h, w), mode="area")
    return x.reshape(b, f, c, h, w).permute(0, 2, 1, 3, 4).to(dtype=video.dtype)


def detail_refine_scene(chain, model, vae, sampler, positive, negative, latent,
                        detail_targets, upsampler, seed, cfg,
                        threshold=DEFAULT_THRESHOLD, strength=1.0, debug=False):
    """Run the segmented-detail pass on one finished scene latent.

    Returns (latent, note): the refined latent dict plus a run_mechanisms note,
    or (latent, None) untouched - same object - when disabled or nothing found.
    `chain` is the FunPackLTXAVSceneChainSampler instance (reuses _latent_tensors
    and _sample_chunk so the refine denoise goes through the exact same path as
    the scene it is polishing).
    """
    targets = parse_targets(detail_targets)
    if not targets or upsampler is None or strength <= 0:
        return latent, None

    tensors = chain._latent_tensors(latent)
    video = tensors[0]
    if video.dim() != 5:
        _log(debug, f"unexpected video latent rank {video.dim()}; skipping")
        return latent, None
    b, c, f, lat_h, lat_w = video.shape

    # 1) Detect on a few decoded keyframes.
    frames = _decode_detection_frames(vae, video, DETECTION_FRAMES, debug=debug)
    heat = _clipseg_heat(frames, targets, debug=debug)
    tube = find_tube(heat, lat_h, lat_w, threshold=threshold, debug=debug)
    if tube is None:
        return latent, None
    y0, y1, x0, x1, mask = tube
    _log(debug, f"tube y[{y0}:{y1}] x[{x0}:{x1}] of {lat_h}x{lat_w} for {targets}")

    # 2) Crop -> 2x latent upsample (Lightricks stage-2 model).
    crop = video[:, :, :, y0:y1, x0:x1]
    crop_up = _run_upsampler(upsampler, crop, vae, debug=debug)

    # 3) Re-denoise the crop over the official stage-2 tail. The clean crop goes
    # in as the latent; CONST noise_scaling (sigma*noise + (1-sigma)*x) re-noises
    # it to sigma=0.85 exactly like the official two-stage workflow's stage 2.
    # The scene's audio stream rides along so joint attention sees a legal AV
    # latent, but its refined output is discarded below - audio is protected by
    # construction, not by masking.
    import comfy.nested_tensor

    crop_latent = {"samples": crop_up}
    if len(tensors) > 1:
        crop_latent["samples"] = comfy.nested_tensor.NestedTensor(
            [crop_up] + [t.detach().clone() for t in tensors[1:]])
    tail_sigmas = torch.tensor(STAGE2_SIGMAS, dtype=torch.float32)
    refined = chain._sample_chunk(
        model, sampler, tail_sigmas, int(seed) + 7777, cfg,
        _strip_layout_conds(positive), _strip_layout_conds(negative), crop_latent)
    refined_crop = chain._latent_tensors(refined)[0]

    # 4) Back to grid size, feathered paste through the CLIPSeg silhouette.
    refined_down = _downscale_to(refined_crop, y1 - y0, x1 - x0)
    m = (mask.to(device=video.device, dtype=video.dtype) * float(strength)).clamp(0.0, 1.0)
    out_video = video.clone()
    region = out_video[:, :, :, y0:y1, x0:x1]
    out_video[:, :, :, y0:y1, x0:x1] = region + (refined_down.to(region.device) - region) * m

    out = dict(latent)
    if len(tensors) > 1:
        out["samples"] = comfy.nested_tensor.NestedTensor([out_video] + list(tensors[1:]))
    else:
        out["samples"] = out_video
    note = (f"segmented_detail({','.join(targets)}, tube={y1 - y0}x{x1 - x0}"
            f"/{lat_h}x{lat_w}, s={strength})")
    return out, note
