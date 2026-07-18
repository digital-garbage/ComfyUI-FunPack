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

# Official Lightricks stage-2 spatial upscaler — the exact file the two-stage
# distilled workflows use. ~1 GB, fetched once into models/latent_upscale_models
# when nothing suitable is installed.
DEFAULT_UPSAMPLER_REPO = "Lightricks/LTX-2.3"
DEFAULT_UPSAMPLER_FILE = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"


def resolve_upsampler_name(model_name=None):
    """Turn the node's detail_upsampler widget value into a loadable filename.

    Explicit filename -> returned as-is. "auto" (and the legacy "None" default)
    -> prefer an installed spatial upscaler (highest version wins), else any
    installed latent upscale model, else download the official file from HF.
    Raises with a user-actionable message when nothing can be obtained — callers
    surface it, they must NOT silently skip.
    """
    import os

    import folder_paths

    name = str(model_name or "").strip()
    if name and name not in ("None", "auto"):
        return name
    files = folder_paths.get_filename_list("latent_upscale_models")
    if DEFAULT_UPSAMPLER_FILE in files:
        return DEFAULT_UPSAMPLER_FILE
    spatial = [f for f in files if "spatial" in f.lower() and "upscal" in f.lower()]
    if spatial:
        # Highest embedded version wins (…-1.1 over …-1.0); plain lexicographic would
        # rank "ltx2_3_…" above "ltx-2.3-…-1.1" ('2' sorts after '-').
        import re
        return max(spatial, key=lambda f: ([int(n) for n in re.findall(r"\d+", f)], f))
    if files:
        return files[0]
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "no model in models/latent_upscale_models and huggingface_hub is not "
            f"installed — pip install huggingface_hub, or download {DEFAULT_UPSAMPLER_FILE} "
            f"from {DEFAULT_UPSAMPLER_REPO} into that folder manually") from exc
    dest_dir = folder_paths.get_folder_paths("latent_upscale_models")[0]
    os.makedirs(dest_dir, exist_ok=True)
    print(f"[FunPackDetail] no latent upscale model installed — downloading "
          f"{DEFAULT_UPSAMPLER_FILE} (~1 GB) from {DEFAULT_UPSAMPLER_REPO} into {dest_dir} …")
    try:
        hf_hub_download(repo_id=DEFAULT_UPSAMPLER_REPO, filename=DEFAULT_UPSAMPLER_FILE,
                        local_dir=dest_dir)
    except Exception as exc:
        raise RuntimeError(
            f"auto-download of {DEFAULT_UPSAMPLER_FILE} failed ({exc}) — if the repo is "
            "gated, accept the license on huggingface.co and set HF_TOKEN, or place the "
            f"file in {dest_dir} manually") from exc
    print(f"[FunPackDetail] download complete: {DEFAULT_UPSAMPLER_FILE}")
    return DEFAULT_UPSAMPLER_FILE


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
            # padding=True is mandatory once len(prompts) > 1: CLIPSeg's tokenizer
            # otherwise returns one variable-length id list per prompt and torch can't
            # stack them into a batch ("excessive nesting... type list where int is
            # expected"). truncation=True guards the rare very-long user-typed target.
            inputs = processor(
                text=list(prompts), images=[pil] * len(prompts), return_tensors="pt",
                padding=True, truncation=True)
            logits = model(**inputs).logits  # [P, 352, 352] (or [352, 352] for one prompt)
            if logits.dim() == 2:
                logits = logits[None]
            frame_heat = torch.sigmoid(logits).amax(dim=0)[None, None]  # union over prompts
            frame_heat = F.interpolate(frame_heat, size=(h, w), mode="bilinear", align_corners=False)[0, 0]
            heat = frame_heat if heat is None else torch.maximum(heat, frame_heat)
    return heat


def find_tube(heat, latent_h, latent_w, threshold=DEFAULT_THRESHOLD,
              area_cap=MAX_TUBE_AREA, debug=False, diag=None):
    """Threshold pixel-space heat into a latent-space tube box + soft paste mask.

    Returns (y0, y1, x0, x1, mask) in latent coords - mask is [1, 1, 1, th, tw]
    matching the tube, feathered from the CLIPSeg heat itself so the paste
    follows the actual silhouette instead of the rectangle. None when nothing
    crossed the threshold or the tube breached the area cap.

    `diag`, when given a dict, is filled with why a None came back (max_heat
    always; reason + area on a miss) so a caller can report something more
    useful than "nothing found" - CLIPSeg's raw sigmoid score for a real,
    correctly-named region is often well under a naive 0.5, so a silent miss
    is indistinguishable from "the threshold is simply too high" without it.
    """
    if heat is None:
        return None
    # Heat -> latent grid first; box math happens where the crop happens.
    lat_heat = F.interpolate(heat[None, None], size=(latent_h, latent_w),
                             mode="bilinear", align_corners=False)[0, 0]
    max_heat = float(lat_heat.max())
    if diag is not None:
        diag["max_heat"] = max_heat
    hot = lat_heat >= threshold
    if not bool(hot.any()):
        _log(debug, f"no region above threshold {threshold} (max heat seen: {max_heat:.3f})")
        if diag is not None:
            diag["reason"] = "below_threshold"
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
        if diag is not None:
            diag["reason"] = "area_cap"
            diag["area"] = area
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

    Returns (latent, note): the refined latent dict plus a run_mechanisms note.
    Two different "nothing happened" shapes, both same-object no-ops:
      - (latent, None): the pass is effectively OFF (no targets, no upsampler,
        strength 0) - not worth a note on every scene.
      - (latent, "segmented_detail(no match: ...)"): the pass RAN and found
        nothing - always reported, with the max CLIPSeg score seen, because a
        miss is otherwise indistinguishable from "nothing to detail here" when
        it's actually just the threshold being wrong for this content.
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
    target_str = ", ".join(targets)

    # 1) Detect on a few decoded keyframes.
    frames = _decode_detection_frames(vae, video, DETECTION_FRAMES, debug=debug)
    heat = _clipseg_heat(frames, targets, debug=debug)
    if heat is None:
        return latent, f"segmented_detail(no match: '{target_str}' - no keyframe decoded)"
    diag = {}
    tube = find_tube(heat, lat_h, lat_w, threshold=threshold, debug=debug, diag=diag)
    if tube is None:
        if diag.get("reason") == "area_cap":
            note = (f"segmented_detail(no match: '{target_str}' region covers "
                    f"{diag['area']:.0%} of frame > {MAX_TUBE_AREA:.0%} cap - refused, "
                    "not a detail pass at that size)")
        else:
            note = (f"segmented_detail(no match: '{target_str}', max CLIPSeg score "
                    f"{diag.get('max_heat', 0.0):.2f} < threshold {threshold:.2f} - "
                    "try a lower detail_threshold or a different target word)")
        return latent, note
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
