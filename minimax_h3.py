"""MiniMax-H3 adapter — the one place that knows how H3 differs from LTXAV.

FunPack was built against LTX2.3/LTXAV. H3 (Comfy-Org/ComfyUI#15224) is close
enough that most of the stack works untouched, and different in four ways that
break silently if nobody accounts for them:

1. **Audio latent axis.** LTXAV's audio latent is ``[B, C, T, freq]`` — time on
   dim 2, the same axis as the video latent's frames, which is why the Chain
   Sampler could slice every stream at ``[:, :, a:b]``. H3's audio latent is
   ``[B, 32, stereo=2, T]`` — dim 2 is the STEREO CHANNEL. Slicing it as time
   silently crops one speaker instead of trimming the tail. Both are 4-D, so the
   tensor alone cannot tell you which; it has to come from the model.

2. **Frame grid.** LTXAV is ``8k+1`` pixel frames -> ``k+1`` latent frames. H3 is
   ``17k+5`` -> ``5k+2`` (and the VAE reports ``downscale_index_formula = (4,16,16)``,
   which is the *index* map, not the count map — reading a frame count off it is
   off by ~20%). The count always lives in ``vae.downscale_ratio[0]``.

3. **No cross-attention.** H3 is a single packed self-attention stream
   ``[text | cond/ref | audio | video]``. There is no ``attn2``, no ``audio_attn2``,
   and no audio text-channel half — so the LTXAV conditioning channel split is a
   no-op here (correctly: the whole tensor conditions both streams).

4. **Batch size 1.** ``MiniMaxH3Model._forward`` raises on ``x.shape[0] != 1``,
   and comfy batches cond+uncond into one forward whenever ``cfg != 1.0``. Left
   alone that makes every CFG generation crash; :func:`install_batch_split`
   splits the batch back apart around the model call.

Nothing here imports the H3 core modules at module scope — the PR is not in
ComfyUI master yet, so every entry point degrades to "not H3" when it is absent.
"""
from __future__ import annotations

import math

import torch

# ── the model's own constants (comfy/ldm/minimax/model.py, nodes_minimax_h3.py) ──
IMAGE_MODEL = "minimax_h3"
FPS = 24                     # H3 generates at a fixed 24 fps
FRAME_GRID = 17              # pixel frames per grid step ...
FRAME_BASE = 5               # ... on top of this base (frame_count % 17 == 5)
LATENT_GRID = 5              # latent frames per grid step ...
LATENT_BASE = 2              # ... on top of this base
AUDIO_LATENT_FPS = 40        # audio latent frames per second
SPATIAL_DOWNSCALE = 16
VIDEO_LATENT_CHANNELS = 24
AUDIO_LATENT_CHANNELS = 32
CANVAS_MULTIPLE = 32         # width/height must be a multiple of this
DEFAULT_SHIFT_VIDEO = 12.0
DEFAULT_SHIFT_AUDIO = 3.0

# Audio rows are [B, 32, stereo, T]: time is the LAST axis, not dim 2.
AUDIO_TIME_DIM = -1
VIDEO_TIME_DIM = 2


# ── detection ────────────────────────────────────────────────────────────────
# All three of these have to work when the H3 PR is not installed at all, so they
# only ever read attributes that any comfy object has.

def _unet_config(model):
    inner = getattr(model, "model", None)
    cfg = getattr(inner, "model_config", None)
    return getattr(cfg, "unet_config", {}) or {}


def is_h3_model(model) -> bool:
    """True for a ModelPatcher (or BaseModel) wrapping the MiniMax H3 DiT."""
    if model is None:
        return False
    try:
        inner = getattr(model, "model", None)
        if type(inner).__name__ == "MiniMaxH3":
            return True
        if type(model).__name__ == "MiniMaxH3":
            return True
        if str(_unet_config(model).get("image_model", "")).lower() == IMAGE_MODEL:
            return True
        diff = getattr(inner, "diffusion_model", None)
        if type(diff).__name__ == "MiniMaxH3Model":
            return True
    except Exception:
        pass
    return False


def is_h3_clip(clip) -> bool:
    """True for the Qwen3-VL-32B conditioning encoder H3 ships with."""
    if clip is None:
        return False
    try:
        tok = getattr(clip, "tokenizer", None)
        if type(tok).__name__ == "MiniMaxH3Tokenizer":
            return True
        # SD1Tokenizer names its sub-tokenizer after the TE; H3's is unique.
        if tok is not None and hasattr(tok, "qwen3vl_32b"):
            return True
        cond_stage = getattr(clip, "cond_stage_model", None)
        if type(cond_stage).__name__ in ("MiniMaxH3TEModel", "MiniMaxH3TEModel_"):
            return True
    except Exception:
        pass
    return False


def is_h3_video_vae(vae) -> bool:
    if vae is None:
        return False
    try:
        return type(getattr(vae, "first_stage_model", None)).__name__ == "MiniMaxH3VideoVAE"
    except Exception:
        return False


def is_h3_audio_vae(vae) -> bool:
    if vae is None:
        return False
    try:
        return type(getattr(vae, "first_stage_model", None)).__name__ == "MiniMaxH3AudioVAE"
    except Exception:
        return False


# ── frame-grid geometry ──────────────────────────────────────────────────────

def align_frame_count(length: int) -> int:
    """Snap a pixel frame count UP to the model's 17k+5 grid (the node's rule)."""
    n = max(FRAME_BASE, int(length))
    while n % FRAME_GRID != FRAME_BASE:
        n += 1
    return n


def video_latent_frames(frame_count: int) -> int:
    """Pixel frames -> video latent frames, for an already-aligned count."""
    n = int(frame_count)
    if n <= FRAME_BASE:
        return LATENT_BASE
    return ((n - FRAME_BASE) // FRAME_GRID) * LATENT_GRID + LATENT_BASE


def audio_latent_frames(frame_count: int) -> int:
    """Pixel frames -> audio latent frames (40 Hz against a 24 fps video clock)."""
    return int(round(int(frame_count) / float(FPS) * AUDIO_LATENT_FPS))


def temporal_shape(length: int):
    """(aligned pixel frames, video latent frames, audio latent frames)."""
    frames = align_frame_count(length)
    return frames, video_latent_frames(frames), audio_latent_frames(frames)


def latent_frames_from_vae(vae, pixel_frames: int):
    """Video latent frame count for `pixel_frames`, asked of the VAE itself.

    ``vae.downscale_ratio[0]`` is the count map every video VAE in comfy carries
    (LTX: ``ceil(n/8)``; H3: ``(n-5)//17*5+2``). Reading it beats hardcoding a
    per-family formula and beats ``downscale_index_formula``, which is the index
    map and gives the wrong count on H3. Returns None when the VAE doesn't
    expose one, so callers can keep their old fallback.
    """
    try:
        ratio = getattr(vae, "downscale_ratio", None)
        if isinstance(ratio, (list, tuple)) and ratio and callable(ratio[0]):
            return max(1, int(ratio[0](max(1, int(pixel_frames)))))
        if isinstance(ratio, (int, float)) and ratio:
            return ((max(1, int(pixel_frames)) - 1) // max(1, int(ratio))) + 1
    except Exception:
        pass
    return None


def adapt_canvas(width: int, height: int, base_short_edge=768, max_pixels=768 * 1344):
    """H3's reference canvas rule: 768 short edge, area-capped, rounded to 32."""
    width = max(1, int(width))
    height = max(1, int(height))
    ratio = width / height
    if ratio >= 1.0:
        nom_w, nom_h = base_short_edge * ratio, float(base_short_edge)
    else:
        nom_w, nom_h = float(base_short_edge), base_short_edge / ratio
    if nom_w * nom_h > max_pixels:
        s = math.sqrt(max_pixels / (nom_w * nom_h))
        nom_w, nom_h = nom_w * s, nom_h * s
    snap = lambda v: max(CANVAS_MULTIPLE, int(round(v / CANVAS_MULTIPLE)) * CANVAS_MULTIPLE)
    return snap(nom_w), snap(nom_h)


# ── per-stream time axes ─────────────────────────────────────────────────────

def stream_time_dims(stream_count: int, h3: bool):
    """Time axis per latent stream, in comfy's pack order (video first).

    LTXAV: every stream carries time on dim 2. H3: video on dim 2, audio on the
    last axis (dim 2 there is the stereo channel). Returned as a plain list so
    callers can index it by stream position without re-deciding the family.
    """
    dims = [VIDEO_TIME_DIM] * max(1, int(stream_count))
    if h3:
        for i in range(1, len(dims)):
            dims[i] = AUDIO_TIME_DIM
    return dims


def time_len(tensor, dim: int) -> int:
    return int(tensor.shape[dim])


def time_slice(tensor, start, end, dim: int):
    idx = [slice(None)] * tensor.dim()
    idx[dim] = slice(start, end)
    return tensor[tuple(idx)]


def set_time_slice(tensor, start, end, value, dim: int):
    idx = [slice(None)] * tensor.dim()
    idx[dim] = slice(start, end)
    tensor[tuple(idx)] = value
    return tensor


def time_cat(tensors, dim: int):
    return torch.cat(tensors, dim=dim if dim >= 0 else tensors[0].dim() + dim)


# ── attention bridge ─────────────────────────────────────────────────────────
# LTX calls ``comfy.ldm.modules.attention.optimized_attention(...)`` through the
# module, so FunPack's temperature / capture / K-V-lock patches take effect by
# rebinding that one symbol. H3 does ``from ... import optimized_attention`` at
# module load, which snapshots the function — rebinding the source module does
# nothing. Any patch that wants to reach H3 has to rebind H3's own copy too.

def attention_patch_targets():
    """Modules whose ``optimized_attention`` symbol must be rebound together.

    Always includes comfy's attention module; includes the H3 DiT module only
    when the PR is installed. Callers save/restore whatever they find.
    """
    targets = []
    try:
        import comfy.ldm.modules.attention as attn_mod
        targets.append(attn_mod)
    except Exception:
        return targets
    try:
        import comfy.ldm.minimax.model as h3_mod
        if hasattr(h3_mod, "optimized_attention"):
            targets.append(h3_mod)
    except Exception:
        pass
    return targets


# H3 calls ``optimized_attention`` with ``skip_reshape=True`` and q/k/v already at
# ``[1, heads, S, head_dim]``. A patch that reads ``v.shape[1]`` as a token count (the
# LTX layout) is reading the HEAD count instead, so anything indexed by token position
# has to check that kwarg before assuming a layout. Temperature is layout agnostic.


# ── batch-size-1 constraint ──────────────────────────────────────────────────

def _split_cond_batch(value, index, batch):
    """Slice one entry of the ``c`` dict down to sample `index` of `batch`."""
    if isinstance(value, torch.Tensor) and value.dim() >= 1 and value.shape[0] == batch:
        return value[index:index + 1]
    if getattr(value, "is_nested", False):
        try:
            import comfy.nested_tensor
            parts = [t[index:index + 1] if t.shape[0] == batch else t for t in value.unbind()]
            return comfy.nested_tensor.NestedTensor(parts)
        except Exception:
            return value
    return value


def install_batch_split(model):
    """Run the H3 DiT one sample at a time when comfy hands it a batched forward.

    ``MiniMaxH3Model._forward`` raises on ``x.shape[0] != 1``. comfy batches the
    positive and negative conds into a single forward whenever ``cfg != 1.0``, so
    without this every CFG>1 generation dies on the first step. Splitting here
    costs exactly what CFG already costs (one forward per cond) and keeps the
    ``cfg`` knob live instead of pinning it to 1.0.

    Returns the previous ``model_function_wrapper`` so the caller can restore it.
    """
    old_wrapper = model.model_options.get("model_function_wrapper")

    def _call(apply_fn, args):
        if old_wrapper is not None:
            return old_wrapper(apply_fn, args)
        return apply_fn(args["input"], args["timestep"], **args.get("c", {}))

    def _batch_split_wrapper(apply_fn, args, _old=old_wrapper):
        x = args.get("input")
        try:
            batch = int(x.shape[0]) if isinstance(x, torch.Tensor) else 1
        except Exception:
            batch = 1
        if batch <= 1:
            return _call(apply_fn, args)
        outs = []
        ts = args.get("timestep")
        c = args.get("c") or {}
        for i in range(batch):
            sub = dict(args)
            sub["input"] = x[i:i + 1]
            if isinstance(ts, torch.Tensor) and ts.shape[0] == batch:
                sub["timestep"] = ts[i:i + 1]
            sub["c"] = {k: _split_cond_batch(v, i, batch) for k, v in c.items()}
            outs.append(_call(apply_fn, sub))
        return torch.cat(outs, dim=0)

    model.model_options["model_function_wrapper"] = _batch_split_wrapper
    return old_wrapper, _batch_split_wrapper


# ── conditioning payload ─────────────────────────────────────────────────────
# H3 carries its visual conditioning on the CONDITIONING dict, not in the latent:
#   minimax_keyframes  — fl2va first/last frame pins (never denoised)
#   minimax_frame_count— pixel length, so the layout can place the last-frame pin
#   minimax_refs       — ref2va identity/style/audio reference blocks
# model_base.MiniMaxH3.extra_conds turns these into the DiT payload.

def _set_values(conditioning, values):
    out = []
    for entry in conditioning or []:
        cond, meta = entry[0], (entry[1] if len(entry) > 1 else {})
        new_meta = dict(meta) if isinstance(meta, dict) else {}
        for key, value in values.items():
            if value is None:
                new_meta.pop(key, None)
            else:
                new_meta[key] = value
        out.append([cond, new_meta])
    return out


def set_keyframes(conditioning, keyframes, frame_count):
    """Attach fl2va keyframe pins. `keyframes` = [{resolved_frame_index, latent}]."""
    if not keyframes:
        return conditioning
    return _set_values(conditioning, {
        "minimax_keyframes": list(keyframes),
        "minimax_frame_count": int(frame_count),
    })


def set_refs(conditioning, ref_blocks):
    """Attach ref2va reference blocks (images / videos / audio)."""
    if not ref_blocks:
        return conditioning
    return _set_values(conditioning, {"minimax_refs": list(ref_blocks)})


def keyframe_indices_supported(frame_index, frame_count):
    """H3's packed layout only places a pin at the FIRST or LAST pixel frame.

    ``PackedLayout`` raises for anything else, so callers must check rather than
    discover it mid-sample. Mid-clip guidance has to go through a reference block
    (identity/style, not frame-pinned) instead.
    """
    idx = int(frame_index)
    return idx == 0 or idx == int(frame_count) - 1


def encode_keyframe(vae, image, width, height, frame_index, crop="disabled"):
    """VAE-encode one pixel frame into an H3 keyframe pin entry.

    `crop` mirrors the reference node: the frame-0 anchor is stretched to the
    canvas (it defines the geometry), a last-frame follower is cover-cropped.
    """
    import comfy.utils
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, int(width), int(height), "lanczos", crop)
    pixels = samples.movedim(1, -1)
    return {"resolved_frame_index": int(frame_index), "latent": vae.encode(pixels)}


def image_ref_block(vae, image, width, height):
    """Encode a pixel image into a ref2va IMAGE block (identity / style reference)."""
    import comfy.utils
    h, w = int(image.shape[1]), int(image.shape[2])
    scale = min(1.0, math.sqrt((int(width) * int(height)) / max(1, w * h)))
    snap = lambda v: max(CANVAS_MULTIPLE, int(round(v / CANVAS_MULTIPLE)) * CANVAS_MULTIPLE)
    tw, th = snap(w * scale), snap(h * scale)
    samples = image[:1, ..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, tw, th, "lanczos", "disabled")
    resized = samples.movedim(1, -1)
    return ({"type": "image", "data": resized},
            {"kind": "image", "latent_h": th // SPATIAL_DOWNSCALE,
             "latent_w": tw // SPATIAL_DOWNSCALE, "latent": vae.encode(resized)})


def audio_ref_block(audio_vae, audio):
    """Encode an AUDIO dict into a ref2va AUDIO block."""
    import torchaudio
    waveform = audio["waveform"]
    sr = int(audio.get("sample_rate", 32000))
    vae_sr = int(getattr(audio_vae, "audio_sample_rate", 32000))
    if sr != vae_sr:
        waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
    z = audio_vae.encode(waveform[:1].movedim(1, -1))
    return ({"type": "audio"},
            {"kind": "audio", "ref_audio_t": int(z.shape[-1]), "audio_latent": z})


def token_tags_length(meta):
    """Length of the ``minimax_token_tags`` run this conditioning carries, or None.

    The DiT walks the tag vector position-by-position across the text span, so a
    conditioning whose token count no longer matches its tags is a hard failure
    (IndexError mid-forward), not a soft one. Anything that changes the token
    count — appending identity tokens, truncating, concatenating prompts — must
    either update the tags or drop the whole feature for H3.
    """
    if not isinstance(meta, dict):
        return None
    tags = meta.get("minimax_token_tags")
    if tags is None:
        return None
    try:
        return int(tags.reshape(-1).shape[0])
    except Exception:
        return None


def tags_match(cond, meta) -> bool:
    """True when `meta`'s token tags still line up with `cond`'s token count."""
    n = token_tags_length(meta)
    if n is None:
        return True  # no tags to invalidate
    try:
        return int(cond.shape[1]) == n
    except Exception:
        return False


def extend_token_tags(meta, added_tokens, tag=1):
    """Grow the tag vector by `added_tokens` positions so appended tokens stay legal.

    Tag 1 is the text modality (0 = video, 2 = audio) — appended context tokens
    are text-like, so they take the text adaLN row.
    """
    n = token_tags_length(meta)
    if n is None or added_tokens <= 0:
        return meta
    out = dict(meta)
    tags = meta["minimax_token_tags"].reshape(-1)
    pad = torch.full((int(added_tokens),), int(tag), dtype=tags.dtype, device=tags.device)
    out["minimax_token_tags"] = torch.cat([tags, pad], dim=0)
    return out
