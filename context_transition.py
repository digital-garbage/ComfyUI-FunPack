import logging

import torch
import comfy.ldm.modules.attention as _attn_mod


# ---------------------------------------------------------------------------
# Scene-cut attention masking
# ---------------------------------------------------------------------------

def _scene_split_self_attn(orig_fn, q, k, v, heads, num_scenes, cache, *args, **kwargs):
    """Run self-attention in per-scene chunks so tokens in different scenes never attend each other."""
    seq_len = q.shape[1]
    cache_key = (seq_len, num_scenes, q.device.type)

    if cache_key not in cache:
        tokens_per_scene = max(1, seq_len // num_scenes)
        ids = torch.arange(seq_len, device=q.device)
        cache[cache_key] = (ids // tokens_per_scene).clamp(max=num_scenes - 1)

    scene_ids = cache[cache_key]
    out = torch.zeros_like(q)

    for scene_id in range(num_scenes):
        mask = (scene_ids == scene_id)
        if not mask.any():
            continue
        out[:, mask, :] = orig_fn(
            q[:, mask, :], k[:, mask, :], v[:, mask, :], heads, *args, **kwargs
        )

    return out


class FunPackSceneCutHandler:
    """Reads num_scenes from conditioning on each step; never activates context windows."""

    def __init__(self):
        self.num_scenes = 1

    def should_use_context(self, model, conds, x_in, timestep, model_options):
        try:
            self.num_scenes = max((len(c) for c in conds if c), default=1)
        except Exception:
            self.num_scenes = 1
        logging.info(
            "FunPack scene cut: %s scene(s) detected from conditioning.",
            self.num_scenes,
        )
        return False  # full-pass denoising; masking applied inside each forward


class FunPackSceneCutWindows:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "The model to apply scene-boundary attention masking to.",
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "FunPack/Sampling"
    DESCRIPTION = (
        "Enforces scene cuts by blocking temporal self-attention across scene boundaries. "
        "All frames are denoised in one forward pass (full temporal attention preserved "
        "within each scene). Works for both LTXV and LTXAV. "
        "Connect after Refiner V2 or Studio with 'Split by Transitions' enabled. "
        "Scene count is auto-detected from the number of conditioning entries."
    )

    def apply(self, model):
        model = model.clone()
        attn_mask_cache = {}

        handler = FunPackSceneCutHandler()
        model.model_options["context_handler"] = handler

        old_wrapper = model.model_options.get("model_function_wrapper")

        def _scene_wrapper(apply_fn, args, _old=old_wrapper):
            num_scenes = handler.num_scenes

            if num_scenes <= 1:
                if _old is not None:
                    return _old(apply_fn, args)
                return apply_fn(args["input"], args["timestep"], **args.get("c", {}))

            orig_attn = _attn_mod.optimized_attention
            orig_attn_masked = _attn_mod.optimized_attention_masked

            def _patched(q, k, v, heads, *a, **kw):
                if q.shape[1] == k.shape[1]:
                    return _scene_split_self_attn(orig_attn, q, k, v, heads, num_scenes, attn_mask_cache, *a, **kw)
                return orig_attn(q, k, v, heads, *a, **kw)

            def _patched_masked(q, k, v, heads, mask, *a, **kw):
                if q.shape[1] == k.shape[1]:
                    return _scene_split_self_attn(orig_attn_masked, q, k, v, heads, num_scenes, attn_mask_cache, *a, **kw)
                return orig_attn_masked(q, k, v, heads, mask, *a, **kw)

            _attn_mod.optimized_attention = _patched
            _attn_mod.optimized_attention_masked = _patched_masked
            try:
                if _old is not None:
                    result = _old(apply_fn, args)
                else:
                    result = apply_fn(args["input"], args["timestep"], **args.get("c", {}))
            finally:
                _attn_mod.optimized_attention = orig_attn
                _attn_mod.optimized_attention_masked = orig_attn_masked

            return result

        model.model_options["model_function_wrapper"] = _scene_wrapper
        return (model,)


# ---------------------------------------------------------------------------
# Seeded temporal noise: reinforce scene cuts via distinct noise per region
# ---------------------------------------------------------------------------

class FunPackSceneNoise:
    """Generate initial noise with different seeds per scene temporal region.

    Combined with the full prompt (transition words preserved), the distinct
    noise per region biases LTX to place different scene content at the right
    temporal positions more reliably across runs.

    Connect to KSamplerAdvanced (set add_noise to 'disable') with this node's
    output as latent_image, or pair with FunPack Scene Cut Windows.
    Scene count auto-detected from positive conditioning entries.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Empty/reference latent that defines shape, device, and dtype."}),
                "seed":   ("INT",    {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "positive": ("CONDITIONING", {"tooltip": "Auto-detects scene count from the number of conditioning entries."}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("noise",)
    FUNCTION = "generate"
    CATEGORY = "FunPack/Sampling"

    def generate(self, latent, seed, positive=None):
        samples = latent["samples"]
        num_scenes = max(1, len(positive)) if positive else 1

        def _make_noise(shape, s):
            rng = torch.Generator(device=samples.device)
            rng.manual_seed(s)
            return torch.randn(shape, generator=rng, dtype=samples.dtype, device=samples.device)

        if num_scenes <= 1:
            return ({"samples": _make_noise(samples.shape, seed)},)

        temporal_dim = None
        for d in [2, 1, 0] + list(range(3, samples.dim())):
            if 1 < int(samples.size(d)) <= 4096:
                temporal_dim = d
                break

        if temporal_dim is None:
            return ({"samples": _make_noise(samples.shape, seed)},)

        total_frames = int(samples.size(temporal_dim))
        noise = torch.zeros_like(samples)
        scene_size = total_frames // num_scenes

        for i in range(num_scenes):
            scene_seed = (seed + i * 31337) & 0xffffffffffffffff
            start = i * scene_size
            end = (i + 1) * scene_size if i < num_scenes - 1 else total_frames
            slices = [slice(None)] * samples.dim()
            slices[temporal_dim] = slice(start, end)
            scene_shape = list(samples.shape)
            scene_shape[temporal_dim] = end - start
            noise[tuple(slices)] = _make_noise(scene_shape, scene_seed)

        logging.info(
            "FunPack scene noise: %s scenes, temporal_dim=%s, total_frames=%s.",
            num_scenes, temporal_dim, total_frames,
        )
        return ({"samples": noise},)
