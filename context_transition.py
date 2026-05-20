import logging

import torch
import comfy.context_windows
import comfy.ldm.modules.attention as _attn_mod


# ---------------------------------------------------------------------------
# Original context-window node (kept for long-video windowed denoising)
# ---------------------------------------------------------------------------

_DIM = 2
_FALLBACK_CONTEXT_LENGTH = 16


def _offset_windows(local_windows, offset, num_frames):
    shifted = []
    for window in local_windows:
        w = [idx + offset for idx in window if 0 <= idx + offset < num_frames]
        if w:
            shifted.append(w)
    return shifted


def _make_auto_schedule():
    base = comfy.context_windows.get_matching_context_schedule(
        comfy.context_windows.ContextSchedules.STATIC_STANDARD
    )

    def auto_schedule(num_frames, handler, model_options):
        num_windows = getattr(handler, "auto_num_windows", 1)
        if num_windows <= 1 or num_frames <= 1:
            return base.func(num_frames, handler, model_options)

        boundaries = [
            int(round(num_frames * i / num_windows))
            for i in range(1, num_windows)
        ]
        boundaries = [b for b in boundaries if 0 < b < num_frames]
        if not boundaries:
            return base.func(num_frames, handler, model_options)

        cuts = [0] + boundaries + [num_frames]
        windows = [
            list(range(cuts[i], cuts[i + 1]))
            for i in range(len(cuts) - 1)
            if cuts[i + 1] > cuts[i]
        ]
        logging.info(
            "FunPack context windows: %s isolated scenes, frame boundaries at %s.",
            num_windows, boundaries,
        )
        return windows or base.func(num_frames, handler, model_options)

    return comfy.context_windows.ContextSchedule("funpack_isolated_scenes", auto_schedule)


_AUTO_SCHEDULE = _make_auto_schedule()


class FunPackTransitionContextHandler(comfy.context_windows.IndexListContextHandler):
    def __init__(self, auto_total_frames=0):
        super().__init__(
            context_schedule=_AUTO_SCHEDULE,
            fuse_method=comfy.context_windows.get_matching_fuse_method(
                comfy.context_windows.ContextFuseMethods.PYRAMID
            ),
            context_length=_FALLBACK_CONTEXT_LENGTH,
            context_overlap=0,
            context_stride=1,
            closed_loop=False,
            dim=_DIM,
            freenoise=False,
            cond_retain_index_list="",
            split_conds_to_windows=True,
        )
        self.auto_total_frames = int(auto_total_frames or 0)
        self.auto_num_windows = 1

    def _detect_temporal_dim(self, x_in):
        ndim = x_in.dim()
        priority = [d for d in [2, 1, 0] if d < ndim]
        rest = [d for d in range(ndim) if d not in priority]
        for d in priority + rest:
            size = int(x_in.size(d))
            if 1 < size <= 4096:
                return d
        return None

    def should_use_context(self, model, conds, x_in, timestep, model_options):
        detected = self._detect_temporal_dim(x_in)
        if detected is None:
            logging.debug(
                "FunPack context windows: x_in shape %s has no suitable temporal dim; skipping.",
                list(x_in.shape),
            )
            return False
        if detected != self.dim:
            logging.info(
                "FunPack context windows: auto-detected temporal dim=%s (shape %s).",
                detected, list(x_in.shape),
            )
            self.dim = detected

        try:
            group_sizes = [len(c) for c in conds if c]
            num_windows = max(group_sizes, default=1)
        except Exception:
            group_sizes = []
            num_windows = 1
        self.auto_num_windows = num_windows
        logging.info(
            "FunPack context windows: cond group sizes=%s, num_windows=%s, dim=%s, x_in=%s.",
            group_sizes, num_windows, self.dim, list(x_in.shape),
        )

        if num_windows > 1:
            total = self.auto_total_frames if self.auto_total_frames > 0 else int(x_in.size(self.dim))
            self.context_length = max(1, total // num_windows)

        return super().should_use_context(model, conds, x_in, timestep, model_options) or num_windows > 1


class FunPackContextTransitionWindows:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model to wrap with per-scene context windows."}),
            },
            "optional": {
                "total_frames": ("INT", {
                    "default": 0, "min": 0, "max": 16384, "step": 1,
                    "tooltip": (
                        "Latent frame count of your video. Used for VRAM estimation. "
                        "Leave 0 to auto-detect from the latent at sampling time."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "FunPack/Sampling"
    DESCRIPTION = (
        "Wraps a model with per-scene context windows (denoising-loop level). "
        "Works for standard LTX but NOT for LTXAV (packed latent format). "
        "For LTXAV, use FunPack Scene Cut Windows instead."
    )

    def apply(self, model, total_frames=0):
        model = model.clone()
        handler = FunPackTransitionContextHandler(auto_total_frames=total_frames)
        model.model_options["context_handler"] = handler
        comfy.context_windows.create_prepare_sampling_wrapper(model)
        return (model,)


# ---------------------------------------------------------------------------
# Scene-cut attention masking: correct approach for LTXAV and standard LTX
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
            "FunPack scene cut: num_scenes=%s detected from conditioning.",
            self.num_scenes,
        )
        return False  # full-pass denoising; masking applied inside each forward


class FunPackSceneCutWindows:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "The model to wrap with scene-boundary attention masking.",
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
        # Use context_handler slot so should_use_context runs each step
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
                # Only mask self-attention (q and k same seq-len = same sequence = video tokens)
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

        if num_scenes <= 1:
            # Single scene: standard noise, same as empty latent into KSampler
            rng = torch.Generator(device=samples.device)
            rng.manual_seed(seed)
            noise = torch.randn(samples.shape, generator=rng,
                                dtype=samples.dtype, device=samples.device)
            return ({"samples": noise},)

        # Find the temporal dimension: the axis with size > 1 and <= 4096
        temporal_dim = None
        for d in [2, 1, 0] + list(range(3, samples.dim())):
            s = int(samples.size(d))
            if 1 < s <= 4096:
                temporal_dim = d
                break

        if temporal_dim is None:
            # Packed format - fall back to single noise
            rng = torch.Generator(device=samples.device)
            rng.manual_seed(seed)
            noise = torch.randn(samples.shape, generator=rng,
                                dtype=samples.dtype, device=samples.device)
            return ({"samples": noise},)

        total_frames = int(samples.size(temporal_dim))
        noise = torch.zeros_like(samples)
        scene_size = total_frames // num_scenes

        for i in range(num_scenes):
            scene_seed = (seed + i * 31337) & 0xffffffffffffffff
            rng = torch.Generator(device=samples.device)
            rng.manual_seed(scene_seed)

            start = i * scene_size
            end = (i + 1) * scene_size if i < num_scenes - 1 else total_frames

            # Build a slice for the temporal dimension
            slices = [slice(None)] * samples.dim()
            slices[temporal_dim] = slice(start, end)
            scene_shape = list(samples.shape)
            scene_shape[temporal_dim] = end - start

            noise[tuple(slices)] = torch.randn(
                scene_shape, generator=rng,
                dtype=samples.dtype, device=samples.device,
            )

        logging.info(
            "FunPack scene noise: %s scenes, temporal_dim=%s, total_frames=%s.",
            num_scenes, temporal_dim, total_frames,
        )
        return ({"samples": noise},)
