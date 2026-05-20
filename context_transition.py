import logging

import comfy.context_windows


_BASE_SCHEDULE = comfy.context_windows.ContextSchedules.STATIC_STANDARD
_FUSE_METHOD = comfy.context_windows.ContextFuseMethods.PYRAMID
_DIM = 2
_CONTEXT_OVERLAP = 4
_FALLBACK_CONTEXT_LENGTH = 16


def _offset_windows(local_windows, offset, num_frames):
    shifted = []
    for window in local_windows:
        w = [idx + offset for idx in window if 0 <= idx + offset < num_frames]
        if w:
            shifted.append(w)
    return shifted


def _make_auto_schedule():
    """Schedule that reads handler.auto_num_windows at call time to place equal-width boundaries."""
    base = comfy.context_windows.get_matching_context_schedule(_BASE_SCHEDULE)

    def auto_schedule(num_frames, handler, model_options):
        num_windows = getattr(handler, "auto_num_windows", 1)
        if num_windows <= 1 or num_frames <= 1:
            return base.func(num_frames, handler, model_options)

        # Equal-width hard cuts between windows
        boundaries = [
            int(round(num_frames * i / num_windows))
            for i in range(1, num_windows)
        ]
        boundaries = [b for b in boundaries if 0 < b < num_frames]
        if not boundaries:
            return base.func(num_frames, handler, model_options)

        cuts = [0] + boundaries + [num_frames]
        windows = []
        seen = set()
        for i in range(len(cuts) - 1):
            seg_start, seg_end = cuts[i], cuts[i + 1]
            seg_len = seg_end - seg_start
            if seg_len <= 0:
                continue
            for w in _offset_windows(base.func(seg_len, handler, model_options), seg_start, num_frames):
                key = tuple(w)
                if key not in seen:
                    seen.add(key)
                    windows.append(w)

        if not windows:
            return base.func(num_frames, handler, model_options)

        logging.info(
            "FunPack context windows: %s segments, boundaries at %s, context_length=%s.",
            num_windows, boundaries, handler.context_length,
        )
        return windows

    return comfy.context_windows.ContextSchedule("funpack_auto", auto_schedule)


_AUTO_SCHEDULE = _make_auto_schedule()


class FunPackTransitionContextHandler(comfy.context_windows.IndexListContextHandler):
    def __init__(self, auto_total_frames=0):
        super().__init__(
            context_schedule=_AUTO_SCHEDULE,
            fuse_method=comfy.context_windows.get_matching_fuse_method(_FUSE_METHOD),
            context_length=max(1, int(auto_total_frames) // 2) if auto_total_frames else _FALLBACK_CONTEXT_LENGTH,
            context_overlap=_CONTEXT_OVERLAP,
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
        """Return the first dim that looks like a temporal frame axis.

        A temporal dim has a size that is:
        - greater than context_overlap (otherwise nothing meaningful to window)
        - at most 4096 (packed flat formats like LTXAV combine all tokens and reach millions)

        Search order: 2 (standard [B,C,T,H,W]), 1, 0 (LTXAV packs frames at dim 0),
        then any remaining dims. Returns None when no suitable dim is found.
        """
        min_size = max(2, self.context_overlap + 1)
        ndim = x_in.dim()
        priority = [d for d in [2, 1, 0] if d < ndim]
        rest = [d for d in range(ndim) if d not in priority]
        for d in priority + rest:
            size = int(x_in.size(d))
            if min_size < size <= 4096:
                return d
        return None

    def should_use_context(self, model, conds, x_in, timestep, model_options):
        # Detect a usable temporal dimension. Packed flat formats (e.g. LTXAV) have
        # no suitable dim and must be skipped - windowing over a flat packed sequence
        # produces sub-tensors that cannot be unpacked by the model.
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

        # Detect how many conditioning entries Refiner/Studio produced
        try:
            num_windows = max(1, len(conds[0])) if conds and conds[0] else 1
        except Exception:
            num_windows = 1
        self.auto_num_windows = num_windows

        # Auto-size context_length to fit windows evenly
        if num_windows > 1:
            total = self.auto_total_frames if self.auto_total_frames > 0 else int(x_in.size(self.dim))
            self.context_length = max(1, total // num_windows)

        return super().should_use_context(model, conds, x_in, timestep, model_options) or num_windows > 1


class FunPackContextTransitionWindows:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "The model to wrap with per-scene context windows.",
                }),
            },
            "optional": {
                "total_frames": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 16384,
                    "step": 1,
                    "tooltip": (
                        "Latent frame count of your video. When set, window size is "
                        "auto-computed as total_frames / number_of_conditioning_entries. "
                        "Leave 0 to derive the count from the actual latent at sampling time."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "FunPack/Sampling"
    DESCRIPTION = (
        "Wraps a model with per-scene context windows. Connect after Refiner V2 or Studio "
        "with 'Split by Transitions' enabled. Window count and size are derived automatically "
        "from the number of conditioning entries at sampling time. Only total_frames needs to "
        "be set (optional - leave 0 to auto-detect from the latent)."
    )

    def apply(self, model, total_frames=0):
        model = model.clone()
        handler = FunPackTransitionContextHandler(auto_total_frames=total_frames)
        model.model_options["context_handler"] = handler
        comfy.context_windows.create_prepare_sampling_wrapper(model)
        return (model,)
