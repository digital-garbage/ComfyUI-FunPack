import logging

import comfy.context_windows


_DIM = 2
_FALLBACK_CONTEXT_LENGTH = 16


def _make_auto_schedule():
    """One isolated window per scene - no overlap, no stride, no carryover between scenes.

    Each scene segment covers exactly its slice of frames. The character description
    in the conditioning is what keeps the subject consistent, not temporal bleed.
    """
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
            context_overlap=0,   # scenes are fully isolated - no overlap
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

        A usable temporal dim has size > 1 and <= 4096.
        Packed/flat formats like LTXAV combine all tokens into millions of elements
        and have no single axis that maps to frame count - those return None.

        Search order: 2 (standard [B,C,T,H,W]), 1, 0 (LTXAV packs frames at dim 0),
        then any remaining dims.
        """
        ndim = x_in.dim()
        priority = [d for d in [2, 1, 0] if d < ndim]
        rest = [d for d in range(ndim) if d not in priority]
        for d in priority + rest:
            size = int(x_in.size(d))
            if 1 < size <= 4096:
                return d
        return None

    def should_use_context(self, model, conds, x_in, timestep, model_options):
        # Detect a usable temporal dimension. Packed flat formats (e.g. LTXAV) produce
        # sub-tensors that cannot be unpacked by the model - skip windowing entirely.
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

        # Detect how many conditioning entries Refiner/Studio produced.
        # conds = [uncond_list, cond_list, ...] - we read the MAX across all groups
        # because conds[0] is the negative conditioning which always has 1 entry.
        try:
            num_windows = max((len(c) for c in conds if c), default=1)
        except Exception:
            num_windows = 1
        self.auto_num_windows = num_windows

        # Set context_length to segment size (total / N) for accurate VRAM estimation
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
                        "Latent frame count of your video. Used for VRAM estimation and to "
                        "pre-size windows before sampling begins. Leave 0 to auto-detect from "
                        "the latent at sampling time."
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
        "with 'Split by Transitions' enabled. Window count is derived from the number of "
        "conditioning entries at sampling time - one fully isolated window per scene, "
        "no overlap, no temporal carryover. Character consistency is handled by the "
        "shared character description in each window's conditioning."
    )

    def apply(self, model, total_frames=0):
        model = model.clone()
        handler = FunPackTransitionContextHandler(auto_total_frames=total_frames)
        model.model_options["context_handler"] = handler
        comfy.context_windows.create_prepare_sampling_wrapper(model)
        return (model,)
