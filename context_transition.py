import logging

import comfy.context_windows


TRANSITION_MODES = ["off", "balanced", "aggressive", "custom"]
CONTEXT_SCHEDULES = [
    comfy.context_windows.ContextSchedules.STATIC_STANDARD,
    comfy.context_windows.ContextSchedules.UNIFORM_STANDARD,
    comfy.context_windows.ContextSchedules.UNIFORM_LOOPED,
    comfy.context_windows.ContextSchedules.BATCHED,
]
FUSE_METHODS = comfy.context_windows.ContextFuseMethods.LIST_STATIC


def _clamp_float(value, minimum, maximum):
    return max(minimum, min(maximum, float(value)))


def _clamp_int(value, minimum, maximum):
    return max(minimum, min(maximum, int(value)))


def _resolve_transition_options(mode, start_pct, count, spacing_pct, strength):
    mode = (mode or "off").lower()
    if mode not in TRANSITION_MODES:
        mode = "off"

    if mode == "off":
        return {
            "enabled": False,
            "mode": mode,
            "start_pct": 0.30,
            "count": 0,
            "spacing_pct": 0.22,
            "strength": 0.0,
        }

    start_pct = _clamp_float(start_pct, 0.0, 0.95)
    count = _clamp_int(count, 1, 8)
    spacing_pct = _clamp_float(spacing_pct, 0.02, 0.50)
    strength = _clamp_float(strength, 0.0, 1.0)

    if mode == "balanced":
        count = min(count, 1)
        strength = min(max(strength, 0.60), 0.75)
    elif mode == "aggressive":
        count = max(count, 2)
        strength = max(strength, 0.90)

    return {
        "enabled": True,
        "mode": mode,
        "start_pct": start_pct,
        "count": count,
        "spacing_pct": spacing_pct,
        "strength": strength,
    }


def _resolve_transition_boundaries(num_frames, options):
    if not options["enabled"] or num_frames <= 1:
        return []

    boundaries = []
    for transition_index in range(options["count"]):
        pct = options["start_pct"] + options["spacing_pct"] * transition_index
        if pct >= 1.0:
            break

        boundary = int(round(pct * max(1, num_frames - 1)))
        boundary = _clamp_int(boundary, 1, num_frames - 1)
        if boundary not in boundaries:
            boundaries.append(boundary)

    boundaries.sort()
    return boundaries


def _offset_windows(local_windows, offset, num_frames):
    shifted_windows = []
    for window in local_windows:
        shifted = [idx + offset for idx in window]
        shifted = [idx for idx in shifted if 0 <= idx < num_frames]
        if shifted:
            shifted_windows.append(shifted)
    return shifted_windows


def _make_segmented_context_schedule(base_schedule_name, transition_options):
    base_schedule = comfy.context_windows.get_matching_context_schedule(base_schedule_name)
    schedule_name = f"funpack_transition_{base_schedule.name}"

    def segmented_schedule(num_frames, handler, model_options):
        boundaries = _resolve_transition_boundaries(num_frames, transition_options)
        if not boundaries:
            return base_schedule.func(num_frames, handler, model_options)

        isolation = transition_options["strength"]
        boundary_bleed = int(round(handler.context_overlap * (1.0 - isolation)))
        cuts = [0] + boundaries + [num_frames]
        windows = []
        seen = set()

        for segment_index in range(len(cuts) - 1):
            segment_start = cuts[segment_index]
            segment_end = cuts[segment_index + 1]

            if boundary_bleed > 0:
                if segment_index > 0:
                    segment_start = max(0, segment_start - boundary_bleed)
                if segment_index < len(cuts) - 2:
                    segment_end = min(num_frames, segment_end + boundary_bleed)

            segment_length = segment_end - segment_start
            if segment_length <= 0:
                continue

            local_windows = base_schedule.func(segment_length, handler, model_options)
            for window in _offset_windows(local_windows, segment_start, num_frames):
                key = tuple(window)
                if key in seen:
                    continue
                seen.add(key)
                windows.append(window)

        if not windows:
            return base_schedule.func(num_frames, handler, model_options)

        logging.info(
            "FunPack context transitions at latent indexes %s with %.2f isolation.",
            boundaries,
            isolation,
        )
        return windows

    return comfy.context_windows.ContextSchedule(schedule_name, segmented_schedule)


class FunPackTransitionContextHandler(comfy.context_windows.IndexListContextHandler):
    def __init__(self, *args, transition_options=None, auto_total_frames=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.transition_options = transition_options or _resolve_transition_options(
            "off", 0.30, 0, 0.22, 0.0
        )
        self.auto_total_frames = int(auto_total_frames or 0)

    def _auto_resize_context_length(self, conds):
        """When total_frames is set, compute context_length from the number of conditioning entries."""
        if self.auto_total_frames <= 0 or not self.split_conds_to_windows:
            return
        # conds is [[pos_entry1, pos_entry2, ...], [neg_entry1, ...]]
        try:
            num_windows = len(conds[0]) if conds and conds[0] else 1
        except Exception:
            num_windows = 1
        if num_windows > 1:
            self.context_length = max(1, self.auto_total_frames // num_windows)
            logging.info(
                "FunPack auto context_length=%s from total_frames=%s / %s windows.",
                self.context_length,
                self.auto_total_frames,
                num_windows,
            )

    def should_use_context(self, model, conds, x_in, timestep, model_options):
        self._auto_resize_context_length(conds)

        if super().should_use_context(model, conds, x_in, timestep, model_options):
            return True

        frame_count = x_in.size(self.dim)
        if _resolve_transition_boundaries(frame_count, self.transition_options):
            logging.info(
                "Using FunPack transition context windows for %s latent frames.",
                frame_count,
            )
            return True

        return False


class FunPackContextTransitionWindows:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "The model to wrap with transition-aware context windows."
                }),
                "transition_mode": (TRANSITION_MODES, {
                    "default": "aggressive",
                    "tooltip": "Off uses ordinary context windows. Balanced/aggressive/custom split context windows at transition points."
                }),
                "context_length": ("INT", {
                    "default": 16,
                    "min": 1,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "Latent-frame length of each context window. Use a value smaller than the clip length to force windowed denoising."
                }),
                "context_overlap": ("INT", {
                    "default": 4,
                    "min": 0,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "Latent-frame overlap between neighboring context windows inside each transition segment."
                }),
                "context_schedule": (CONTEXT_SCHEDULES, {
                    "default": comfy.context_windows.ContextSchedules.STATIC_STANDARD,
                    "tooltip": "Base context window schedule borrowed from Comfy's manual context windows."
                }),
                "fuse_method": (FUSE_METHODS, {
                    "default": comfy.context_windows.ContextFuseMethods.PYRAMID,
                    "tooltip": "How overlapping context-window results are blended inside each segment."
                }),
                "dim": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 5,
                    "step": 1,
                    "tooltip": "Tensor dimension to window. For LTX/WAN-like video latents this is usually 2."
                }),
                "transition_start_pct": ("FLOAT", {
                    "default": 0.30,
                    "min": 0.0,
                    "max": 0.95,
                    "step": 0.01,
                    "tooltip": "Clip progress where the first context reset begins. 0.0 means immediately after latent frame 0."
                }),
                "transition_count": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 8,
                    "step": 1,
                    "tooltip": "How many context reset boundaries to create."
                }),
                "transition_spacing_pct": ("FLOAT", {
                    "default": 0.22,
                    "min": 0.02,
                    "max": 0.50,
                    "step": 0.01,
                    "tooltip": "Clip progress spacing between context reset boundaries."
                }),
                "transition_strength": ("FLOAT", {
                    "default": 0.85,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "How strongly transition boundaries block context carryover. 1.0 is a hard split; lower values allow a small overlap bleed."
                }),
            },
            "optional": {
                "context_stride": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 16,
                    "step": 1,
                    "tooltip": "Stride for uniform context schedules."
                }),
                "closed_loop": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Whether looped schedules may wrap around the clip."
                }),
                "freenoise": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use Comfy's FreeNoise wrapper for context-window noise shuffling."
                }),
                "total_frames": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 16384,
                    "step": 1,
                    "tooltip": "When > 0 and split_conds_to_windows is enabled, context_length is auto-computed as total_frames divided by the number of conditioning entries from Refiner/Studio. Connect your latent frame count or leave 0 to set context_length manually."
                }),
                "cond_retain_index_list": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated latent indexes to retain in conditioning tensors for every window. Leave empty for transitions so frame 0 does not over-anchor later segments."
                }),
                "split_conds_to_windows": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "When Refiner or Studio outputs multiple conditioning entries (via Split by Transitions), route each entry to its corresponding context window region."
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "FunPack/Sampling"
    DESCRIPTION = "Manual context windows with transition boundaries that reduce temporal carryover across selected parts of a video clip."

    def apply(self, model, transition_mode, context_length, context_overlap,
              context_schedule, fuse_method, dim, transition_start_pct,
              transition_count, transition_spacing_pct, transition_strength,
              total_frames=0, context_stride=1, closed_loop=False, freenoise=False,
              cond_retain_index_list="", split_conds_to_windows=False):
        transition_options = _resolve_transition_options(
            transition_mode,
            transition_start_pct,
            transition_count,
            transition_spacing_pct,
            transition_strength,
        )
        schedule = _make_segmented_context_schedule(context_schedule, transition_options)
        model = model.clone()
        model.model_options["context_handler"] = FunPackTransitionContextHandler(
            context_schedule=schedule,
            fuse_method=comfy.context_windows.get_matching_fuse_method(fuse_method),
            context_length=context_length,
            context_overlap=context_overlap,
            context_stride=context_stride,
            closed_loop=closed_loop,
            dim=dim,
            freenoise=freenoise,
            cond_retain_index_list=cond_retain_index_list,
            split_conds_to_windows=split_conds_to_windows,
            transition_options=transition_options,
            auto_total_frames=total_frames,
        )
        comfy.context_windows.create_prepare_sampling_wrapper(model)
        if freenoise:
            comfy.context_windows.create_sampler_sample_wrapper(model)
        return (model,)
