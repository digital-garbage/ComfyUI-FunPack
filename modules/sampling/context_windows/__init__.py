"""Context windows: scenes longer than the model can hold at once.

Nothing is ported: ComfyUI's own `comfy.context_windows` owns the whole mechanism. This builds its
handler and hangs it on the model, the way the stock LTXVContextWindows node does. Core picks it up
inside `calc_cond_batch`, which every sampler already goes through.

The capability gate is the important part. The handler has long existed for plain video, but the
packed audio+video latent needs `BaseModel.map_context_window_to_modalities` to unpack the stream,
map each video window onto its audio window and re-slice the guides. Without it, windowing an LTX
latent would cut the packed tensor blindly and quietly wreck the sound and the guides, so that is
refused, in words, rather than tried.

Core's schedule names put the words the other way round from what this once offered
(`standard_uniform`, not `uniform_standard`); the old spellings are still read, from projects that
saved them.
"""

import inspect

from ..._core import control, patching

ID = "context_windows"
TITLE = "Context windows"
MOUNT = "generation.sampling"
STAGE = "sampling"
CATEGORY = "sampling"
STATUS = "experimental"
REQUIRES = ["ltx_av"]

SCHEDULES = ["standard_uniform", "standard_static", "looped_uniform", "batched"]
FUSES = ["pyramid", "relative", "flat", "overlap-linear"]
_ALIASES = {"uniform_standard": "standard_uniform", "static_standard": "standard_static",
            "uniform_looped": "looped_uniform"}
_TEMPORAL = 8                      # LTX: real frames per latent frame

SETTINGS = {
    "enabled": {"type": "bool", "default": False, "label": "Context windows (long scenes)",
                "hint": "Sample a long scene in overlapping windows so it fits the model's memory."},
    "length": {"type": "int", "default": 145, "min": 9, "max": 2049, "step": 8, "label": "Window length (frames)",
               "when": {"enabled": True}},
    "overlap": {"type": "int", "default": 40, "min": 0, "max": 512, "step": 8, "label": "Window overlap (frames)",
                "when": {"enabled": True}},
    "schedule": {"type": "enum", "default": "standard_uniform", "label": "Window schedule",
                 "options": [{"value": v, "label": v} for v in SCHEDULES], "when": {"enabled": True}},
    "fuse": {"type": "enum", "default": "pyramid", "label": "Window blend",
             "options": [{"value": v, "label": v} for v in FUSES], "when": {"enabled": True}},
    "freenoise": {"type": "bool", "default": True, "label": "FreeNoise blending", "when": {"enabled": True},
                  "hint": "Shuffle the starting noise so neighbouring windows share it."},
    "retain_first": {"type": "bool", "default": False, "label": "Pin anchor in every window",
                     "when": {"enabled": True}},
}


def _cleanup(patcher, key):
    """A handler left on this model by an earlier install of ours (a clone carries it forward)."""
    options = patcher.model_options
    if patching._ours(options.get("context_handler"), key):
        options.pop("context_handler", None)


def install(patcher, values, key):
    _cleanup(patcher, key)
    if not values.get("enabled"):
        return None
    try:
        import comfy.context_windows as cw
        import comfy.patcher_extension as pe
    except ImportError:
        raise control.Unavailable("this ComfyUI has no context-window support (needs ComfyUI >= v0.29.0)")
    if not hasattr(getattr(patcher, "model", None), "map_context_window_to_modalities"):
        raise control.Unavailable("this ComfyUI cannot window an audio+video latent (it lacks "
                           "map_context_window_to_modalities); update ComfyUI")

    schedule = _ALIASES.get(str(values.get("schedule")), str(values.get("schedule")))
    try:
        sched = cw.get_matching_context_schedule(schedule)
    except ValueError:
        raise control.Unavailable(f"schedule {schedule!r} is not one this ComfyUI knows")
    try:
        fuse = cw.get_matching_fuse_method(str(values.get("fuse")))
    except ValueError:
        raise control.Unavailable(f"blend {values.get('fuse')!r} is not one this ComfyUI knows")

    latent_len = max(((int(values["length"]) - 1) // _TEMPORAL) + 1, 1)
    overlap = max(int(values["overlap"]) // _TEMPORAL, 0)
    clamped = overlap >= latent_len                       # core's schedules loop forever / return no windows
    overlap = min(overlap, latent_len - 1)
    retain = "0" if values.get("retain_first") else ""
    kwargs = dict(context_schedule=sched, fuse_method=fuse, context_length=latent_len,
                  context_overlap=overlap, context_stride=1,
                  closed_loop=False, dim=2, freenoise=bool(values.get("freenoise")),
                  cond_retain_index_list=retain, latent_retain_index_list=retain,
                  split_conds_to_windows=False)
    accepted = set(inspect.signature(cw.IndexListContextHandler).parameters)
    dropped = [k for k in kwargs if k not in accepted]       # this core's handler lacks some keywords
    for k in dropped:
        kwargs.pop(k)
    handler = patching.tag(cw.IndexListContextHandler(**kwargs), key)
    patcher.model_options["context_handler"] = handler
    patcher.add_wrapper_with_key(pe.WrappersMP.PREPARE_SAMPLING, f"{key}.prepare", cw._prepare_sampling_wrapper)
    if values.get("freenoise"):
        patcher.add_wrapper_with_key(pe.WrappersMP.SAMPLER_SAMPLE, f"{key}.freenoise", cw._sampler_sample_wrapper)
    note = f"windows of {values['length']} frames, {values['overlap']} overlap ({schedule}, {values['fuse']})"
    if clamped:
        note = (f"windows of {values['length']} frames, overlap cut to {overlap * _TEMPORAL} "
                f"(it must be shorter than the window) ({schedule}, {values['fuse']})")
    if retain and "latent_retain_index_list" in dropped:
        note += "; this ComfyUI cannot pin the anchor in the latent, only in the conditioning"
    return note


PROVIDES = {"modifier": install}
