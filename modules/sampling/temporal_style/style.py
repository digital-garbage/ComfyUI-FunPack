"""How time is felt: the frame rate the model THINKS it is making.

LTX positions every frame in time from the frame rate in its conditioning. Telling it a higher rate
for the same frames means smaller steps between frames (smoother, more held motion); a lower one means
larger steps (punchier). The names track the clock, not the subject.
"""

# frame_rate multiplier per style; >1 holds, <1 punches.
MULT = {"accelerate": 1.35, "decelerate": 0.72, "freeze": 2.0}

# Pulse: repeated ease-down segments, each starting punchy and easing to a hold.
PULSE_SEGMENTS, PULSE_PEAK, PULSE_FLOOR = 3, 0.88, 1.65

# Rapid start/end: a punchy multiplier over the first and/or last slice of the denoise, natural elsewhere.
RAPID_MULT, RAPID_FRACTION = 0.65, 0.35


def _ease(t):
    t = max(0.0, min(1.0, float(t)))
    return t * t * (3.0 - 2.0 * t)


def pulse_mult(progress, segments=PULSE_SEGMENTS, peak=PULSE_PEAK, floor=PULSE_FLOOR):
    progress = max(0.0, min(1.0, float(progress)))
    index = min(int(progress * segments), segments - 1)
    return peak + (floor - peak) * _ease(progress * segments - index)


def rapid_mult(progress, mode, fraction=RAPID_FRACTION, mult=RAPID_MULT):
    fraction = max(1e-3, min(0.5, float(fraction)))
    progress = max(0.0, min(1.0, float(progress)))
    if mode in ("rapid_start", "rapid_start_end") and progress < fraction:
        return mult + (1.0 - mult) * _ease(progress / fraction)
    if mode in ("rapid_end", "rapid_start_end") and progress > 1.0 - fraction:
        return mult + (1.0 - mult) * _ease((1.0 - progress) / fraction)
    return 1.0


def scale_frame_rate(args, mult):
    """`args` with the conditioning's frame_rate times `mult`; unchanged when it has none."""
    c = args.get("c")
    if not (isinstance(c, dict) and "frame_rate" in c):
        return args
    cond = c["frame_rate"]
    if not hasattr(cond, "cond"):
        return args
    scaled = dict(c)
    scaled["frame_rate"] = type(cond)(float(cond.cond) * float(mult))
    return {**args, "c": scaled}


def progress_of(args, state):
    """How far through the denoise this call is, 0..1: from the schedule when the sampler published
    it, else from how far sigma has fallen since this run's first call."""
    from ..._core import dit_hooks
    where = dit_hooks.current_step((args.get("c") or {}).get("transformer_options"))
    if where is not None:
        index, total = where
        return index / max(1, total - 1) if total > 1 else 1.0
    ts = args.get("timestep")
    try:
        sigma = float(ts.max().item()) if ts is not None else 1.0
    except Exception:                                  # noqa: BLE001
        sigma = 1.0
    if state.get("start") is None or sigma > state.get("last", 0.0) + 1e-4:
        state["start"] = max(sigma, 1e-6)              # sigma went back up: a new run
    state["last"] = sigma
    return max(0.0, min(1.0, 1.0 - sigma / state["start"]))


def make_wrapper(old, mult_for):
    """A model_function_wrapper that scales the frame rate by `mult_for(args)`, then runs `old` (the
    wrapper that was there) or the model."""
    def wrapper(apply_fn, args):
        mult = mult_for(args)
        if abs(mult - 1.0) > 1e-3:
            args = scale_frame_rate(args, mult)
        if old is not None:
            return old(apply_fn, args)
        return apply_fn(args["input"], args["timestep"], **args.get("c", {}))
    return wrapper
