"""What the timeline offers: clip effects and seam transitions.

Fixed lists, not a file on disk: nothing edits them, and a stored copy only ever
meant an install that predates a preset never saw it (v4 had to merge new
built-ins into the saved file for exactly that reason).
"""

_FRAMES = {"label": "Frames", "default": 16, "min": 1, "max": 120, "step": 1}


def _param(label, default, lo, hi, step):
    return {"label": label, "default": default, "min": lo, "max": hi, "step": step}


EFFECTS = [
    {"id": "zoom_in", "name": "Zoom in (push)", "description": "Timed push-in (set ratio, length, start in inspector)"},
    {"id": "zoom_out", "name": "Zoom out (pull back)", "description": "Timed pull-back (set ratio, length, start in inspector)"},
    {"id": "blur", "name": "Gaussian blur", "param": _param("Strength (0–1)", 0.3, 0, 1, 0.05)},
    {"id": "fade_in", "name": "Fade in", "param": _param("Seconds", 0.5, 0, 10, 0.1)},
    {"id": "fade_out", "name": "Fade out", "param": _param("Seconds", 0.5, 0, 10, 0.1)},
    {"id": "flip_h", "name": "Flip horizontal", "description": "Mirror left-to-right. Apply again to turn it off."},
    {"id": "flip_v", "name": "Flip vertical", "description": "Mirror top-to-bottom. Apply again to turn it off."},
    {"id": "fill_frame", "name": "Fill frame (crop to fit)",
     "description": "Cover the output frame and crop the overflow instead of letterboxing. "
                    "Apply again to go back to letterbox."},
    {"id": "crop", "name": "Crop edges (punch in)", "param": _param("Trim per edge (%)", 10, 0, 40, 1)},
    {"id": "reverse", "name": "Reverse", "description": "Play the clip backwards, audio included. Apply again to turn it off."},
    {"id": "reset", "name": "Remove all effects", "description": "Clear every effect on the selected clip."},
]

VIDEO_TRANSITIONS = [
    {"id": "crossfade", "name": "Crossfade", "type": "crossfade", "param": dict(_FRAMES)},
    {"id": "fadeblack", "name": "Fade to black", "type": "fadeblack", "param": dict(_FRAMES)},
    {"id": "wipeleft", "name": "Wipe left", "type": "wipeleft", "param": dict(_FRAMES)},
    {"id": "wiperight", "name": "Wipe right", "type": "wiperight", "param": dict(_FRAMES)},
]


def payload() -> dict:
    return {"version": 1, "effects": EFFECTS, "video_transitions": VIDEO_TRANSITIONS}
