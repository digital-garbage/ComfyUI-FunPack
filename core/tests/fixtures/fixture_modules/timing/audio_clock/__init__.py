"""A good module: the shape everything else is measured against."""

ID = "audio_clock"
TITLE = "Audio clock"
MOUNT = "generation.timing"
STAGE = "conditioning"
STATUS = "proven"
REQUIRES = ["audio_stream"]

SETTINGS = {
    "enabled": {
        "type": "bool", "default": True,
        "label": "Sync to audio clock",
        "hint": "Locks frame timing to the audio stream.",
    },
    "strength": {
        "type": "float", "default": 0.65, "min": 0.0, "max": 1.0, "step": 0.05,
        "label": "Lock strength", "unit": "x", "ui": "slider",
        "when": {"enabled": True},
    },
    "mode": {
        "type": "enum", "default": "beat", "label": "Alignment", "ui": "segmented",
        "options": [
            {"value": "beat", "label": "Beat grid"},
            {"value": "onset", "label": "Onset"},
            {"value": "flat", "label": "Flat"},
        ],
        "when": {"enabled": True},
    },
}
