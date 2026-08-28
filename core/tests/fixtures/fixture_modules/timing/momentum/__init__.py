"""Declares a relation, so ordering has something to sort."""

ID = "momentum"
TITLE = "Momentum"
MOUNT = "generation.timing"
STAGE = "conditioning"
AFTER = ["audio_clock"]
SETTINGS = {"enabled": {"type": "bool", "default": False, "label": "Smooth motion"}}
