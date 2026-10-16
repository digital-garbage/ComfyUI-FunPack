"""Valid, but wants a trait the current model may not have."""

ID = "needs_trait"
TITLE = "Needs a trait"
MOUNT = "generation.model"
REQUIRES = ["cfg_free", "audio_stream"]
SETTINGS = {"enabled": {"type": "bool", "default": True, "label": "On"}}
