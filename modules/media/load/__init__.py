"""Media loading.

A node-only module: no settings, no panel, no mount. It exists to contribute the
ComfyUI nodes that load a library file, which is why `SETTINGS` is absent rather than empty.
"""

from .nodes import FunPackLoadMedia, FunPackLoadMediaAudio, FunPackLoadMediaVideo

ID = "media_load"
TITLE = "Media loader"
STAGE = "load"
CATEGORY = "system"
STATUS = "proven"

NODES = [FunPackLoadMedia, FunPackLoadMediaVideo, FunPackLoadMediaAudio]
