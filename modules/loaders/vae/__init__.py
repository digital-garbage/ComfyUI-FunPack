"""VAE loader.

A node-only module: no settings, no panel, no mount. It exists to contribute one
ComfyUI node, which is why `SETTINGS` is absent rather than empty.
"""

from .nodes import FunPackVAELoader

ID = "loader_vae"
TITLE = "VAE loader"
STAGE = "load"
STATUS = "proven"

NODES = [FunPackVAELoader]
