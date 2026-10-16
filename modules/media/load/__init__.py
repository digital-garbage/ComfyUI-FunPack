"""Media loading.

A node-only module: no settings, no panel, no mount. It exists to contribute one
ComfyUI node, which is why `SETTINGS` is absent rather than empty.
"""

from .nodes import FunPackLoadMedia

ID = "media_load"
TITLE = "Media loader"
STAGE = "load"
CATEGORY = "system"
STATUS = "proven"

NODES = [FunPackLoadMedia]
